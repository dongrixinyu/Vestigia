# MySQL 指纹与特征表

建表语句位于 [`../sql/create_model_fingerprints.sql`](../sql/create_model_fingerprints.sql)，适用于 MySQL 8.0 及以上版本：

```bash
mysql --default-character-set=utf8mb4 -u <user> -p <database> \
  < sql/create_model_fingerprints.sql
```

该脚本按依赖顺序创建两张表：

- `fingerprint_features`：特征（探针）定义及其版本；
- `model_fingerprints`：模型在某个特征定义版本下采集得到的一份指纹。

所有表和字段均带有 MySQL `COMMENT`，可执行以下命令查看：

```sql
SHOW FULL COLUMNS FROM fingerprint_features;
SHOW FULL COLUMNS FROM model_fingerprints;
SHOW TABLE STATUS LIKE 'fingerprint_features';
```

## 特征表与版本控制

`fingerprint_features` 的一行对应一个**不可变的特征定义版本**，而非只对应一个 `prompt_id`。同一个 `prompt_id` 的 prompt、解析器、检查器、提取字段或稳定性策略发生变化时，必须计算新的 `definition_hash` 并插入新行；旧行保留，将其 `is_active` 设为 `0`，可填写 `retired_at`。不能直接覆盖旧定义，否则历史指纹将无法复现。

应使用规范化 JSON 后的 SHA-256 作为 `definition_hash` 的输入。建议输入至少包括：

```json
{
  "prompt_id": "project_success_score",
  "category": "business_assessment",
  "prompt_variants": ["..."],
  "system_prompt": null,
  "feature_kind": "parsed",
  "feature_field": "parsed.score.value",
  "length_field": "content",
  "parser_reference": "vestigia.prompts.project_success_score.parse",
  "checker_reference": "vestigia.prompts.project_success_score.check",
  "parser_configuration": {},
  "checker_configuration": {},
  "stability_policy": {
    "sample_size": 20,
    "resamples": 1000,
    "seed": 0,
    "max_p95_tv_distance": 0.2,
    "acceptance_distance_type": "total_variation"
  }
}
```

`model_fingerprints.feature_id` 与 `prompt_id` 通过联合外键关联特征表，因此数据库会阻止某份指纹关联到不同 `prompt_id` 的特征定义。导入指纹前，应先创建特征定义并取得 `feature_id`。

## 指纹 JSON 的导入映射

`model_fingerprints` 的一行对应 `fingerprints/` 下的一份 JSON 指纹文件。例如：

```text
fingerprints/qwen-direct/qwen3.7-max/
  qwen3.7-max__project_success_score__b5565cdec4a7ac35__2026-07-29T08-03-47.json
```

| JSON 或路径来源 | 表字段 |
| --- | --- |
| `fingerprints/<vendor>/...` 中的 `<vendor>` | `vendor` |
| 已登记的特征定义 | `feature_id` |
| JSON 的 `model` | `model` |
| JSON 的 `prompt_id` | `prompt_id` |
| JSON 的 `parameters_hash` | `parameters_hash` |
| JSON 的 `prompt` | `prompt` |
| JSON 的 `feature_kind` | `feature_kind` |
| JSON 的 `field` | `feature_field` |
| JSON 的 `request_params` | `request_params`（JSON） |
| JSON 的 `values` | `sample_values`（JSON） |
| JSON 的 `distribution` | `distribution`（JSON） |
| JSON 的 `stability` | `stability`（JSON） |
| JSON 的 `started_at` / `finished_at` | `started_at` / `finished_at` |
| JSON 文件相对仓库的路径 | `source_path` |

`source_path` 是唯一键，重复导入可使用 `INSERT ... ON DUPLICATE KEY UPDATE`。预测查询可利用 `(feature_id, parameters_hash, model)` 或 `(prompt_id, parameters_hash, model)` 索引；按厂商或模型筛选可利用 `(vendor, model)` 索引。

## 稳定性：现有实现与控制项

不能只凭一个 prompt 文本保证特征稳定。稳定性至少取决于：实际 prompt 文本/变体、系统提示词、模型与请求参数、解析规则、特征字段、样本量和阈值。因此它们应与指纹一起被版本化和保存。

### 解析函数和检查函数

- 内置特征定义是 `src/vestigia/prompts/base.py` 的 `PromptTemplate`：`id`、`category`、`variants`、`parser`、`checker`、`system`、`field`、`feature_kind`、`length_field` 都会影响特征含义。
- `parser(response.content)` 将模型文本转成结构化字典；例如 `project_success_score.parse` 只从第一个非空行提取 `[0, 1]` 范围内的分数，`favorite_number.parse` 解析数字。
- `checker(response, parsed)` 只在 CLI 采集流程 `src/vestigia/collect.py` 中记录 `check_passed`；当前 `create_fingerprint()` → `build_model_fingerprint()` 路径会直接把 parser 提取的字段加入分布，**不会以 checker 过滤样本**。因此 checker 规则变更仍应作为定义变更记录，但它目前不改变 workflow 生成的 `values`。
- `field` 是 `resolve_field()` 使用的点分路径，例如 `parsed.score.value`。字段缺失会被显式编码为 JSON 的 `null` 并进入分布，而不是静默丢弃；解析失败比例过高会直接改变分布。

### 稳定性检查函数

`src/vestigia/identify.py::build_model_fingerprint()` 在采样后调用：

```python
validate_stability(
    values,
    sample_size=min(STABILITY_SUBSET_SIZE, len(values)),
    resamples=STABILITY_RESAMPLES,
    seed=STABILITY_SEED,
    max_p95_tv_distance=MAX_P95_TV_DISTANCE,
)
```

对应实现位于 `src/vestigia/validation.py::validate_stability()`：它从完整样本中无放回随机抽取 `resamples` 个大小为 `sample_size` 的子集，分别计算子集与完整经验分布的 TVD 和 JSD；当 **TVD 的 P95 不大于** `max_p95_tv_distance` 时，结果 JSON 的 `stability.reliable` 为 `true`。

当前全局默认值在 `src/vestigia/config.py`：

| 配置 | 当前值 | 作用 |
| --- | ---: | --- |
| `STABILITY_SUBSET_SIZE` | `20` | 每次重采样子集大小；不可大于成功样本数 |
| `STABILITY_RESAMPLES` | `1000` | Monte-Carlo 重采样次数；越大估计越稳定但越慢 |
| `STABILITY_SEED` | `0` | 随机种子；固定可复现，`None` 则每次不同 |
| `MAX_P95_TV_DISTANCE` | `0.20` | 稳定性通过阈值；更小更严格 |
| `LLM_COLLECTION_CONCURRENCY` | `10` | 同批并发请求数；不改变统计公式，但可能影响服务端时间漂移/限流行为 |

模型请求参数也必须固定并保存：`temperature`、`top_p`、`top_k`、`max_tokens`、各类 penalty、`extra_body`、`extra_headers` 以及实际 system prompt 都会影响输出分布。它们被保存到指纹 JSON 的 `request_params`，并参与 `parameters_hash`。特别是 `SYSTEM_PROMPT` 在 `src/vestigia/config.py` 中；改变它会改变模型行为。

## 稳定性实践建议

1. 固定一个特征定义版本后，不要修改其 prompt、parser、checker 或 field；变更即新版本。
2. 使用足够大的 `count`。当前默认采样数是 50，而检查时以最多 20 个样本的子集估计稳定性；低频类别很多时应增加 `count` 和 `sample_size`。
3. 将 `stability.reliable = false` 的指纹标记为不用于严格识别，或单独复采后再决定。
4. 不要只关注 `reliable` 布尔值；保存并比较 `stability.total_variation_distance.p95`、`stability.jensen_shannon_distance.p95`、成功样本数及解析失败/`null` 的占比。
5. 对同一模型、同一特征、同一请求参数在不同时间重复采样，比较跨批次距离。现有 `validate_stability()` 测量的是**单次采样内部子集**稳定性，不是服务端模型版本变化、负载波动或跨时间漂移。
