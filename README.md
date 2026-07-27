# Vestigia

`Vestigia` 是一个可安装的 Python 包，用于通过一组问题采集多个 LLM 的非流式回答，并为后续的模型来源识别（fingerprinting）提供统一数据入口。

这里的“指纹”不是单独的模型名称，也不是单次回答。一个可比较的指纹由以下内容共同确定：

1. **模型身份**：`provider` 与供应商实际接受的 `model` 标识；
2. **固定输入**：完整 user prompt、system instruction、探针版本/措辞；
3. **完整采样配置**：`temperature`、`max_tokens`，以及 `extra_body` 内的 `top_p`、`top_k`、`seed`、`frequency_penalty`、`presence_penalty` 或网关支持的其他生成参数；
4. **缓存策略/协议版本**：响应缓存禁用策略、cache-buster 参数、Anthropic API version；
5. **统计输出特征**：解析后的分类分布，以及按 2 的幂分桶的响应文本长度分布。

因此，`model="claude...", temperature=0.1, top_p=0.9` 与同模型的 `temperature=0.7` 是**两组不同指纹**，必须分别采集、验证和比较；不能混合样本。

当前版本通过 **LiteLLM** 统一调用层：所有供应商、OpenAI-compatible 中转站和 Anthropic Messages 服务均由 LiteLLM 适配；Vestigia 不直接实现或调用供应商 HTTP API。

## 安装

开发安装：

```bash
python -m pip install -e ".[dev]"
```

作为用户安装：

```bash
pip install .
```

需要 Python 3.10 或更新版本。

## 快速开始

### OpenAI Chat Completions 兼容服务

适用于官方 OpenAI、OpenAI-compatible 中转站及多数兼容网关：

```python
from vestigia import LLMClient, LLMConfig

client = LLMClient(LLMConfig(
    provider="openai_compatible",
    base_url="https://api.example.com/v1",
    api_key="your-api-key",
    model="example-model",
))

response = client.complete("用一句话解释什么是模型指纹。")
print(response.text)
print(response.model, response.request_id)
```

客户端通过 LiteLLM 将请求路由至该服务；对于 OpenAI-compatible 配置，使用的 LiteLLM 路由为 `openai/<model>`，API base 为 `base_url`。若已传入完整接口地址，也可使用 `endpoint` 覆盖：

```python
config = LLMConfig(
    provider="openai_compatible",
    base_url="https://gateway.example.com",
    endpoint="https://gateway.example.com/custom/chat/completions",
    api_key="your-api-key",
    model="example-model",
)
```

### Anthropic Messages API

```python
from vestigia import LLMClient, LLMConfig

client = LLMClient(LLMConfig(
    provider="anthropic",
    base_url="https://api.anthropic.com",
    api_key="your-api-key",
    model="claude-sonnet-4-20250514",
    max_tokens=512,
))
response = client.complete("你好")
print(response.text)
```

Anthropic 配置同样通过 LiteLLM 的 `anthropic/<model>` 路由调用；可通过 `api_version` 修改版本，或通过 `extra_headers` 增加网关要求的头。

## 固定题库与批量采集

`vestigia.prompts` 内置了 8 个探针类别。**每一个探针都在 `src/vestigia/prompts/` 中拥有独立模块**，模块提供 `PROMPT`、`parse(response)` 与 `check(response, parsed)`：

```text
prompts/
  favorite_number.py         # 数字抽取与校验
  short_self_description.py
  creative_association.py
  ambiguous_choice.py
  instruction_following.py
  everyday_advice.py
  word_association.py
  simple_explanation.py
```

题目在每个模块内保留多个等义问法；采集时按确定性顺序轮换类别与同类措辞，使不同模型使用完全相同的请求序列比较。`parser` 输出可用于模型特征分析的结构化字段，`checker` 则判定回答是否满足探针要求。

例如 `favorite_number.py` 能从响应中抽取阿拉伯数字和中文数字，输出原始写法、记法和规范化数值；支持如 `7`、`-12.5`、`七`、`一百零二`、`两千零二十`、`负十二点五`。

安装后使用 `vestigia-collect`。下面示例向 OpenAI-compatible 中转站发起 20 次调用，并将每条结果写入 JSONL：

```bash
# Windows PowerShell
$env:LLM_API_KEY = "your-api-key"
vestigia-collect `
  --base-url "https://gateway.example.com/v1" `
  --model "example-model" `
  --count 20 `
  --temperature 0.7 `
  --max-tokens 256 `
  --output samples/example-model-20.jsonl
```

也可以不安装命令入口，直接运行模块：

```bash
python -m vestigia.collect \
  --base-url "https://gateway.example.com/v1" \
  --api-key "your-api-key" \
  --model "example-model" \
  --count 50 \
  --output samples/example-model-50.jsonl
```

常用参数：

- `--count 20`：调用总次数，可改为 `50` 或任意正整数。
- `--endpoint URL`：中转站不是标准 `/chat/completions` 路径时指定完整请求 URL。
- `--provider anthropic`：改用 Anthropic Messages 协议。
- `--system TEXT`：对每次调用追加相同 system instruction。
- `--extra-headers-json '{"HTTP-Referer":"https://example.com"}'`：网关所需的附加请求头。
- `--extra-body-json '{"top_p":0.9}'`：透传额外生成参数；传入值会覆盖客户端生成的同名请求字段。
- `--fail-fast`：默认失败仍记录错误并继续；该选项使其在首次失败时退出。

输出为 UTF-8 JSONL（每行一个请求）。成功记录含 `prompt_id`、`category`、实际 `prompt`、`parsed`（parser 提取的结构化特征）、`check_passed`（checker 检查结果）、最终 `response.text`、服务端模型名、token 用量和请求 ID；失败记录含错误状态码及响应正文，方便后续清洗或重试。

例如，要把 temperature 为 0.1 时“最喜欢的数字”题的回答分布保存为指纹，可固定**同一题目和同一措辞**重复调用：

```bash
vestigia-collect \
  --base-url "https://gateway.example.com/v1" \
  --model "example-model" \
  --prompt-id favorite_number \
  --variant-index 0 \
  --count 20 \
  --temperature 0.1 \
  --max-tokens 64 \
  --output samples/example-number-raw.jsonl \
  --fingerprint-output samples/example-number-fingerprint.json \
  --fingerprint-field parsed.first_number.value
```

原始 JSONL 会保留每一次调用；`--fingerprint-output` 会额外写入聚合后的经验分布。结果会按完整请求签名（模型、协议、题目与原文、temperature、max tokens、system、`extra_body`）隔离，避免把不同配置混在一个指纹中。上述例子中，若 11 条成功结果中 `76` 有 10 次、`34` 有 1 次，输出包含：

```json
{
  "sample_count": 11,
  "field": "parsed.first_number.value",
  "values": [
    {"value": "76", "count": 10, "proportion": 0.9090909090909091},
    {"value": "34", "count": 1, "proportion": 0.09090909090909091}
  ]
}
```

省略 `--prompt-id` 时仍按固定题库轮换；此时输出会为每个题目/措辞/参数组合分别生成分布。对开放题可把 `--fingerprint-field` 设为 `parsed.text`；默认值 `parsed` 会统计整个结构化解析结果。

### 防止网关返回旧响应

Vestigia 默认会在每个请求中发送以下 HTTP 响应缓存控制头：

```http
Cache-Control: no-cache, no-store, max-age=0
Pragma: no-cache
```

它们用于阻止中转网关或代理直接返回此前的完整文本响应；这不会修改 `messages` 或 prompt，因此不会污染分布指纹。默认行为由 `LLMConfig(disable_response_cache=True)` 控制；只有明确需要时才设置为 `False`，或在 CLI 使用 `--allow-response-cache`。

部分不遵守 HTTP 缓存头的网关会以 URL 作为响应缓存键。此时使用每次请求都不同、但不进入模型上下文的查询参数：

```bash
vestigia-collect ... --cache-bust-query-param vestigia_request
```

Python 中：

```python
config = LLMConfig(
    provider="openai_compatible",
    base_url="https://gateway.example.com/v1",
    api_key="...",
    model="example-model",
    cache_bust_query_param="vestigia_request",
)
```

客户端会追加类似 `?vestigia_request=<随机 UUID>` 的 URL 参数。它不会改变请求 body、prompt 或模型采样条件，因此同一分布指纹仍可比较；请求签名也会记录此策略。若网关有私有缓存开关（如请求体 `cache: false`），仍应按其文档通过 `extra_body` 或 `extra_headers` 传入。Anthropic 的显式 Prompt Caching 只有附加 `cache_control` 内容块才会启用；Vestigia 不会添加该字段。

### 指纹稳定性验证与模型比较

仅有“Claude 20 次中 76 出现 10 次”还不足以作为可用指纹：需要先验证该比例在同一模型、同一请求配置下是否稳定。对某模型先采集至少 50 条成功样本，再运行：

```bash
vestigia-validate \
  --input samples/claude-number-50.jsonl \
  --field parsed.first_number.value \
  --sample-size 20 \
  --resamples 1000 \
  --seed 42 \
  --max-p95-tv-distance 0.20 \
  --output samples/claude-number-validation.json
```

该命令会从 50 条记录中随机、**无放回**抽取 1,000 个 20 条子集，并分别与完整 50 条的分布比较。报告给出 total variation（TV）距离和 Jensen-Shannon（JS）距离；二者都是 `0` 表示相同，数值越大表示越不同。默认规则是：20 条子集相对完整样本的 TV 距离第 95 百分位（`p95`）不大于 `0.20`，该特征才标为 `reliable: true`。`--seed` 使这项 Monte-Carlo 检验可复现。

要比较两个已经稳定的模型：

```bash
vestigia-validate \
  --input samples/claude-number-50.jsonl \
  --compare-input samples/kimi-number-50.jsonl \
  --field parsed.first_number.value \
  --sample-size 20 \
  --resamples 1000 \
  --seed 42
```

报告中的 `comparison.between_model.total_variation_distance` 就是两模型的分布差异。例如 Claude 的 `76` 概率约为 50%、Kimi 的 `36` 概率约为 60%，会产生明显的 TV/JS 距离。只有两边都通过稳定性检验，并且模型间 TV 距离大于两边各自的子集波动 `p95`，才会得到 `distinguishable: true`。这避免将采样噪声误判成模型差异。

### Python API：一站式采集、保存和复测

如果只需要提供 URL、API Key、模型名和一个 prompt，可直接使用封装工作流；不需要自己创建 `LLMClient`、调用循环或处理 JSON：

```python
from vestigia import create_fingerprint, load_fingerprint, verify_fingerprint

# 对参考模型重复调用 50 次，计算分布并持久化为 JSON。
reference = create_fingerprint(
    base_url="https://gateway.example.com/v1",
    api_key="reference-api-key",
    model="reference-model",
    prompt="只回答你最喜欢的一个数字。",
    output="fingerprints/reference.json",
    temperature=0.1,
    max_tokens=64,
    extra_body={"top_p": 0.9, "seed": 42},
)

# 可在其他进程中加载；prompt、system 和全部采样参数自动复用。
reference = load_fingerprint("fingerprints/reference.json")
result = verify_fingerprint(
    reference,
    base_url="https://other-gateway.example.com/v1",
    api_key="candidate-api-key",
    model="candidate-model",
    count=20,
)
print(result.matches_reference)
```

默认用完整回答文本建立经验分布。对于数字、分类等可提取稳定特征的题目，传入 `parser` 和 `field`，例如 `parser=vestigia.prompts.favorite_number.parse`、`field="parsed.first_number.value"`。`verify_fingerprint` 默认继承参考指纹的 `extra_body`；若显式传入不同采样参数，会拒绝比较，避免混合不同请求条件。

### Python API：建立基准并测试待测模型

除了 CLI，也可直接在代码中建立一个可验证的参考指纹，再对待测模型发起同样的请求：

```python
from vestigia import (
    LLMClient,
    LLMConfig,
    build_model_fingerprint,
    test_model_against_fingerprint,
)
from vestigia.prompts.favorite_number import parse

reference = LLMClient(LLMConfig(
    provider="anthropic",
    base_url="https://api.anthropic.com",
    api_key="...",
    model="claude-sonnet-4-20250514",
    temperature=0.1,
    max_tokens=64,
    # 所有额外的采样旋钮也属于此指纹，必须固定并记录。
    extra_body={"top_p": 0.9},
))

fingerprint = build_model_fingerprint(
    reference,
    "请只回答你最喜欢的一个数字。",
    parse,
    field="parsed.first_number.value",
    count=50,
    subset_size=20,
    resamples=1000,
    seed=42,
)
assert fingerprint.stability["reliable"]

candidate = LLMClient(LLMConfig(
    provider="openai_compatible",
    base_url="https://gateway.example.com/v1",
    api_key="...",
    model="unknown-model",
))
result = test_model_against_fingerprint(candidate, fingerprint, parse, count=20)
print(result.matches_reference)
print(result.distances["total_variation_distance"])
```

`build_model_fingerprint` 会进行多次调用，并在 `fingerprint.request_configuration` 中记录不含密钥的完整请求控制配置（provider、model、temperature、max tokens、`extra_body`、协议版本与缓存策略）。它提取 `parser` 返回结果中的 `field`，并执行子集稳定性检验。它还会将每次 `response.text` 的原始 Unicode 字符数（**包含空白、标点和换行**）作为第二项特征，但不统计每一个精确长度：长度会按 2 的幂分区间，例如 `1`、`2–3`、`4–7`、`8–15`、`16–31`。这避免极长回答使直方图稀疏；该对数分桶分布也会以 TV 距离执行 50→20 的稳定性验证。

`test_model_against_fingerprint` 自动复用参考指纹的 prompt、system、temperature 和 max tokens，并拒绝 `extra_body`（包括 `top_p`、seed 等）不一致的候选配置。待测模型名称可以不同——这正是识别任务的目标——但分类分布和长度分桶分布都必须落在参考样本的自身波动范围内，`matches_reference` 才为真。原始平均长度、标准差、最小值和最大值仍保留在报告中作辅助解释。结果对象均可用 `.to_dict()` 保存为 JSON。

数字题的典型输出片段：

```json
{
  "prompt_id": "favorite_number",
  "parsed": {
    "numbers": [
      {"source": "负十二点五", "notation": "chinese", "value": "-12.5"}
    ],
    "first_number": {"source": "负十二点五", "notation": "chinese", "value": "-12.5"}
  },
  "check_passed": true
}
```

可在 Python 中检查题库或生成同样的请求序列：

```python
from vestigia.prompts import DEFAULT_PROMPTS, iter_prompts

for prompt, template in iter_prompts(20):
    print(template.id, prompt)
```

## API 说明

- `LLMConfig`：连接信息和默认生成参数，由 LiteLLM 统一执行；`provider` 可为 `"openai_compatible"` 或 `"anthropic"`，并映射到相应 LiteLLM 路由。使用 `extra_body` 透传网关支持的额外请求体字段。
- `LLMClient.complete(...)`：同步、非流式地取得最终文本，返回 `LLMResponse`。
- `LLMClient.complete_messages(...)`：当需要完整多轮 `messages` 时使用。
- `LLMRequestError`：网络、HTTP 或响应格式异常。其 `status_code`、`response_body` 属性有助于排障。

`messages` 使用供应商通用的结构：

```python
[
    {"role": "system", "content": "回答要简洁"},
    {"role": "user", "content": "给我一个测试题"},
]
```

> 不要将密钥写进仓库。推荐从环境变量读取：`api_key=os.environ["LLM_API_KEY"]`。

## 项目结构

```text
src/vestigia/       可发布的包源码
  llm/              统一的 LLM HTTP 客户端与数据模型
  prompts/           每个文件一个探针，含题目、parser、checker
  collect.py         JSONL 批量采集 CLI
tests/              单元测试
pyproject.toml      打包、依赖和工具配置
requirements*.txt   便捷依赖入口
```

## 测试与检查

```bash
python -m pytest
python -m ruff check .
```

后续的提问集、特征提取和模型判别逻辑可以直接消费 `LLMResponse.text`、`model`、`usage` 与 `raw` 字段。
