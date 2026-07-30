# Vestigia 使用说明

[English version](usage_en.md)

本文说明如何安装、采集模型指纹，以及用已观测的输出分布进行离线预测。示例使用内置探针，因此解析器和指纹特征字段会自动选择。

## 1. 安装

```bash
python -m pip install -e ".[dev]"  # 开发安装
# 或
pip install .
```

需要 Python 3.10 或更高版本。请勿把密钥写进代码或提交到仓库；建议从环境变量读取。

## 2. 请求配置

Vestigia 通过 LiteLLM 路由请求。单次调用可创建 `LLMConfig`：

```python
import os
from vestigia import LLMClient, LLMConfig

config = LLMConfig(
    provider="openai_compatible",  # 或 "anthropic"
    base_url="https://gateway.example.com/v1",
    api_key=os.environ["LLM_API_KEY"],
    model="example-model",
    temperature=0.1,
    top_p=1.0,
    top_k=None,       # None 表示不发送；显式设置 0 会被保留
    max_tokens=64,
    extra_body={
        # 服务商或网关私有字段放在这里：
        # "reasoning": True,
        # "reasoning_effort": "low",
        # "seed": 42,
    },
)

with LLMClient(config) as client:
    response = client.complete("只回答一个数字。")
    print(response.content)
```

`temperature`、`max_tokens`、`top_p`、`top_k`、`presence_penalty`、`frequency_penalty` 是 Vestigia 的请求参数。由于各服务商对 `top_k` 的支持方式不同，Vestigia 会在内部将它写入 LiteLLM 的 `extra_body`。`reasoning`、`reasoning_effort`、`seed`、缓存控制等服务商/网关私有参数，必须自行写入 `extra_body`。

单次调用覆盖参数时，统一通过一个字典传入，不要使用独立生成参数关键字：

```python
response = client.complete_messages(
    [{"role": "user", "content": "请简短回答。"}],
    request_parameters={"temperature": 0.2, "max_tokens": 32},
)
```

每次模型请求成功后，程序会记录一条 INFO 日志，包含模型、endpoint 和请求 ID；不会记录 API Key、prompt 或回答正文。

## 3. 创建参考指纹

`create_fingerprint()` 会对一个内置探针进行重复调用，并保存输出分布。每个 `prompt_id` 自己绑定 parser 和特征字段，所以**不要传入 `field`**。

```python
import os
from vestigia import create_fingerprint

fingerprint = create_fingerprint(
    base_url="https://gateway.example.com/v1",
    api_key=os.environ["LLM_API_KEY"],
    model="reference-model",
    provider="openai_compatible",
    prompt_id="favorite_number",
    variant_index=0,
    count=50,
    output="fingerprints",
    request_params={
        "temperature": 0.1,
        "max_tokens": 64,
        "top_p": 1.0,
        "top_k": None,
        "extra_body": {},
    },
)
print(fingerprint.distribution)
```

当前主要内置探针：

| `prompt_id` | 自动统计的指纹特征 |
| --- | --- |
| `favorite_number` | `parsed.first_number.value` |
| `project_success_score` | `parsed.score.value` |
| 其他内置 prompt | 对应完整 `parsed` 对象 |

`project_success_score` 有多个项目描述 variants。该探针有意把 `"0.6"` 和 `"0.60"` 当作不同特征，以保留模型的数值表达风格。

指纹会记录 prompt、选用特征与请求控制参数（不含 API Key）。不同 prompt、不同 variant、不同 system instruction 或不同采样参数的样本不可混合。建议尽可能为每个参考模型采集至少 50 条成功样本；库会随指纹保存子集稳定性统计。

要批量采集多个模型，可编辑并运行：

```bash
python examples/get_fingerprint.py
```

> 此示例文件是本地配置模板。请替换示例值，且绝不要提交真实密钥。

## 4. 根据已观测分布预测

当你已拥有**多个探针**抽取出的输出值，希望离线进行模型级别匹配时，使用 `predict_distribution()`；该函数不会调用 LLM。每个输入分布必须声明 `prompt_id` 和 `params_hash`（对应参考指纹 JSON 中的 `parameters_hash`）。`params_hash` 为空字符串时，会匹配该 `prompt_id` 下所有参数配置的指纹：

```python
from vestigia import predict_distribution

result = predict_distribution(
    [
        {
            "prompt_id": "favorite_number",
            "params_hash": "复制已保存指纹的 parameters_hash",
            "values": ["163", "142", "163", "168", "142"],
        },
        {
            "prompt_id": "model_identity",
            "params_hash": "复制已保存指纹的 parameters_hash",
            "values": ["gpt", "gpt", "null", "gpt"],
        },
    ],
    "fingerprints",
    distance_type="jensen_shannon",  # 或 "total_variation"
    softmax_temperature=0.1,
)
```

函数会比较 `prompt_id` 相同的参考特征；非空 `params_hash` 必须完全相同，空字符串则可匹配任意 `parameters_hash`。候选模型必须覆盖**每一项**输入特征；最终模型距离是各特征距离的等权平均值。`feature_matches` 会保留每一个探针实际选中的参数哈希、距离和来源指纹路径，便于解释判断结果。

每个模型结果都会同时返回两种距离。`distance_type` 决定排序、每项特征存在多份历史指纹时的选择方式，以及 softmax 相对分数的计算方式。`probability` 仅是**相对相似度分数**，不是模型身份的校准概率。

可直接编辑并运行文本表格示例：

```bash
python examples/predict_distribution.py
```

输入值必须与参考指纹采用相同的探针和特征规则。例如，`favorite_number` 的输入应是 `"142"` 这类字符串；`project_success_score` 会保留 `"0.6"` 与 `"0.60"` 的格式差异。

## 5. 用 CLI 校验 JSONL 采集结果

若需要保留原始 JSONL 记录和稳定性报告，可使用 CLI：

```bash
vestigia-collect \
  --base-url "https://gateway.example.com/v1" \
  --api-key "$LLM_API_KEY" \
  --model "example-model" \
  --prompt-id favorite_number \
  --variant-index 0 \
  --count 50 \
  --temperature 0.1 \
  --max-tokens 64 \
  --output samples/favorite-number.jsonl

vestigia-validate \
  --input samples/favorite-number.jsonl \
  --field parsed.first_number.value \
  --sample-size 20 \
  --resamples 1000 \
  --output samples/favorite-number-validation.json
```

CLI 的服务商私有参数使用 `--extra-body-json`：

```bash
--extra-body-json '{"reasoning":true,"reasoning_effort":"low"}'
```

## 可复现性注意事项

- 同一份指纹必须固定 prompt variant、system instruction 和全部采样参数。
- 若网关可能重放完整响应，应关闭或绕过响应缓存。
- 采样参数不同即代表不同指纹，不能合并样本。
- 项目通过 LiteLLM 支持 OpenAI-compatible 与 Anthropic 协议；`provider` 描述的是调用协议，并不代表底层模型供应商身份。
