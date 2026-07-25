# Vestigia

`Vestigia` 是一个可安装的 Python 包，用于通过一组问题采集多个 LLM 的非流式回答，并为后续的模型来源识别（fingerprinting）提供统一数据入口。

当前版本先提供稳定的 **LLM 调用层**：支持 OpenAI Chat Completions 兼容接口（模型官网或中转站）以及 Anthropic Messages 接口；不会依赖特定厂商 SDK。

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

客户端会把请求发送至 `<base_url>/chat/completions`。若已传入完整接口地址，也可使用 `endpoint` 覆盖：

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

Anthropic 请求会发送至 `<base_url>/v1/messages`，并带上 `anthropic-version: 2023-06-01`。可通过 `api_version` 修改该版本，或通过 `extra_headers` 增加网关要求的头。

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

- `LLMConfig`：连接信息和默认生成参数。`provider` 可为 `"openai_compatible"` 或 `"anthropic"`；使用 `extra_body` 透传网关支持的额外请求体字段。
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
