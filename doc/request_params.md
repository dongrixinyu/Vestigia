# 固化的指纹请求参数预设

请求参数预设位于 `src/vestigia/request_params/`，其结构和 `prompts/` 一样作为独立、可复用的定义目录。它不会修改 `examples/get_fingerprint.py`，也不会读取或覆盖该脚本中已有的内联参数。

## 使用方式

```python
from vestigia import create_fingerprint, get_request_params

fingerprint = create_fingerprint(
    base_url="https://gateway.example.com/v1",
    api_key="...",
    model="example-model",
    prompt_id="favorite_number",
    request_params=get_request_params("fingerprint_standard_v1"),
)
```

可用预设：

```python
from vestigia import available_request_param_presets

print(available_request_param_presets())
# ('fingerprint_low_variance_v1', 'fingerprint_standard_v1')
```

## 当前预设

### `fingerprint_standard_v1`

这是默认的指纹采样实验：

```python
{
    "temperature": 0.1,
    "max_tokens": None,
    "top_p": 1.0,
    "top_k": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "extra_body": {
        "reasoning": True,
        "reasoning_effort": "low",
    },
    "extra_headers": {},
}
```

`None` 表示请求层省略该参数。`extra_body` 固定启用 `reasoning=True` 与 `reasoning_effort="low"`，使支持这些控制项的端点以一致的低强度推理模式采样。该字段并非所有 OpenAI-compatible 服务都支持；不支持的供应商应新增不含此字段的独立版本预设，而不是修改本预设。

### `fingerprint_low_variance_v1`

`fingerprint_low_variance_v1` 当前与标准预设完全相同，均设定：

```python
"temperature": 0.1
```

该名称保留用于兼容已有调用。由于其当前内容与 `fingerprint_standard_v1` 相同，两者产生的参数哈希也相同；如需新的低随机性配置，应创建新的版本名并明确不同参数。

## 固化与版本规则

- 预设定义在 `src/vestigia/request_params/__init__.py` 的 `REQUEST_PARAM_PRESETS`。
- 通过 `get_request_params(name)` 获取参数；返回的是深层独立的普通 `dict`，修改返回值不会污染预设或后续请求。
- 已存在的预设不得修改。新增或调整任何值时，创建新的带版本名预设，例如 `fingerprint_standard_v2`。
- `create_fingerprint()` 保存的 `request_params` 和生成的 `parameters_hash` 已包含这些参数。因此不同预设生成的指纹会具有不同的参数哈希，预测时不会被误当成同一个实验配置。
- 若某个供应商必须使用私有控制项，请新建明确的供应商专用版本，而不要修改通用预设；这类指纹只能和相同专用预设下的指纹比较。
