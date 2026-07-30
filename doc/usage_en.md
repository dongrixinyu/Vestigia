# Vestigia Usage Guide

[中文版本](usage_cn.md)

This guide covers installation, fingerprint collection, and offline prediction. The API examples use built-in probes so the parser and fingerprint feature are selected automatically.

## 1. Install

```bash
python -m pip install -e ".[dev]"  # development install
# or
pip install .
```

Use Python 3.10 or later. Keep credentials outside source control, preferably in environment variables.

## 2. Request configuration

Vestigia routes requests through LiteLLM. Create an `LLMConfig` for direct, single requests:

```python
import os
from vestigia import LLMClient, LLMConfig

config = LLMConfig(
    provider="openai_compatible",  # or "anthropic"
    base_url="https://gateway.example.com/v1",
    api_key=os.environ["LLM_API_KEY"],
    model="example-model",
    temperature=0.1,
    top_p=1.0,
    top_k=None,       # None: omit the field; an explicit 0 is retained
    max_tokens=64,
    extra_body={
        # Endpoint-specific fields belong here:
        # "reasoning": True,
        # "reasoning_effort": "low",
        # "seed": 42,
    },
)

with LLMClient(config) as client:
    response = client.complete("Reply with one number.")
    print(response.content)
```

`temperature`, `max_tokens`, `top_p`, `top_k`, `presence_penalty`, and `frequency_penalty` are Vestigia request parameters. `top_k` is sent inside LiteLLM's `extra_body`, because provider support differs. Provider- or gateway-specific settings—such as `reasoning`, `reasoning_effort`, `seed`, and cache controls—must be placed in `extra_body`.

For a one-call override, use one dictionary rather than separate generation keyword arguments:

```python
response = client.complete_messages(
    [{"role": "user", "content": "Reply concisely."}],
    request_parameters={"temperature": 0.2, "max_tokens": 32},
)
```

Each successful model call writes an INFO log with model, endpoint, and request ID. Prompts, responses, and API keys are not logged.

## 3. Create reference fingerprints

`create_fingerprint()` repeatedly calls a built-in probe and saves the distribution. Each `prompt_id` owns its parser and feature field, so **do not pass `field`**.

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

Available built-in probes include:

| `prompt_id` | Fingerprint feature |
| --- | --- |
| `favorite_number` | `parsed.first_number.value` |
| `project_success_score` | `parsed.score.value` |
| Other built-in prompts | their complete `parsed` object |

`project_success_score` has multiple project-description variants. Its score parser deliberately treats lexical forms such as `"0.6"` and `"0.60"` as distinct features.

A fingerprint records the prompt, selected feature, and request controls (without the API key). Do not compare samples collected with different prompts or sampling settings. Collect at least 50 successful reference samples when possible; the library stores subset-stability statistics with each fingerprint.

For a multi-model collection template, edit and run:

```bash
python examples/get_fingerprint.py
```

> The example file is a local configuration template. Replace placeholder values and never commit credentials.

## 4. Predict from externally observed values

Use `predict_distribution()` when you have extracted values from **multiple probes** and want an offline, model-level comparison against saved fingerprints. Each input distribution must declare `prompt_id` and `params_hash` (`parameters_hash` in its saved reference JSON). An empty `params_hash` string matches fingerprints with any parameter configuration for that `prompt_id`:

```python
from vestigia import predict_distribution

result = predict_distribution(
    [
        {
            "prompt_id": "favorite_number",
            "params_hash": "copy-the-saved-parameters_hash",
            "values": ["163", "142", "163", "168", "142"],
        },
        {
            "prompt_id": "model_identity",
            "params_hash": "copy-the-saved-parameters_hash",
            "values": ["gpt", "gpt", "null", "gpt"],
        },
    ],
    "fingerprints",
    distance_type="jensen_shannon",  # or "total_variation"
    softmax_temperature=0.1,
)
```

The function compares references with the same `prompt_id`; a non-empty `params_hash` must match exactly, while an empty string matches any `parameters_hash`. A candidate model must have a matching saved fingerprint for **every** supplied feature. The final model distance is the equal-weight mean of its per-feature distances; `feature_matches` retains the actually selected parameter hash, per-probe distance, and source fingerprint path.

Every model result returns the selected metric through generic `distance_type` and `distance` fields. `distance_type` controls ranking, selection of the closest duplicate reference for each feature, and the softmax score. `probability` is a relative similarity score, **not** a calibrated probability that the model has a particular identity.

A ready-to-edit ASCII-table example is available at:

```bash
python examples/predict_distribution.py
```

The input values must be collected with the same probe and feature convention as the reference fingerprints. For example, favorite-number values should be strings such as `"142"`; project-success-score values preserve score formatting, such as `"0.6"` versus `"0.60"`.

## 5. Validate JSONL collections from the CLI

The CLI is useful when you need raw JSONL records and stability reports:

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

For endpoint-specific CLI controls, use `--extra-body-json`, for example:

```bash
--extra-body-json '{"reasoning":true,"reasoning_effort":"low"}'
```

## Notes on reproducibility

- Fix prompt variant, system instruction, and every sampling control for a given fingerprint.
- Disable or bypass response caches when the gateway can replay complete responses.
- Different sampling parameters create different fingerprints; do not merge their samples.
- The project supports OpenAI-compatible and Anthropic protocol routing through LiteLLM; `provider` describes the wire protocol, not a claim about the underlying model vendor.
