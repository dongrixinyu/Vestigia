"""Small, self-contained example: call a relay gateway 20 times and save
distributions of a single probe.

Probe: "pick your favorite number between 0 and 100 and reply with ONLY the
number". For each of 20 non-streaming calls we record:

* the final answer text and its Unicode length,
* the model's reasoning / thinking text, its length, and its first 10 chars,
* the server-reported model name and token usage.

Then we aggregate three distributions:

1. the chosen number (counts + proportions),
2. the answer length bucketed by powers of two (and the same for the
   reasoning length),
3. the frequency of the first 10 characters of the reasoning across calls.

Outputs (UTF-8, under ``samples/``):

* ``favorite-number-raw.jsonl``  -- one record per call,
* ``favorite-number-dist.json``  -- aggregated distributions.

Usage (Windows PowerShell)::

    $env:LLM_BASE_URL = "https://gateway.example.com/v1"
    $env:LLM_API_KEY  = "your-api-key"
    $env:LLM_MODEL    = "example-model"
    python examples/favorite_number_small.py

Every connection default can be overridden with environment variables; see
``load_settings`` below.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

# Make the package importable when running this script in place.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vestigia.llm.client import LLMClient
from vestigia.llm.models import LLMConfig, LLMRequestError, LLMResponse
from vestigia.prompts.favorite_number import parse as parse_numbers

PROMPT = (
    "请在 0 到 100 之间选择一个你最喜欢的整数，并以\"#数字\"的格式回答，"
    "例如 #42。回答里只能有一处 #数字 标记。"
)

# First 10 characters ("characters" here = Unicode code points) of the
# reasoning that we keep per call, and the aggregate key.
REASONING_PREFIX_LEN = 10

# Buckets use 2^k .. 2^(k+1)-1 as in README's length-scope feature.
MIN_BUCKET = 1  # length 1 is its own bucket; smaller -> "0".


@dataclass(slots=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    provider: str = "openai_compatible"
    count: int = 20
    temperature: float = 0.7
    max_tokens: int | None = 512
    system: str | None = None
    extra_body_json: str | None = None  # raw JSON string for --extra-body style knobs
    disable_response_cache: bool = True
    cache_bust_query_param: str | None = "vestigia_request"
    retry_seconds: float = 2.0
    output_dir: str = "samples"

    @property
    def extra_body(self) -> dict[str, Any]:
        if not self.extra_body_json:
            return {}
        loaded = json.loads(self.extra_body_json)
        if not isinstance(loaded, dict):
            raise ValueError("EXTRA_BODY_JSON must be a JSON object")
        return loaded


def load_settings() -> Settings:
    base_url = os.environ["LLM_BASE_URL"]
    api_key = os.environ["LLM_API_KEY"]
    model = os.environ["LLM_MODEL"]
    settings = Settings(
        base_url=base_url,
        api_key=api_key,
        model=model,
        provider=os.environ.get("LLM_PROVIDER", "openai_compatible"),
        count=int(os.environ.get("LLM_COUNT", "20")),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(v) if (v := os.environ.get("LLM_MAX_TOKENS")) else 512,
        system=os.environ.get("LLM_SYSTEM") or None,
        extra_body_json=os.environ.get("LLM_EXTRA_BODY_JSON") or None,
        disable_response_cache=os.environ.get("LLM_DISABLE_RESPONSE_CACHE", "1") == "1",
        cache_bust_query_param=os.environ.get("LLM_CACHE_BUST") or "vestigia_request",
    )
    return settings


def make_client(settings: Settings) -> LLMClient:
    config = LLMConfig(
        provider=settings.provider,  # type: ignore[arg-type]
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        max_tokens=settings.max_tokens,
        # temperature is supplied per-call for clarity
        disable_response_cache=settings.disable_response_cache,
        cache_bust_query_param=settings.cache_bust_query_param,
        extra_body=settings.extra_body,
    )
    return LLMClient(config)


def extract_reasoning(response: LLMResponse) -> str:
    """Best-effort pull of a thinking / reasoning string from the raw response.

    OpenAI-compatible relays vary; common layouts handled:

    * ``choices[0].message.reasoning`` / ``reasoning_content`` (string),
    * ``choices[0].message.content`` is an array of parts, some having
      ``type == "reasoning"`` (or ``reasoning`` blocks); the text lives in
      ``part["text"]`` or ``part["content"]``,
    * some relays expose it under a top-level ``reasoning`` key.
    """
    raw = response.raw

    def _from_choice_message(msg: dict[str, Any]) -> str:
        for key in ("reasoning", "reasoning_content", "thinking"):
            value = msg.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                joined = _join_reasoning_parts(value)
                if joined:
                    return joined
        content = msg.get("content")
        if isinstance(content, list):
            joined = _join_reasoning_parts(content)
            if joined:
                return joined
        return ""

    def _join_reasoning_parts(parts: list[Any]) -> str:
        chunks: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in {"reasoning", "thinking", "thinking_delta"} or part.get("reasoning"):
                text = part.get("text") or part.get("content") or part.get("reasoning")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)

    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict):
                reasoning = _from_choice_message(msg)
                if reasoning:
                    return reasoning
    for key in ("reasoning", "reasoning_content", "thinking"):
        value = raw.get(key)
        if isinstance(value, str):
            return value
    return ""


def bucketed_length(length: int) -> str:
    """Bin a length into a power-of-two bucket label, matching README's scheme."""
    if length <= 0:
        return "0"
    if length == 1:
        return "1"
    upper_excluded = 1 << math.ceil(math.log2(length + 1))
    lower_inclusive = upper_excluded // 2
    return f"{lower_inclusive}-{upper_excluded - 1}"


def chosen_number(text: str) -> dict[str, Any] | None:
    """Return the normalized first ``#number`` occurrence, or first number."""
    # Prefer the explicit "#NN" marker the prompt asked for.
    marker = None
    for piece in text.split():
        token = piece.strip()
        if token.startswith("#") and token[1:]:
            candidate = token[1:]
            if candidate.lstrip("+-").isdigit():
                marker = candidate
                break
    if marker is not None:
        return {"source": marker, "notation": "arabic", "value": marker}
    parsed = parse_numbers(text)
    return parsed.get("first_number")


def one_call(client: LLMClient, settings: Settings, index: int) -> dict[str, Any]:
    attempt = 0
    last_error: str | None = None
    while True:
        attempt += 1
        t0 = time.time()
        try:
            response = client.complete(
                PROMPT,
                system=settings.system,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            elapsed = time.time() - t0
        except LLMRequestError as exc:
            elapsed = time.time() - t0
            last_error = f"{exc} status={exc.status_code} body={exc.response_body}"
            print(f"[{index + 1:02d}] attempt {attempt} failed: {exc}", file=sys.stderr)
            if attempt >= 3:
                return {
                    "index": index,
                    "ok": False,
                    "error": last_error,
                    "elapsed_seconds": round(elapsed, 3),
                }
            time.sleep(settings.retry_seconds)
            continue

        text = response.text
        reasoning = extract_reasoning(response)
        chosen = chosen_number(text)
        record = {
            "index": index,
            "ok": True,
            "elapsed_seconds": round(elapsed, 3),
            "model_reported": response.model,
            "finish_reason": response.finish_reason,
            "usage": dict(response.usage) if response.usage else None,
            "request_id": response.request_id,
            "answer_text": text,
            "answer_length": len(text),
            "chosen_number": chosen,
            "reasoning_text": reasoning,
            "reasoning_length": len(reasoning),
            "reasoning_prefix": reasoning[:REASONING_PREFIX_LEN],
        }
        print(
            f"[{index + 1:02d}] ok model={response.model} "
            f"choice={(chosen or {}).get('value','-')} "
            f"ans_len={len(text)} reason_len={len(reasoning)}",
            file=sys.stderr,
        )
        return record


def distribute(values: list[Any]) -> list[dict[str, Any]]:
    """Turn a list into a count+proportion distribution, sorted by count desc."""
    total = len(values)
    counter = Counter(map(str, values))
    return [
        {"value": value, "count": count, "proportion": count / total if total else 0.0}
        for value, count in counter.most_common()
    ]


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("ok")]

    numbers: list[str] = []
    answer_buckets: list[str] = []
    reasoning_buckets: list[str] = []
    reasoning_prefixes: list[str] = []
    for r in ok:
        chosen = r.get("chosen_number")
        if chosen:
            numbers.append(chosen.get("value"))
        answer_buckets.append(bucketed_length(int(r.get("answer_length", 0))))
        reasoning_buckets.append(bucketed_length(int(r.get("reasoning_length", 0))))
        prefix = r.get("reasoning_prefix") or ""
        reasoning_prefixes.append(prefix)

    # Frequency of each character that ever appears in a reasoning prefix.
    char_counter: Counter[str] = Counter()
    for prefix in reasoning_prefixes:
        char_counter.update(prefix)

    return {
        "sample_count": len(ok),
        "failed_count": len(records) - len(ok),
        "chosen_number_distribution": distribute(numbers),
        "answer_length_bucket_distribution": distribute(answer_buckets),
        "reasoning_length_bucket_distribution": distribute(reasoning_buckets),
        "reasoning_prefix_distribution": distribute(reasoning_prefixes),
        "reasoning_prefix_char_frequency": [
            {"char": ch, "count": cnt}
            for ch, cnt in sorted(char_counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
    }


def main() -> int:
    settings = load_settings()
    os.makedirs(settings.output_dir, exist_ok=True)
    raw_path = os.path.join(settings.output_dir, "favorite-number-raw.jsonl")
    dist_path = os.path.join(settings.output_dir, "favorite-number-dist.json")

    print(f"Calling {settings.model} at {settings.base_url} {settings.count} times ...",
          file=sys.stderr)
    records: list[dict[str, Any]] = []
    with make_client(settings) as client, open(raw_path, "w", encoding="utf-8") as raw_out:
        for i in range(settings.count):
            record = one_call(client, settings, i)
            records.append(record)
            raw_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    distributions = aggregate(records)
    # Attach enough request configuration to interpret the distributions later.
    distributions["request_configuration"] = {
        "provider": settings.provider,
        "model": settings.model,
        "base_url": settings.base_url,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "system": settings.system,
        "extra_body": settings.extra_body,
        "disable_response_cache": settings.disable_response_cache,
        "cache_bust_query_param": settings.cache_bust_query_param,
        "count": settings.count,
        "prompt": PROMPT,
    }
    with open(dist_path, "w", encoding="utf-8") as out:
        json.dump(distributions, out, ensure_ascii=False, indent=2)

    print(
        f"\nDone. ok={distributions['sample_count']}/{len(records)} "
        f"failed={distributions['failed_count']}\n"
        f"raw -> {os.path.abspath(raw_path)}\n"
        f"dist -> {os.path.abspath(dist_path)}",
        file=sys.stderr,
    )
    # Also echo a compact view of the number distribution to stdout.
    print(json.dumps(
        {
            "chosen_number_distribution": distributions["chosen_number_distribution"],
            "answer_length_bucket_distribution": distributions["answer_length_bucket_distribution"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if distributions["sample_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
