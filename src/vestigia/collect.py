"""Command-line collector for repeated LLM fingerprint probes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vestigia.llm import LLMClient, LLMConfig, LLMRequestError
from vestigia.prompts import iter_prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Call an LLM endpoint repeatedly with Vestigia's fixed prompt templates."
    )
    parser.add_argument("--base-url", required=True, help="Gateway API root, e.g. https://host/v1")
    parser.add_argument("--model", required=True, help="Model identifier accepted by the gateway")
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_API_KEY"),
        help="API key; defaults to the LLM_API_KEY environment variable",
    )
    parser.add_argument(
        "--provider",
        choices=("openai_compatible", "anthropic"),
        default="openai_compatible",
        help="API protocol used by the endpoint (default: openai_compatible)",
    )
    parser.add_argument(
        "--endpoint", help="Optional complete endpoint URL for non-standard gateways"
    )
    parser.add_argument("--count", type=int, default=20, help="Number of requests (default: 20)")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL file")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, help="Maximum generated tokens")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument("--system", help="Optional system instruction sent with every request")
    parser.add_argument(
        "--extra-headers-json",
        default="{}",
        help='Additional request headers as JSON, e.g. \'{"HTTP-Referer":"https://app.example"}\'',
    )
    parser.add_argument(
        "--extra-body-json",
        default="{}",
        help='Additional JSON body fields, e.g. \'{"top_p":0.9}\'',
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed request instead of recording it",
    )
    return parser


def json_object(value: str, option_name: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} must be valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{option_name} must be a JSON object")
    return parsed


def response_record(
    index: int, prompt: str, template_id: str, category: str, response: Any
) -> dict[str, Any]:
    return {
        "index": index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_id": template_id,
        "category": category,
        "prompt": prompt,
        "status": "ok",
        "response": {
            "text": response.text,
            "model": response.model,
            "provider": response.provider,
            "finish_reason": response.finish_reason,
            "usage": response.usage,
            "request_id": response.request_id,
        },
    }


def run(args: argparse.Namespace) -> int:
    if not args.api_key:
        raise ValueError("--api-key is required (or set LLM_API_KEY)")
    if args.count < 1:
        raise ValueError("--count must be greater than zero")

    headers = json_object(args.extra_headers_json, "--extra-headers-json")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in headers.items()):
        raise ValueError("--extra-headers-json keys and values must be strings")
    extra_body = json_object(args.extra_body_json, "--extra-body-json")
    config = LLMConfig(
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        endpoint=args.endpoint,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_headers=headers,
        extra_body=extra_body,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with LLMClient(config) as client, args.output.open("w", encoding="utf-8") as output:
        for index, (prompt, template) in enumerate(iter_prompts(args.count), start=1):
            try:
                response = client.complete(prompt, system=args.system)
                record = response_record(index, prompt, template.id, template.category, response)
                print(f"[{index}/{args.count}] ok: {template.id}", file=sys.stderr)
            except LLMRequestError as exc:
                failures += 1
                record = {
                    "index": index,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt_id": template.id,
                    "category": template.category,
                    "prompt": prompt,
                    "status": "error",
                    "error": {
                        "message": str(exc),
                        "status_code": exc.status_code,
                        "response_body": exc.response_body,
                    },
                }
                print(f"[{index}/{args.count}] error: {exc}", file=sys.stderr)
                if args.fail_fast:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    return 1
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    try:
        exit_code = run(parser.parse_args(argv))
    except ValueError as exc:
        parser.error(str(exc))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
