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

from vestigia.fingerprint import build_fingerprint
from vestigia.llm import LLMClient, LLMConfig, LLMRequestError, LLMResponse, RequestSignature
from vestigia.prompts import DEFAULT_PROMPTS, PromptTemplate, iter_prompts


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
    parser.add_argument(
        "--prompt-id",
        choices=tuple(template.id for template in DEFAULT_PROMPTS),
        help="Repeat one probe instead of cycling through the complete prompt set",
    )
    parser.add_argument(
        "--variant-index",
        type=int,
        default=0,
        help="Zero-based prompt wording to use with --prompt-id (default: 0)",
    )
    parser.add_argument(
        "--fingerprint-output",
        type=Path,
        help="Optional JSON file for the empirical response distribution",
    )
    parser.add_argument(
        "--fingerprint-field",
        default="parsed",
        help="Dotted record field to count, e.g. parsed.first_number.value (default: parsed)",
    )
    parser.add_argument(
        "--allow-response-cache",
        action="store_true",
        help="Do not send standard no-cache HTTP headers (not recommended for fingerprinting)",
    )
    parser.add_argument(
        "--cache-bust-query-param",
        help="Unique query parameter for every request, for gateways that ignore no-cache headers",
    )
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, help="Nucleus sampling probability")
    parser.add_argument("--top-k", type=int, help="Restrict sampling to the top K tokens")
    parser.add_argument("--presence-penalty", type=float, help="Presence penalty")
    parser.add_argument("--frequency-penalty", type=float, help="Frequency penalty")
    parser.add_argument(
        "--reasoning-json",
        help='Reasoning configuration object, e.g. \'{"effort":"high"}\'',
    )
    parser.add_argument("--reasoning-effort", help="Reasoning effort, e.g. low, medium, high")
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
        help="Additional JSON body fields, e.g. '{\"top_p\":0.9}'",
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
    index: int,
    prompt: str,
    template: PromptTemplate,
    response: LLMResponse,
    signature: RequestSignature,
) -> dict[str, Any]:
    parsed = dict(template.parser(response.text))
    return {
        "index": index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_id": template.id,
        "category": template.category,
        "prompt": prompt,
        "status": "ok",
        "request_signature": signature.to_dict(),
        "parsed": parsed,
        "check_passed": template.checker(response.text, parsed),
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

    if args.prompt_id:
        template = next(template for template in DEFAULT_PROMPTS if template.id == args.prompt_id)
        if not 0 <= args.variant_index < len(template.variants):
            raise ValueError(
                f"--variant-index must be between 0 and {len(template.variants) - 1} "
                f"for --prompt-id {args.prompt_id!r}"
            )
        prompt_sequence = (
            (template.variants[args.variant_index], template) for _ in range(args.count)
        )
    else:
        if args.variant_index != 0:
            raise ValueError("--variant-index requires --prompt-id")
        prompt_sequence = iter_prompts(args.count)

    headers = json_object(args.extra_headers_json, "--extra-headers-json")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in headers.items()):
        raise ValueError("--extra-headers-json keys and values must be strings")
    reasoning = (
        json_object(args.reasoning_json, "--reasoning-json") if args.reasoning_json else None
    )
    config = LLMConfig(
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        endpoint=args.endpoint,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        reasoning=reasoning,
        reasoning_effort=args.reasoning_effort,
        extra_headers=headers,
        extra_body=extra_body,
        disable_response_cache=not args.allow_response_cache,
        cache_bust_query_param=args.cache_bust_query_param,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    records: list[dict[str, Any]] = []
    with LLMClient(config) as client, args.output.open("w", encoding="utf-8") as output:
        for index, (prompt, template) in enumerate(prompt_sequence, start=1):
            signature_context = client.request_signature_context(prompt, system=args.system)
            signature = RequestSignature(
                model=str(signature_context.pop("request_model")),
                provider=config.provider,
                prompt=prompt,
                prompt_id=template.id,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                system=args.system,
                extra_body=config.extra_body,
                disable_response_cache=config.disable_response_cache,
                cache_bust_query_param=config.cache_bust_query_param,
                **signature_context,
            )
            try:
                response = client.complete(prompt, system=args.system)
                record = response_record(index, prompt, template, response, signature)
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
            records.append(record)

    if args.fingerprint_output:
        args.fingerprint_output.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = build_fingerprint(records, field=args.fingerprint_field)
        args.fingerprint_output.write_text(
            json.dumps(fingerprint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
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
