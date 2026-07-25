from __future__ import annotations

import json

import httpx
import respx

from vestigia.collect import build_parser, run
from vestigia.prompts import DEFAULT_PROMPTS, iter_prompts


def test_prompt_sequence_cycles_templates_then_variants() -> None:
    prompts = list(iter_prompts(len(DEFAULT_PROMPTS) + 1))

    assert prompts[0] == (DEFAULT_PROMPTS[0].variants[0], DEFAULT_PROMPTS[0])
    assert prompts[len(DEFAULT_PROMPTS)] == (DEFAULT_PROMPTS[0].variants[1], DEFAULT_PROMPTS[0])


@respx.mock
def test_collect_writes_one_jsonl_record_per_request(tmp_path) -> None:
    route = respx.post("https://gateway.example/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            },
        )
    )
    output = tmp_path / "results.jsonl"
    args = build_parser().parse_args(
        [
            "--base-url",
            "https://gateway.example/v1",
            "--api-key",
            "secret",
            "--model",
            "test-model",
            "--count",
            "2",
            "--output",
            str(output),
            "--extra-body-json",
            '{"top_p":0.9}',
        ]
    )

    assert run(args) == 0

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["prompt_id"] == "favorite_number"
    assert records[0]["status"] == "ok"
    assert records[0]["response"]["text"] == "answer"
    assert route.call_count == 2
    assert json.loads(route.calls.last.request.content)["top_p"] == 0.9
