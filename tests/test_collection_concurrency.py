from __future__ import annotations

from unittest.mock import patch

from vestigia.identify import _batch_sizes, _collect_feature_values
from vestigia.llm import LLMResponse


class _Client:
    def complete(self, _: str) -> LLMResponse:
        return LLMResponse(
            content="beach",
            model="test-model",
            provider="openai_compatible",
            finish_reason="stop",
            usage=None,
            request_id=None,
            raw={},
        )


def test_batch_sizes_uses_full_batches_and_one_remainder() -> None:
    assert _batch_sizes(48, 10) == (10, 10, 10, 10, 8)
    assert _batch_sizes(10, 10) == (10,)
    assert _batch_sizes(1, 10) == (1,)


def test_feature_collection_submits_requests_in_configured_batches() -> None:
    workers: list[int] = []

    class TrackingExecutor:
        def __init__(self, *, max_workers: int) -> None:
            workers.append(max_workers)

        def __enter__(self) -> TrackingExecutor:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def map(self, function, iterable):  # type: ignore[no-untyped-def]
            return map(function, iterable)

    with patch("vestigia.identify.LLM_COLLECTION_CONCURRENCY", 10), patch(
        "vestigia.identify.ThreadPoolExecutor", TrackingExecutor
    ):
        values, _ = _collect_feature_values(
            _Client(), "prompt", lambda content: {"choice": content},
            feature_kind="parsed", field="parsed.choice", length_field="content", count=48,
        )

    assert workers == [10, 10, 10, 10, 8]
    assert values == ["beach"] * 48
