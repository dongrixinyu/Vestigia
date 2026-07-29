from __future__ import annotations

import pytest

from vestigia.prompts.project_success_score import PROMPT


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("0.65\n尚未获得付费客户，因此存在较高的市场验证风险。", "0.65"),
        ("零点四\n需要先验证付费 Pilot 的转化能力。", "0.4"),
        ("1\n市场需求明确，但仍需要商业验证。", "1"),
    ],
)
def test_project_success_score_parser_extracts_first_line_score(response: str, expected: str) -> None:
    parsed = PROMPT.parser(response)

    assert parsed["score"]["value"] == expected
    assert PROMPT.checker(response, parsed)


@pytest.mark.parametrize("response", ["很可能成功。", "1.2\n分数超出范围。", "-0.1\n分数超出范围。"])
def test_project_success_score_checker_rejects_missing_or_out_of_range_score(response: str) -> None:
    parsed = PROMPT.parser(response)

    assert parsed["score"] is None
    assert not PROMPT.checker(response, parsed)


def test_project_success_score_preserves_arabic_decimal_precision() -> None:
    score_a = PROMPT.parser("0.6\n理由")["score"]
    score_b = PROMPT.parser("0.60\n理由")["score"]

    assert score_a["value"] == "0.6"
    assert score_b["value"] == "0.60"
    assert score_a["value"] != score_b["value"]


def test_project_success_score_is_available_from_default_catalog() -> None:
    assert PROMPT.id == "project_success_score"
    assert len(PROMPT.variants) == 3
