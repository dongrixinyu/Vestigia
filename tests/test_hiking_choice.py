from __future__ import annotations

import pytest

from vestigia.prompts.outdoor_trip_choice import CATEGORIES, PARSE_ERROR, PROMPT


@pytest.mark.parametrize("response", CATEGORIES)
def test_outdoor_trip_choice_parser_accepts_each_exact_category(response: str) -> None:
    parsed = PROMPT.parser(response)

    assert parsed == {"choice": response}
    assert PROMPT.checker(response, parsed)


@pytest.mark.parametrize(
    "response",
    [
        "海边",
        "beach\n",
        " beach",
        "beach。",
        "beach，因为可以放松。",
        "beach or mountain",
        "7",
        "",
        "invalid_choice",
    ],
)
def test_outdoor_trip_choice_parser_maps_every_invalid_response_to_parse_error(response: str) -> None:
    parsed = PROMPT.parser(response)

    assert parsed == {"choice": PARSE_ERROR}
    assert not PROMPT.checker(response, parsed)


def test_outdoor_trip_choice_prompt_requires_one_ascii_category_only() -> None:
    assert PROMPT.id == "outdoor_trip_choice"
    assert PROMPT.field == "parsed.choice"
    assert len(PROMPT.variants) == 5
    assert all("只能输出该类别码本身" in variant for variant in PROMPT.variants)
