from decimal import Decimal

import pytest

from vestigia.prompts.favorite_number import PROMPT, chinese_numeral_to_decimal


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("七", Decimal("7")),
        ("十二", Decimal("12")),
        ("一百零二", Decimal("102")),
        ("两千零二十", Decimal("2020")),
        ("负十二点五", Decimal("-12.5")),
        ("一万零三", Decimal("10003")),
    ],
)
def test_chinese_numeral_to_decimal(source: str, expected: Decimal) -> None:
    assert chinese_numeral_to_decimal(source) == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("我喜欢 7。", [{"source": "7", "notation": "arabic", "value": "7"}]),
        ("我选七，因为它很常见。", [{"source": "七", "notation": "chinese", "value": "7"}]),
        (
            "可以是 -12.5 或负十二点五。",
            [
                {"source": "-12.5", "notation": "arabic", "value": "-12.5"},
                {"source": "负十二点五", "notation": "chinese", "value": "-12.5"},
            ],
        ),
    ],
)
def test_favorite_number_parser_extracts_arabic_and_chinese_numbers(response, expected) -> None:
    parsed = PROMPT.parser(response)

    assert parsed["numbers"] == expected
    assert PROMPT.checker(response, parsed)


def test_favorite_number_checker_rejects_response_without_number() -> None:
    parsed = PROMPT.parser("我没有特别偏好的数字。")

    assert parsed["numbers"] == []
    assert not PROMPT.checker("我没有特别偏好的数字。", parsed)

