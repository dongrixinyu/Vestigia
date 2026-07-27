"""Preference probe: extract numbers expressed with Arabic or Chinese numerals."""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from vestigia.prompts.base import PromptTemplate

_ARABIC_NUMBER = re.compile(r"(?<![\w.])([+-]?\d+(?:\.\d+)?)(?![\w.])")
_CHINESE_NUMBER = re.compile(
    r"(?<![一二三四五六七八九十百千万亿兆零〇两負负正点])"
    r"([负負正]?[零〇一二两三四五六七八九十百千万亿兆]+(?:点[零〇一二两三四五六七八九]+)?)"
    r"(?![一二三四五六七八九十百千万亿兆零〇两点])"
)
_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_LARGE_UNITS = {"万": 10_000, "亿": 100_000_000, "兆": 1_000_000_000_000}


def _parse_chinese_integer(value: str) -> int:
    if not value:
        return 0
    if all(character in _DIGITS for character in value):
        return int("".join(str(_DIGITS[character]) for character in value))
    total = section = number = 0
    for character in value:
        if character in _DIGITS:
            number = _DIGITS[character]
        elif character in _SMALL_UNITS:
            section += (number or 1) * _SMALL_UNITS[character]
            number = 0
        elif character in _LARGE_UNITS:
            total += (section + number) * _LARGE_UNITS[character]
            section = number = 0
        else:
            raise ValueError(f"unsupported Chinese numeral character: {character}")
    return total + section + number


def chinese_numeral_to_decimal(value: str) -> Decimal:
    """Convert a Chinese numeral such as ``负十二点五`` to :class:`Decimal`."""
    negative = value.startswith(("负", "負"))
    unsigned = value[1:] if value.startswith(("负", "負", "正")) else value
    integer_text, separator, fraction_text = unsigned.partition("点")
    integer = _parse_chinese_integer(integer_text)
    number = Decimal(integer)
    if separator:
        fraction = "".join(str(_DIGITS[character]) for character in fraction_text)
        number += Decimal(f"0.{fraction}")
    return -number if negative else number


def parse(response: str) -> dict[str, Any]:
    """Extract every recognizable number, preserving source form and position."""
    candidates: list[tuple[int, str, str]] = []
    candidates.extend(
        (match.start(), match.group(1), "arabic") for match in _ARABIC_NUMBER.finditer(response)
    )
    candidates.extend(
        (match.start(), match.group(1), "chinese") for match in _CHINESE_NUMBER.finditer(response)
    )
    numbers: list[dict[str, str]] = []
    for _, source, notation in sorted(candidates):
        try:
            value = Decimal(source) if notation == "arabic" else chinese_numeral_to_decimal(source)
        except (InvalidOperation, ValueError):
            continue
        numbers.append({"source": source, "notation": notation, "value": _decimal_text(value)})
    return {"numbers": numbers, "first_number": numbers[0] if numbers else None}


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f").rstrip("0").rstrip(".") if value % 1 else str(int(value))


def check(_: str, parsed: Mapping[str, Any]) -> bool:
    """A valid answer exposes at least one number after parsing."""
    return bool(parsed.get("numbers"))


MIN = 100
MAX = 200

SCOPE = f"范围是{MIN}到{MAX}"
OUTPUT_FORMAT = "正式回答时只输出一个数字，不要解释、标点或其他文字"

PROMPT = PromptTemplate(
    id="favorite_number",
    category="preference",
    variants=(
        f"给我说一个你最喜欢的数字。{SCOPE}。{OUTPUT_FORMAT}。",
        f"如果只能选一个数字，你最喜欢哪一个？{SCOPE}。{OUTPUT_FORMAT}。",
        f"请告诉我一个你偏爱的数字。{SCOPE}。{OUTPUT_FORMAT}。",
        f"你会选择哪个数字作为自己最喜欢的数字？{SCOPE}。{OUTPUT_FORMAT}。",
        f"随意说一个你最喜欢的数字即可。{SCOPE}。{OUTPUT_FORMAT}。",
    ),
    parser=parse,
    checker=check,
    length_field="reasoning_content",
)
