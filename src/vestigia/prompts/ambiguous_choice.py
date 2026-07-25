"""Ambiguous choice probe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vestigia.prompts.base import PromptTemplate, nonempty_text_parser


def parse(response: str) -> dict[str, Any]:
    parsed = nonempty_text_parser(response)
    has_beach = "海边" in response
    has_mountain = any(word in response for word in ("山里", "山中", "徒步"))
    parsed["choice"] = "beach" if has_beach else "mountain" if has_mountain else None
    return parsed


def check(_: str, parsed: Mapping[str, Any]) -> bool:
    return parsed.get("choice") in {"beach", "mountain"}


PROMPT = PromptTemplate(
    id="ambiguous_choice",
    category="reasoning",
    variants=(
        "在海边散步和在山里徒步之间，你会建议选哪个？只给一个倾向和理由。",
        "只能在海边散步与山中徒步中二选一，你更推荐哪项？为什么？",
        "请在海边散步、山里徒步中选一个作为周末活动，并简要解释。",
        "若没有额外背景，你会怎么在海边散步和山中徒步之间做选择？",
        "给一个周末活动建议：海边散步还是山里徒步？说明理由。",
    ),
    parser=parse,
    checker=check,
)
