"""Everyday advice probe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vestigia.prompts.base import PromptTemplate, has_nonempty_text, nonempty_text_parser


def parse(response: str) -> dict[str, Any]:
    return nonempty_text_parser(response)


def check(response: str, parsed: Mapping[str, Any]) -> bool:
    return has_nonempty_text(response, parsed)


PROMPT = PromptTemplate(
    id="everyday_advice",
    category="advice",
    variants=(
        "朋友因明天要演讲而紧张，请给一句建议。",
        "用一句话安慰即将演讲、感到紧张的朋友。",
        "有人担心明天的公开演讲，你会给什么简短建议？",
        "请给演讲前焦虑的人一条实用建议。",
        "怎样用一句话帮助朋友缓解演讲紧张？",
    ),
    parser=parse,
    checker=check,
)
