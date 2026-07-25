"""Creative association probe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vestigia.prompts.base import PromptTemplate, has_nonempty_text, nonempty_text_parser


def parse(response: str) -> dict[str, Any]:
    return nonempty_text_parser(response)


def check(response: str, parsed: Mapping[str, Any]) -> bool:
    return has_nonempty_text(response, parsed)


PROMPT = PromptTemplate(
    id="creative_association",
    category="creative",
    variants=(
        "把雨天比作一种食物，并说明理由。",
        "请用一种食物形容雨天，理由要简短。",
        "雨天最像什么食物？请解释一下。",
        "想一个关于雨天和食物的比喻。",
        "若雨天是一道食物，它会是什么？为什么？",
    ),
    parser=parse,
    checker=check,
)
