"""Word-association probe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vestigia.prompts.base import PromptTemplate, has_nonempty_text, nonempty_text_parser


def parse(response: str) -> dict[str, Any]:
    return nonempty_text_parser(response)


def check(response: str, parsed: Mapping[str, Any]) -> bool:
    return has_nonempty_text(response, parsed)


PROMPT = PromptTemplate(
    id="word_association",
    category="language",
    variants=(
        "看到“灯塔”这个词，你首先想到什么？",
        "“灯塔”让你联想到什么？简短回答。",
        "请说出你对灯塔的第一个联想。",
        "用一句话写下“灯塔”带给你的联想。",
        "灯塔这个词最容易让你想到什么？",
    ),
    parser=parse,
    checker=check,
)
