"""Simple explanation probe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vestigia.prompts.base import PromptTemplate, has_nonempty_text, nonempty_text_parser


def parse(response: str) -> dict[str, Any]:
    return nonempty_text_parser(response)


def check(response: str, parsed: Mapping[str, Any]) -> bool:
    return has_nonempty_text(response, parsed)


PROMPT = PromptTemplate(
    id="simple_explanation",
    category="explanation",
    variants=(
        "向八岁孩子解释为什么会有影子。",
        "请用小朋友能懂的话说明影子是怎么来的。",
        "为什么人会有影子？请做一个面向儿童的简短解释。",
        "假设听众是八岁儿童，解释一下影子的形成。",
        "用简单的话回答：影子为什么会出现？",
    ),
    parser=parse,
    checker=check,
)
