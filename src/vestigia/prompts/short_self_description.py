"""Self-reference probe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vestigia.prompts.base import PromptTemplate, has_nonempty_text, nonempty_text_parser


def parse(response: str) -> dict[str, Any]:
    return nonempty_text_parser(response)


def check(response: str, parsed: Mapping[str, Any]) -> bool:
    return has_nonempty_text(response, parsed)


PROMPT = PromptTemplate(
    id="short_self_description",
    category="self_reference",
    variants=(
        "请用一句话介绍你自己。",
        "用不超过二十个字描述你是什么。",
        "你会怎样简短地向陌生人说明自己？",
        "给出一句关于你自身能力的概述。",
        "请以一句简洁的话说明你的角色。",
    ),
    parser=parse,
    checker=check,
)
