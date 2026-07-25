"""Format-following probe requiring exactly three words."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from vestigia.prompts.base import PromptTemplate


def parse(response: str) -> dict[str, Any]:
    words = re.findall(r"[\w\u4e00-\u9fff]+", response)
    return {"words": words, "word_count": len(words)}


def check(_: str, parsed: Mapping[str, Any]) -> bool:
    return parsed.get("word_count") == 3


PROMPT = PromptTemplate(
    id="instruction_following",
    category="format",
    variants=(
        "只用三个词描述一只猫。",
        "请恰好写三个词，形容猫。",
        "不要解释；用三个词描绘一只猫。",
        "用三个词回答：猫是什么样的？",
        "给出三个描述猫的词，不多不少。",
    ),
    parser=parse,
    checker=check,
)
