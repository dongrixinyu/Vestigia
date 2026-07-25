"""Fixed, varied prompt templates for collecting LLM behavioral samples."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """One probe family and equivalent ways to ask the same question."""

    id: str
    category: str
    variants: tuple[str, ...]


DEFAULT_PROMPTS: tuple[PromptTemplate, ...] = (
    PromptTemplate(
        id="favorite_number",
        category="preference",
        variants=(
            "给我说一个你最喜欢的数字。",
            "如果只能选一个数字，你最喜欢哪一个？",
            "请告诉我一个你偏爱的数字，并简单说明原因。",
            "你会选择哪个数字作为自己最喜欢的数字？",
            "随意说一个你最喜欢的数字即可。",
        ),
    ),
    PromptTemplate(
        id="short_self_description",
        category="self_reference",
        variants=(
            "请用一句话介绍你自己。",
            "用不超过二十个字描述你是什么。",
            "你会怎样简短地向陌生人说明自己？",
            "给出一句关于你自身能力的概述。",
            "请以一句简洁的话说明你的角色。",
        ),
    ),
    PromptTemplate(
        id="creative_association",
        category="creative",
        variants=(
            "把雨天比作一种食物，并说明理由。",
            "请用一种食物形容雨天，理由要简短。",
            "雨天最像什么食物？请解释一下。",
            "想一个关于雨天和食物的比喻。",
            "若雨天是一道食物，它会是什么？为什么？",
        ),
    ),
    PromptTemplate(
        id="ambiguous_choice",
        category="reasoning",
        variants=(
            "在海边散步和在山里徒步之间，你会建议选哪个？只给一个倾向和理由。",
            "只能在海边散步与山中徒步中二选一，你更推荐哪项？为什么？",
            "请在海边散步、山里徒步中选一个作为周末活动，并简要解释。",
            "若没有额外背景，你会怎么在海边散步和山中徒步之间做选择？",
            "给一个周末活动建议：海边散步还是山里徒步？说明理由。",
        ),
    ),
    PromptTemplate(
        id="instruction_following",
        category="format",
        variants=(
            "只用三个词描述一只猫。",
            "请恰好写三个词，形容猫。",
            "不要解释；用三个词描绘一只猫。",
            "用三个词回答：猫是什么样的？",
            "给出三个描述猫的词，不多不少。",
        ),
    ),
    PromptTemplate(
        id="everyday_advice",
        category="advice",
        variants=(
            "朋友因明天要演讲而紧张，请给一句建议。",
            "用一句话安慰即将演讲、感到紧张的朋友。",
            "有人担心明天的公开演讲，你会给什么简短建议？",
            "请给演讲前焦虑的人一条实用建议。",
            "怎样用一句话帮助朋友缓解演讲紧张？",
        ),
    ),
    PromptTemplate(
        id="word_association",
        category="language",
        variants=(
            "看到“灯塔”这个词，你首先想到什么？",
            "“灯塔”让你联想到什么？简短回答。",
            "请说出你对灯塔的第一个联想。",
            "用一句话写下“灯塔”带给你的联想。",
            "灯塔这个词最容易让你想到什么？",
        ),
    ),
    PromptTemplate(
        id="simple_explanation",
        category="explanation",
        variants=(
            "向八岁孩子解释为什么会有影子。",
            "请用小朋友能懂的话说明影子是怎么来的。",
            "为什么人会有影子？请做一个面向儿童的简短解释。",
            "假设听众是八岁儿童，解释一下影子的形成。",
            "用简单的话回答：影子为什么会出现？",
        ),
    ),
)


def iter_prompts(
    count: int, templates: Sequence[PromptTemplate] = DEFAULT_PROMPTS
) -> Iterator[tuple[str, PromptTemplate]]:
    """Yield exactly ``count`` probes, cycling template and wording evenly."""
    if count < 1:
        raise ValueError("count must be greater than zero")
    if not templates:
        raise ValueError("templates must not be empty")
    for index in range(count):
        template = templates[index % len(templates)]
        variant = template.variants[(index // len(templates)) % len(template.variants)]
        yield variant, template
