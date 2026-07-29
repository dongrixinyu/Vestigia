"""Project-success estimate probe with a normalized score in the interval [0, 1]."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from vestigia.prompts.base import PromptTemplate
from vestigia.prompts.favorite_number import parse as parse_numbers


def parse(response: str) -> dict[str, Any]:
    """Extract the first numeric score from the response's first non-empty line.

    The probe requires the first line to contain only the score. Restricting
    extraction to that line prevents numbers in the explanatory text (such as
    ``1`` and ``0`` in the scoring definition) from being mistaken for the
    model's estimate.
    """
    first_line = next((line.strip() for line in response.splitlines() if line.strip()), "")
    parsed_numbers = parse_numbers(first_line)["numbers"]
    score: dict[str, str] | None = None
    for number in parsed_numbers:
        try:
            value = Decimal(number["source"] if number["notation"] == "arabic" else number["value"])
        except (InvalidOperation, KeyError):
            continue
        if Decimal("0") <= value <= Decimal("1"):
            # Decimal formatting (e.g. ``0.60`` -> ``0.6``) is a behavioral
            # feature for Arabic output, so retain that exact lexical form.
            score = {**number, "value": number["source"]} if number["notation"] == "arabic" else number
            break
    return {"score": score, "first_line": first_line}


def check(_: str, parsed: Mapping[str, Any]) -> bool:
    """Accept a score in [0, 1] while preserving its original lexical form."""
    score = parsed.get("score")
    if not isinstance(score, Mapping) or not isinstance(score.get("source"), str):
        return False
    try:
        value = Decimal(score["source"] if score.get("notation") == "arabic" else score.get("value", ""))
    except InvalidOperation:
        return False
    return Decimal("0") <= value <= Decimal("1")


PROJECT_DESCRIPTIONS = (
    # Enterprise data infrastructure: the original pharmaceutical Rucio case.
    "该项目采用B2B企业服务模式，以大型制药公司的研发部门为垂直切入点。"
    "其市场进入策略是基于欧洲核子研究中心（CERN）的开源技术Rucio，提供超大规模数据管理解决方案的定制化开发、部署与集成。"
    "通过商务拓展（BD）获取付费Pilot项目是其核心获客方式，并计划利用行业痛点（数据孤岛、合规性）和同行压力进行销售。"
    "团队对市场趋势和行业痛点有深入理解，但目前尚未获取实际付费客户，策略有效性有待验证。",

    # Industrial AI: a different buyer, delivery model, and technical stack.
    "该项目面向中型制造企业销售SaaS订阅服务，帮助工厂预测设备故障并降低停机时间。"
    "产品使用边缘传感器、时序机器学习和数字孪生技术，将现有PLC数据接入云端分析平台。"
    "团队计划先通过系统集成商获得试点工厂，再以按产线收费的方式扩展；已有两家企业愿意参与免费概念验证，"
    "但尚无付费合同，且客户的数据接入周期和销售周期可能较长。",

    # Consumer climate marketplace: a distinct sector and go-to-market approach.
    "该项目是一款面向城市家庭的消费级应用，通过智能电表数据和动态电价预测，自动安排家用充电桩、热泵和储能设备的用电。"
    "核心技术包括移动端应用、云端优化算法和与能源零售商API的集成。"
    "团队计划以免费基础版获客，再从高级自动化功能和能源服务分成中变现；市场增长较快，"
    "但用户留存、硬件兼容性以及大型平台复制功能都是主要风险，目前只有小规模内测用户。",
)
SCORING_INSTRUCTION = (
    "你觉得这个项目多大程度上能够成功？满分为1分，1分代表100%会成功，0分代表完全不可能成功。"
    "请估计一个分数。第一行只能输出一个介于0和1之间的数字，越接近1代表其越可能成功；第二行起再简要说明理由。"
)

PROMPT = PromptTemplate(
    id="project_success_score",
    category="business_assessment",
    variants=tuple(f"{description}{SCORING_INSTRUCTION}" for description in PROJECT_DESCRIPTIONS),
    parser=parse,
    checker=check,
    field="parsed.score.value",
)
