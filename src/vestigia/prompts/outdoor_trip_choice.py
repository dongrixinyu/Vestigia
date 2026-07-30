"""Six-way outdoor-trip choice probe with an explicit parse-error bucket."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vestigia.prompts.base import PromptTemplate

# ASCII category codes deliberately avoid Chinese-character variants, punctuation,
# and whitespace ambiguity in the feature distribution.
CATEGORIES = (
    "beach",
    "mountain",
    "lakeside",
    "forest",
    "old_town",
    "grassland",
)
PARSE_ERROR = "parse_error"

OPTIONS = """可选方案及其唯一类别码：
- lakeside：环湖慢行，沿着水岸平缓步道行走，湖水纯蓝平静，抚平心灵涟漪。
- beach：海边平缓散步，听海浪、闻海风湿润气息。
- mountain：山野徒步，攀登隐秘小径，登高眺望云山雾绕秀美远景。
- forest：原始森林林间漫游，树荫茂密，植被丰富，天然森林氧吧。
- old_town：老城街巷闲逛，探访精致文化小店、感受人文烟火，品尝传统美食。
- grassland：草原旷野漫步，视野开阔，地势平坦，远望牛羊遍野，天空悠远。
"""

OUTPUT_RULE = (
    "请从上述六个类别码中选择唯一一个最优选项。正式回答必须且只能输出该类别码本身："
    "beach、mountain、lakeside、forest、old_town 或 grassland。"
    "不要输出中文名称、数字、理由、标点、Markdown 或任何其他字符。"
)


def parse(response: str) -> dict[str, Any]:
    """Return one approved category code, or ``parse_error`` for every invalid reply.

    Exact matching is intentional: it preserves strict output-following behavior
    as part of the fingerprint. A response containing a valid code plus any
    additional character is not silently accepted.
    """
    choice = response if response in CATEGORIES else PARSE_ERROR
    return {"choice": choice}


def check(_: str, parsed: Mapping[str, Any]) -> bool:
    """Report whether the model returned one of the six required categories."""
    return parsed.get("choice") in CATEGORIES


PROMPT = PromptTemplate(
    id="outdoor_trip_choice",
    category="preference",
    variants=(
        f"你计划安排半天户外放空行程，你有如下的选择。\n\\n{OPTIONS}\n{OUTPUT_RULE}",
        f"假设你必须为一个周末下午选择一种出游方式。{OPTIONS}\n{OUTPUT_RULE}",
        f"没有其他背景信息时，请做一次唯一的户外活动推荐。{OPTIONS}\n{OUTPUT_RULE}",
        f"请从下列六种轻量出游方案中选出你认为最优的一项。{OPTIONS}\n{OUTPUT_RULE}",
        f"现在需要确定一个半日出游方案，不能并列或保留多个备选。{OPTIONS}\n{OUTPUT_RULE}",
    ),
    parser=parse,
    checker=check,
    field="parsed.choice",
)
