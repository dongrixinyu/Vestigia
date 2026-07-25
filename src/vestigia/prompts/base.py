"""Shared contracts and helpers for individual fingerprint probes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

Parser = Callable[[str], Mapping[str, Any]]
Checker = Callable[[str, Mapping[str, Any]], bool]


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A probe with variant phrasings and response-analysis functions."""

    id: str
    category: str
    variants: tuple[str, ...]
    parser: Parser
    checker: Checker


def nonempty_text_parser(text: str) -> dict[str, Any]:
    """Extract normalized text for open-ended probes."""
    normalized = " ".join(text.split())
    return {"text": normalized, "length": len(normalized)}


def has_nonempty_text(_: str, parsed: Mapping[str, Any]) -> bool:
    """Accept a response when its normalized text is non-empty."""
    return bool(parsed.get("text"))
