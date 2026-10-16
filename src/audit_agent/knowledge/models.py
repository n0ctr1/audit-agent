"""Canonical data models for the knowledge base."""

from __future__ import annotations

import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Severity = Literal["critical", "high", "medium", "low", "info"]
Source = Literal[
    "code4rena", "solodit", "sherlock", "immunefi", "rekt",
    "defihacklabs", "arxiv", "manual",
]


class ProtocolMeta(BaseModel):
    name: str = ""
    category: str = ""  # DEX / lending / bridge / etc.


class DescriptionStructured(BaseModel):
    summary: str = ""
    root_cause: str = ""
    attack_vector: str = ""
    impact: str = ""
    preconditions: list[str] = Field(default_factory=list)


class CodeSnippet(BaseModel):
    file: str = ""
    lines: str = ""
    code: str = ""


class Fix(BaseModel):
    description: str = ""
    code_diff: str = ""


class Finding(BaseModel):
    """Canonical KB entry — the single output schema for all scrapers + structurer."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    severity: Severity
    source: Source
    url: str = ""

    vulnerability_class: str = ""   # SWC-XXX or custom tag
    custom_tags: list[str] = Field(default_factory=list)

    protocol: ProtocolMeta = Field(default_factory=ProtocolMeta)
    description_structured: DescriptionStructured = Field(
        default_factory=DescriptionStructured
    )
    code_snippets: list[CodeSnippet] = Field(default_factory=list)
    poc: str = ""
    fix: Fix = Field(default_factory=Fix)
    references: list[str] = Field(default_factory=list)

    # Internal quality / pipeline fields
    raw_text: str = ""
    quality_score: float = 0.0
    embedding_text: str = ""   # the text that was embedded (set by pipeline)

    def to_search_result(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "severity": self.severity,
            "vulnerability_class": self.vulnerability_class,
            "custom_tags": self.custom_tags,
            "summary": self.description_structured.summary,
        }
