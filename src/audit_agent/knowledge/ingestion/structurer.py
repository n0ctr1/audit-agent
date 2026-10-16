"""LLM-based structurer: raw Markdown/HTML finding → canonical Finding JSON.

Extracts structured data using Claude's tool_use for reliable JSON output.
Works with any raw audit report text.
"""

from __future__ import annotations

import json
import re
from typing import Any

from audit_agent.llm import get_provider
from audit_agent.llm.base import LLMProvider, ToolDefinition
from audit_agent.knowledge.models import (
    CodeSnippet,
    DescriptionStructured,
    Finding,
    Fix,
    ProtocolMeta,
    Severity,
    Source,
)
from audit_agent.observability.logging import get_logger

log = get_logger("knowledge.structurer")

_TOOL_DEF = ToolDefinition(
    name="extract_finding",
    description="Extract a structured security finding from raw audit report text.",
    parameters={
        "type": "object",
        "required": ["title", "severity", "vulnerability_class", "custom_tags",
                     "protocol_name", "protocol_category",
                     "summary", "root_cause", "attack_vector", "impact", "preconditions",
                     "code_snippets", "poc", "fix_description", "fix_code_diff", "references"],
        "properties": {
            "title": {"type": "string", "description": "Short, descriptive title of the vulnerability"},
            "severity": {
                "type": "string",
                "enum": ["critical", "high", "medium", "low", "info"],
                "description": "Severity level as reported",
            },
            "vulnerability_class": {
                "type": "string",
                "description": "SWC number (e.g. SWC-107) or category name (e.g. Reentrancy, Oracle Manipulation)",
            },
            "custom_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Keyword tags: e.g. ['price-manipulation', 'twap', 'oracle']",
            },
            "protocol_name": {"type": "string"},
            "protocol_category": {
                "type": "string",
                "description": "e.g. DEX, lending, bridge, NFT, governance",
            },
            "summary": {"type": "string", "description": "1-3 sentence plain-language summary"},
            "root_cause": {"type": "string", "description": "Technical root cause of the bug"},
            "attack_vector": {"type": "string", "description": "How the attacker triggers the bug"},
            "impact": {"type": "string", "description": "What damage the attacker can cause"},
            "preconditions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Required conditions for the attack to succeed",
            },
            "code_snippets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "lines": {"type": "string"},
                        "code": {"type": "string"},
                    },
                    "required": ["file", "lines", "code"],
                },
                "description": "Relevant vulnerable code snippets",
            },
            "poc": {"type": "string", "description": "Proof-of-concept exploit code, if present"},
            "fix_description": {"type": "string", "description": "Recommended fix in plain language"},
            "fix_code_diff": {"type": "string", "description": "Suggested code fix or diff"},
            "references": {
                "type": "array",
                "items": {"type": "string"},
                "description": "GitHub links, EIPs, or other references",
            },
        },
    },
)

_SYSTEM_PROMPT = """\
You are a senior smart contract security auditor. Your task is to extract structured \
information from raw audit finding reports. Be precise: extract only what is stated in \
the report. If a field is missing from the source text, return an empty string or empty list.
"""

_USER_TEMPLATE = """\
Extract the structured finding from this audit report:

SOURCE: {source}
URL: {url}

---
{raw_text}
---
"""


class Structurer:
    """LLM-based extractor: Markdown/HTML → canonical Finding."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        model: str | None = None,
    ) -> None:
        self._provider = provider or get_provider()
        if model is None:
            from audit_agent.config import get_settings
            model = get_settings().scout_model
        self._model = model

    async def structure(
        self,
        raw_text: str,
        source: Source,
        url: str = "",
        finding_id: str | None = None,
    ) -> Finding:
        """Extract a canonical Finding from raw Markdown/HTML text."""
        log.debug("structurer.structure", source=source, url=url, chars=len(raw_text))

        user_msg = _USER_TEMPLATE.format(source=source, url=url, raw_text=raw_text[:12_000])

        from audit_agent.llm.base import LLMResponse
        response = await self._provider.complete(
            model=self._model,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            tools=[_TOOL_DEF],
            tool_choice={"name": "extract_finding"},
            messages=[{"role": "user", "content": user_msg}],
        )

        extracted = self._parse_tool_response(response)
        return self._build_finding(extracted, raw_text, source, url, finding_id)

    def _parse_tool_response(self, response: Any) -> dict[str, Any]:
        from audit_agent.llm.base import LLMResponse
        if isinstance(response, LLMResponse):
            for tc in response.tool_calls:
                if tc.name == "extract_finding":
                    return tc.arguments
            raise ValueError(f"LLM did not call extract_finding tool; tool_calls: {response.tool_calls}")
        # Fallback for raw anthropic response (backward compat)
        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_finding":
                return block.input  # type: ignore[return-value]
        raise ValueError(f"LLM did not call extract_finding tool")

    def _build_finding(
        self,
        data: dict[str, Any],
        raw_text: str,
        source: Source,
        url: str,
        finding_id: str | None,
    ) -> Finding:
        snippets = [
            CodeSnippet(
                file=s.get("file", ""),
                lines=s.get("lines", ""),
                code=s.get("code", ""),
            )
            for s in (data.get("code_snippets") or [])
        ]
        finding = Finding(
            title=data.get("title", ""),
            severity=_coerce_severity(data.get("severity", "medium")),
            source=source,
            url=url,
            vulnerability_class=data.get("vulnerability_class", ""),
            custom_tags=data.get("custom_tags") or [],
            protocol=ProtocolMeta(
                name=data.get("protocol_name", ""),
                category=data.get("protocol_category", ""),
            ),
            description_structured=DescriptionStructured(
                summary=data.get("summary", ""),
                root_cause=data.get("root_cause", ""),
                attack_vector=data.get("attack_vector", ""),
                impact=data.get("impact", ""),
                preconditions=data.get("preconditions") or [],
            ),
            code_snippets=snippets,
            poc=data.get("poc", ""),
            fix=Fix(
                description=data.get("fix_description", ""),
                code_diff=data.get("fix_code_diff", ""),
            ),
            references=data.get("references") or [],
            raw_text=raw_text,
        )
        if finding_id:
            finding.id = finding_id
        return finding


def _coerce_severity(s: str) -> Severity:
    s = s.lower().strip()
    if s in ("critical", "high", "medium", "low", "info"):
        return s  # type: ignore[return-value]
    if s in ("h", "high-risk"):
        return "high"
    if s in ("m", "med", "medium-risk"):
        return "medium"
    if s in ("c",):
        return "critical"
    if s in ("l", "low-risk"):
        return "low"
    return "medium"


def extract_severity_from_markdown(text: str) -> Severity:
    """Heuristic severity extraction without LLM call — used by scrapers for pre-filtering."""
    for line in text.splitlines()[:20]:
        lower = line.lower()
        for sev in ("critical", "high", "medium", "low"):
            if sev in lower and ("severity" in lower or line.strip().startswith("#")):
                return sev  # type: ignore[return-value]
    # Fallback: H-XX prefix in title
    m = re.search(r"\b([HMC])-\d+\b", text[:200])
    if m:
        prefix = m.group(1)
        return {"H": "high", "M": "medium", "C": "critical"}.get(prefix, "medium")  # type: ignore[return-value]
    return "medium"
