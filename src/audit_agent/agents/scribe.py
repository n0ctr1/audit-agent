"""Scribe agent — platform-specific report generator (§3.6).

Pipeline per ValidatedFinding (status == "validated"):
  1. Locate matching Hypothesis and ExploitResult
  2. Call LLM via generate_report tool to produce structured prose fields
  3. On LLM failure: construct fields deterministically from existing data
  4. Render fields into platform-specific Markdown template
  5. Store rendered Markdown in state["reports"][finding_id]
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from audit_agent.agents.base import BaseAgent
from audit_agent.graph.state import AuditState
from audit_agent.knowledge.scope_filter import (
    assess_scope,
    extract_known_issues,
    scope_banner,
)
from audit_agent.models.schemas import (
    AuditPlatform,
    ExploitResult,
    Hypothesis,
    ValidatedFinding,
)
from audit_agent.llm import get_provider
from audit_agent.llm.base import LLMProvider, ToolDefinition
from audit_agent.observability.audit_logger import get_audit_logger
from audit_agent.observability.logging import get_logger

log = get_logger("agents.scribe")


# ─────────────────────────────────────────────────────────────────────────────
# LLM tool definition
# ─────────────────────────────────────────────────────────────────────────────

_REPORT_TOOL = ToolDefinition(
    name="generate_report",
    description=(
        "Generate structured prose fields for a vulnerability report. "
        "Call this exactly once per finding."
    ),
    parameters={
        "type": "object",
        "required": ["summary", "vulnerability_detail", "impact", "recommendation"],
        "properties": {
            "summary": {
                "type": "string",
                "description": "1–2 sentence summary of the vulnerability.",
            },
            "vulnerability_detail": {
                "type": "string",
                "description": "Technical root-cause explanation with exact function/line references.",
            },
            "impact": {
                "type": "string",
                "description": "Concrete worst-case impact (who, what assets, under what conditions).",
            },
            "recommendation": {
                "type": "string",
                "description": "Specific, actionable fix with code-level detail.",
            },
        },
    },
)

_SCRIBE_SYSTEM = (
    "You are a professional smart contract security report writer. "
    "Produce precise, evidence-driven vulnerability reports. "
    "Every claim must be supported by the provided exploit evidence. "
    "Call generate_report exactly once."
)


# ─────────────────────────────────────────────────────────────────────────────
# Platform Markdown renderers
# ─────────────────────────────────────────────────────────────────────────────

def _severity_prefix(severity: str, platform: AuditPlatform) -> str:
    """Return platform-specific severity label."""
    if platform == "c4":
        label = {"critical": "H", "high": "H", "medium": "M", "low": "L", "info": "I"}.get(
            severity.lower(), "M"
        )
        return f"[{label}]"
    return severity.capitalize()


def render_report(
    finding: ValidatedFinding,
    fields: dict[str, str],
    platform: AuditPlatform,
    poc_code: Optional[str] = None,
    reproduction_steps: Optional[list[str]] = None,
) -> str:
    """Render a structured report dict into platform-specific Markdown."""
    summary = fields.get("summary", "")
    vuln_detail = fields.get("vulnerability_detail", "")
    impact = fields.get("impact", "")
    recommendation = fields.get("recommendation", "")
    severity = finding.severity

    poc_block = ""
    if poc_code:
        poc_block = f"```solidity\n{poc_code}\n```"

    steps_block = ""
    if reproduction_steps:
        steps_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(reproduction_steps))

    if platform == "c4":
        prefix = _severity_prefix(severity, platform)
        return (
            f"## {prefix} {finding.title}\n\n"
            f"### Summary\n{summary}\n\n"
            f"### Vulnerability Detail\n{vuln_detail}\n\n"
            f"### Impact\n{impact}\n\n"
            f"### Proof of Concept\n"
            + (f"{poc_block}\n\n" if poc_block else "(No PoC available)\n\n")
            + (f"Reproduction steps:\n{steps_block}\n\n" if steps_block else "")
            + f"### Recommended Mitigation Steps\n{recommendation}\n"
        )

    if platform == "immunefi":
        return (
            f"# Vulnerability Report\n\n"
            f"## Title\n{finding.title}\n\n"
            f"## Severity\n{severity.capitalize()}\n\n"
            f"## Description\n{vuln_detail}\n\n"
            f"## Proof of Concept\n"
            + (f"{poc_block}\n\n" if poc_block else "(No PoC available)\n\n")
            + f"## Impact\n{impact}\n\n"
            f"## Recommended Fix\n{recommendation}\n"
        )

    if platform == "sherlock":
        return (
            f"## Summary\n{summary}\n\n"
            f"## Vulnerability Detail\n{vuln_detail}\n\n"
            f"## Impact\n{impact}\n\n"
            f"## Code Snippet\n"
            + (f"{poc_block}\n\n" if poc_block else "(No snippet available)\n\n")
            + f"## Tool used\nManual Review"
            + (" / Foundry" if poc_code else "")
            + f"\n\n## Recommendation\n{recommendation}\n"
        )

    # cantina / spearbit / generic — full professional report
    return (
        f"# {finding.title}\n\n"
        f"**Severity:** {severity.capitalize()}  \n"
        f"**Confidence:** {finding.confidence:.0%}  \n"
        f"**Affected:** {', '.join(finding.affected_functions) or 'N/A'}\n\n"
        f"## Summary\n{summary}\n\n"
        f"## Vulnerability Detail\n{vuln_detail}\n\n"
        f"## Impact\n{impact}\n\n"
        f"## Proof of Concept\n"
        + (f"{poc_block}\n\n" if poc_block else "(No PoC available)\n\n")
        + (f"### Reproduction Steps\n{steps_block}\n\n" if steps_block else "")
        + f"## Recommendation\n{recommendation}\n"
        + (f"\n## Skeptic Notes\n{finding.skeptic_notes}\n" if finding.skeptic_notes else "")
    )


# ─────────────────────────────────────────────────────────────────────────────
# ScribeAgent
# ─────────────────────────────────────────────────────────────────────────────

class ScribeAgent(BaseAgent):
    """Generates platform-specific Markdown reports for validated findings."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: str | None = None,
    ) -> None:
        super().__init__()
        self._provider = provider or get_provider()
        if model is None:
            from audit_agent.config import get_settings
            model = get_settings().scribe_model
        self._model = model

    # ── LangGraph node ────────────────────────────────────────────────────

    async def run(self, state: AuditState) -> AuditState:
        audit_id = state.get("audit_id", "unknown")
        validated: list[ValidatedFinding] = list(state.get("validated_findings") or [])
        hypotheses: list[Hypothesis] = list(state.get("hypotheses") or [])
        exploits: list[ExploitResult] = list(state.get("exploits") or [])
        platform: AuditPlatform = state.get("platform", "generic")  # type: ignore[assignment]

        true_positives = [f for f in validated if f.status == "validated"]
        needs_review = [f for f in validated if f.status == "unverified"]

        self.log.info(
            "Scribe: starting",
            audit_id=audit_id,
            true_positives=len(true_positives),
            needs_review=len(needs_review),
            platform=platform,
        )

        al = get_audit_logger()
        ar = al.log_agent_start("scribe", 6, {
            "audit_id": audit_id,
            "true_positives": len(true_positives),
            "needs_review": len(needs_review),
            "platform": platform,
        }) if al else None

        hyp_by_id: dict[str, Hypothesis] = {h.id: h for h in hypotheses}
        exploit_by_hyp_id: dict[str, ExploitResult] = {
            e.hypothesis_id: e for e in exploits
        }

        reports: dict[str, str] = dict(state.get("reports") or {})
        in_scope_files: list[str] = list(state.get("scope_in_scope") or [])
        # Project-declared known issues, mined once from the collected docs.
        known_issues = extract_known_issues(state.get("documentation") or "")

        def _scope(f: ValidatedFinding) -> str:
            """Set the scope self-flag on the finding and return a header banner."""
            a = assess_scope(
                f.title, f.vulnerability_class, f.description, f.severity,
                list(f.affected_functions or []), platform, known_issues,
            )
            f.scope_flag, f.scope_note = a.flag, a.note
            return scope_banner(a)

        for finding in true_positives:
            hyp = hyp_by_id.get(finding.hypothesis_id or "")
            exploit = exploit_by_hyp_id.get(finding.hypothesis_id or "")

            banner = _scope(finding)
            fields = await self._generate_fields(finding, hyp, exploit, in_scope_files)
            markdown = render_report(
                finding=finding,
                fields=fields,
                platform=platform,
                poc_code=exploit.poc_code if exploit else None,
                reproduction_steps=exploit.reproduction_steps if exploit else None,
            )
            markdown = banner + "\n\n" + markdown
            reports[finding.id] = markdown
            self.log.info(
                "Scribe: report generated",
                finding_id=finding.id, platform=platform, chars=len(markdown),
            )
            if al:
                al.log_decision("scribe", "report_generated",
                    f"[{finding.severity}] {finding.title} → {platform}",
                    {"finding_id": finding.id, "severity": finding.severity,
                     "platform": platform, "report_length": len(markdown)})

        for finding in needs_review:
            hyp = hyp_by_id.get(finding.hypothesis_id or "")
            exploit = exploit_by_hyp_id.get(finding.hypothesis_id or "")
            banner = _scope(finding)
            markdown = banner + "\n\n" + _render_needs_review(finding, hyp, exploit)
            reports[finding.id] = markdown
            self.log.info(
                "Scribe: needs_review entry added",
                finding_id=finding.id, chars=len(markdown),
            )

        self.log.info("Scribe: complete", reports=len(reports))

        if al and ar:
            al.log_agent_end(ar, {
                "reports_generated": len(reports),
                "true_positives": len(true_positives),
                "needs_review_stubs": len(needs_review),
            })

        return {**state, "reports": reports}

    # ── Field generation ──────────────────────────────────────────────────

    async def _generate_fields(
        self,
        finding: ValidatedFinding,
        hypothesis: Optional[Hypothesis],
        exploit: Optional[ExploitResult],
        in_scope_files: Optional[list[str]] = None,
    ) -> dict[str, str]:
        """Try LLM first; fall back to deterministic construction."""
        llm_fields = await self._llm_generate(finding, hypothesis, exploit, in_scope_files)
        if llm_fields is not None:
            return llm_fields
        return _deterministic_fields(finding, hypothesis, exploit)

    async def _llm_generate(
        self,
        finding: ValidatedFinding,
        hypothesis: Optional[Hypothesis],
        exploit: Optional[ExploitResult],
        in_scope_files: Optional[list[str]] = None,
    ) -> Optional[dict[str, str]]:
        prompt = _build_scribe_prompt(finding, hypothesis, exploit, in_scope_files)
        al = get_audit_logger()
        llm_record = None
        if al:
            llm_record = al.log_llm_call_start(
                agent="scribe",
                purpose=f"report_{finding.hypothesis_id or finding.id}",
                model=self._model,
                system_prompt=_SCRIBE_SYSTEM,
                user_message=prompt,
                tools=[_REPORT_TOOL],
                max_tokens=4096,
            )
        t0 = time.time()
        try:
            response = await asyncio.wait_for(
                self._provider.complete(
                    model=self._model,
                    max_tokens=4096,
                    system=_SCRIBE_SYSTEM,
                    tools=[_REPORT_TOOL],
                    tool_choice={"name": "generate_report"},
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=240.0,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            duration = time.time() - t0
            err = "timeout" if isinstance(exc, asyncio.TimeoutError) else str(exc)
            if al and llm_record:
                al.log_llm_call_end(llm_record, error=err, duration_seconds=duration)
            log.warning("scribe.llm_unavailable", error=err)
            return None

        duration = response.duration_seconds or (time.time() - t0)
        fields: Optional[dict[str, str]] = None
        for tc in response.tool_calls:
            if tc.name == "generate_report":
                fields = {
                    "summary": tc.arguments.get("summary", ""),
                    "vulnerability_detail": tc.arguments.get("vulnerability_detail", ""),
                    "impact": tc.arguments.get("impact", ""),
                    "recommendation": tc.arguments.get("recommendation", ""),
                }

        if al and llm_record:
            al.log_llm_call_end(
                llm_record,
                tool_calls=[{"name": "generate_report"}] if fields else [],
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                stop_reason=response.stop_reason,
                duration_seconds=duration,
            )
        return fields


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_scribe_prompt(
    finding: ValidatedFinding,
    hypothesis: Optional[Hypothesis],
    exploit: Optional[ExploitResult],
    in_scope_files: Optional[list[str]] = None,
) -> str:
    from pathlib import Path as _Path
    parts: list[str] = [
        f"## Validated Finding\n"
        f"Title: {finding.title}\n"
        f"Severity: {finding.severity}\n"
        f"Confidence: {finding.confidence:.0%}\n"
        f"Affected functions: {finding.affected_functions}\n"
        f"Description: {finding.description}"
    ]

    if in_scope_files:
        file_names = [_Path(f).name for f in in_scope_files]
        parts.append(
            f"\n## In-Scope Files (reference only these in 'Affected Code' sections)\n"
            + "\n".join(f"- {n}" for n in file_names)
        )

    if finding.skeptic_notes:
        parts.append(f"\n## Skeptic Notes\n{finding.skeptic_notes}")

    if hypothesis:
        parts.append(
            f"\n## Original Hypothesis\n"
            f"Attack scenario: {hypothesis.attack_scenario}\n"
            f"Preconditions: {hypothesis.required_preconditions}"
        )

    if exploit:
        parts.append(
            f"\n## Exploit Evidence\n"
            f"Status: {exploit.status}\n"
            f"Gas used: {exploit.gas_used}\n"
            f"Reproduction steps: {exploit.reproduction_steps[:5]}"
        )

    return "\n".join(parts)


def _render_needs_review(
    finding: ValidatedFinding,
    hypothesis: Optional[Hypothesis],
    exploit: Optional[ExploitResult],
) -> str:
    """Render a lightweight stub for findings that require manual triage."""
    affected = ", ".join(finding.affected_functions) or "N/A"
    poc_block = ""
    if exploit and exploit.poc_code:
        poc_block = f"\n```solidity\n{exploit.poc_code[:1000]}\n```\n"
    scenario = hypothesis.attack_scenario if hypothesis else "No scenario available."
    return (
        f"## Findings Requiring Manual Review\n\n"
        f"### {finding.title}\n\n"
        f"**Severity (estimated):** {finding.severity.capitalize()}  \n"
        f"**Confidence:** {finding.confidence:.0%}  \n"
        f"**Affected:** {affected}\n\n"
        f"**Why needs review:** The PoC did not succeed, but the vulnerability hypothesis "
        f"is plausible based on code analysis. A human auditor must determine whether this "
        f"is a real issue.\n\n"
        f"**Attack scenario:** {scenario}\n\n"
        f"**Skeptic notes:** {finding.skeptic_notes or 'None.'}\n"
        + (f"\n**Partial PoC:**{poc_block}" if poc_block else "")
    )


def _deterministic_fields(
    finding: ValidatedFinding,
    hypothesis: Optional[Hypothesis],
    exploit: Optional[ExploitResult],
) -> dict[str, str]:
    """Construct report fields without LLM from existing structured data."""
    affected = ", ".join(finding.affected_functions) or "affected function"

    summary = (
        f"The {affected} is vulnerable to {finding.vulnerability_class or finding.title}. "
        f"An attacker can exploit this to cause direct harm to users or the protocol."
    )

    if hypothesis:
        vuln_detail = (
            f"**Root cause:** {hypothesis.attack_scenario}\n\n"
            f"**Preconditions:** {', '.join(hypothesis.required_preconditions) or 'None documented'}"
        )
    else:
        vuln_detail = finding.description or "No technical detail available."

    if exploit and exploit.status == "exploited":
        impact = (
            f"Severity: **{finding.severity.capitalize()}**. "
            f"PoC test confirmed the exploit succeeds. "
            + (f"Gas consumed: {exploit.gas_used}. " if exploit.gas_used else "")
            + (finding.description or "")
        )
    else:
        impact = (
            f"Severity: **{finding.severity.capitalize()}**. "
            f"{finding.description or 'Impact requires further investigation.'}"
        )

    recommendation = (
        f"Review and fix the vulnerability in {affected}. "
        f"Apply the principle of checks-effects-interactions (CEI), "
        f"use OpenZeppelin's `ReentrancyGuard` or equivalent access controls as appropriate, "
        f"and add comprehensive test coverage for the fix."
    )

    return {
        "summary": summary,
        "vulnerability_detail": vuln_detail,
        "impact": impact,
        "recommendation": recommendation,
    }
