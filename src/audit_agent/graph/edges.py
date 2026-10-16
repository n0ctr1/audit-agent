"""Conditional routing between LangGraph nodes."""

from __future__ import annotations

from audit_agent.graph.state import AuditState


def after_scout(state: AuditState) -> str:
    """Route after Scout. In review mode skip heavy analysis."""
    mode = state.get("mode", "audit")
    if mode == "review":
        return "static_analysis"
    return "mapper"


def after_skeptic(state: AuditState) -> str:
    """Route after Skeptic.

    - Any validated (true-positive) or needs_review (unverified) findings → scribe.
    - All false-positives only → log and end.
    """
    validated = state.get("validated_findings") or []
    has_reportable = any(f.status in ("validated", "unverified") for f in validated)
    if has_reportable:
        return "scribe"
    return "end"
