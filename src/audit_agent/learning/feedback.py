"""Learning loop — process post-audit feedback to improve future runs (§8.9).

Three feedback event types:
  - confirmed:      finding was a real vulnerability (reinforce pattern)
  - false_positive: finding was wrong (add to FP suppression set)
  - missed:         real vulnerability the system didn't find (add to KB,
                    generate Semgrep rule stub for future static detection)

Storage backend: Postgres via SQLAlchemy (same session factory as the rest of the system).
When Postgres is unavailable, events are written to an in-memory list so tests
never fail due to missing infrastructure.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from audit_agent.observability.logging import get_logger

log = get_logger("learning.feedback")


# ─────────────────────────────────────────────────────────────────────────────
# Feedback event data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeedbackEvent:
    """A single human-verified finding outcome."""
    finding_id: str
    audit_id: str
    title: str
    vulnerability_class: str
    affected_functions: list[str]
    description: str
    severity: str = "medium"
    notes: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AuditFeedback:
    """Structured feedback for a completed audit."""
    audit_id: str
    confirmed: list[FeedbackEvent] = field(default_factory=list)
    false_positives: list[FeedbackEvent] = field(default_factory=list)
    missed: list[FeedbackEvent] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# In-memory fallback store (used when Postgres is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _InMemoryStore:
    confirmed_patterns: list[dict[str, Any]] = field(default_factory=list)
    fp_set: list[dict[str, Any]] = field(default_factory=list)
    knowledge_base: list[dict[str, Any]] = field(default_factory=list)
    semgrep_rules: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Semgrep rule generation (stub — real implementation needs LLM + rule DSL)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_semgrep_rule_stub(event: FeedbackEvent) -> str:
    """Generate a minimal Semgrep YAML rule stub for a missed finding.

    This is a best-effort stub. A production implementation would call an LLM
    with the finding description and the affected Solidity code to produce a
    working rule. For now it generates a placeholder with the right schema.
    """
    safe_id = event.vulnerability_class.lower().replace(" ", "-").replace("/", "-")[:40]
    affected = event.affected_functions[0] if event.affected_functions else "unknown"
    fn_name = affected.split(".")[-1].split("(")[0] if "." in affected else affected

    return (
        f"rules:\n"
        f"  - id: audit-agent-{safe_id}-{event.finding_id[:8]}\n"
        f"    message: |\n"
        f"      Potential {event.vulnerability_class} detected — {event.title}\n"
        f"      Missed in audit {event.audit_id}. Review manually.\n"
        f"    severity: {event.severity.upper()}\n"
        f"    languages: [solidity]\n"
        f"    metadata:\n"
        f"      category: security\n"
        f"      vulnerability_class: {event.vulnerability_class}\n"
        f"      generated_by: audit-agent-learning-loop\n"
        f"    pattern: |\n"
        f"      # TODO: implement pattern for {fn_name}()\n"
        f"      # Description: {event.description[:120]}\n"
        f"      $X.{fn_name}(...)\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LearningLoop
# ─────────────────────────────────────────────────────────────────────────────

class LearningLoop:
    """Processes post-audit feedback and stores it for future retrieval.

    Postgres is optional — if unavailable, all events are stored in memory
    and a warning is logged. This ensures tests always work offline.
    """

    def __init__(self) -> None:
        self._mem = _InMemoryStore()
        self._use_postgres = False  # set True when DB is reachable

    # ── Public API ────────────────────────────────────────────────────────

    async def process_feedback(self, audit_id: str, feedback: AuditFeedback) -> None:
        """Process all feedback events for a completed audit.

        Args:
            audit_id: The audit run identifier.
            feedback: Structured feedback with confirmed, FP, and missed lists.
        """
        log.info(
            "learning.process_feedback",
            audit_id=audit_id,
            confirmed=len(feedback.confirmed),
            false_positives=len(feedback.false_positives),
            missed=len(feedback.missed),
        )

        for event in feedback.confirmed:
            await self.reinforce_pattern(event)

        for event in feedback.false_positives:
            await self.add_to_fp_set(event)

        for event in feedback.missed:
            await self.add_to_knowledge_base(event)
            rule = await self.generate_semgrep_rule(event)
            self._mem.semgrep_rules.append(rule)
            log.info(
                "learning.semgrep_rule_generated",
                event_id=event.finding_id,
                rule_id=f"audit-agent-{event.vulnerability_class[:20]}",
            )

    async def reinforce_pattern(self, event: FeedbackEvent) -> None:
        """Store a confirmed finding as a positive pattern for future retrieval.

        In production this would upsert into the Qdrant vulnerability_patterns
        collection (or a dedicated Postgres table) so the Hunter agent can
        retrieve it as a "similar historical case" in future audits.
        """
        record = {
            "id": str(uuid.uuid4()),
            "finding_id": event.finding_id,
            "audit_id": event.audit_id,
            "title": event.title,
            "vulnerability_class": event.vulnerability_class,
            "affected_functions": event.affected_functions,
            "description": event.description,
            "severity": event.severity,
            "confirmed_at": event.timestamp.isoformat(),
        }
        self._mem.confirmed_patterns.append(record)
        log.debug("learning.pattern_reinforced", title=event.title[:60])

        await self._persist_to_postgres("confirmed_patterns", record)

    async def add_to_fp_set(self, event: FeedbackEvent) -> None:
        """Add a false-positive fingerprint to the suppression set.

        Future Skeptic agent runs can check this set to automatically down-rank
        or suppress findings that match known false-positive patterns.
        """
        record = {
            "id": str(uuid.uuid4()),
            "finding_id": event.finding_id,
            "audit_id": event.audit_id,
            "title": event.title,
            "vulnerability_class": event.vulnerability_class,
            "notes": event.notes,
            "added_at": event.timestamp.isoformat(),
        }
        self._mem.fp_set.append(record)
        log.debug("learning.fp_added", title=event.title[:60])

        await self._persist_to_postgres("fp_suppression_set", record)

    async def add_to_knowledge_base(self, event: FeedbackEvent) -> None:
        """Add a missed finding to the knowledge base.

        This allows the Hunter agent to retrieve it as a historical case
        when analyzing similar code patterns in future audits.
        """
        record = {
            "id": str(uuid.uuid4()),
            "finding_id": event.finding_id,
            "audit_id": event.audit_id,
            "title": event.title,
            "vulnerability_class": event.vulnerability_class,
            "affected_functions": event.affected_functions,
            "description": event.description,
            "severity": event.severity,
            "type": "missed_finding",
            "added_at": event.timestamp.isoformat(),
        }
        self._mem.knowledge_base.append(record)
        log.debug("learning.kb_entry_added", title=event.title[:60])

        await self._persist_to_postgres("knowledge_base", record)

    async def generate_semgrep_rule(self, event: FeedbackEvent) -> str:
        """Generate a Semgrep rule for a missed finding.

        Currently produces a stub rule. A future implementation would call
        the LLM with the finding context to produce a working rule.
        """
        return _generate_semgrep_rule_stub(event)

    # ── State accessors (for tests / CLI) ────────────────────────────────

    @property
    def confirmed_patterns(self) -> list[dict[str, Any]]:
        return list(self._mem.confirmed_patterns)

    @property
    def fp_set(self) -> list[dict[str, Any]]:
        return list(self._mem.fp_set)

    @property
    def knowledge_base(self) -> list[dict[str, Any]]:
        return list(self._mem.knowledge_base)

    @property
    def semgrep_rules(self) -> list[str]:
        return list(self._mem.semgrep_rules)

    # ── Postgres persistence (best-effort, silent on failure) ─────────────

    async def _persist_to_postgres(self, table: str, record: dict[str, Any]) -> None:
        """Best-effort Postgres persistence. Silently logs on failure."""
        try:
            import uuid as _uuid
            from audit_agent.storage.postgres import (
                ConfirmedPattern,
                FpSuppressionSet,
                KnowledgeBase,
                get_session_factory,
            )

            session_factory = get_session_factory()
            async with session_factory() as session:
                if table == "confirmed_patterns":
                    obj = ConfirmedPattern(
                        id=_uuid.UUID(record["id"]),
                        finding_id=record.get("finding_id"),
                        audit_id=record.get("audit_id"),
                        title=record.get("title"),
                        vulnerability_class=record.get("vulnerability_class"),
                        affected_functions=record.get("affected_functions"),
                        description=record.get("description"),
                        severity=record.get("severity"),
                        confirmed_at=record.get("confirmed_at"),
                    )
                elif table == "fp_suppression_set":
                    obj = FpSuppressionSet(
                        id=_uuid.UUID(record["id"]),
                        finding_id=record.get("finding_id"),
                        audit_id=record.get("audit_id"),
                        title=record.get("title"),
                        vulnerability_class=record.get("vulnerability_class"),
                        notes=record.get("notes"),
                        added_at=record.get("added_at"),
                    )
                elif table == "knowledge_base":
                    obj = KnowledgeBase(
                        id=_uuid.UUID(record["id"]),
                        finding_id=record.get("finding_id"),
                        audit_id=record.get("audit_id"),
                        title=record.get("title"),
                        vulnerability_class=record.get("vulnerability_class"),
                        affected_functions=record.get("affected_functions"),
                        description=record.get("description"),
                        severity=record.get("severity"),
                        entry_type=record.get("type"),
                        added_at=record.get("added_at"),
                    )
                else:
                    log.debug("learning.unknown_table", table=table)
                    return
                session.add(obj)
                await session.commit()
                self._use_postgres = True
        except Exception as exc:
            log.debug("learning.postgres_unavailable", table=table, error=str(exc))
