"""LangGraph node implementations."""

from __future__ import annotations

from audit_agent.agents.chainer import ChainerAgent
from audit_agent.agents.exploit import ExploitAgent
from audit_agent.agents.hunter import HunterAgent
from audit_agent.agents.invariant_prover import InvariantProverAgent
from audit_agent.agents.mapper import MapperAgent
from audit_agent.agents.refuter import RefuterAgent
from audit_agent.agents.scout import ScoutAgent
from audit_agent.agents.skeptic import SkepticAgent
from audit_agent.agents.threat_modeler import ThreatModelerAgent
from audit_agent.agents.scribe import ScribeAgent
from audit_agent.config import get_settings
from audit_agent.graph.state import AuditState
from audit_agent.models.schemas import ValidatedFinding
from audit_agent.observability.logging import get_logger
from audit_agent.observability.metrics import get_metrics

log = get_logger("graph.nodes")


def _record_node_metrics(node_name: str, state_before: AuditState, state_after: AuditState) -> None:
    metrics = get_metrics()
    tokens_before = state_before.get("tokens_used", 0) or 0
    tokens_after = state_after.get("tokens_used", 0) or 0
    delta = tokens_after - tokens_before
    if delta > 0:
        metrics.llm_tokens_total.labels(model="claude-sonnet-4-6", direction="total").inc(delta)
    metrics.tool_runs_total.labels(tool=node_name, status="success").inc()


async def scout_node(state: AuditState) -> AuditState:
    agent = ScoutAgent()
    result = await agent.run(state)
    _record_node_metrics("scout", state, result)
    return result


async def mapper_node(state: AuditState) -> AuditState:
    agent = MapperAgent()
    result = await agent.run(state)
    _record_node_metrics("mapper", state, result)
    return result


async def threat_modeler_node(state: AuditState) -> AuditState:
    agent = ThreatModelerAgent()
    result = await agent.run(state)
    _record_node_metrics("threat_modeler", state, result)
    return result


async def hunter_node(state: AuditState) -> AuditState:
    agent = HunterAgent()
    result = await agent.run(state)
    _record_node_metrics("hunter", state, result)
    return result


async def invariant_prover_node(state: AuditState) -> AuditState:
    agent = InvariantProverAgent()
    result = await agent.run(state)
    _record_node_metrics("invariant_prover", state, result)
    return result


async def chainer_node(state: AuditState) -> AuditState:
    agent = ChainerAgent()
    result = await agent.run(state)
    _record_node_metrics("chainer", state, result)
    return result


def _hypotheses_as_unverified(state: AuditState) -> list[ValidatedFinding]:
    """Skeptic disabled: report every hypothesis as an `unverified` finding."""
    audit_id = state.get("audit_id", "")
    return [
        ValidatedFinding(
            audit_id=audit_id,
            hypothesis_id=h.id,
            title=h.title,
            vulnerability_class=h.vulnerability_class,
            severity=h.estimated_severity,
            confidence=h.confidence if h.confidence is not None else 0.0,
            status="unverified",
            skeptic_status="unverified",
            affected_functions=list(h.affected_functions),
            description=h.attack_scenario,
            skeptic_notes="Not validated: SKEPTIC_ENABLED=false.",
        )
        for h in (state.get("hypotheses") or [])
    ]


async def exploit_node(state: AuditState) -> AuditState:
    if not get_settings().exploit_enabled:
        log.info("exploit_node: disabled via EXPLOIT_ENABLED=false")
        return {**state, "exploits": []}  # type: ignore[return-value]
    agent = ExploitAgent()
    result = await agent.run(state)
    _record_node_metrics("exploit", state, result)
    return result


async def skeptic_node(state: AuditState) -> AuditState:
    if not get_settings().skeptic_enabled:
        log.info("skeptic_node: disabled via SKEPTIC_ENABLED=false")
        return {**state, "validated_findings": _hypotheses_as_unverified(state)}  # type: ignore[return-value]
    agent = SkepticAgent()
    result = await agent.run(state)
    # Track findings by severity/status
    metrics = get_metrics()
    for vf in (result.get("validated_findings") or []):
        metrics.findings_total.labels(severity=vf.severity, status=vf.status).inc()
    _record_node_metrics("skeptic", state, result)
    return result


async def refuter_node(state: AuditState) -> AuditState:
    # Record the Skeptic verdict before Refuter (which mutates status in place)
    # so benchmarks can attribute downgrades to the right validator.
    for vf in (state.get("validated_findings") or []):
        if vf.skeptic_status is None:
            vf.skeptic_status = vf.status
    agent = RefuterAgent()
    result = await agent.run(state)
    _record_node_metrics("refuter", state, result)
    return result


async def scribe_node(state: AuditState) -> AuditState:
    agent = ScribeAgent()
    result = await agent.run(state)
    await _persist_findings(result)
    return result


async def _persist_findings(state: AuditState) -> None:
    """Best-effort persistence of validated findings to Postgres."""
    validated = state.get("validated_findings") or []
    if not validated:
        return
    try:
        import uuid as _uuid
        from datetime import datetime as _dt

        from audit_agent.storage.postgres import Finding, get_session_factory

        audit_id_str = state.get("audit_id", "")
        try:
            audit_uuid = _uuid.UUID(audit_id_str)
        except ValueError:
            return

        session_factory = get_session_factory()
        async with session_factory() as session:
            for vf in validated:
                hyp_id: _uuid.UUID | None = None
                if vf.hypothesis_id:
                    try:
                        hyp_id = _uuid.UUID(vf.hypothesis_id)
                    except ValueError:
                        pass
                finding = Finding(
                    audit_id=audit_uuid,
                    hypothesis_id=hyp_id,
                    title=vf.title,
                    vulnerability_class=vf.vulnerability_class,
                    severity=vf.severity,
                    confidence=vf.confidence,
                    status=vf.status,
                    affected_functions=vf.affected_functions or [],
                    description=vf.description,
                    poc_uri=vf.poc_uri or None,
                    trace_uri=vf.trace_uri or None,
                    validated_at=vf.validated_at or _dt.utcnow(),
                )
                session.add(finding)
            await session.commit()
            log.info("persist_findings: saved", count=len(validated), audit_id=audit_id_str)
    except Exception as exc:
        log.debug("persist_findings: postgres_unavailable", error=str(exc))


async def static_analysis_node(state: AuditState) -> AuditState:
    """Review-mode node: converts static findings directly to validated findings and runs Scribe."""
    audit_id = state.get("audit_id", "")
    static_findings = state.get("static_findings") or []

    validated: list[ValidatedFinding] = []
    for sf in static_findings:
        vf = ValidatedFinding(
            audit_id=audit_id,
            title=sf.title,
            vulnerability_class=sf.vulnerability_class,
            severity=sf.severity,
            confidence=0.6,
            status="unverified",
            affected_functions=[f for f in [sf.affected_function] if f],
            description=sf.description,
        )
        validated.append(vf)

    log.info(
        "static_analysis_node: promoted static findings",
        count=len(validated),
        audit_id=audit_id,
    )
    return {**state, "validated_findings": validated}  # type: ignore[return-value]
