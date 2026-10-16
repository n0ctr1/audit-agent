"""LangGraph state definition."""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict

from audit_agent.models.schemas import (
    AttackChain,
    AuditMode,
    AuditPlatform,
    ErrorEvent,
    ExploitResult,
    ForkAnalysis,
    Hypothesis,
    Invariant,
    ProtocolSummary,
    ScopeMap,
    StaticFinding,
    ThreatModel,
    ValidatedFinding,
)


class AuditState(TypedDict, total=False):
    # Configuration (required)
    audit_id: str
    mode: AuditMode
    target_dir: str
    docs_dir: Optional[str]
    known_issues: Optional[list[str]]
    platform: AuditPlatform
    budget_tokens: int
    budget_usd: float

    # Scope
    scope_file: Optional[str]           # path to scope.txt (from CLI or auto-detected)
    scope_in_scope: list[str]           # absolute paths of in-scope .sol files
    scope_context_files: list[str]      # absolute paths of dependency files
    scope_source: str                   # "explicit_file" | "auto_detected" | "all_non_library"
    fuzz_enabled: bool

    # Accumulated context
    protocol_summary: Optional[ProtocolSummary]
    fork_analysis: Optional[ForkAnalysis]
    threat_model: Optional[ThreatModel]
    scope_map: Optional[ScopeMap]
    graph_id: Optional[str]
    invariants: list[Invariant]
    static_findings: list[StaticFinding]
    hypotheses: list[Hypothesis]
    chains: list[AttackChain]
    exploits: list[ExploitResult]
    validated_findings: list[ValidatedFinding]

    # Transient analysis data — passed between agents, not persisted to Postgres
    # Raw Slither JSON dict: set by Mapper, consumed by Hunter to build InMemoryGraph
    slither_analysis_raw: Optional[dict]
    # Project documentation (README, whitepaper, etc.) collected by Scout
    documentation: Optional[str]

    # Scribe output: finding_id → rendered Markdown report
    reports: dict[str, str]

    # Tracking
    tokens_used: int
    cost_usd: float
    errors: list[ErrorEvent]
    run_checkpoint_id: str
    # Tools that were unavailable/mocked during this run (degraded mode only)
    degraded_tools: list[str]
