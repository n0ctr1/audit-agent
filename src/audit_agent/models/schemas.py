"""All Pydantic models used throughout the system."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


AuditMode = Literal["audit", "bugbounty", "ctf", "review"]
AuditPlatform = Literal["immunefi", "c4", "sherlock", "cantina", "spearbit", "generic"]
Severity = Literal["critical", "high", "medium", "low", "info"]
FindingStatus = Literal["unverified", "exploited", "validated", "false_positive", "duplicate"]
ExploitStatus = Literal["exploited", "partial", "blocked", "error"]
TestType = Literal["unit", "fork", "fuzz", "symbolic"]
InvariantType = Literal["global", "transition", "economic"]
InvariantSource = Literal["natspec", "docs", "inferred"]


class ProtocolSummary(BaseModel):
    name: str
    category: str  # DEX / lending / bridge / etc.
    key_mechanisms: list[str]
    user_roles: list[str]
    sol_files: list[str]
    raw_notes: str = ""


class ForkAnalysis(BaseModel):
    is_fork: bool
    base_protocol: Optional[str] = None
    similarity_score: float = 0.0
    modified_files: list[str] = Field(default_factory=list)
    added_files: list[str] = Field(default_factory=list)
    critical_modifications: list[str] = Field(default_factory=list)


class ScopeMap(BaseModel):
    in_scope: list[str]
    out_of_scope: list[str]  # dependencies, OZ, etc.
    total_sol_files: int
    total_loc: int


class Invariant(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    audit_id: str
    invariant_type: InvariantType
    natural_language: str
    affected_functions: list[str] = Field(default_factory=list)
    solidity_assert: Optional[str] = None
    echidna_property: Optional[str] = None
    halmos_spec: Optional[str] = None
    source: InvariantSource


class StaticFinding(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    audit_id: str
    tool: str  # slither / aderyn / semgrep / mythril
    title: str
    severity: Severity
    vulnerability_class: Optional[str] = None
    description: str
    affected_contract: Optional[str] = None
    affected_function: Optional[str] = None
    line: Optional[int] = None
    raw: dict[str, Any] = Field(default_factory=dict)


class Hypothesis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    audit_id: str
    title: str
    vulnerability_class: Optional[str] = None
    affected_functions: list[str] = Field(default_factory=list)
    attack_scenario: str
    required_preconditions: list[str] = Field(default_factory=list)
    estimated_severity: Severity
    similar_historical_cases: list[str] = Field(default_factory=list)
    priority_score: float = 0.0
    confidence: Optional[float] = None  # 0.0–1.0; set externally (e.g. by Hunter scoring)
    subgraph_context: Optional[str] = None
    status: Literal["pending", "testing", "confirmed", "rejected"] = "pending"


class ExploitResult(BaseModel):
    hypothesis_id: str
    test_type: TestType
    status: ExploitStatus
    poc_code: str
    echidna_property: Optional[str] = None
    execution_trace: str = ""
    gas_used: Optional[int] = None
    profit_usd: Optional[float] = None
    artifacts_uri: str = ""
    reproduction_steps: list[str] = Field(default_factory=list)


class ValidatedFinding(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    audit_id: str
    hypothesis_id: Optional[str] = None
    title: str
    vulnerability_class: Optional[str] = None
    severity: Severity
    confidence: float  # 0.0 – 1.0
    status: FindingStatus
    affected_functions: list[str] = Field(default_factory=list)
    description: str
    poc_code: Optional[str] = None
    execution_trace: Optional[str] = None
    skeptic_notes: Optional[str] = None
    # Status assigned by Skeptic, snapshotted before Refuter runs (graph/nodes.py).
    # Lets evaluation separate Skeptic and Refuter decisions; None outside the graph.
    skeptic_status: Optional[FindingStatus] = None
    poc_uri: str = ""
    trace_uri: str = ""
    report_uri: str = ""
    # Scope/known-issue self-flag (orthogonal to validity; set by Scribe).
    scope_flag: Optional[str] = None   # likely_in_scope | out_of_scope | known_issue | below_paid_tier | needs_scope_review
    scope_note: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    validated_at: Optional[datetime] = None


class ThreatActor(BaseModel):
    name: str                    # e.g. "Flash-loan attacker", "Malicious oracle", "Whale LP"
    capability: str = ""         # what this actor can do
    relevance: str = ""          # why they matter for this protocol


class PrioritizedTarget(BaseModel):
    """A function ranked by value-at-risk — drives Hunter's analysis order."""
    function: str
    contract: str = ""
    reason: str = ""
    value_at_risk: Severity = "medium"


class ThreatModel(BaseModel):
    """Where the value is and who can attack it (roadmap #6).

    Produced after Mapper from the protocol summary + docs + the rich code facts
    (value-moving functions, privileged functions, external dependencies). Used to
    prioritize Hunter toward value-concentrated code rather than raw attack surface.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    audit_id: str
    actors: list[ThreatActor] = Field(default_factory=list)
    assets: list[str] = Field(default_factory=list)                 # what holds/represents value
    trust_boundaries: list[str] = Field(default_factory=list)
    external_dependencies: list[str] = Field(default_factory=list)
    value_flows: list[str] = Field(default_factory=list)            # how value enters/leaves
    prioritized_targets: list[PrioritizedTarget] = Field(default_factory=list)
    notes: str = ""


class AttackChainStep(BaseModel):
    order: int
    hypothesis_id: Optional[str] = None  # component hypothesis this step came from
    function: str = ""                   # function exercised in this step
    action: str = ""                     # what the attacker does
    effect: str = ""                     # resulting state change consumed by the next step


class AttackChain(BaseModel):
    """A multi-step exploit composed from individual hypotheses (roadmap #5).

    Example: H1 lets the attacker set an unchecked state var → H2 trusts that var
    when pricing → H3 drains funds. None of H1/H2/H3 alone is critical, but the
    composition is.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    audit_id: str
    title: str
    steps: list[AttackChainStep] = Field(default_factory=list)
    combined_severity: Severity
    combined_scenario: str
    root_cause: str = ""
    component_hypothesis_ids: list[str] = Field(default_factory=list)
    affected_functions: list[str] = Field(default_factory=list)
    priority_score: float = 0.0


class ErrorEvent(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    node: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
