"""Attack surface scoring function — §8.5 of the spec.

Pure function: no I/O, no async. Works on SlitherFunction dataclasses from builders.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


CallType = Literal["call", "delegatecall", "staticcall", "transfer", "send"]
TargetType = Literal["known_contract", "user_controlled", "address_literal"]


@dataclass
class ExternalCallInfo:
    id: str = ""
    call_type: CallType = "call"
    target_type: TargetType = "user_controlled"
    line: int = 0


@dataclass
class ScoredFunction:
    """Minimal function description for scoring. Populated from SlitherFunction."""

    id: str
    name: str
    visibility: Literal["external", "public", "internal", "private"]
    state_mutability: Literal["pure", "view", "nonpayable", "payable"]
    modifiers: list[str] = field(default_factory=list)
    external_calls: list[ExternalCallInfo] = field(default_factory=list)
    writes_state_vars: list[str] = field(default_factory=list)
    handles_value: bool = False
    reads_oracle: bool = False
    is_privileged: bool = False
    has_division: bool = False
    has_multiplication: bool = False
    # Fork annotations (set by DifferentialAnalyzer / ForkAnalysis)
    is_modified_in_fork: bool = False
    is_new_in_fork: bool = False


def attack_surface_score(fn: ScoredFunction) -> float:
    """Compute priority heuristic score per §8.5.

    Higher score → Hunter should analyze first.
    Deterministic given the same inputs — no randomness, no LLM.
    """
    score = 0.0

    # ── Accessibility ──────────────────────────────────────────────
    if fn.visibility == "external":
        score += 3.0
    elif fn.visibility == "public":
        score += 2.0
    # internal / private: 0 (not directly reachable externally)

    # ── Value handling ──────────────────────────────────────────────
    if fn.handles_value:
        score += 4.0
    if fn.state_mutability == "payable":
        score += 2.0

    # ── External interactions (reentrancy surface) ──────────────────
    score += len(fn.external_calls) * 2.0
    for call in fn.external_calls:
        if call.call_type == "delegatecall":
            score += 5.0
        if call.target_type == "user_controlled":
            score += 3.0

    # ── Oracle reads ────────────────────────────────────────────────
    if fn.reads_oracle:
        score += 5.0

    # ── State variables written ─────────────────────────────────────
    score += len(fn.writes_state_vars) * 1.5

    # ── Privileged operations ───────────────────────────────────────
    if fn.is_privileged:
        score += 3.0

    # ── Precision loss surface ──────────────────────────────────────
    if fn.has_division or fn.has_multiplication:
        score += 2.0
    _FEE_KEYWORDS = ("fee", "rate", "price", "reward", "yield", "interest", "share", "ratio")
    if any(kw in fn.name.lower() for kw in _FEE_KEYWORDS):
        score += 3.0

    # ── Missing reentrancy guard ────────────────────────────────────
    has_reentrancy_guard = any(
        "nonReentrant" in m or "reentrancy" in m.lower()
        for m in fn.modifiers
    )
    if fn.external_calls and not has_reentrancy_guard:
        score += 6.0

    # ── Fork multipliers ────────────────────────────────────────────
    if fn.is_modified_in_fork:
        score *= 2.0
    if fn.is_new_in_fork:
        score *= 3.0

    return round(score, 2)
