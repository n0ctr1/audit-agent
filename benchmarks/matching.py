"""Finding match logic for benchmark evaluation.

A candidate (Hunter hypothesis, or a final finding resolved to its hypothesis)
matches a ground-truth (GT) finding when it points at the same location and
describes the same bug:

    match = function_match AND (class_compatible OR title_jaccard >= TITLE_JACCARD_MIN)

* ``function_match`` is mandatory: at least one affected function must be the
  same. Functions are compared by exact name (case-insensitive, leading ``_``
  ignored). Contracts are compared only when both sides name one, so
  ``Vault.withdraw`` matches ``withdraw`` but not ``Pool.withdraw``, and two
  different functions of the same contract never match.
* ``class_compatible`` is a soft signal resolved through ``benchmarks.taxonomy``
  (aliases, SWC codes, related classes).
* ``title_jaccard`` is word-level Jaccard similarity of the titles.

``score`` is used only to rank candidate pairs for one-to-one assignment and
for debugging; it never decides a match on its own.

One-to-one assignment collects every matching (candidate, GT) pair, sorts them
by descending score with a content-based tie-break, and assigns greedily. The
result is independent of the order of the candidate list.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from benchmarks.taxonomy import ClassRelation, class_relation

TITLE_JACCARD_MIN = 0.2

# Ranking weights (only order candidate pairs; not a match threshold).
W_FUNCTION = 0.4
W_CLASS = 0.4
W_TITLE = 0.2
_CLASS_SCORE: dict[str, float] = {"exact": 1.0, "related": 0.5, "none": 0.0}

_TITLE_STOPWORDS = frozenset(
    {"", "in", "of", "the", "a", "an", "via", "to", "and", "or", "for", "on", "can", "be", "is",
     "by", "with", "from", "at", "its", "it", "that", "this", "when", "due", "may", "allows"}
)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GroundTruthFinding:
    title: str
    severity: str
    vulnerability_class: str
    affected_functions: list[str] = field(default_factory=list)
    description: str = ""
    id: str = ""


class Candidate(Protocol):
    """Anything with these attributes can be matched (``Hypothesis`` qualifies)."""

    id: str
    title: str
    vulnerability_class: Optional[str]
    affected_functions: list[str]


@dataclass(frozen=True)
class FunctionRef:
    contract: Optional[str]
    function: str

    def __str__(self) -> str:
        return f"{self.contract}.{self.function}" if self.contract else self.function


@dataclass
class MatchResult:
    function_match: bool
    matched_functions: list[str]
    class_relation: ClassRelation
    title_jaccard: float
    score: float

    @property
    def class_compatible(self) -> bool:
        return self.class_relation != "none"

    @property
    def is_match(self) -> bool:
        return self.function_match and (
            self.class_compatible or self.title_jaccard >= TITLE_JACCARD_MIN
        )


@dataclass
class Assignment:
    candidate_index: int
    gt_index: int
    candidate_id: str
    gt_id: str
    score: float


@dataclass
class MatchingResult:
    assignments: list[Assignment]
    num_candidates: int
    num_ground_truth: int

    @property
    def matched_gt_indices(self) -> set[int]:
        return {a.gt_index for a in self.assignments}

    @property
    def matched_candidate_indices(self) -> set[int]:
        return {a.candidate_index for a in self.assignments}

    @property
    def candidate_flags(self) -> list[bool]:
        hit = self.matched_candidate_indices
        return [i in hit for i in range(self.num_candidates)]

    def gt_for_candidate(self, candidate_index: int) -> Optional[int]:
        for a in self.assignments:
            if a.candidate_index == candidate_index:
                return a.gt_index
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_function_ref(fn_id: str) -> Optional[FunctionRef]:
    """'src/Vault.sol:Vault._withdraw(uint256)' → FunctionRef('vault', 'withdraw').

    Accepts ``func``, ``Contract.func``, ``Contract::func``, ``Contract#func``,
    ``File.sol:Contract.func`` and an optional parameter list.
    """
    s = (fn_id or "").strip()
    s = s.split("(", 1)[0].strip()
    s = re.sub(r"^function\s+", "", s, flags=re.IGNORECASE)
    s = s.rsplit("/", 1)[-1]
    s = re.sub(r"\.sol\b", "", s, flags=re.IGNORECASE)
    parts = [p for p in re.split(r"[.:#]+", s) if p.strip()]
    if not parts:
        return None
    function = parts[-1].strip().lower().lstrip("_")
    if not function:
        return None
    contract = parts[-2].strip().lower() if len(parts) >= 2 else None
    return FunctionRef(contract=contract or None, function=function)


def _refs(fns: Iterable[str]) -> list[FunctionRef]:
    return [r for r in (parse_function_ref(f) for f in fns or []) if r is not None]


def _same_location(a: FunctionRef, b: FunctionRef) -> bool:
    if a.function != b.function:
        return False
    if a.contract and b.contract:
        return a.contract == b.contract
    return True


def title_jaccard(title_a: str, title_b: str) -> float:
    """Jaccard similarity of word tokens between two titles (stopwords removed)."""
    ta = set(re.split(r"\W+", (title_a or "").lower())) - _TITLE_STOPWORDS
    tb = set(re.split(r"\W+", (title_b or "").lower())) - _TITLE_STOPWORDS
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise match
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pair(candidate: Any, ground_truth: GroundTruthFinding) -> MatchResult:
    c_refs = _refs(getattr(candidate, "affected_functions", []) or [])
    g_refs = _refs(ground_truth.affected_functions)
    matched_gt = [g for g in g_refs if any(_same_location(c, g) for c in c_refs)]
    function_match = bool(matched_gt)
    fn_fraction = len(matched_gt) / len(g_refs) if g_refs else 0.0

    relation = class_relation(getattr(candidate, "vulnerability_class", None),
                              ground_truth.vulnerability_class)
    jac = title_jaccard(getattr(candidate, "title", ""), ground_truth.title)
    score = W_FUNCTION * fn_fraction + W_CLASS * _CLASS_SCORE[relation] + W_TITLE * jac
    return MatchResult(
        function_match=function_match,
        matched_functions=[str(g) for g in matched_gt],
        class_relation=relation,
        title_jaccard=jac,
        score=round(score, 6),
    )


def match_score(candidate: Any, ground_truth: GroundTruthFinding) -> float:
    """Ranking score in [0, 1]; for debugging only — use ``finding_matches`` to decide."""
    return evaluate_pair(candidate, ground_truth).score


def finding_matches(candidate: Any, ground_truth: GroundTruthFinding) -> bool:
    return evaluate_pair(candidate, ground_truth).is_match


# ─────────────────────────────────────────────────────────────────────────────
# One-to-one assignment
# ─────────────────────────────────────────────────────────────────────────────

def _tie_key(candidate: Any) -> tuple[str, str, str]:
    return (
        str(getattr(candidate, "title", "")),
        str(getattr(candidate, "id", "")),
        ",".join(sorted(getattr(candidate, "affected_functions", []) or [])),
    )


def match_one_to_one(
    candidates: Sequence[Any],
    ground_truths: Sequence[GroundTruthFinding],
) -> MatchingResult:
    """Assign each GT to at most one candidate and vice versa, best score first."""
    pairs: list[tuple[float, int, tuple[str, str, str], int]] = []
    for ci, cand in enumerate(candidates):
        tie = _tie_key(cand)
        for gi, gt in enumerate(ground_truths):
            res = evaluate_pair(cand, gt)
            if res.is_match:
                pairs.append((res.score, gi, tie, ci))
    # Highest score first; ties broken by GT order (fixed by the dataset) and by
    # candidate content — never by the candidate's position in the input list.
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    used_c: set[int] = set()
    used_g: set[int] = set()
    assignments: list[Assignment] = []
    for score, gi, _tie, ci in pairs:
        if ci in used_c or gi in used_g:
            continue
        used_c.add(ci)
        used_g.add(gi)
        assignments.append(Assignment(
            candidate_index=ci,
            gt_index=gi,
            candidate_id=str(getattr(candidates[ci], "id", "")),
            gt_id=ground_truths[gi].id or str(gi),
            score=score,
        ))
    assignments.sort(key=lambda a: a.gt_index)
    return MatchingResult(assignments, len(candidates), len(ground_truths))


def match_hypotheses_to_ground_truth(
    hypotheses: Sequence[Any],
    ground_truths: Sequence[GroundTruthFinding],
) -> list[bool]:
    """Per-candidate hit flags of the one-to-one assignment (compat helper)."""
    return match_one_to_one(hypotheses, ground_truths).candidate_flags
