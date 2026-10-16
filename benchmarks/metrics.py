"""Benchmark metrics computed from a saved pipeline state.

All quality numbers are measured against the contest ground truth (GT); no
metric uses the pipeline's own verdicts as evidence of correctness.

Per contest run:

  recall_hypotheses_at_30  share of GT closed by the top-30 hypotheses (by
                           priority_score) from Hunter + InvariantProver + Chainer,
                           i.e. before Exploit/Skeptic/Refuter
  recall_hypotheses_all    same, all hypotheses
  recall_final             share of GT closed by final findings with status
                           validated or unverified
  recall_validated         same, validated only
  validator_loss           GT closed by hypotheses but lost after Skeptic/Refuter
  pipeline_loss            GT closed by hypotheses that never got a verdict (e.g.
                           beyond Exploit's top-N) or whose finding was used for
                           another GT
  precision_gt             validated findings matched to GT / validated findings
                           (lower bound; None when there are no validated findings)
  precision_adjusted       (matched + labelled tp_not_in_gt) / (validated - duplicate);
                           "unsure" counts as not a true positive (lower bound),
                           only when every unmatched validated finding is labelled
  recall_by_severity       recall_* per GT severity (always reports high, medium)
  exploit_stats            Exploit status counts and exploited share among
                           GT-matching vs non-matching hypotheses
  validator_confusion      Skeptic verdict and final (post-Refuter) status ×
                           "matches some GT"
  cost_usd/tokens/time_s   measured values recorded by the runner

Findings are matched through their hypothesis (``ValidatedFinding.hypothesis_id``):
functions, class and title come from the hypothesis, falling back to the
finding's own fields when the hypothesis is missing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, Optional

from pydantic import BaseModel, Field

from audit_agent.models.schemas import ExploitResult, Hypothesis, ValidatedFinding
from benchmarks.matching import (
    GroundTruthFinding,
    MatchingResult,
    evaluate_pair,
    match_one_to_one,
)

TOP_N_HYPOTHESES = 30
_FROM_STATE: Any = object()  # sentinel: take the value from the state
FINAL_STATUSES = ("validated", "unverified")
LABEL_VALUES = ("tp_not_in_gt", "fp", "duplicate", "unsure")
EXPLOIT_BUCKETS = ("exploited", "partial", "blocked", "error", "compile_error")
REPORTED_SEVERITIES = ("high", "medium")

# Minimum expected sections per platform
_PLATFORM_REQUIRED_SECTIONS: dict[str, list[str]] = {
    "c4":       ["Summary", "Vulnerability Detail", "Impact", "Proof of Concept", "Recommended Mitigation Steps"],
    "immunefi": ["Title", "Severity", "Description", "Proof of Concept", "Impact", "Recommended Fix"],
    "sherlock":  ["Summary", "Vulnerability Detail", "Impact", "Code Snippet", "Tool used", "Recommendation"],
    "cantina":  ["Summary", "Vulnerability Detail", "Impact", "Recommendation"],
    "spearbit": ["Summary", "Vulnerability Detail", "Impact", "Recommendation"],
    "generic":  ["Summary", "Vulnerability Detail", "Impact", "Recommendation"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Result models
# ─────────────────────────────────────────────────────────────────────────────

class MatchRecord(BaseModel):
    gt_id: str
    candidate_id: str  # hypothesis id (hypothesis recall) or finding id (final recall)
    score: float


class LossRecord(BaseModel):
    gt_id: str
    reason: str  # rejected_by_skeptic | rejected_by_refuter | never_validated | assigned_elsewhere
    hypothesis_ids: list[str]
    finding_statuses: list[Optional[str]]


class SeverityRecall(BaseModel):
    num_gt: int
    recall_hypotheses_at_30: Optional[float]
    recall_final: Optional[float]
    recall_validated: Optional[float]


class ExploitStats(BaseModel):
    counts: dict[str, int]
    exploited_rate_gt_matched: Optional[float]
    exploited_rate_gt_unmatched: Optional[float]
    num_gt_matched: int
    num_gt_unmatched: int


class AdjustedPrecision(BaseModel):
    value: Optional[float]
    labelled: dict[str, int]
    unlabelled: int


class ContestRunMetrics(BaseModel):
    contest_id: str
    run_index: int = 0
    status: str = "ok"  # ok | error
    error: Optional[str] = None

    num_ground_truth: int = 0
    num_hypotheses: int = 0
    num_final_findings: int = 0
    num_validated: int = 0

    recall_hypotheses_at_30: Optional[float] = None
    recall_hypotheses_all: Optional[float] = None
    recall_final: Optional[float] = None
    recall_validated: Optional[float] = None

    hypothesis_matches: list[MatchRecord] = Field(default_factory=list)
    final_matches: list[MatchRecord] = Field(default_factory=list)
    validated_matches: list[MatchRecord] = Field(default_factory=list)

    validator_loss_count: int = 0
    validator_loss: list[LossRecord] = Field(default_factory=list)
    pipeline_loss_count: int = 0
    pipeline_loss: list[LossRecord] = Field(default_factory=list)

    precision_gt: Optional[float] = None
    num_validated_matched: int = 0
    unmatched_validated_ids: list[str] = Field(default_factory=list)
    precision_adjusted: AdjustedPrecision = Field(
        default_factory=lambda: AdjustedPrecision(value=None, labelled={}, unlabelled=0))

    recall_by_severity: dict[str, SeverityRecall] = Field(default_factory=dict)
    exploit_stats: Optional[ExploitStats] = None
    validator_confusion: dict[str, dict[str, dict[str, int]]] = Field(default_factory=dict)

    cost_usd: Optional[float] = None
    tokens: Optional[int] = None
    time_s: Optional[float] = None

    # From manual per-hypothesis labels (benchmarks/human_labels.py); None until
    # every row of the run's labels file is reviewed.
    human_recall_hypotheses: Optional[float] = None
    human_recall_final: Optional[float] = None
    human_labels: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _as_models(items: Iterable[Any], model: type[BaseModel]) -> list[Any]:
    return [i if isinstance(i, model) else model.model_validate(i) for i in (items or [])]


def _ratio(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def top_hypotheses(hypotheses: Sequence[Hypothesis], n: int = TOP_N_HYPOTHESES) -> list[Hypothesis]:
    """Top-n by priority_score; ties by id so the result does not depend on list order."""
    return sorted(hypotheses, key=lambda h: (-(h.priority_score or 0.0), h.id))[:n]


class _FindingView:
    """A ValidatedFinding seen through its hypothesis for matching purposes."""

    def __init__(self, finding: ValidatedFinding, hyp: Optional[Hypothesis]) -> None:
        self.finding = finding
        self.id = finding.id
        self.title = hyp.title if hyp else finding.title
        self.vulnerability_class = (hyp.vulnerability_class if hyp else None) or finding.vulnerability_class
        self.affected_functions = (hyp.affected_functions if hyp else None) or finding.affected_functions


def _records(result: MatchingResult, gts: Sequence[GroundTruthFinding]) -> list[MatchRecord]:
    return [MatchRecord(gt_id=a.gt_id, candidate_id=a.candidate_id, score=a.score)
            for a in result.assignments]


def _gt_id(gts: Sequence[GroundTruthFinding], i: int) -> str:
    return gts[i].id or str(i)


def exploit_bucket(e: ExploitResult) -> str:
    """Exploit status, with compile failures split out of ``error``."""
    if e.status != "error":
        return e.status
    from audit_agent.exploit.trace_parser import parse_forge_output

    steps = " ".join(e.reproduction_steps or []).lower()
    if "compile" in steps or parse_forge_output(e.execution_trace or "", "").is_compile_error:
        return "compile_error"
    return "error"


def _matches_any(candidate: Any, gts: Sequence[GroundTruthFinding]) -> bool:
    return any(evaluate_pair(candidate, g).is_match for g in gts)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_contest_metrics(
    final_state: dict[str, Any],
    ground_truths: Sequence[GroundTruthFinding],
    *,
    contest_id: str,
    run_index: int = 0,
    labels: Optional[dict[str, str]] = None,
    cost_usd: Optional[float] = _FROM_STATE,
    tokens: Optional[int] = _FROM_STATE,
    time_s: Optional[float] = None,
) -> ContestRunMetrics:
    """Compute every metric for one contest run.

    Args:
        final_state: final pipeline state (Pydantic objects or plain dicts).
        ground_truths: GT findings of the contest.
        labels: finding_id → label (tp_not_in_gt | fp | duplicate | unsure) for unmatched
            validated findings, from unmatched_findings.csv.
        cost_usd / tokens / time_s: measured values; when omitted, cost and
            tokens are read from the state's ``cost_usd`` / ``tokens_used``.
            Pass ``None`` explicitly for "not measured".
    """
    gts = list(ground_truths)
    n_gt = len(gts)
    hypotheses: list[Hypothesis] = _as_models(final_state.get("hypotheses"), Hypothesis)
    findings: list[ValidatedFinding] = _as_models(final_state.get("validated_findings"), ValidatedFinding)
    exploits: list[ExploitResult] = _as_models(final_state.get("exploits"), ExploitResult)
    hyp_by_id = {h.id: h for h in hypotheses}

    views = [_FindingView(f, hyp_by_id.get(f.hypothesis_id or "")) for f in findings]
    final_views = [v for v in views if v.finding.status in FINAL_STATUSES]
    validated_views = [v for v in views if v.finding.status == "validated"]

    m_top = match_one_to_one(top_hypotheses(hypotheses), gts)
    m_all = match_one_to_one(hypotheses, gts)
    m_final = match_one_to_one(final_views, gts)
    m_val = match_one_to_one(validated_views, gts)

    out = ContestRunMetrics(
        contest_id=contest_id,
        run_index=run_index,
        num_ground_truth=n_gt,
        num_hypotheses=len(hypotheses),
        num_final_findings=len(final_views),
        num_validated=len(validated_views),
        recall_hypotheses_at_30=_ratio(len(m_top.assignments), n_gt),
        recall_hypotheses_all=_ratio(len(m_all.assignments), n_gt),
        recall_final=_ratio(len(m_final.assignments), n_gt),
        recall_validated=_ratio(len(m_val.assignments), n_gt),
        hypothesis_matches=_records(m_top, gts),
        final_matches=_records(m_final, gts),
        validated_matches=_records(m_val, gts),
    )

    # ── losses between hypotheses and final findings ──────────────────────
    findings_by_hyp: dict[str, list[ValidatedFinding]] = {}
    for f in findings:
        if f.hypothesis_id:
            findings_by_hyp.setdefault(f.hypothesis_id, []).append(f)

    lost = sorted(m_all.matched_gt_indices - m_final.matched_gt_indices)
    for gi in lost:
        matching_hyps = sorted(
            (h for h in hypotheses if evaluate_pair(h, gts[gi]).is_match),
            key=lambda h: (-evaluate_pair(h, gts[gi]).score, h.id),
        )
        fs = [f for h in matching_hyps for f in findings_by_hyp.get(h.id, [])]
        if not fs:
            reason = "never_validated"
        elif any(f.status in FINAL_STATUSES for f in fs):
            reason = "assigned_elsewhere"
        elif any((f.skeptic_status or f.status) in FINAL_STATUSES for f in fs):
            reason = "rejected_by_refuter"
        else:
            reason = "rejected_by_skeptic"
        rec = LossRecord(
            gt_id=_gt_id(gts, gi), reason=reason,
            hypothesis_ids=[h.id for h in matching_hyps],
            finding_statuses=[f.status for f in fs],
        )
        if reason in ("rejected_by_skeptic", "rejected_by_refuter"):
            out.validator_loss.append(rec)
        else:
            out.pipeline_loss.append(rec)
    out.validator_loss_count = len(out.validator_loss)
    out.pipeline_loss_count = len(out.pipeline_loss)

    # ── precision against GT ──────────────────────────────────────────────
    matched_val = m_val.matched_candidate_indices
    out.num_validated_matched = len(matched_val)
    out.precision_gt = _ratio(len(matched_val), len(validated_views))
    unmatched_ids = [v.id for i, v in enumerate(validated_views) if i not in matched_val]
    out.unmatched_validated_ids = unmatched_ids

    labels = labels or {}
    lab = Counter(labels[i] for i in unmatched_ids if labels.get(i) in LABEL_VALUES)
    unlabelled = sum(1 for i in unmatched_ids if labels.get(i) not in LABEL_VALUES)
    adj_value: Optional[float] = None
    if validated_views and unlabelled == 0:
        den = len(validated_views) - lab["duplicate"]
        adj_value = _ratio(len(matched_val) + lab["tp_not_in_gt"], den)
    out.precision_adjusted = AdjustedPrecision(
        value=adj_value, labelled={k: lab.get(k, 0) for k in LABEL_VALUES}, unlabelled=unlabelled)

    # ── recall by severity ────────────────────────────────────────────────
    severities = sorted(set(REPORTED_SEVERITIES) | {g.severity.lower() for g in gts})
    for sev in severities:
        idx = {i for i, g in enumerate(gts) if g.severity.lower() == sev}
        out.recall_by_severity[sev] = SeverityRecall(
            num_gt=len(idx),
            recall_hypotheses_at_30=_ratio(len(m_top.matched_gt_indices & idx), len(idx)),
            recall_final=_ratio(len(m_final.matched_gt_indices & idx), len(idx)),
            recall_validated=_ratio(len(m_val.matched_gt_indices & idx), len(idx)),
        )

    # ── exploit stats ─────────────────────────────────────────────────────
    counts = Counter({b: 0 for b in EXPLOIT_BUCKETS})
    exp_matched = exp_unmatched = n_matched = n_unmatched = 0
    for e in exploits:
        bucket = exploit_bucket(e)
        counts[bucket] += 1
        hyp = hyp_by_id.get(e.hypothesis_id)
        if hyp is not None and _matches_any(hyp, gts):
            n_matched += 1
            exp_matched += bucket == "exploited"
        else:
            n_unmatched += 1
            exp_unmatched += bucket == "exploited"
    out.exploit_stats = ExploitStats(
        counts=dict(counts),
        exploited_rate_gt_matched=_ratio(exp_matched, n_matched),
        exploited_rate_gt_unmatched=_ratio(exp_unmatched, n_unmatched),
        num_gt_matched=n_matched,
        num_gt_unmatched=n_unmatched,
    )

    # ── validator confusion: verdict × matches some GT ────────────────────
    statuses = ("validated", "unverified", "false_positive", "duplicate")
    confusion = {stage: {s: {"gt_match": 0, "no_gt_match": 0} for s in statuses}
                 for stage in ("skeptic", "final")}
    for v in views:
        col = "gt_match" if _matches_any(v, gts) else "no_gt_match"
        skeptic = v.finding.skeptic_status or v.finding.status
        if skeptic in statuses:
            confusion["skeptic"][skeptic][col] += 1
        if v.finding.status in statuses:
            confusion["final"][v.finding.status][col] += 1
    out.validator_confusion = confusion

    # ── measured cost / time ──────────────────────────────────────────────
    out.cost_usd = final_state.get("cost_usd") if cost_usd is _FROM_STATE else cost_usd
    out.tokens = final_state.get("tokens_used") if tokens is _FROM_STATE else tokens
    out.time_s = time_s
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation across runs / contests
# ─────────────────────────────────────────────────────────────────────────────

AGGREGATED_FIELDS = (
    "human_recall_hypotheses",
    "human_recall_final",
    "recall_hypotheses_at_30",
    "recall_hypotheses_all",
    "recall_final",
    "recall_validated",
    "precision_gt",
    "validator_loss_count",
    "cost_usd",
    "tokens",
    "time_s",
)


class Stat(BaseModel):
    mean: Optional[float]
    min: Optional[float]
    max: Optional[float]
    n: int


def summarize(values: Iterable[Optional[float]]) -> Stat:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return Stat(mean=None, min=None, max=None, n=0)
    return Stat(mean=sum(vals) / len(vals), min=min(vals), max=max(vals), n=len(vals))


def aggregate_runs(runs: Sequence[ContestRunMetrics]) -> dict[str, Stat]:
    """mean / min / max per metric over successful runs (errors excluded)."""
    ok = [r for r in runs if r.status == "ok"]
    out = {f: summarize(getattr(r, f) for r in ok) for f in AGGREGATED_FIELDS}
    out["precision_adjusted"] = summarize(r.precision_adjusted.value for r in ok)
    return out


def pooled_recall(runs: Sequence[ContestRunMetrics], field: str) -> Optional[float]:
    """Micro-averaged recall: total GT closed / total GT over the given runs."""
    key = {"recall_hypotheses_at_30": "hypothesis_matches",
           "recall_final": "final_matches",
           "recall_validated": "validated_matches"}[field]
    ok = [r for r in runs if r.status == "ok"]
    total = sum(r.num_ground_truth for r in ok)
    hit = sum(len(getattr(r, key)) for r in ok)
    return _ratio(hit, total)


# ─────────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────────

def time_per_audit(start_ts: float, end_ts: float) -> float:
    """Wall-clock seconds for an audit run."""
    return max(0.0, end_ts - start_ts)


def report_format_valid(report_markdown: str, platform: str) -> bool:
    """Check that a rendered Markdown report contains all required sections."""
    required = _PLATFORM_REQUIRED_SECTIONS.get(platform, _PLATFORM_REQUIRED_SECTIONS["generic"])
    lower = report_markdown.lower()
    return all(section.lower() in lower for section in required)
