"""Human ground-truth labels per hypothesis, human recall and matcher agreement.

For every stored run, ``labels_<contest>_run<k>.csv`` in the results directory has
one row per hypothesis (rank by priority, id, title, functions, the first 200
characters of the attack scenario, final status). A human fills two columns:

  matches_gt  GT ids this hypothesis describes, separated by ";" (empty = none)
  reviewed    any non-empty value once the row has been checked

``reviewed`` separates "checked, matches nothing" from "not checked yet": human
metrics are only reported for a run once every row is reviewed.

Metrics (per run, from human labels):
  human_recall_hypotheses  GT covered by the top-30 hypotheses (by priority)
  human_recall_final       GT covered by hypotheses whose final finding status is
                           validated or unverified (after Refuter)
Matcher agreement compares, for every reviewed hypothesis × GT pair, the human
label with the matcher's decision.

Rewriting a labels file keeps existing ``matches_gt`` / ``reviewed`` values.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from audit_agent.models.schemas import Hypothesis, ValidatedFinding
from benchmarks.matching import GroundTruthFinding, evaluate_pair
from benchmarks.metrics import FINAL_STATUSES, TOP_N_HYPOTHESES, top_hypotheses

COLUMNS = ["rank", "in_top30", "hypothesis_id", "title", "functions", "description",
           "final_status", "matches_gt", "reviewed"]
DESCRIPTION_CHARS = 200


def labels_filename(contest_id: str, run_index: int) -> str:
    return f"labels_{contest_id}_run{run_index}.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Writing / reading label files
# ─────────────────────────────────────────────────────────────────────────────

def label_rows(state: dict[str, Any]) -> list[dict[str, str]]:
    hyps = [Hypothesis.model_validate(h) for h in state.get("hypotheses") or []]
    findings = [ValidatedFinding.model_validate(f) for f in state.get("validated_findings") or []]
    status_by_hyp = {f.hypothesis_id: f.status for f in findings if f.hypothesis_id}
    ranked = top_hypotheses(hyps, n=len(hyps))
    rows = []
    for rank, h in enumerate(ranked, 1):
        rows.append({
            "rank": str(rank),
            "in_top30": "yes" if rank <= TOP_N_HYPOTHESES else "no",
            "hypothesis_id": h.id,
            "title": h.title,
            "functions": "; ".join(h.affected_functions),
            "description": " ".join(h.attack_scenario.split())[:DESCRIPTION_CHARS],
            "final_status": status_by_hyp.get(h.id, "no_finding"),
            "matches_gt": "",
            "reviewed": "",
        })
    return rows


def write_labels(path: Path, rows: list[dict[str, str]]) -> None:
    """Write rows, keeping matches_gt / reviewed already present in ``path``."""
    existing = {r["hypothesis_id"]: r for r in read_rows(path)} if path.exists() else {}
    for r in rows:
        old = existing.get(r["hypothesis_id"])
        if old:
            r["matches_gt"] = old.get("matches_gt", "")
            r["reviewed"] = old.get("reviewed", "")
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def parse_gt_ids(cell: str) -> set[str]:
    return {x.strip() for x in (cell or "").replace(",", ";").split(";") if x.strip()}


# ─────────────────────────────────────────────────────────────────────────────
# Human recall + matcher agreement
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Disagreement:
    hypothesis_id: str
    gt_id: str
    hyp_title: str
    gt_title: str
    matcher: str
    human: str


@dataclass
class HumanLabelResult:
    rows: int = 0
    unreviewed: int = 0
    unknown_gt_ids: list[str] = field(default_factory=list)
    human_recall_hypotheses: Optional[float] = None
    human_recall_final: Optional[float] = None
    covered_hypotheses: list[str] = field(default_factory=list)
    covered_final: list[str] = field(default_factory=list)
    matched_hypothesis_ids: list[str] = field(default_factory=list)  # reviewed rows with a GT id
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    disagreements: list[Disagreement] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.rows > 0 and self.unreviewed == 0 and not self.unknown_gt_ids

    @property
    def labelled_pairs(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def matcher_precision(self) -> Optional[float]:
        d = self.tp + self.fp
        return self.tp / d if d else None

    @property
    def matcher_recall(self) -> Optional[float]:
        d = self.tp + self.fn
        return self.tp / d if d else None

    @property
    def matcher_accuracy(self) -> Optional[float]:
        n = self.labelled_pairs
        return (self.tp + self.tn) / n if n else None

    @property
    def cohen_kappa(self) -> Optional[float]:
        n = self.labelled_pairs
        if not n:
            return None
        po = (self.tp + self.tn) / n
        pm = (self.tp + self.fp) / n
        ph = (self.tp + self.fn) / n
        pe = pm * ph + (1 - pm) * (1 - ph)
        return None if pe == 1 else (po - pe) / (1 - pe)


def evaluate_labels(
    rows: Sequence[dict[str, str]],
    state: dict[str, Any],
    ground_truths: Sequence[GroundTruthFinding],
) -> HumanLabelResult:
    gts = {g.id: g for g in ground_truths}
    hyps = {h["id"]: Hypothesis.model_validate(h) for h in state.get("hypotheses") or []}
    res = HumanLabelResult(rows=len(rows))
    covered_hyp: set[str] = set()
    covered_final: set[str] = set()
    unknown: set[str] = set()
    for r in rows:
        ids = parse_gt_ids(r.get("matches_gt", ""))
        unknown |= ids - set(gts)
        if not (r.get("reviewed") or "").strip():
            res.unreviewed += 1
            continue
        ids &= set(gts)
        if ids:
            res.matched_hypothesis_ids.append(r["hypothesis_id"])
        if r.get("in_top30") == "yes":
            covered_hyp |= ids
        if r.get("final_status") in FINAL_STATUSES:
            covered_final |= ids
        h = hyps.get(r["hypothesis_id"])
        if h is None:
            continue
        for gid, g in gts.items():
            human = gid in ids
            matcher = evaluate_pair(h, g).is_match
            if matcher and human:
                res.tp += 1
            elif matcher:
                res.fp += 1
            elif human:
                res.fn += 1
            else:
                res.tn += 1
            if matcher != human:
                res.disagreements.append(Disagreement(
                    h.id, gid, h.title, g.title,
                    "match" if matcher else "no_match", "match" if human else "no_match"))
    res.unknown_gt_ids = sorted(unknown)
    res.covered_hypotheses = sorted(covered_hyp)
    res.covered_final = sorted(covered_final)
    if res.complete and gts:
        res.human_recall_hypotheses = len(covered_hyp) / len(gts)
        res.human_recall_final = len(covered_final) / len(gts)
    return res
