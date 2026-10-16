"""Manual labels for validated findings that do not match any GT finding.

``results/<run>/unmatched_findings.csv`` lists every unmatched ``validated``
finding. A human fills the ``label`` column with one of ``tp_not_in_gt``,
``fp``, ``duplicate`` or ``unsure`` (counted as not a true positive);
``precision_adjusted`` is computed from it. Once a run's per-hypothesis labels are
complete, validated findings that the human labels put outside GT are listed too
(``unmatched_by`` = matcher / human / matcher+human); the human precision in
metrics.json uses those labels. Rewriting
the file (e.g. on ``rescore``) keeps labels already present for the same
(contest, run, finding_id). A labelled finding that the current matcher now
matches to GT is kept with ``matcher_status=matched_now`` (its label is then
ignored), so labels survive matcher changes in either direction.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from benchmarks.metrics import LABEL_VALUES

UNMATCHED_CSV = "unmatched_findings.csv"


@dataclass
class UnmatchedRow:
    contest: str
    run: int
    finding_id: str
    hypothesis_id: str
    title: str
    functions: str
    severity: str
    report_path: str
    matcher_status: str = "unmatched"  # unmatched | matched (outside GT by human labels only) | matched_now
    unmatched_by: str = "matcher"  # matcher | human | matcher+human (human = manual labels)
    label: str = ""  # last column: filled by a human


COLUMNS = [f.name for f in fields(UnmatchedRow)]
LabelKey = tuple[str, int, str]


def read_labels(path: Path) -> dict[LabelKey, str]:
    """(contest, run, finding_id) → label; unknown/empty labels are skipped."""
    if not path.exists():
        return {}
    out: dict[LabelKey, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("label") or "").strip()
            if label not in LABEL_VALUES:
                continue
            out[(row["contest"], int(row["run"]), row["finding_id"])] = label
    return out


def labels_for_run(labels: dict[LabelKey, str], contest: str, run: int) -> dict[str, str]:
    return {fid: lab for (c, r, fid), lab in labels.items() if c == contest and r == run}


def write_unmatched(path: Path, rows: list[UnmatchedRow]) -> None:
    """Write rows, carrying over labels (and labelled rows) already in ``path``."""
    existing: dict[LabelKey, dict[str, str]] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if (r.get("label") or "").strip() in LABEL_VALUES:
                    existing[(r["contest"], int(r["run"]), r["finding_id"])] = r
    current = {(r.contest, r.run, r.finding_id) for r in rows}
    for row in rows:
        old = existing.get((row.contest, row.run, row.finding_id))
        if old and not row.label:
            row.label = old["label"].strip()
    for key, old in existing.items():
        if key in current:
            continue
        rows.append(UnmatchedRow(**{c: old.get(c, "") for c in COLUMNS if c not in ("run",)},
                                 run=int(old["run"])))
        rows[-1].matcher_status = "matched_now"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for row in sorted(rows, key=lambda r: (r.contest, r.run, r.finding_id)):
            w.writerow(asdict(row))
