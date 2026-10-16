"""Benchmark dataset schema, loading and contest selection.

A dataset is a JSON file ``{"name": ..., "contests": [...]}`` in
``benchmarks/datasets/`` (looked up by name) or any path. Each contest records
where its ground truth comes from (``gt_source``) and which split it belongs to:

  test   counted in reported metrics
  dev    the pipeline was tuned on it (e.g. Intuition); never in test metrics
  smoke  plumbing checks only; usually ``contaminated`` (public solutions that
         are in model training data, e.g. Damn Vulnerable DeFi)

Contests with ``needs_review: true`` are drafts produced by
``benchmarks/tools/import_contest.py`` and are skipped unless explicitly
requested.

Relative ``source_dir`` / ``docs_dir`` paths are resolved against
``$BENCHMARK_CONTESTS_DIR`` when set, otherwise against the repository root;
``~`` and ``$VARS`` are expanded.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from benchmarks.matching import GroundTruthFinding

DATASETS_DIR = Path(__file__).parent / "datasets"
REPO_ROOT = Path(__file__).resolve().parent.parent

Split = Literal["test", "dev", "smoke"]
GTSeverity = Literal["critical", "high", "medium", "low"]


class GroundTruthFindingModel(BaseModel):
    id: str = ""
    title: str
    severity: GTSeverity
    vulnerability_class: str
    affected_functions: list[str] = Field(default_factory=list)
    description: str = ""

    def to_matching(self) -> GroundTruthFinding:
        return GroundTruthFinding(
            id=self.id,
            title=self.title,
            severity=self.severity,
            vulnerability_class=self.vulnerability_class,
            affected_functions=list(self.affected_functions),
            description=self.description,
        )


class BenchmarkContest(BaseModel):
    id: str
    name: str
    source_dir: str
    scope_file: Optional[str] = None
    docs_dir: Optional[str] = None
    split: Split = "test"
    contest_end_date: Optional[date] = None
    contaminated: bool = False
    # Why the contest is (or may be) contaminated, e.g. "code public before model release".
    contamination_note: Optional[str] = None
    # Date the official report (= ground truth) was published, if known.
    report_published: Optional[date] = None
    gt_source: Optional[str] = None
    gt_note: Optional[str] = None
    needs_review: bool = False
    known_issues: list[str] = Field(default_factory=list)
    official_findings: list[GroundTruthFindingModel] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> BenchmarkContest:
        ids = [f.id for f in self.official_findings]
        dupes = sorted({i for i in ids if i and ids.count(i) > 1})
        if dupes:
            raise ValueError(f"{self.id}: duplicate finding ids {dupes}")
        if self.split == "smoke" or self.needs_review:
            return self
        # Reviewed test/dev contests: GT must be traceable and matchable.
        problems: list[str] = []
        if not self.gt_source:
            problems.append("gt_source is required")
        if self.contest_end_date is None:
            problems.append("contest_end_date is required")
        for f in self.official_findings:
            if not f.id:
                problems.append(f"finding {f.title!r} has no id")
            if not f.affected_functions:
                problems.append(f"finding {f.id or f.title!r} has no affected_functions "
                                "(location match is mandatory, it could never be matched)")
        if problems:
            raise ValueError(f"{self.id} ({self.split}): " + "; ".join(problems))
        return self

    def ground_truth(self) -> list[GroundTruthFinding]:
        return [f.to_matching() for f in self.official_findings]

    def resolved_source_dir(self) -> Path:
        return resolve_path(self.source_dir)

    def resolved_docs_dir(self) -> Optional[Path]:
        return resolve_path(self.docs_dir) if self.docs_dir else None

    def resolved_scope_file(self) -> Optional[Path]:
        if not self.scope_file:
            return None
        p = Path(os.path.expandvars(os.path.expanduser(self.scope_file)))
        return p if p.is_absolute() else self.resolved_source_dir() / p


class BenchmarkDataset(BaseModel):
    name: str
    contests: list[BenchmarkContest] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> BenchmarkDataset:
        ids = [c.id for c in self.contests]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate contest ids {dupes}")
        return self


def resolve_path(raw: str) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(raw)))
    if p.is_absolute():
        return p
    base = os.environ.get("BENCHMARK_CONTESTS_DIR")
    if base is None:
        from audit_agent.config import get_settings

        base = get_settings().benchmark_contests_dir  # also honours .env
    return (Path(base) if base else REPO_ROOT) / p


def dataset_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".json" or p.exists():
        return p
    return DATASETS_DIR / f"{name_or_path}.json"


def load_dataset(name_or_path: str) -> BenchmarkDataset:
    """Load and validate a dataset by name (``benchmarks/datasets/<name>.json``) or path.

    Raises:
        FileNotFoundError: the dataset file does not exist.
        pydantic.ValidationError: the dataset violates the schema.
    """
    path = dataset_path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return BenchmarkDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


def select_contests(
    dataset: BenchmarkDataset,
    *,
    split: str = "test",
    contest_ids: Optional[Iterable[str]] = None,
    after_date: Optional[date] = None,
    include_needs_review: bool = False,
) -> list[BenchmarkContest]:
    """Contests to run. Defaults: split=test, drafts (needs_review) excluded.

    ``split="all"`` keeps every split. ``after_date`` keeps contests whose
    ``contest_end_date`` is strictly after it (contests without a date are
    dropped, since they cannot be shown to postdate the model's cutoff).
    Explicit ``contest_ids`` still obey the other filters.
    """
    wanted = set(contest_ids or [])
    unknown = wanted - {c.id for c in dataset.contests}
    if unknown:
        raise ValueError(f"unknown contest ids: {sorted(unknown)}")
    out = []
    for c in dataset.contests:
        if wanted and c.id not in wanted:
            continue
        if split != "all" and c.split != split:
            continue
        if c.needs_review and not include_needs_review:
            continue
        if after_date is not None and (c.contest_end_date is None or c.contest_end_date <= after_date):
            continue
        out.append(c)
    return out
