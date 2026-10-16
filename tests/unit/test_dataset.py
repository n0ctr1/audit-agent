"""Tests for the benchmark dataset schema and contest selection (benchmarks/dataset.py)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from benchmarks.dataset import (
    REPO_ROOT,
    BenchmarkContest,
    BenchmarkDataset,
    load_dataset,
    resolve_path,
    select_contests,
)
from pydantic import ValidationError


def _finding(fid="H-01", fns=("Vault.withdraw",)):
    return {"id": fid, "title": "Reentrancy", "severity": "high",
            "vulnerability_class": "reentrancy", "affected_functions": list(fns)}


def _contest(cid="c1", split="test", **kw):
    d = {"id": cid, "name": cid, "source_dir": "x", "split": split,
         "contest_end_date": "2026-05-01", "gt_source": "https://example.org/report",
         "official_findings": [_finding()]}
    d.update(kw)
    return d


class TestSchema:
    def test_valid_test_contest(self):
        c = BenchmarkContest.model_validate(_contest())
        assert c.split == "test"
        assert c.contest_end_date == date(2026, 5, 1)
        assert c.contaminated is False
        assert c.ground_truth()[0].id == "H-01"

    @pytest.mark.parametrize("override,msg", [
        ({"gt_source": None}, "gt_source"),
        ({"contest_end_date": None}, "contest_end_date"),
        ({"official_findings": [_finding(fid="")]}, "has no id"),
        ({"official_findings": [_finding(fns=())]}, "affected_functions"),
        ({"official_findings": [_finding(), _finding()]}, "duplicate finding ids"),
        ({"split": "holdout"}, "split"),
    ])
    def test_invalid_test_contest(self, override, msg):
        with pytest.raises(ValidationError, match=msg):
            BenchmarkContest.model_validate(_contest(**override))

    def test_invalid_severity(self):
        bad = _finding()
        bad["severity"] = "urgent"
        with pytest.raises(ValidationError):
            BenchmarkContest.model_validate(_contest(official_findings=[bad]))

    def test_drafts_and_smoke_are_lenient(self):
        BenchmarkContest.model_validate(_contest(needs_review=True, gt_source=None,
                                                 official_findings=[_finding(fns=())]))
        BenchmarkContest.model_validate(_contest(split="smoke", gt_source=None,
                                                 contest_end_date=None))

    def test_duplicate_contest_ids(self):
        with pytest.raises(ValidationError, match="duplicate contest ids"):
            BenchmarkDataset.model_validate({"name": "d", "contests": [_contest(), _contest()]})

    def test_load_validates_file(self, tmp_path: Path):
        p = tmp_path / "ds.json"
        p.write_text(json.dumps({"name": "d", "contests": [_contest(gt_source=None)]}))
        with pytest.raises(ValidationError):
            load_dataset(str(p))

    def test_load_by_path(self, tmp_path: Path):
        p = tmp_path / "ds.json"
        p.write_text(json.dumps({"name": "d", "contests": [_contest()]}))
        assert load_dataset(str(p)).contests[0].id == "c1"

    def test_shipped_datasets_validate(self):
        for p in (REPO_ROOT / "benchmarks" / "datasets").glob("*.json"):
            load_dataset(str(p))


class TestSelection:
    def _ds(self):
        return BenchmarkDataset.model_validate({"name": "d", "contests": [
            _contest("t1", contest_end_date="2026-01-10"),
            _contest("t2", contest_end_date="2026-06-10"),
            _contest("draft", needs_review=True),
            _contest("dev1", split="dev"),
            _contest("smoke1", split="smoke", contest_end_date=None, gt_source=None),
        ]})

    def test_default_is_reviewed_test_only(self):
        assert [c.id for c in select_contests(self._ds())] == ["t1", "t2"]

    def test_split_dev_and_all(self):
        assert [c.id for c in select_contests(self._ds(), split="dev")] == ["dev1"]
        assert [c.id for c in select_contests(self._ds(), split="all")] == ["t1", "t2", "dev1", "smoke1"]

    def test_include_needs_review(self):
        ids = [c.id for c in select_contests(self._ds(), include_needs_review=True)]
        assert "draft" in ids

    def test_after_date_strict_and_drops_undated(self):
        ids = [c.id for c in select_contests(self._ds(), split="all", after_date=date(2026, 1, 10))]
        assert ids == ["t2", "dev1"]  # t1 ends on the cutoff; smoke1 has no date

    def test_contest_ids_filter(self):
        assert [c.id for c in select_contests(self._ds(), contest_ids=["t2"])] == ["t2"]
        # explicit ids still obey the split filter
        assert select_contests(self._ds(), contest_ids=["dev1"]) == []
        with pytest.raises(ValueError, match="unknown contest ids"):
            select_contests(self._ds(), contest_ids=["nope"])


class TestPaths:
    def test_relative_to_repo_root_by_default(self, monkeypatch):
        monkeypatch.delenv("BENCHMARK_CONTESTS_DIR", raising=False)
        assert resolve_path("a/b") == REPO_ROOT / "a" / "b"

    def test_relative_to_env_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BENCHMARK_CONTESTS_DIR", str(tmp_path))
        assert resolve_path("c4/x") == tmp_path / "c4" / "x"

    def test_scope_file_relative_to_source(self, tmp_path):
        c = BenchmarkContest.model_validate(_contest(source_dir=str(tmp_path), scope_file="scope.txt"))
        assert c.resolved_scope_file() == tmp_path / "scope.txt"


def test_contamination_fields_roundtrip():
    c = BenchmarkContest.model_validate(_contest(contaminated=True, report_published="2025-11-21",
                                                 contamination_note="report published before model release"))
    assert c.report_published == date(2025, 11, 21)
    d = c.model_dump(mode="json")
    assert d["contamination_note"] == "report published before model release"
    assert d["report_published"] == "2025-11-21"
