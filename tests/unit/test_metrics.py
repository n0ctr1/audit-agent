"""Tests for benchmark metrics (benchmarks/metrics.py) on hand-computed synthetic states."""

from __future__ import annotations

import pytest
from benchmarks.matching import GroundTruthFinding
from benchmarks.metrics import (
    ContestRunMetrics,
    aggregate_runs,
    compute_contest_metrics,
    exploit_bucket,
    pooled_recall,
    summarize,
)

from audit_agent.llm.base import LLMResponse
from audit_agent.llm.mock_provider import MockProvider
from audit_agent.llm.usage import UsageTrackingProvider
from audit_agent.models.schemas import ExploitResult, Hypothesis, ValidatedFinding


def _gt(gid, title, vc, fns, sev):
    return GroundTruthFinding(id=gid, title=title, severity=sev, vulnerability_class=vc,
                              affected_functions=fns)


def _h(hid, title, vc, fns, prio):
    return Hypothesis(id=hid, audit_id="a", title=title, vulnerability_class=vc,
                      affected_functions=fns, attack_scenario="x", estimated_severity="high",
                      priority_score=prio)


def _f(fid, hid, status, skeptic_status=None, conf=0.95, title="finding"):
    return ValidatedFinding(id=fid, audit_id="a", hypothesis_id=hid, title=title,
                            severity="high", confidence=conf, status=status,
                            skeptic_status=skeptic_status, description="d")


def _e(hid, status, steps=None):
    return ExploitResult(hypothesis_id=hid, test_type="unit", status=status, poc_code="",
                         reproduction_steps=steps or [])


GTS = [
    _gt("H-01", "Reentrancy in withdraw", "SWC-107", ["Vault.withdraw"], "high"),
    _gt("H-02", "Oracle price manipulation", "oracle", ["Oracle.getPrice", "Pool.borrow"], "high"),
    _gt("M-01", "Missing access control on setFee", "access-control", ["Pool.setFee"], "medium"),
    _gt("M-02", "Rounding error in redeem", "rounding", ["Vault.redeem"], "medium"),
]


def _state():
    hyps = [
        _h("h1", "Reentrancy in withdraw", "reentrancy", ["Vault.withdraw"], 90),
        _h("h2", "Spot price manipulation in borrow", "price manipulation", ["Pool.borrow"], 80),
        _h("h3", "Anyone can set fee", "unauthorized", ["Pool.setFee"], 70),
        _h("h4", "Gas griefing", "dos", ["Pool.loop"], 60),
        _h("h5", "Rounding in redeem", "rounding", ["Vault.redeem"], 50),
        _h("h6", "Unrelated", "gas", ["X.y"], 40),
    ]
    findings = [
        _f("f1", "h1", "validated", "validated"),
        _f("f2", "h2", "false_positive", "validated"),       # killed by Refuter
        _f("f3", "h3", "false_positive", "false_positive"),  # killed by Skeptic
        _f("f4", "h4", "validated", "validated"),            # not in GT
        _f("f5", "h6", "unverified", "unverified"),
    ]
    exploits = [
        _e("h1", "exploited"), _e("h2", "partial"), _e("h3", "blocked"),
        _e("h4", "exploited"), _e("h6", "error", ["Failed after 3 compile attempts"]),
    ]
    return {"hypotheses": hyps, "validated_findings": findings, "exploits": exploits,
            "tokens_used": 12345, "cost_usd": 1.25}


@pytest.fixture
def m() -> ContestRunMetrics:
    return compute_contest_metrics(_state(), GTS, contest_id="c", time_s=10.0)


class TestHandComputed:
    def test_counts(self, m):
        assert m.num_ground_truth == 4
        assert m.num_hypotheses == 6
        assert m.num_final_findings == 3   # f1, f4 validated + f5 unverified
        assert m.num_validated == 2

    def test_recalls(self, m):
        assert m.recall_hypotheses_at_30 == 1.0
        assert m.recall_hypotheses_all == 1.0
        assert m.recall_final == 0.25
        assert m.recall_validated == 0.25

    def test_match_records_name_hypothesis_and_finding(self, m):
        assert {(r.gt_id, r.candidate_id) for r in m.hypothesis_matches} == {
            ("H-01", "h1"), ("H-02", "h2"), ("M-01", "h3"), ("M-02", "h5")}
        assert [(r.gt_id, r.candidate_id) for r in m.final_matches] == [("H-01", "f1")]
        assert [(r.gt_id, r.candidate_id) for r in m.validated_matches] == [("H-01", "f1")]

    def test_validator_and_pipeline_loss(self, m):
        assert m.validator_loss_count == 2
        assert {(r.gt_id, r.reason) for r in m.validator_loss} == {
            ("H-02", "rejected_by_refuter"), ("M-01", "rejected_by_skeptic")}
        assert m.pipeline_loss_count == 1
        assert m.pipeline_loss[0].gt_id == "M-02"
        assert m.pipeline_loss[0].reason == "never_validated"
        assert m.pipeline_loss[0].hypothesis_ids == ["h5"]

    def test_precision_gt(self, m):
        assert m.precision_gt == 0.5
        assert m.num_validated_matched == 1
        assert m.unmatched_validated_ids == ["f4"]

    def test_precision_adjusted_requires_labels(self, m):
        assert m.precision_adjusted.value is None
        assert m.precision_adjusted.unlabelled == 1

    @pytest.mark.parametrize("label,expected", [
        ("tp_not_in_gt", 1.0),   # (1 + 1) / 2
        ("fp", 0.5),             # 1 / 2
        ("duplicate", 1.0),      # 1 / (2 - 1)
        ("unsure", 0.5),         # conservative: not a true positive → 1 / 2
    ])
    def test_precision_adjusted_with_labels(self, label, expected):
        m = compute_contest_metrics(_state(), GTS, contest_id="c", labels={"f4": label})
        assert m.precision_adjusted.value == expected
        assert m.precision_adjusted.unlabelled == 0
        assert m.precision_adjusted.labelled[label] == 1

    def test_recall_by_severity(self, m):
        hi, med = m.recall_by_severity["high"], m.recall_by_severity["medium"]
        assert (hi.num_gt, hi.recall_hypotheses_at_30, hi.recall_final, hi.recall_validated) == (2, 1.0, 0.5, 0.5)
        assert (med.num_gt, med.recall_hypotheses_at_30, med.recall_final, med.recall_validated) == (2, 1.0, 0.0, 0.0)

    def test_exploit_stats(self, m):
        es = m.exploit_stats
        assert es.counts == {"exploited": 2, "partial": 1, "blocked": 1, "error": 0, "compile_error": 1}
        assert es.num_gt_matched == 3 and es.num_gt_unmatched == 2
        assert es.exploited_rate_gt_matched == pytest.approx(1 / 3)
        assert es.exploited_rate_gt_unmatched == 0.5

    def test_validator_confusion(self, m):
        sk, fin = m.validator_confusion["skeptic"], m.validator_confusion["final"]
        assert sk["validated"] == {"gt_match": 2, "no_gt_match": 1}
        assert sk["unverified"] == {"gt_match": 0, "no_gt_match": 1}
        assert sk["false_positive"] == {"gt_match": 1, "no_gt_match": 0}
        assert fin["validated"] == {"gt_match": 1, "no_gt_match": 1}
        assert fin["unverified"] == {"gt_match": 0, "no_gt_match": 1}
        assert fin["false_positive"] == {"gt_match": 2, "no_gt_match": 0}

    def test_cost_tokens_time_from_state(self, m):
        assert m.cost_usd == 1.25
        assert m.tokens == 12345
        assert m.time_s == 10.0

    def test_explicit_cost_overrides_state(self):
        m = compute_contest_metrics(_state(), GTS, contest_id="c", cost_usd=2.0, tokens=7)
        assert (m.cost_usd, m.tokens) == (2.0, 7)

    def test_explicit_none_means_not_measured(self):
        m = compute_contest_metrics(_state(), GTS, contest_id="c", cost_usd=None, tokens=None)
        assert (m.cost_usd, m.tokens) == (None, None)

    def test_accepts_plain_dicts(self):
        s = _state()
        d = {k: ([x.model_dump(mode="json") for x in v] if isinstance(v, list) else v)
             for k, v in s.items()}
        m = compute_contest_metrics(d, GTS, contest_id="c")
        assert m.recall_final == 0.25 and m.precision_gt == 0.5


class TestPrecisionIsNotSelfAssessment:
    def test_zero_gt_matches_gives_zero_precision(self):
        gts = [_gt("H-01", "Reentrancy in withdraw", "SWC-107", ["Vault.withdraw"], "high")]
        hyps = [_h(f"h{i}", f"Issue {i}", "logic", [f"Other.fn{i}"], 50) for i in range(3)]
        findings = [_f(f"f{i}", f"h{i}", "validated", "validated", conf=0.95) for i in range(3)]
        m = compute_contest_metrics({"hypotheses": hyps, "validated_findings": findings},
                                    gts, contest_id="c")
        assert m.num_validated == 3
        assert m.precision_gt == 0.0

    def test_no_validated_findings_precision_undefined(self):
        m = compute_contest_metrics({"hypotheses": [], "validated_findings": []}, GTS,
                                    contest_id="c")
        assert m.precision_gt is None
        assert m.recall_final == 0.0


class TestTopN:
    def test_only_top_30_by_priority_count(self):
        gts = [_gt("H-01", "Reentrancy in withdraw", "SWC-107", ["Vault.withdraw"], "high")]
        filler = [_h(f"x{i}", "Other", "gas", [f"A.f{i}"], 100 - i) for i in range(30)]
        hit = _h("hit", "Reentrancy in withdraw", "reentrancy", ["Vault.withdraw"], 1)
        for hyps in (filler + [hit], [hit] + filler):
            m = compute_contest_metrics({"hypotheses": hyps}, gts, contest_id="c")
            assert m.recall_hypotheses_at_30 == 0.0
            assert m.recall_hypotheses_all == 1.0


class TestLossAssignedElsewhere:
    def test_finding_used_for_other_gt(self):
        gts = [
            _gt("A", "Withdraw reentrancy", "SWC-107", ["Vault.withdraw"], "high"),
            _gt("B", "Withdraw rounding", "rounding", ["Vault.withdraw"], "medium"),
        ]
        hyps = [
            _h("h1", "Withdraw reentrancy", "SWC-107", ["Vault.withdraw"], 10),
            _h("h2", "Withdraw rounding", "rounding", ["Vault.withdraw"], 9),
        ]
        findings = [_f("f1", "h1", "validated")]  # h2 never validated
        m = compute_contest_metrics({"hypotheses": hyps, "validated_findings": findings},
                                    gts, contest_id="c")
        assert m.recall_final == 0.5
        # B matched h2 (never validated) and h1 (whose finding closed A).
        assert m.pipeline_loss[0].gt_id == "B"
        assert m.pipeline_loss[0].reason == "assigned_elsewhere"


class TestExploitBucket:
    def test_compile_error_from_trace(self):
        e = ExploitResult(hypothesis_id="h", test_type="unit", status="error", poc_code="",
                          execution_trace="Compiler run failed:\nError (7576): Undeclared identifier.")
        assert exploit_bucket(e) == "compile_error"

    def test_plain_error(self):
        assert exploit_bucket(_e("h", "error", ["docker timeout"])) == "error"


class TestAggregation:
    def test_summarize(self):
        s = summarize([0.5, None, 1.0])
        assert (s.mean, s.min, s.max, s.n) == (0.75, 0.5, 1.0, 2)
        assert summarize([]).mean is None

    def test_aggregate_excludes_errors_and_pools(self):
        ok = compute_contest_metrics(_state(), GTS, contest_id="c", run_index=0)
        err = ContestRunMetrics(contest_id="c", run_index=1, status="error", error="boom")
        agg = aggregate_runs([ok, err])
        assert agg["recall_final"].n == 1
        assert agg["recall_final"].mean == 0.25
        assert pooled_recall([ok, err], "recall_final") == 0.25


class TestUsageTracking:
    async def test_tokens_counted_and_unpriced_model_flagged(self):
        resp = LLMResponse(text="ok", tool_calls=[], input_tokens=100, output_tokens=20,
                           stop_reason="end_turn", model="m")
        p = UsageTrackingProvider(MockProvider([resp, resp]))
        await p.complete(model="m", system="s", messages=[])
        assert await p.complete_simple(model="m", system="s", user_message="u") == "ok"
        u = p.usage.to_dict()
        assert u["calls"] == 2
        assert u["input_tokens"] == 200 and u["output_tokens"] == 40 and u["tokens"] == 240
        assert u["cost_usd"] is None
        assert u["unpriced_models"] == ["m"]

    async def test_priced_model_cost(self):
        class Priced(MockProvider):
            def has_pricing(self, model):
                return True

            def estimate_cost(self, model, i, o):
                return (i + o) * 1e-6

        resp = LLMResponse(text="", tool_calls=[], input_tokens=1000, output_tokens=0,
                           stop_reason="end_turn", model="m")
        p = UsageTrackingProvider(Priced([resp]))
        await p.complete(model="m", system="s", messages=[])
        assert p.usage.cost_usd == pytest.approx(0.001)


class TestSkepticStatusSnapshot:
    async def test_refuter_node_records_skeptic_status(self, monkeypatch):
        from audit_agent.graph.nodes import refuter_node
        from audit_agent.llm.factory import set_provider

        monkeypatch.setenv("REFUTER_ENABLED", "false")
        set_provider(MockProvider())
        try:
            f = _f("f1", "h1", "validated")
            out = await refuter_node({"audit_id": "a", "validated_findings": [f]})
            assert out["validated_findings"][0].skeptic_status == "validated"
        finally:
            set_provider(None)


class TestUnmatchedCsv:
    def test_roundtrip_keeps_existing_labels(self, tmp_path):
        from benchmarks.labels import UnmatchedRow, labels_for_run, read_labels, write_unmatched

        path = tmp_path / "unmatched_findings.csv"
        def row(fid):
            return UnmatchedRow(contest="c", run=0, finding_id=fid, hypothesis_id="h",
                                title="t", functions="A.f", severity="high", report_path="r.md")
        write_unmatched(path, [row("f1"), row("f2")])
        assert read_labels(path) == {}

        lines = path.read_text().splitlines()
        lines[1] += "fp"  # human labels f1
        path.write_text("\n".join(lines) + "\n")
        assert read_labels(path) == {("c", 0, "f1"): "fp"}

        write_unmatched(path, [row("f1"), row("f2"), row("f3")])  # rescore rewrites the file
        labels = read_labels(path)
        assert labels == {("c", 0, "f1"): "fp"}
        assert labels_for_run(labels, "c", 0) == {"f1": "fp"}
        assert labels_for_run(labels, "c", 1) == {}

    def test_labelled_row_kept_when_matcher_now_matches(self, tmp_path):
        from benchmarks.labels import UnmatchedRow, read_labels, write_unmatched

        path = tmp_path / "unmatched_findings.csv"

        def row(fid, label=""):
            return UnmatchedRow(contest="c", run=0, finding_id=fid, hypothesis_id="h", title="t",
                                functions="A.f", severity="high", report_path="", label=label)

        write_unmatched(path, [row("f1", "fp"), row("f2")])
        write_unmatched(path, [row("f2")])  # f1 is matched by a newer matcher
        text = path.read_text()
        assert "matched_now" in text
        assert read_labels(path) == {("c", 0, "f1"): "fp"}
        write_unmatched(path, [row("f1"), row("f2")])  # and unmatched again
        assert read_labels(path) == {("c", 0, "f1"): "fp"}
        assert "matched_now" not in path.read_text()

    def test_invalid_label_ignored(self, tmp_path):
        from benchmarks.labels import read_labels

        path = tmp_path / "u.csv"
        path.write_text("contest,run,finding_id,hypothesis_id,title,functions,severity,report_path,label\n"
                        "c,0,f1,h,t,f,high,r,maybe\n")
        assert read_labels(path) == {}
