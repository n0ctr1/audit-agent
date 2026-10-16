"""Manual per-hypothesis GT labels: file generation, human recall, matcher agreement."""

from __future__ import annotations

from pathlib import Path

from benchmarks.human_labels import (
    COLUMNS,
    evaluate_labels,
    label_rows,
    parse_gt_ids,
    read_rows,
    write_labels,
)
from benchmarks.matching import GroundTruthFinding

from audit_agent.models.schemas import Hypothesis, ValidatedFinding


def _h(hid, title, fns, prio):
    return Hypothesis(id=hid, audit_id="a", title=title, vulnerability_class="reentrancy",
                      affected_functions=fns, attack_scenario="step " * 100,
                      estimated_severity="high", priority_score=prio).model_dump(mode="json")


def _f(hid, status):
    return ValidatedFinding(audit_id="a", hypothesis_id=hid, title="t", severity="high",
                            confidence=0.5, status=status, description="d").model_dump(mode="json")


GTS = [
    GroundTruthFinding(id="H-01", title="Reentrancy in withdraw", severity="high",
                       vulnerability_class="reentrancy", affected_functions=["Vault.withdraw"]),
    GroundTruthFinding(id="M-01", title="Fee rounding", severity="medium",
                       vulnerability_class="rounding", affected_functions=["Vault.fee"]),
]


def _state(n_filler=0):
    hyps = [_h("h1", "Reentrancy in withdraw", ["Vault.withdraw"], 90),
            _h("h2", "Fee rounds to zero", ["Vault.other"], 80),   # human match, matcher miss
            _h("h3", "Reentrancy again", ["Vault.withdraw"], 70)]  # matcher match, human says no
    hyps += [_h(f"x{i}", "filler", ["X.f"], 50 - i) for i in range(n_filler)]
    return {"hypotheses": hyps,
            "validated_findings": [_f("h1", "false_positive"), _f("h2", "unverified"), _f("h3", "validated")]}


def _labelled(tmp_path, state, labels):
    p = tmp_path / "labels.csv"
    write_labels(p, label_rows(state))
    rows = read_rows(p)
    for r in rows:
        r["matches_gt"], r["reviewed"] = labels.get(r["hypothesis_id"], ("", "x"))
    import csv
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return p


def test_rows_ranked_with_status_and_short_description():
    rows = label_rows(_state(n_filler=30))
    assert [r["hypothesis_id"] for r in rows[:3]] == ["h1", "h2", "h3"]
    assert rows[0]["in_top30"] == "yes" and rows[30]["in_top30"] == "no"
    assert rows[0]["final_status"] == "false_positive"
    assert rows[3]["final_status"] == "no_finding"
    assert len(rows[0]["description"]) == 200
    assert rows[0]["matches_gt"] == "" and rows[0]["reviewed"] == ""


def test_rewrite_keeps_labels(tmp_path: Path):
    p = _labelled(tmp_path, _state(), {"h1": ("H-01", "x")})
    write_labels(p, label_rows(_state()))  # e.g. rescore after a matcher change
    r = {x["hypothesis_id"]: x for x in read_rows(p)}
    assert (r["h1"]["matches_gt"], r["h1"]["reviewed"]) == ("H-01", "x")


def test_parse_ids():
    assert parse_gt_ids(" H-01; M-02 ,") == {"H-01", "M-02"}


def test_unreviewed_rows_block_metrics(tmp_path: Path):
    p = tmp_path / "labels.csv"
    write_labels(p, label_rows(_state()))
    res = evaluate_labels(read_rows(p), _state(), GTS)
    assert res.unreviewed == 3 and not res.complete
    assert res.human_recall_hypotheses is None and res.human_recall_final is None


def test_human_recall_and_agreement(tmp_path: Path):
    p = _labelled(tmp_path, _state(), {"h1": ("H-01", "x"), "h2": ("M-01", "x"), "h3": ("", "x")})
    res = evaluate_labels(read_rows(p), _state(), GTS)
    assert res.complete
    assert res.human_recall_hypotheses == 1.0             # H-01 (h1) + M-01 (h2), both in top 30
    assert res.human_recall_final == 0.5                  # only h2 is validated/unverified → M-01
    assert res.covered_final == ["M-01"]
    # pairs: h1×H-01 TP; h2×M-01 FN; h3×H-01 FP; the other 3 pairs TN
    assert (res.tp, res.fp, res.fn, res.tn) == (1, 1, 1, 3)
    assert res.matcher_precision == 0.5 and res.matcher_recall == 0.5
    assert {(d.hypothesis_id, d.gt_id) for d in res.disagreements} == {("h2", "M-01"), ("h3", "H-01")}


def test_unknown_gt_id_blocks_metrics(tmp_path: Path):
    p = _labelled(tmp_path, _state(), {"h1": ("H-09", "x")})
    res = evaluate_labels(read_rows(p), _state(), GTS)
    assert res.unknown_gt_ids == ["H-09"] and res.human_recall_final is None
