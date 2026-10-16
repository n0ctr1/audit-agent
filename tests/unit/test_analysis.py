"""benchmarks/analysis.py on synthetic stored runs (no LLM, no Docker)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.analysis import CAUSES, build_analysis, classify_compile_error, load_runs
from benchmarks.human_labels import COLUMNS, label_rows, labels_filename
from typer.testing import CliRunner

from audit_agent.cli import app
from audit_agent.models.schemas import ExploitResult, Hypothesis, ValidatedFinding

GT = [{"id": f"G{i}", "title": f"Bug {i}", "severity": "high" if i < 3 else "medium",
       "vulnerability_class": "logic", "affected_functions": [f"Vault.f{i}"]} for i in range(1, 6)]


def solc_error(code: str, msg: str, line: str, col: int, width: int) -> str:
    return (f"Error ({code}): {msg}\n  --> test/PoC.t.sol:10:{col}:\n   |\n"
            f"10 | {line}\n   | {' ' * (col - 1)}{'^' * width}\n")


NON_TEST = solc_error("7576", "Undeclared identifier.", "        vm.prank(user);", 9, 2)
STATE = solc_error("7576", "Undeclared identifier.", "        vault.deposit(1);", 9, 5)
API = solc_error("9582", 'Member "x" not found or not visible.', "        harness.x();", 9, 9)
DUP = solc_error("2333", "Identifier already declared.", "    uint256 a;", 5, 9)


def _poc(parent: str) -> str:
    return f'import "forge-std/Test.sol";\ncontract AuditAgentPoC is {parent} {{ function test_x() public {{}} }}'


def _hyp(i, prio):
    return Hypothesis(id=f"h{i}", audit_id="a", title=f"Hyp {i}", vulnerability_class="logic",
                      affected_functions=[f"Vault.f{i}"], attack_scenario="scenario " * 40,
                      estimated_severity="high", priority_score=prio).model_dump(mode="json")


def _finding(i, skeptic, final):
    return ValidatedFinding(id=f"f{i}", audit_id="a", hypothesis_id=f"h{i}", title=f"Hyp {i}",
                            severity="high", confidence=0.5, status=final, skeptic_status=skeptic,
                            description="d").model_dump(mode="json")


def _exploit(i, status, steps=()):
    return ExploitResult(hypothesis_id=f"h{i}", test_type="unit", status=status, poc_code=_poc("Test"),
                         reproduction_steps=list(steps)).model_dump(mode="json")


def make_run(root: Path, name: str, *, preset="full", labels=None, logs=(), exploits=None) -> Path:
    src = root / "src_project"
    (src / "test").mkdir(parents=True, exist_ok=True)
    (src / "test" / "Base.sol").write_text("contract ProjBase is Test { function setUp() public {} }\n"
                                           "contract Helper { function setUp() public {} }")
    rd = root / name
    cdir = rd / "c1"
    cdir.mkdir(parents=True)
    contest = {"id": "c1", "name": "c1", "source_dir": str(src), "split": "test",
               "contest_end_date": "2026-06-01", "gt_source": "https://example.org",
               "official_findings": GT}
    flags = {"SKEPTIC_ENABLED": preset != "hunter_only"}
    (rd / "meta.json").write_text(json.dumps({"contests": [contest], "preset": {"name": preset},
                                              "flags": flags, "git": {"commit": name * 3}}))
    hyps = [_hyp(i, 100 - i) for i in range(1, 6)]
    state = {
        "hypotheses": hyps,
        # h1 validated; h2 Skeptic validated → Refuter FP; h3 no finding; h4 Skeptic FP; h5 unverified (no GT)
        "validated_findings": [_finding(1, "validated", "validated"),
                               _finding(2, "validated", "false_positive"),
                               _finding(4, "false_positive", "false_positive"),
                               _finding(5, "unverified", "unverified")],
        "exploits": exploits if exploits is not None else [
            _exploit(1, "exploited"), _exploit(2, "error", ["Failed after 3 compile attempts"]),
            _exploit(4, "error", ["Failed after 3 compile attempts"]), _exploit(5, "partial")],
    }
    (cdir / "run_0.json").write_text(json.dumps({"contest_id": "c1", "run_index": 0, "status": "ok",
                                                 "state": state}))
    tools = cdir / "run_0_logs" / "audit_logs" / "x" / "tools"
    tools.mkdir(parents=True)
    for n, (hid, attempt, stdout, parent) in enumerate(logs):
        (tools / f"tool_{n:03d}_forge.json").write_text(json.dumps({
            "tool_name": "forge", "status": "compile_error", "hypothesis_id": hid,
            "attempt_number": attempt, "stdout": "Compiler run failed:\n" + stdout, "stderr": "",
            "poc_code": _poc(parent)}))
    rows = label_rows(state)
    for r in rows:
        if labels is not None:
            r["reviewed"] = "x"
            r["matches_gt"] = labels.get(r["hypothesis_id"], "")
    with (rd / labels_filename("c1", 0)).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rd


LABELS = {"h1": "G1", "h2": "G2", "h3": "G3", "h4": "G4"}  # G5: no hypothesis
LOGS = [("h2", 1, NON_TEST, "Helper"), ("h2", 2, STATE, "Test"),     # final attempt for h2: STATE
        ("h4", 1, API, "Test")]


def test_classifier_causes():
    assert classify_compile_error({"stdout": NON_TEST, "poc_code": _poc("Helper")}) == CAUSES[0]
    assert classify_compile_error({"stdout": STATE, "poc_code": _poc("Test")}) == CAUSES[1]
    assert classify_compile_error({"stdout": API, "poc_code": _poc("Test")}) == CAUSES[2]
    assert classify_compile_error({"stdout": 'Error (6275): Source "x" not found', "poc_code": ""}) == CAUSES[3]
    assert classify_compile_error({"stdout": DUP, "poc_code": _poc("Test")}) == CAUSES[4]


def test_pending_labels_never_fall_back_to_matcher(tmp_path: Path):
    rd = make_run(tmp_path, "r1", labels=None, logs=LOGS)
    md = build_analysis([rd])
    assert "*pending labels* — 0/5 rows reviewed" in md
    assert "Lost at" not in md                       # no GT path table without labels
    assert "| validated | 1 (pending labels) | 0 | 1 (pending labels) |" in md  # counts, no GT split
    assert "GT 1/" not in md


def test_gt_paths_with_labels(tmp_path: Path):
    rd = make_run(tmp_path, "r1", labels=LABELS, logs=LOGS)
    md = build_analysis([rd])
    for gid, stage in [("G1", "reported as validated"), ("G2", "rejected by Refuter"),
                       ("G3", "not sent to Exploit/Skeptic (no finding)"), ("G4", "rejected by Skeptic"),
                       ("G5", "not found (no matching hypothesis)")]:
        assert any(line.startswith(f"| {gid} |") and line.rstrip().endswith(f"{stage} |")
                   for line in md.splitlines()), gid
    assert "reported as validated: 1/5 GT" in md and "not found (no matching hypothesis): 1/5 GT" in md
    assert "(preset full, n = 5 GT)" in md


def test_validator_matrix_with_labels(tmp_path: Path):
    rd = make_run(tmp_path, "r1", labels=LABELS, logs=LOGS)
    md = build_analysis([rd])
    assert "(n = 4 findings)" in md
    assert "| validated | 1 (GT 1/1) | 0 | 1 (GT 1/1) |" in md
    assert "| false_positive | 0 | 0 | 1 (GT 1/1) |" in md
    assert "Refuter turned 1/3 findings it reviewed into false positives (1/1 of them match GT)." in md


def test_hunter_only_has_no_validator_matrix(tmp_path: Path):
    rd = make_run(tmp_path, "r1", preset="hunter_only", labels=LABELS, exploits=[])
    md = build_analysis([rd])
    assert "Not applicable (preset hunter_only" in md


def test_compile_section_counts_sample_and_marker(tmp_path: Path):
    rd = make_run(tmp_path, "r1", labels=None, logs=LOGS)
    md = build_analysis([rd], seed=1)
    assert "**Automatic classification**" in md
    assert "compile_error 2/4 PoCs" in md
    assert f"| {CAUSES[1]} | 1 | 1 |" in md          # h2 final attempt → state/helpers
    assert f"| {CAUSES[2]} | 1 | 1 |" in md          # h4
    assert f"| {CAUSES[0]} | 0 | 1 |" in md          # h2's first attempt only
    assert "Manual check sample (2 of 2 PoCs, seed 1)" in md
    assert build_analysis([rd], seed=1) == md         # deterministic for a given seed


def test_sample_capped_at_five(tmp_path: Path):
    logs = [(f"h{i}", 1, API, "Test") for i in (1, 2, 4, 5)]
    ex = [_exploit(i, "error", ["Failed after 3 compile attempts"]) for i in (1, 2, 4, 5)]
    rds = [make_run(tmp_path / str(n), "r", logs=logs, exploits=ex) for n in range(2)]
    md = build_analysis(rds, seed=3)
    assert "Manual check sample (5 of 8 PoCs, seed 3)" in md
    assert sum(1 for line in md.splitlines() if line[:3] in {f"{i}. " for i in range(1, 9)}) == 5


def test_before_after(tmp_path: Path):
    before = make_run(tmp_path / "b", "before", logs=LOGS)
    fixed = [_exploit(i, s) for i, s in ((1, "exploited"), (2, "partial"), (4, "exploited"), (5, "partial"))]
    after = make_run(tmp_path / "a", "after", exploits=fixed)
    md = build_analysis([before, after], before=before, after=after)
    assert "| Exploit: compile_error | 2/4 PoCs | 0/4 PoCs |" in md
    assert "| Exploit: exploited | 1/4 PoCs | 2/4 PoCs |" in md
    assert "Same contest, one run each" in md


def test_load_runs_reads_labels(tmp_path: Path):
    rd = make_run(tmp_path, "r1", labels=LABELS)
    (run,) = load_runs([rd])
    assert run.labels_complete and run.matches()["h1"] == {"G1"}


def test_cli_writes_markdown(tmp_path: Path):
    rd = make_run(tmp_path, "r1", labels=None, logs=LOGS)
    out = tmp_path / "A.md"
    res = CliRunner().invoke(app, ["benchmark", "analysis", str(rd), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert out.read_text().startswith("# Pilot analysis")

