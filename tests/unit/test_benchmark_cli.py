"""`audit-agent benchmark` end-to-end in mock mode (MockProvider, AUDIT_AGENT_TOOL_MODE=mock).

The full LangGraph pipeline runs offline on the VulnerableBank fixture; Slither
output comes from the cached fixture JSON.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from audit_agent.cli import app
from audit_agent.llm.base import LLMResponse, ToolCall
from audit_agent.llm.factory import set_provider
from audit_agent.llm.mock_provider import MockProvider

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

HYPOTHESES = [
    {"title": "Reentrancy in withdraw", "vulnerability_class": "reentrancy",
     "affected_functions": ["VulnerableBank.withdraw"], "attack_scenario": "re-enter withdraw",
     "required_preconditions": [], "estimated_severity": "high"},
    {"title": "Fee truncates to zero", "vulnerability_class": "rounding",
     "affected_functions": ["VulnerableBank.collectFee"], "attack_scenario": "tiny amounts",
     "required_preconditions": [], "estimated_severity": "medium"},
]


async def _llm(**kw):
    names = [t.name for t in (kw.get("tools") or [])]

    def tool(name, args):
        return LLMResponse(text="", tool_calls=[ToolCall(id="1", name=name, arguments=args)],
                           input_tokens=100, output_tokens=20, stop_reason="tool_use",
                           model=kw["model"])

    if "report_hypotheses" in names:
        return tool("report_hypotheses", {"hypotheses": HYPOTHESES})
    if "validate_exploit" in names:
        return tool("validate_exploit", {"verdict": "true_positive", "confidence": 0.9,
                                         "reasoning": "r", "real_impact": "i",
                                         "suggested_severity": "high"})
    return LLMResponse(text="", tool_calls=[], input_tokens=1, output_tokens=1,
                       stop_reason="end_turn", model=kw["model"])


@pytest.fixture(autouse=True)
def _restore_event_loop():
    # The CLI uses asyncio.run(), which leaves no current loop; some older tests
    # in the suite still call asyncio.get_event_loop().
    yield
    asyncio.set_event_loop(asyncio.new_event_loop())


@pytest.fixture
def mock_llm():
    set_provider(MockProvider(callback=_llm))
    yield
    set_provider(None)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    src = tmp_path / "contest"
    shutil.copytree(FIXTURES / "contracts", src)
    shutil.copy(FIXTURES / "slither_output.json", src / ".slither_analysis.json")
    ds = json.loads((FIXTURES / "benchmarks" / "smoke_fixture.json").read_text())
    ds["contests"][0]["source_dir"] = str(src)
    broken = json.loads(json.dumps(ds["contests"][0]))
    broken["id"] = "fixture-missing-source"
    broken["source_dir"] = str(tmp_path / "does-not-exist")
    ds["contests"].append(broken)
    p = tmp_path / "smoke.json"
    p.write_text(json.dumps(ds))
    return p


def _invoke(*args: str):
    res = CliRunner().invoke(app, list(args), catch_exceptions=False)
    assert res.exit_code == 0, res.output
    return res


def _only_results_dir(out: Path) -> Path:
    dirs = [d for d in out.iterdir() if d.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def _strip(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "generated_at"}


def test_benchmark_runs_and_rescore_reproduces(dataset: Path, tmp_path: Path, mock_llm):
    out = tmp_path / "results"
    _invoke("benchmark", "--dataset", str(dataset), "--split", "smoke", "--runs", "1",
            "--out", str(out), "--no-fuzz")
    rd = _only_results_dir(out)
    assert rd.name.endswith("_full_claude-opus-4-7")

    meta = json.loads((rd / "meta.json").read_text())
    assert meta["preset"]["name"] == "full"
    assert meta["runs"] == 1
    assert "hunter" in meta["models"] and "REFUTER_ENABLED" in meta["flags"]
    assert {"commit", "dirty"} <= set(meta["git"])
    assert [c["id"] for c in meta["contests"]] == ["fixture-vulnerable-bank", "fixture-missing-source"]

    run = json.loads((rd / "fixture-vulnerable-bank" / "run_0.json").read_text())
    assert run["status"] == "ok"
    assert {"hypotheses", "exploits", "validated_findings", "tokens_used", "cost_usd", "errors"} <= set(run["state"])
    assert run["usage"]["tokens"] > 0
    assert run["usage"]["cost_usd"] is None  # MockProvider has no price table
    assert (rd / "fixture-vulnerable-bank" / "run_0_reports").is_dir()

    failed = json.loads((rd / "fixture-missing-source" / "run_0.json").read_text())
    assert failed["status"] == "error"
    assert "source_dir not found" in failed["error"]

    metrics = json.loads((rd / "metrics.json").read_text())
    ok = [r for r in metrics["runs"] if r["contest_id"] == "fixture-vulnerable-bank"][0]
    assert ok["recall_hypotheses_at_30"] == 0.5
    assert ok["recall_final"] == 0.5
    assert ok["num_validated"] == 2
    assert ok["precision_gt"] == 0.5
    assert ok["cost_usd"] is None and ok["tokens"] > 0
    assert metrics["splits"]["smoke"]["num_runs_error"] == 1

    _invoke("benchmark", "rescore", str(rd))
    assert _strip(json.loads((rd / "metrics.json").read_text())) == _strip(metrics)


def test_labels_drive_precision_adjusted(dataset: Path, tmp_path: Path, mock_llm):
    out = tmp_path / "results"
    _invoke("benchmark", "--dataset", str(dataset), "--split", "smoke",
            "--contest", "fixture-vulnerable-bank", "--out", str(out), "--no-fuzz")
    rd = _only_results_dir(out)
    csv_path = rd / "unmatched_findings.csv"
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 2 and "collectFee" in lines[1]
    assert json.loads((rd / "metrics.json").read_text())["unlabelled_unmatched_findings"] == 1

    csv_path.write_text("\n".join([lines[0], lines[1] + "tp_not_in_gt"]) + "\n")
    _invoke("benchmark", "rescore", str(rd))
    run = json.loads((rd / "metrics.json").read_text())["runs"][0]
    assert run["precision_adjusted"]["value"] == 1.0
    assert run["precision_adjusted"]["labelled"]["tp_not_in_gt"] == 1
    assert csv_path.read_text().splitlines()[1].endswith("tp_not_in_gt")  # label kept


def test_multiple_runs(dataset: Path, tmp_path: Path, mock_llm):
    out = tmp_path / "results"
    _invoke("benchmark", "--dataset", str(dataset), "--split", "smoke",
            "--contest", "fixture-vulnerable-bank", "--runs", "2", "--out", str(out), "--no-fuzz")
    rd = _only_results_dir(out)
    assert sorted(p.name for p in (rd / "fixture-vulnerable-bank").glob("run_*.json")) == [
        "run_0.json", "run_1.json"]
    summ = json.loads((rd / "metrics.json").read_text())["splits"]["smoke"]["summary"]
    assert summ["recall_final"] == {"mean": 0.5, "min": 0.5, "max": 0.5, "n": 2}


def test_default_split_selects_nothing_for_smoke_dataset(dataset: Path, tmp_path: Path, mock_llm):
    res = CliRunner().invoke(app, ["benchmark", "--dataset", str(dataset), "--out", str(tmp_path)])
    assert res.exit_code == 1
    assert "No contests selected" in res.output


def test_unknown_dataset_and_missing_results(tmp_path: Path):
    res = CliRunner().invoke(app, ["benchmark", "--dataset", "nope", "--split", "smoke"])
    assert res.exit_code == 1
    res = CliRunner().invoke(app, ["benchmark", "rescore", str(tmp_path)])
    assert res.exit_code == 1


# ── presets (P1-7) ───────────────────────────────────────────────────────────

def test_preset_env_sets_every_flag_explicitly():
    from benchmarks.presets import ABLATION_FLAGS, PRESETS, preset_env

    assert set(PRESETS) == {"full", "no_refuter", "no_experts", "no_threat_model", "hunter_only",
                            "baseline"}
    for name in PRESETS:
        assert set(preset_env(name)) == set(ABLATION_FLAGS)
    assert all(v == "true" for v in preset_env("full").values())
    assert preset_env("no_refuter")["REFUTER_ENABLED"] == "false"
    with pytest.raises(ValueError):
        preset_env("nope")


def test_applied_preset_restores_environment(monkeypatch):
    import os

    from benchmarks.presets import applied_preset

    from audit_agent.config import get_settings

    monkeypatch.setenv("REFUTER_ENABLED", "true")
    monkeypatch.delenv("SKEPTIC_ENABLED", raising=False)
    with applied_preset("hunter_only"):
        s = get_settings()
        assert (s.refuter_enabled, s.skeptic_enabled, s.exploit_enabled) == (False, False, False)
    assert os.environ["REFUTER_ENABLED"] == "true"
    assert "SKEPTIC_ENABLED" not in os.environ
    assert get_settings().skeptic_enabled is True


def _run_preset(dataset: Path, tmp_path: Path, preset: str) -> tuple[dict, dict]:
    out = tmp_path / preset
    _invoke("benchmark", "--dataset", str(dataset), "--split", "smoke",
            "--contest", "fixture-vulnerable-bank", "--preset", preset,
            "--out", str(out), "--no-fuzz")
    rd = _only_results_dir(out)
    assert f"_{preset}_" in rd.name
    meta = json.loads((rd / "meta.json").read_text())
    run = json.loads((rd / "fixture-vulnerable-bank" / "run_0.json").read_text())
    return meta, run


def test_preset_hunter_only(dataset: Path, tmp_path: Path, mock_llm):
    meta, run = _run_preset(dataset, tmp_path, "hunter_only")
    assert meta["preset"]["name"] == "hunter_only"
    assert meta["flags"]["SKEPTIC_ENABLED"] is False and meta["flags"]["EXPLOIT_ENABLED"] is False
    state = run["state"]
    assert state["exploits"] == []
    assert len(state["validated_findings"]) == len(state["hypotheses"]) == 2
    assert {f["status"] for f in state["validated_findings"]} == {"unverified"}
    assert {f["hypothesis_id"] for f in state["validated_findings"]} == {h["id"] for h in state["hypotheses"]}


def test_preset_no_refuter_records_flags(dataset: Path, tmp_path: Path, mock_llm):
    from unittest.mock import patch

    from audit_agent.agents.refuter import RefuterAgent

    with patch.object(RefuterAgent, "_challenge") as challenge:
        meta, run = _run_preset(dataset, tmp_path, "no_refuter")
    challenge.assert_not_called()
    assert meta["flags"]["REFUTER_ENABLED"] is False
    assert meta["preset"]["env"]["REFUTER_ENABLED"] == "false"
    assert {f["status"] for f in run["state"]["validated_findings"]} == {"validated"}


@pytest.mark.parametrize("preset,expect_called", [("full", True), ("no_experts", False)])
def test_preset_no_experts_skips_passes(dataset: Path, tmp_path: Path, mock_llm, preset, expect_called):
    from unittest.mock import AsyncMock, patch

    from audit_agent.agents.hunter import HunterAgent

    names = ["_expert_passes", "_crypto_file_scan", "_cross_contract_check",
             "_aggregation_check", "_boundary_check", "_temporal_check", "_crypto_check"]
    mocks = {n: AsyncMock(return_value=[]) for n in names}
    with patch.multiple(HunterAgent, **mocks):
        _run_preset(dataset, tmp_path, preset)
    for n, m in mocks.items():
        assert m.called is expect_called, n


# ── report (P1-8) ────────────────────────────────────────────────────────────

def test_report_compares_presets_and_lists_limitations(dataset: Path, tmp_path: Path, mock_llm):
    out = tmp_path / "results"
    for preset in ("full", "hunter_only"):
        _invoke("benchmark", "--dataset", str(dataset), "--split", "smoke", "--preset", preset,
                "--out", str(out), "--no-fuzz")
    dirs = sorted(d for d in out.iterdir() if d.is_dir())
    report_path = tmp_path / "RESULTS.md"
    _invoke("benchmark", "report", *map(str, dirs), "--out", str(report_path))
    md = report_path.read_text()

    assert "## Smoke split" in md and "## Test split" not in md
    assert "full · claude-opus-4-7" in md and "hunter_only · claude-opus-4-7" in md
    assert "fixture-vulnerable-bank‡" in md          # contaminated marker
    assert "### Per contest" in md and "Validator confusion" in md
    assert "Recall by severity" in md
    assert "1 run per contest" in md
    assert "1 failed run(s)" in md                    # the missing-source contest
    assert "not labelled" in md                       # unmatched validated finding in `full`
    assert "cost unknown for models without a price" in md
    assert "REFUTER_ENABLED=false" in md              # hunter_only flags recorded
    assert "Settings: " in md and "hunter_top_functions=" in md  # run-shaping knobs visible
    # numbers come from metrics.json: recall_final 50% for one of two GT
    assert "50.0%" in md


def test_report_missing_dir(tmp_path: Path):
    res = CliRunner().invoke(app, ["benchmark", "report", str(tmp_path / "nope"),
                                   "--out", str(tmp_path / "R.md")])
    assert res.exit_code == 1


def test_preset_baseline(dataset: Path, tmp_path: Path, mock_llm):
    meta, run = _run_preset(dataset, tmp_path, "baseline")
    assert meta["preset"]["name"] == "baseline"
    assert "baseline_max_tokens" in meta["settings"]
    state = run["state"]
    assert state["exploits"] == []
    assert state["baseline"]["stop_reason"] == "reported"
    assert len(state["hypotheses"]) == 2
    assert {f["status"] for f in state["validated_findings"]} == {"unverified"}
    assert run["usage"]["tokens"] > 0



def test_labels_files_and_matcher_agreement_cli(dataset: Path, tmp_path: Path, mock_llm):
    import csv

    out = tmp_path / "results"
    _invoke("benchmark", "--dataset", str(dataset), "--split", "smoke",
            "--contest", "fixture-vulnerable-bank", "--out", str(out), "--no-fuzz")
    rd = _only_results_dir(out)
    lf = rd / "labels_fixture-vulnerable-bank_run0.csv"
    assert lf.exists()                                   # written by rescore
    m = json.loads((rd / "metrics.json").read_text())["runs"][0]
    assert m["human_recall_final"] is None and m["human_labels"]["unreviewed"] == 2

    rows = list(csv.DictReader(lf.open()))
    for r in rows:
        r["reviewed"] = "x"
        r["matches_gt"] = "F-01" if "withdraw" in r["functions"] else ""
    with lf.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    res = _invoke("benchmark", "matcher-agreement", str(rd))
    assert "human recall final : 50.0%" in res.output
    assert "TP=1 FP=0 FN=0 TN=3" in res.output
    _invoke("benchmark", "labels", str(rd))                 # refresh keeps the labels
    m = json.loads((rd / "metrics.json").read_text())["runs"][0]
    assert m["human_recall_hypotheses"] == 0.5 and m["human_recall_final"] == 0.5
    assert m["human_labels"]["matcher_agreement"]["precision"] == 1.0
    _invoke("benchmark", "report", str(rd), "--out", str(tmp_path / "R.md"))
    assert "Recall final (human)" in (tmp_path / "R.md").read_text()


def test_report_uses_gt_flags_from_rescore_with_dataset(dataset: Path, tmp_path: Path, mock_llm):
    ds = json.loads(dataset.read_text())
    ds["contests"] = ds["contests"][:1]
    ds["contests"][0]["needs_review"] = True
    dataset.write_text(json.dumps(ds))
    out = tmp_path / "results"
    _invoke("benchmark", "--dataset", str(dataset), "--split", "smoke", "--include-needs-review",
            "--out", str(out), "--no-fuzz")
    rd = _only_results_dir(out)
    report_path = tmp_path / "RESULTS.md"
    _invoke("benchmark", "report", str(rd), "--no-rescore", "--out", str(report_path))
    assert "unreviewed ground truth" in report_path.read_text()

    ds["contests"][0]["needs_review"] = False        # GT reviewed after the run
    dataset.write_text(json.dumps(ds))
    _invoke("benchmark", "rescore", str(rd), "--dataset", str(dataset))
    _invoke("benchmark", "report", str(rd), "--no-rescore", "--out", str(report_path))
    assert "unreviewed ground truth" not in report_path.read_text()   # meta.json snapshot ignored
