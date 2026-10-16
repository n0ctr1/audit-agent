"""Benchmark runner: runs the pipeline over dataset contests and stores raw results.

Every run stores the raw final state, so metrics can be recomputed without
re-running the (expensive) pipeline when the matcher or GT changes:

    results/<timestamp>_<preset>_<model>/
        meta.json                  git commit, models per role, *_ENABLED flags,
                                   preset, date, GT snapshot of selected contests
        <contest>/run_<k>.json     raw final state + measured usage/time/errors
        <contest>/run_<k>_reports/ Scribe reports (<finding_id>.md)
        <contest>/run_<k>_logs/    AuditLogger trail
        metrics.json               written by rescore()
        unmatched_findings.csv     unmatched validated findings for manual labels

Runs are sequential; Hunter's own concurrency comes from HUNTER_MAX_CONCURRENCY.
A failing contest run is recorded (status "error") and the benchmark continues.
"""

from __future__ import annotations

import hashlib
import json
import platform as _platform
import re
import subprocess
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from benchmarks.dataset import (  # noqa: F401  (re-exported)
    DATASETS_DIR,
    REPO_ROOT,
    BenchmarkContest,
    BenchmarkDataset,
    GroundTruthFindingModel,
    dataset_path,
    load_dataset,
    select_contests,
)
from benchmarks.labels import (
    UNMATCHED_CSV,
    UnmatchedRow,
    labels_for_run,
    read_labels,
    write_unmatched,
)
from benchmarks.matching import TITLE_JACCARD_MIN, W_CLASS, W_FUNCTION, W_TITLE
from benchmarks.metrics import (
    AGGREGATED_FIELDS,
    LABEL_VALUES,
    ContestRunMetrics,
    Stat,
    aggregate_runs,
    compute_contest_metrics,
    pooled_recall,
    summarize,
)
from benchmarks.presets import applied_preset

AuditRunner = Callable[[BenchmarkContest, str], Awaitable[dict[str, Any]]]

RAW_STATE_KEYS = ("hypotheses", "exploits", "validated_findings", "chains",
                  "tokens_used", "cost_usd", "errors", "degraded_tools", "baseline")
META_FILE = "meta.json"
METRICS_FILE = "metrics.json"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner (injectable for tests)
# ─────────────────────────────────────────────────────────────────────────────

def make_pipeline_runner(*, fuzz: bool = True, log_root: Optional[Path] = None) -> AuditRunner:
    """Full LangGraph pipeline, configured like ``audit-agent audit``."""

    async def run(contest: BenchmarkContest, audit_id: str) -> dict[str, Any]:
        from audit_agent.config import get_settings
        from audit_agent.graph.workflow import build_graph
        from audit_agent.observability.audit_logger import AuditLogger, set_audit_logger

        settings = get_settings()
        target = contest.resolved_source_dir()
        if not target.is_dir():
            raise FileNotFoundError(f"source_dir not found: {target}")
        scope = contest.resolved_scope_file()
        if scope is not None and not scope.exists():
            raise FileNotFoundError(f"scope_file not found: {scope}")
        docs = contest.resolved_docs_dir()

        logger = AuditLogger(log_root, audit_id) if (log_root and settings.audit_log_enabled) else None
        set_audit_logger(logger)
        try:
            graph = build_graph(checkpointer=None)
            initial_state = {
                "audit_id": audit_id,
                "mode": "audit",
                "target_dir": str(target),
                "docs_dir": str(docs) if docs else None,
                "known_issues": contest.known_issues or None,
                "platform": "generic",
                "budget_tokens": settings.default_budget_tokens,
                "budget_usd": settings.default_budget_usd,
                "fuzz_enabled": fuzz,
                "scope_file": str(scope) if scope else None,
                "scope_in_scope": [],
                "scope_context_files": [],
                "scope_source": "",
                "threat_model": None,
                "invariants": [],
                "static_findings": [],
                "hypotheses": [],
                "chains": [],
                "exploits": [],
                "validated_findings": [],
                "tokens_used": 0,
                "cost_usd": 0.0,
                "errors": [],
                "run_checkpoint_id": "",
                "degraded_tools": [],
            }
            config = {"configurable": {"thread_id": audit_id}}
            final = await graph.ainvoke(initial_state, config=config)
            if logger:
                logger.write_summary(dict(final))
            return final
        finally:
            set_audit_logger(None)

    return run


def make_baseline_runner(*, log_root: Optional[Path] = None) -> AuditRunner:
    """Single-agent baseline (agents/baseline.py) with the pipeline's scope and docs.

    Every hypothesis is reported as an ``unverified`` finding, as in ``hunter_only``.
    """

    async def run(contest: BenchmarkContest, audit_id: str) -> dict[str, Any]:
        from audit_agent.agents.baseline import BaselineAgent
        from audit_agent.agents.scout import _collect_documentation
        from audit_agent.config import get_settings
        from audit_agent.graph.nodes import _hypotheses_as_unverified
        from audit_agent.observability.audit_logger import AuditLogger, set_audit_logger
        from audit_agent.scope import resolve_scope

        target = contest.resolved_source_dir()
        if not target.is_dir():
            raise FileNotFoundError(f"source_dir not found: {target}")
        scope = resolve_scope(target, contest.resolved_scope_file())
        in_scope = [str(Path(p).resolve().relative_to(target.resolve())) for p in scope.in_scope]
        docs = _collect_documentation(target)
        docs_dir = contest.resolved_docs_dir()
        if docs_dir and docs_dir.is_dir():
            docs += "\n\n" + _collect_documentation(docs_dir)

        logger = AuditLogger(log_root, audit_id) if (log_root and get_settings().audit_log_enabled) else None
        set_audit_logger(logger)
        try:
            agent = BaselineAgent()
            hyps = await agent.run(str(target), in_scope, audit_id, documentation=docs)
        finally:
            set_audit_logger(None)
        state: dict[str, Any] = {"audit_id": audit_id, "hypotheses": hyps, "exploits": [],
                                 "chains": [], "errors": [], "degraded_tools": [],
                                 "tokens_used": agent.tokens_used, "cost_usd": 0.0}
        state["validated_findings"] = _hypotheses_as_unverified(state)  # type: ignore[arg-type]
        state["baseline"] = {"steps": agent.steps, "stop_reason": agent.stop_reason}
        return state

    return run


# ─────────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────────

def _git_info() -> dict[str, Any]:
    def git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                 text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def settings_snapshot() -> dict[str, Any]:
    """Models per role, every *_ENABLED flag and run-shaping knobs. No secrets, no URLs."""
    from audit_agent.config import get_settings

    s = get_settings()
    names = type(s).model_fields
    models = {k[: -len("_model")]: getattr(s, k) for k in names if k.endswith("_model")}
    flags = {names[k].alias or k.upper(): getattr(s, k) for k in names if k.endswith("_enabled")}
    knobs = ("llm_provider", "llm_disable_thinking", "llm_reasoning_max_tokens", "llm_min_output_tokens", "tool_mode", "hunter_max_concurrency",
             "hunter_top_functions", "hunter_max_hypotheses", "hunter_source_budget_chars",
             "hunter_context_budget_chars", "hunter_navigator_leads", "hunter_navigator_max_steps",
             "chainer_max_chains", "hunter_threat_boost", "triage_batch_size", "triage_weight",
             "default_budget_tokens", "default_budget_usd", "baseline_max_tokens",
             "baseline_max_seconds", "baseline_max_steps", "baseline_command_timeout")
    return {"models": models, "flags": flags, "settings": {k: getattr(s, k) for k in knobs}}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", s).strip("-") or "unknown"


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, list):
        return [_dump(o) for o in obj]
    return obj


def raw_state(final_state: dict[str, Any]) -> dict[str, Any]:
    return {k: _dump(final_state.get(k)) for k in RAW_STATE_KEYS if k in final_state}


# ─────────────────────────────────────────────────────────────────────────────
# Running
# ─────────────────────────────────────────────────────────────────────────────

async def run_suite(
    contests: Sequence[BenchmarkContest],
    *,
    dataset_ref: str,
    runs: int = 1,
    preset: str = "full",
    out_dir: Path = Path("results"),
    audit_runner: Optional[AuditRunner] = None,
    fuzz: bool = True,
    selection: Optional[dict[str, Any]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """Run every contest ``runs`` times (sequentially), store raw results, then rescore."""
    from audit_agent.config import get_settings
    from audit_agent.llm.factory import get_provider, set_provider
    from audit_agent.llm.usage import UsageTrackingProvider

    say = on_progress or (lambda _msg: None)
    if runs < 1:
        raise ValueError("runs must be >= 1")

    with applied_preset(preset) as preset_overrides:
        settings = get_settings()
        started = datetime.now(UTC)
        name = f"{started:%Y%m%dT%H%M%SZ}_{preset}_{_slug(settings.hunter_model)}"
        results_dir = Path(out_dir) / name
        n = 1
        while results_dir.exists():
            n += 1
            results_dir = Path(out_dir) / f"{name}-{n}"
        results_dir.mkdir(parents=True)

        ds_file = dataset_path(dataset_ref)
        meta = {
            "created_at": started.isoformat(),
            "dataset": {"ref": dataset_ref, "path": str(ds_file),
                        "sha256": hashlib.sha256(ds_file.read_bytes()).hexdigest()
                        if ds_file.exists() else None},
            "selection": selection or {},
            "runs": runs,
            "preset": {"name": preset, "env": preset_overrides},
            "fuzz": fuzz,
            "git": _git_info(),
            "python": _platform.python_version(),
            **settings_snapshot(),
            "contests": [c.model_dump(mode="json") for c in contests],
        }
        (results_dir / META_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

        for contest in contests:
            cdir = results_dir / _slug(contest.id)
            cdir.mkdir(exist_ok=True)
            for k in range(runs):
                logs = cdir / f"run_{k}_logs"
                runner = audit_runner or (make_baseline_runner(log_root=logs) if preset == "baseline"
                                          else make_pipeline_runner(fuzz=fuzz, log_root=logs))
                say(f"{contest.id} run {k + 1}/{runs}")
                record = await _run_once(contest, k, runner, get_provider, set_provider,
                                         UsageTrackingProvider)
                _write_run(cdir, k, record)
                status = record["status"]
                say(f"{contest.id} run {k + 1}/{runs}: {status}"
                    + (f" ({record['error'].splitlines()[-1][:200]})" if record.get("error") else ""))

    rescore(results_dir)
    return results_dir


async def _run_once(contest, k, runner, get_provider, set_provider, UsageTrackingProvider) -> dict:
    audit_id = str(uuid.uuid4())
    previous = None
    tracker = None
    record: dict[str, Any] = {"contest_id": contest.id, "run_index": k, "audit_id": audit_id,
                              "started_at": datetime.now(UTC).isoformat()}
    t0 = time.monotonic()
    try:
        previous = get_provider()
        tracker = UsageTrackingProvider(previous)
        set_provider(tracker)
        final = await runner(contest, audit_id)
        record.update(status="ok", error=None, state=raw_state(final or {}),
                      reports=dict((final or {}).get("reports") or {}))
    except Exception as exc:  # a failing contest must not stop the benchmark
        record.update(status="error",
                      error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-4000:]}",
                      state={}, reports={})
    finally:
        if previous is not None:
            set_provider(previous)
    record["elapsed_s"] = round(time.monotonic() - t0, 3)
    record["usage"] = tracker.usage.to_dict() if tracker else None
    return record


def _write_run(cdir: Path, k: int, record: dict[str, Any]) -> None:
    reports = record.pop("reports", {}) or {}
    if reports:
        rdir = cdir / f"run_{k}_reports"
        rdir.mkdir(exist_ok=True)
        for fid, md in reports.items():
            (rdir / f"{_slug(fid)}.md").write_text(md, encoding="utf-8")
    (cdir / f"run_{k}.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Rescoring
# ─────────────────────────────────────────────────────────────────────────────

def load_meta(results_dir: Path) -> dict[str, Any]:
    p = Path(results_dir) / META_FILE
    if not p.exists():
        raise FileNotFoundError(f"not a benchmark results dir (no {META_FILE}): {results_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def iter_run_records(results_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    out = []
    for p in sorted(Path(results_dir).glob("*/run_*.json")):
        if re.fullmatch(r"run_\d+\.json", p.name):
            out.append((p, json.loads(p.read_text(encoding="utf-8"))))
    out.sort(key=lambda t: (t[1]["contest_id"], t[1]["run_index"]))
    return out


def run_metrics(record: dict[str, Any], contest: BenchmarkContest,
                labels: Optional[dict[str, str]] = None) -> ContestRunMetrics:
    if record.get("status") != "ok":
        return ContestRunMetrics(contest_id=contest.id, run_index=record["run_index"],
                                 status="error", error=(record.get("error") or "")[:500],
                                 num_ground_truth=len(contest.official_findings),
                                 time_s=record.get("elapsed_s"))
    # Measured usage only; AuditState.tokens_used / cost_usd are never updated by agents.
    usage = record.get("usage") or {}
    return compute_contest_metrics(
        record.get("state") or {},
        contest.ground_truth(),
        contest_id=contest.id,
        run_index=record["run_index"],
        labels=labels,
        cost_usd=usage.get("cost_usd"),
        tokens=usage.get("tokens"),
        time_s=record.get("elapsed_s"),
    )


def _split_summary(runs: list[ContestRunMetrics], n_contests: int) -> dict[str, Any]:
    by_k: dict[int, list[ContestRunMetrics]] = {}
    for r in runs:
        by_k.setdefault(r.run_index, []).append(r)
    per_k = []
    for k in sorted(by_k):
        ok = [r for r in by_k[k] if r.status == "ok"]
        macro = {f: summarize(getattr(r, f) for r in ok).mean for f in AGGREGATED_FIELDS}
        macro["precision_adjusted"] = summarize(r.precision_adjusted.value for r in ok).mean
        pooled = {f: pooled_recall(by_k[k], f)
                  for f in ("recall_hypotheses_at_30", "recall_final", "recall_validated")}
        totals = {"cost_usd": sum(r.cost_usd for r in ok if r.cost_usd is not None)
                  if ok and all(r.cost_usd is not None for r in ok) else None,
                  "tokens": sum(r.tokens or 0 for r in ok),
                  "time_s": sum(r.time_s or 0 for r in ok)}
        per_k.append({"run_index": k, "num_ok": len(ok), "num_error": len(by_k[k]) - len(ok),
                      "macro": macro, "pooled": pooled, "totals": totals})
    keys = list(AGGREGATED_FIELDS) + ["precision_adjusted"]
    return {
        "num_contests": n_contests,
        "num_runs_ok": sum(1 for r in runs if r.status == "ok"),
        "num_runs_error": sum(1 for r in runs if r.status != "ok"),
        "per_run_index": per_k,
        # mean / min–max over run indices of the dataset-level (macro) value
        "summary": {f: summarize(p["macro"][f] for p in per_k).model_dump() for f in keys},
        "pooled_summary": {f: summarize(p["pooled"][f] for p in per_k).model_dump()
                           for f in ("recall_hypotheses_at_30", "recall_final", "recall_validated")},
    }


def _human_precision(state: dict[str, Any], human_unmatched: list[str],
                     labels: dict[str, str]) -> dict[str, Any]:
    """Precision of validated findings by the manual labels: (findings whose hypothesis a
    human matched to GT + tp_not_in_gt) / (validated − duplicate). None while any finding
    outside GT (by the human labels) has no label in unmatched_findings.csv."""
    n_val = sum(1 for f in state.get("validated_findings") or [] if f.get("status") == "validated")
    lab = Counter(labels.get(fid, "") for fid in human_unmatched)
    unlabelled = sum(1 for fid in human_unmatched if labels.get(fid) not in LABEL_VALUES)
    den = n_val - lab["duplicate"]
    value = (n_val - len(human_unmatched) + lab["tp_not_in_gt"]) / den if den and not unlabelled else None
    return {"value": value, "validated": n_val, "matched_gt": n_val - len(human_unmatched),
            "labelled": {k: lab.get(k, 0) for k in LABEL_VALUES}, "unlabelled": unlabelled}


def _apply_human_labels(results_dir: Path, contest: BenchmarkContest, k: int,
                        state: dict[str, Any], m: ContestRunMetrics) -> list[str]:
    """Create/refresh labels_<contest>_run<k>.csv and fold the human labels into ``m``.

    Returns the ids of validated findings outside GT according to the manual labels
    (empty until every row is reviewed)."""
    from benchmarks.human_labels import (
        evaluate_labels,
        label_rows,
        labels_filename,
        read_rows,
        write_labels,
    )

    path = results_dir / labels_filename(contest.id, k)
    write_labels(path, label_rows(state))
    res = evaluate_labels(read_rows(path), state, contest.ground_truth())
    m.human_recall_hypotheses = res.human_recall_hypotheses
    m.human_recall_final = res.human_recall_final
    m.human_labels = {
        "file": path.name, "rows": res.rows, "unreviewed": res.unreviewed,
        "unknown_gt_ids": res.unknown_gt_ids, "complete": res.complete,
        "covered_hypotheses": res.covered_hypotheses, "covered_final": res.covered_final,
        "matcher_agreement": {"tp": res.tp, "fp": res.fp, "fn": res.fn, "tn": res.tn,
                              "precision": res.matcher_precision, "recall": res.matcher_recall,
                              "accuracy": res.matcher_accuracy, "cohen_kappa": res.cohen_kappa},
    }
    if not res.complete:
        return []
    matched = set(res.matched_hypothesis_ids)
    return [f["id"] for f in state.get("validated_findings") or []
            if f.get("status") == "validated" and f.get("hypothesis_id") not in matched]


def rescore(results_dir: Path, dataset_override: Optional[str] = None) -> dict[str, Any]:
    """Recompute metrics.json (and unmatched_findings.csv) from stored run files.

    GT comes from the snapshot in meta.json unless ``dataset_override`` names a
    dataset whose (current) contests should be used instead.
    """
    results_dir = Path(results_dir)
    meta = load_meta(results_dir)
    contests = {c["id"]: BenchmarkContest.model_validate(c) for c in meta["contests"]}
    gt_origin = "meta.json snapshot"
    if dataset_override:
        current = {c.id: c for c in load_dataset(dataset_override).contests}
        contests.update({cid: current[cid] for cid in contests if cid in current})
        gt_origin = str(dataset_path(dataset_override))

    csv_path = results_dir / UNMATCHED_CSV
    labels = read_labels(csv_path)

    all_runs: list[ContestRunMetrics] = []
    rows: list[UnmatchedRow] = []
    for path, record in iter_run_records(results_dir):
        contest = contests.get(record["contest_id"])
        if contest is None:
            continue
        k = record["run_index"]
        m = run_metrics(record, contest, labels_for_run(labels, contest.id, k))
        all_runs.append(m)
        if m.status != "ok":
            continue
        state = record.get("state") or {}
        human_unmatched = _apply_human_labels(results_dir, contest, k, state, m)
        hyps = {h["id"]: h for h in state.get("hypotheses") or []}
        findings = {f["id"]: f for f in state.get("validated_findings") or []}
        by_matcher = set(m.unmatched_validated_ids)
        for fid in [*m.unmatched_validated_ids, *(i for i in human_unmatched if i not in by_matcher)]:
            f = findings.get(fid, {})
            h = hyps.get(f.get("hypothesis_id") or "", {})
            report = path.parent / f"run_{k}_reports" / f"{_slug(fid)}.md"
            rows.append(UnmatchedRow(
                contest=contest.id, run=k, finding_id=fid,
                hypothesis_id=f.get("hypothesis_id") or "",
                title=h.get("title") or f.get("title", ""),
                functions="; ".join(h.get("affected_functions") or f.get("affected_functions") or []),
                severity=f.get("severity", ""),
                report_path=str(report.relative_to(results_dir)) if report.exists() else "",
                matcher_status="unmatched" if fid in by_matcher else "matched",
                unmatched_by="+".join(x for x, ok in (("matcher", fid in by_matcher),
                                                      ("human", fid in human_unmatched)) if ok),
            ))
        if m.human_labels.get("complete"):
            m.human_labels["precision_validated"] = _human_precision(
                state, human_unmatched, labels_for_run(labels, contest.id, k))
    write_unmatched(csv_path, rows)

    contests_out: dict[str, Any] = {}
    by_split: dict[str, list[ContestRunMetrics]] = {}
    for cid, contest in sorted(contests.items()):
        runs = [r for r in all_runs if r.contest_id == cid]
        contests_out[cid] = {
            "split": contest.split,
            "needs_review": contest.needs_review,
            "contaminated": contest.contaminated,
            "num_ground_truth": len(contest.official_findings),
            "num_runs": len(runs),
            "aggregate": {k: v.model_dump() for k, v in aggregate_runs(runs).items()},
        }
        by_split.setdefault(contest.split, []).extend(runs)

    metrics = {
        "generated_at": datetime.now(UTC).isoformat(),
        "gt_origin": gt_origin,
        "matcher": {"rule": "function_match AND (class_compatible OR title_jaccard >= min)",
                    "title_jaccard_min": TITLE_JACCARD_MIN,
                    "rank_weights": {"function": W_FUNCTION, "class": W_CLASS, "title": W_TITLE}},
        "unlabelled_unmatched_findings": sum(r.precision_adjusted.unlabelled for r in all_runs),
        "unreviewed_label_rows": sum((r.human_labels or {}).get("unreviewed", 0) for r in all_runs),
        "runs": [r.model_dump(mode="json") for r in all_runs],
        "contests": contests_out,
        "splits": {s: _split_summary(rs, sum(1 for c in contests.values() if c.split == s))
                   for s, rs in sorted(by_split.items())},
    }
    (results_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# In-memory convenience API
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    final_state: dict[str, Any],
    contest: BenchmarkContest,
    elapsed_seconds: Optional[float] = None,
    *,
    run_index: int = 0,
    labels: Optional[dict[str, str]] = None,
) -> ContestRunMetrics:
    """Compute all metrics for one contest run from the final pipeline state."""
    return compute_contest_metrics(
        final_state,
        contest.ground_truth(),
        contest_id=contest.id,
        run_index=run_index,
        labels=labels,
        time_s=elapsed_seconds,
    )


class BenchmarkReport(BaseModel):
    dataset: str
    per_contest: list[ContestRunMetrics] = []
    aggregate: dict[str, Stat] = {}

    @property
    def num_contests(self) -> int:
        return len(self.per_contest)


async def run_benchmark(
    dataset_name: str,
    limit: Optional[int] = None,
    audit_runner: Optional[AuditRunner] = None,
    split: str = "test",
) -> BenchmarkReport:
    """Run selected contests once each and return metrics without writing files.

    By default only reviewed ``split: test`` contests are run. Prefer
    ``run_suite`` (used by the CLI), which persists raw states for rescoring.
    """
    runner = audit_runner or make_pipeline_runner()
    dataset = load_dataset(dataset_name)
    contests = select_contests(dataset, split=split)
    contests = contests[:limit] if limit else contests

    per_contest: list[ContestRunMetrics] = []
    for contest in contests:
        t0 = time.monotonic()
        final_state = await runner(contest, str(uuid.uuid4()))
        per_contest.append(compute_metrics(final_state, contest, time.monotonic() - t0))

    return BenchmarkReport(dataset=dataset_name, per_contest=per_contest,
                           aggregate=aggregate_runs(per_contest))
