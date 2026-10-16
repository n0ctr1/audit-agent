"""Re-run stored ``exploited`` PoCs with call traces (``forge test -vvvv``).

No LLM calls: the PoC code comes from the stored run state. Each run's PoCs execute
in a temporary copy of the contest source (the working copy is not modified),
through the same Docker wrapper and PoC safety guard as the pipeline. Results are
saved to ``<results>/<contest>/run_<k>_retrace/<hypothesis_id>.json`` and read by
``benchmark analysis`` for the strict exploited check (benchmarks/strict_exploit.py).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import tomllib
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from audit_agent.models.schemas import ExploitResult
from benchmarks.dataset import BenchmarkContest, load_dataset
from benchmarks.metrics import exploit_bucket
from benchmarks.runner import iter_run_records, load_meta

VERBOSITY = 4
ForgeRunner = Callable[..., Awaitable[Any]]


def retrace_dir(contest_dir: Path, run_index: int) -> Path:
    return Path(contest_dir) / f"run_{run_index}_retrace"


def load_retraces(contest_dir: Path, run_index: int) -> dict[str, dict[str, Any]]:
    """hypothesis_id → stored re-run record."""
    out = {}
    for p in sorted(retrace_dir(contest_dir, run_index).glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        d["_path"] = str(p)
        out[d.get("hypothesis_id", p.stem)] = d
    return out


def exploited_pocs(state: dict[str, Any]) -> list[ExploitResult]:
    exploits = [ExploitResult.model_validate(e) for e in state.get("exploits") or []]
    return [e for e in exploits if exploit_bucket(e) == "exploited"]


def _test_dir(project: Path) -> str:
    try:
        cfg = tomllib.loads((project / "foundry.toml").read_text(encoding="utf-8"))
        return cfg.get("profile", {}).get("default", {}).get("test", "test")
    except (OSError, ValueError):
        return "test"


async def _run_poc(project: Path, e: ExploitResult, forge: ForgeRunner) -> dict[str, Any]:
    # Same placement as the pipeline's forge runner (agents/exploit.py), incl. fork profile.
    from audit_agent.agents.exploit import _detect_fork_profile, _poc_base_dir

    profile = _detect_fork_profile(str(project))
    rel_dir = (_poc_base_dir(e.poc_code, str(project)) if profile else None) or _test_dir(project)
    name = f"AuditAgentPoC_{e.hypothesis_id.replace('-', '')[:12]}.t.sol"
    poc = project / rel_dir / name
    poc.parent.mkdir(parents=True, exist_ok=True)
    poc.write_text(e.poc_code, encoding="utf-8")
    try:
        res = await forge(str(project), match_path=f"{rel_dir}/{name}", profile=profile,
                          timeout=600 if profile else 300, verbosity=VERBOSITY)
    finally:
        poc.unlink(missing_ok=True)
    return {
        "hypothesis_id": e.hypothesis_id,
        "verbosity": VERBOSITY,
        "profile": profile,
        "passed": bool(getattr(res, "passed", False)),
        "mock": not hasattr(res, "passed"),
        "output": getattr(res, "output", "") or "",
        "error": getattr(res, "error", "") or "",
        "retraced_at": datetime.now(UTC).isoformat(),
    }


async def retrace_results(
    results_dir: Path,
    dataset_override: Optional[str] = None,
    forge: Optional[ForgeRunner] = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Re-run every exploited PoC of a results dir; returns the new records."""
    if forge is None:
        from audit_agent.tools.foundry import run_forge_test as forge
    results_dir = Path(results_dir)
    meta = load_meta(results_dir)
    contests = {c["id"]: BenchmarkContest.model_validate(c) for c in meta["contests"]}
    if dataset_override:
        contests.update({c.id: c for c in load_dataset(dataset_override).contests if c.id in contests})
    done = []
    for path, rec in iter_run_records(results_dir):
        contest = contests.get(rec["contest_id"])
        if rec.get("status") != "ok" or contest is None:
            continue
        k = rec["run_index"]
        out_dir = retrace_dir(path.parent, k)
        todo = [e for e in exploited_pocs(rec.get("state") or {})
                if overwrite or not (out_dir / f"{e.hypothesis_id}.json").exists()]
        if not todo:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="retrace_", ignore_cleanup_errors=True) as tmp:
            project = Path(tmp) / "project"
            shutil.copytree(contest.resolved_source_dir(), project, symlinks=True,
                            ignore=shutil.ignore_patterns(".git"))
            for e in todo:
                r = await _run_poc(project, e, forge)
                r.update(contest_id=contest.id, run_index=k)
                (out_dir / f"{e.hypothesis_id}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
                done.append(r)
    return done
