"""Pilot analysis from stored benchmark runs (no LLM calls).

Four views, each with its sample size next to every number:

1. Path of each GT finding through the stages (hypothesis → Exploit → Skeptic →
   Refuter) — needs complete manual labels (labels_<contest>_run<k>.csv); until
   then the view says "pending labels" instead of falling back to the matcher.
2. Skeptic → final status matrix (after Refuter), with the share of findings that
   describe a GT finding according to the manual labels (pending until labelled).
3. Causes of PoCs that never compiled — an AUTOMATIC classification of the last
   solc error per PoC, plus a seeded random sample of PoCs for manual checking.
4. Strict check of ``exploited`` PoCs (benchmarks/strict_exploit.py): exploited
   (raw) / exploited (strict) and why PoCs were downgraded. Uses the -vvvv traces
   saved by ``benchmark retrace``.
5. Before/after comparison of two runs on the same contest (e.g. the harness fix).
6. Near misses — a MANUAL list (near_misses.csv) of hypotheses that point at the
   right code but name another mechanism. Shown for context, never counted in recall.
"""

from __future__ import annotations

import collections
import csv
import glob
import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from audit_agent.exploit.harness_context import _parents, _test_contracts
from audit_agent.models.schemas import ExploitResult
from benchmarks.dataset import BenchmarkContest, load_dataset
from benchmarks.human_labels import labels_filename, parse_gt_ids, read_rows
from benchmarks.labels import UNMATCHED_CSV, labels_for_run, read_labels
from benchmarks.metrics import FINAL_STATUSES, exploit_bucket
from benchmarks.retrace import exploited_pocs, load_retraces
from benchmarks.runner import iter_run_records, load_meta
from benchmarks.strict_exploit import (
    REASONS,
    StrictResult,
    passing_asserts_traced,
    scope_contract_names,
    strict_check,
)

PENDING = "pending labels"
DEFAULT_SEED = 2026
SAMPLE_SIZE = 5

# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RunData:
    results_dir: Path
    contest: BenchmarkContest
    run_index: int
    preset: str
    meta: dict[str, Any]
    state: dict[str, Any]
    label_rows: Optional[list[dict[str, str]]]
    forge_logs: list[dict[str, Any]] = field(default_factory=list)
    retraces: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.results_dir.name} · {self.contest.id} run {self.run_index}"

    @property
    def gt_ids(self) -> list[str]:
        return [g.id for g in self.contest.ground_truth()]

    @property
    def labels_complete(self) -> bool:
        if not self.label_rows:
            return False
        known = set(self.gt_ids)
        return all((r.get("reviewed") or "").strip() for r in self.label_rows) and all(
            parse_gt_ids(r.get("matches_gt", "")) <= known for r in self.label_rows)

    def labels_progress(self) -> str:
        if self.label_rows is None:
            return "labels file missing"
        done = sum(1 for r in self.label_rows if (r.get("reviewed") or "").strip())
        return f"{done}/{len(self.label_rows)} rows reviewed"

    def matches(self) -> dict[str, set[str]]:
        """hypothesis_id → GT ids (manual labels)."""
        return {r["hypothesis_id"]: parse_gt_ids(r.get("matches_gt", "")) for r in self.label_rows or []}

    @property
    def skeptic_enabled(self) -> bool:
        return bool((self.meta.get("flags") or {}).get("SKEPTIC_ENABLED", True))


def _forge_logs(contest_dir: Path, k: int) -> list[dict[str, Any]]:
    out = []
    for p in sorted(glob.glob(str(contest_dir / f"run_{k}_logs" / "**" / "tools" / "*.json"), recursive=True)):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("tool_name") == "forge":
            d["_path"] = p
            out.append(d)
    return out


def load_runs(results_dirs: Sequence[Path], dataset_override: Optional[str] = None) -> list[RunData]:
    runs = []
    for d in results_dirs:
        d = Path(d)
        meta = load_meta(d)
        contests = {c["id"]: BenchmarkContest.model_validate(c) for c in meta["contests"]}
        if dataset_override:
            contests.update({c.id: c for c in load_dataset(dataset_override).contests if c.id in contests})
        for path, rec in iter_run_records(d):
            if rec.get("status") != "ok" or rec["contest_id"] not in contests:
                continue
            k = rec["run_index"]
            lf = d / labels_filename(rec["contest_id"], k)
            runs.append(RunData(
                results_dir=d, contest=contests[rec["contest_id"]], run_index=k,
                preset=(meta.get("preset") or {}).get("name", "?"), meta=meta,
                state=rec.get("state") or {},
                label_rows=read_rows(lf) if lf.exists() else None,
                forge_logs=_forge_logs(path.parent, k),
                retraces=load_retraces(path.parent, k),
            ))
    return runs


def _n(k: int, n: int, unit: str) -> str:
    return f"{k}/{n} {unit}" if n else f"0 {unit}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return (["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(r) + " |" for r in rows])


# ─────────────────────────────────────────────────────────────────────────────
# 1. GT path through the stages
# ─────────────────────────────────────────────────────────────────────────────

LOSS_ORDER = ("not found (no matching hypothesis)", "not sent to Exploit/Skeptic (no finding)",
              "rejected by Skeptic", "rejected by Refuter", "reported as unverified", "reported as validated")


def gt_paths(run: RunData) -> Optional[list[dict[str, Any]]]:
    """Per GT finding: matching hypotheses and where the pipeline lost it. None = pending labels."""
    if not run.labels_complete:
        return None
    matches = run.matches()
    top30 = {r["hypothesis_id"] for r in run.label_rows or [] if r.get("in_top30") == "yes"}
    findings = {f.get("hypothesis_id"): f for f in run.state.get("validated_findings") or []}
    exploits = {e["hypothesis_id"]: ExploitResult.model_validate(e) for e in run.state.get("exploits") or []}
    out = []
    for g in run.contest.ground_truth():
        hyps = [h for h, ids in matches.items() if g.id in ids]
        fs = [findings[h] for h in hyps if h in findings]
        final = {f["status"] for f in fs}
        skeptic = {f.get("skeptic_status") or f["status"] for f in fs}
        if not hyps:
            stage = LOSS_ORDER[0]
        elif "validated" in final:
            stage = LOSS_ORDER[5]
        elif final & set(FINAL_STATUSES):
            stage = LOSS_ORDER[4]
        elif not fs:
            stage = LOSS_ORDER[1]
        elif skeptic & set(FINAL_STATUSES):
            stage = LOSS_ORDER[3]
        else:
            stage = LOSS_ORDER[2]
        out.append({
            "gt_id": g.id, "severity": g.severity, "hypotheses": len(hyps),
            "in_top30": sum(1 for h in hyps if h in top30),
            "exploit": sorted(exploit_bucket(exploits[h]) for h in hyps if h in exploits),
            "skeptic": sorted(skeptic), "final": sorted(final), "stage": stage,
        })
    return out


def section_gt_paths(runs: Sequence[RunData]) -> list[str]:
    lines = ["## 1. Where GT findings are lost", "",
             "Matching hypotheses come from the manual labels only (no matcher fallback).", ""]
    for run in runs:
        n_gt = len(run.gt_ids)
        lines += [f"### {run.label} (preset {run.preset}, n = {n_gt} GT)", ""]
        paths = gt_paths(run)
        if paths is None:
            lines += [f"*{PENDING}* — {run.labels_progress()}.", ""]
            continue
        lines += _table(["GT", "Sev", "Matching hyps", "In top-30", "Exploit", "Skeptic", "Final", "Lost at"],
                        [[p["gt_id"], p["severity"], str(p["hypotheses"]), str(p["in_top30"]),
                          ", ".join(p["exploit"]) or "–", ", ".join(p["skeptic"]) or "–",
                          ", ".join(p["final"]) or "–", p["stage"]] for p in paths])
        counts = collections.Counter(p["stage"] for p in paths)
        lines += ["", "Summary: " + "; ".join(f"{s}: {_n(counts[s], n_gt, 'GT')}"
                                             for s in LOSS_ORDER if counts[s]), ""]
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# 2. Skeptic → Refuter
# ─────────────────────────────────────────────────────────────────────────────

STATUSES = ("validated", "unverified", "false_positive")


def validator_matrix(run: RunData) -> dict[str, Any]:
    fs = run.state.get("validated_findings") or []
    matches = run.matches() if run.labels_complete else None
    cells: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for f in fs:
        cells[(f.get("skeptic_status") or f["status"], f["status"])].append(f)
    out = {"n": len(fs), "cells": {}, "labels": matches is not None}
    for (sk, fin), items in cells.items():
        gt = None if matches is None else sum(1 for f in items if matches.get(f.get("hypothesis_id"), set()))
        out["cells"][(sk, fin)] = {"n": len(items), "gt_match": gt}
    return out


def section_validators(runs: Sequence[RunData]) -> list[str]:
    lines = ["## 2. Skeptic → Refuter", "",
             "Rows: status after Skeptic; columns: final status after Refuter. "
             "\"GT\" = findings whose hypothesis a human labelled as matching a GT finding.", ""]
    for run in runs:
        if not run.skeptic_enabled:
            lines += [f"### {run.label}", "", f"Not applicable (preset {run.preset}: no Skeptic/Refuter).", ""]
            continue
        m = validator_matrix(run)
        lines += [f"### {run.label} (n = {m['n']} findings)", ""]
        rows = []
        for sk in STATUSES:
            row = [sk]
            for fin in STATUSES:
                c = m["cells"].get((sk, fin))
                if not c:
                    row.append("0")
                    continue
                gt = PENDING if c["gt_match"] is None else f"GT {c['gt_match']}/{c['n']}"
                row.append(f"{c['n']} ({gt})")
            rows.append(row)
        lines += _table(["Skeptic \\ Final"] + list(STATUSES), rows)
        sent = sum(c["n"] for (sk, _f), c in m["cells"].items() if sk in FINAL_STATUSES)
        cut = sum(c["n"] for (sk, fin), c in m["cells"].items()
                  if sk in FINAL_STATUSES and fin == "false_positive")
        cut_gt = [c["gt_match"] for (sk, fin), c in m["cells"].items()
                  if sk in FINAL_STATUSES and fin == "false_positive"]
        gt_txt = PENDING if any(x is None for x in cut_gt) else f"{sum(cut_gt)}/{cut} of them match GT"
        lines += ["", f"Refuter turned {_n(cut, sent, 'findings it reviewed')} into false positives ({gt_txt})."]
        unl = labels_for_run(read_labels(run.results_dir / UNMATCHED_CSV), run.contest.id, run.run_index)
        unmatched_final = [f for f in run.state.get("validated_findings") or []
                           if f["status"] == "validated"
                           and not (m["labels"] and run.matches().get(f.get("hypothesis_id")))]
        if m["labels"] and unmatched_final:
            lab = collections.Counter(unl.get(f["id"], "unlabelled") for f in unmatched_final)
            lines.append(f"Validated findings outside GT (n = {len(unmatched_final)}): "
                         + ", ".join(f"{k} {v}" for k, v in sorted(lab.items())) + ".")
        lines.append("")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# 3. PoCs that never compiled (automatic classification)
# ─────────────────────────────────────────────────────────────────────────────

_BLOCK = re.compile(r"Error \((\d+)\): ([^\n]*)\n\s*--> [^\n]*\n[^\n]*\n\s*\d+ \|([^\n]*)\n\s*\|(\s*)(\^+)")
_STATE_NAMES = re.compile(r"^(vault|user|users|operator|governor|admin|owner|susdm|usdm|usdc|accountant|config|"
                          r"redeemEscrow|yieldEscrow|insuranceFund|harness|_\w+)$")

CAUSES = (
    "inherits a non-Test contract (no vm/assert*, no harness state)",
    "uses state/helpers of a project test it does not inherit",
    "invented or misused API (member not found, library used as contract, wrong overload)",
    "missing import / file not found",
    "duplicate declaration",
    "syntax error",
    "other undeclared identifier",
    "other",
)


def classify_compile_error(log: dict[str, Any]) -> str:
    poc = log.get("poc_code") or ""
    txt = (log.get("stdout") or "") + "\n" + (log.get("stderr") or "")
    codes: collections.Counter[str] = collections.Counter()
    names = set()
    for code, _msg, line, sp, car in _BLOCK.findall(txt):
        codes[code] += 1
        s = max(len(sp) - 1, 0)
        names.add(line[s:s + len(car) + 1].strip(" (.;,"))
    m = re.search(r"contract\s+AuditAgentPoC\s+is\s+([^{]+)\{", poc)
    parents = _parents(m.group(1)) if m else []
    if codes.get("6275") or re.search(r'Source "[^"]+" not found', txt):
        return CAUSES[3]
    if codes.get("7576") and "Test" not in parents:
        return CAUSES[0]
    if codes.get("7576") and any(_STATE_NAMES.match(n) for n in names):
        return CAUSES[1]
    if codes.get("9582") or codes.get("6675") or codes.get("1130") or codes.get("7920"):
        return CAUSES[2]
    if codes.get("2333"):
        return CAUSES[4]
    if codes.get("8936") or "ParserError" in txt:
        return CAUSES[5]
    if codes.get("7576"):
        return CAUSES[6]
    return CAUSES[7]


@dataclass
class CompileStats:
    pocs: int
    compile_failed: list[dict[str, Any]]          # final failing attempt per PoC
    by_cause_poc: collections.Counter
    attempts: int
    by_cause_attempt: collections.Counter
    buckets: collections.Counter


def compile_stats(run: RunData) -> CompileStats:
    exploits = [ExploitResult.model_validate(e) for e in run.state.get("exploits") or []]
    buckets = collections.Counter(exploit_bucket(e) for e in exploits)
    failing = [lg for lg in run.forge_logs if lg.get("status") == "compile_error"]
    last: dict[str, dict] = {}
    for lg in sorted(failing, key=lambda x: int(x.get("attempt_number") or 0)):
        last[lg.get("hypothesis_id", "")] = lg
    final = [last[e.hypothesis_id] for e in exploits
             if exploit_bucket(e) == "compile_error" and e.hypothesis_id in last]
    return CompileStats(
        pocs=len(exploits), compile_failed=final,
        by_cause_poc=collections.Counter(classify_compile_error(lg) for lg in final),
        attempts=len(failing),
        by_cause_attempt=collections.Counter(classify_compile_error(lg) for lg in failing),
        buckets=buckets,
    )


def section_compile(runs: Sequence[RunData], seed: int) -> list[str]:
    lines = ["## 3. PoCs that never compiled", "",
             "**Automatic classification** of the last solc error of each PoC (regex on error codes "
             "7576/9582/1130/2333/8936/6275 and the PoC's inheritance). Not manually verified — "
             "see the random sample below.", ""]
    sample_pool: list[tuple[RunData, dict]] = []
    for run in runs:
        st = compile_stats(run)
        if not st.pocs:
            continue
        n_fail = len(st.compile_failed)
        lines += [f"### {run.label}", "",
                  "Exploit outcomes: " + ", ".join(f"{k} {_n(v, st.pocs, 'PoCs')}" for k, v in
                                                   sorted(st.buckets.items())) + ".", ""]
        if not n_fail:
            lines += ["No PoC ended in a compile error.", ""]
            continue
        lines += _table(["Cause (automatic)", f"PoCs (n = {n_fail})", f"All failing attempts (n = {st.attempts})"],
                        [[c, str(st.by_cause_poc[c]), str(st.by_cause_attempt[c])]
                         for c in CAUSES if st.by_cause_poc[c] or st.by_cause_attempt[c]])
        lines.append("")
        sample_pool += [(run, lg) for lg in st.compile_failed]
    if sample_pool:
        rng = random.Random(seed)
        picked = rng.sample(sample_pool, min(SAMPLE_SIZE, len(sample_pool)))
        lines += [f"### Manual check sample ({len(picked)} of {len(sample_pool)} PoCs, seed {seed})", "",
                  "Open the log (it contains the PoC code and the compiler output) and confirm or correct "
                  "the automatic cause.", ""]
        for i, (run, lg) in enumerate(picked, 1):
            lines.append(f"{i}. `{run.contest.id}` run {run.run_index}, hypothesis "
                         f"`{lg.get('hypothesis_id', '')[:8]}` — auto: *{classify_compile_error(lg)}* — "
                         f"`{lg['_path']}`")
        lines.append("")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# 4. Exploited: raw vs strict
# ─────────────────────────────────────────────────────────────────────────────

_SCOPE_NAMES: dict[str, set[str]] = {}
_MARK = {True: "yes", False: "**no**", None: "–"}


def _scope_names(run: RunData) -> set[str]:
    src = run.contest.resolved_source_dir()
    key = f"{src}|{run.contest.resolved_scope_file()}"
    if key not in _SCOPE_NAMES:
        _SCOPE_NAMES[key] = scope_contract_names(src, run.contest.resolved_scope_file())
    return _SCOPE_NAMES[key]


def strict_results(run: RunData) -> list[StrictResult]:
    pocs = exploited_pocs(run.state)
    if not pocs:
        return []
    names = _scope_names(run)
    traced = passing_asserts_traced(run.contest.resolved_source_dir())
    return [strict_check(e.hypothesis_id, e.poc_code, run.retraces.get(e.hypothesis_id), names, traced)
            for e in pocs]


def raw_vs_strict(run: RunData) -> str:
    rs = strict_results(run)
    n = len(run.state.get("exploits") or [])
    strict = sum(r.verdict == "strict exploited" for r in rs)
    other = collections.Counter(r.verdict for r in rs if r.verdict not in ("strict exploited", "downgraded"))
    return f"{len(rs)} / {strict} of {n} PoCs" + "".join(f" ({v} {k})" for k, v in sorted(other.items()))


def section_strict(runs: Sequence[RunData]) -> list[str]:
    lines = ["## 4. Exploited: raw vs strict", "",
             "Raw = Exploit status `exploited` (the forge test passed). Strict = additionally "
             "(a) the test calls a contract defined in scope, (b) no `assertTrue(true)` / `assert(true)` "
             "in any form, (c) no assertion comparing two identical expressions over the same arguments, "
             "(d) the `-vvvv` trace (re-run by `benchmark retrace`, no LLM) shows a passing assertion after "
             "a successful (not reverted) call to an in-scope contract. Checks (b) and (c) read the stored "
             "PoC code, (a) and (d) the re-run trace. forge-std ≥ 1.10 calls `vm.assert*` only when an "
             "assertion fails, so for such projects a passing assertion is invisible: (d) is then "
             "*undetermined* if a successful in-scope call exists and *no* only if there is none. "
             "`–` = not available (no re-run trace / not determinable).", ""]
    rows, details, identical = [], [], []
    for run in runs:
        n = len(run.state.get("exploits") or [])
        if not n:
            rows.append([run.label, "0", "–", f"no Exploit stage (preset {run.preset})"])
            continue
        rs = strict_results(run)
        reasons = collections.Counter(k for r in rs for k in r.reasons)
        traced = passing_asserts_traced(run.contest.resolved_source_dir())
        rows.append([run.label + ("" if traced else " (passing asserts not traced)"), str(n), raw_vs_strict(run),
                     "; ".join(f"{REASONS[k]}: {_n(reasons[k], len(rs), 'PoCs')}" for k in REASONS
                               if reasons[k]) or "–"])
        status = {f.get("hypothesis_id"): f for f in run.state.get("validated_findings") or []}
        for r in rs:
            f = status.get(r.hypothesis_id, {})
            fin = f"{f.get('skeptic_status') or f.get('status', '–')} → {f.get('status', '–')}" if f else "no finding"
            rr = run.retraces.get(r.hypothesis_id)
            details.append([run.label, f"`{r.hypothesis_id[:8]}`", fin,
                            *(_MARK[r.checks[k]] for k in ("a", "b", "c", "d")),
                            "–" if r.rerun_passed is None else ("pass" if r.rerun_passed else "**fail**"),
                            r.verdict, f"`{rr['_path']}`" if rr else "–"])
            identical += [f"`{r.hypothesis_id[:8]}`: `{x}`" for x in r.identical]
    lines += _table(["Run", "PoCs", "exploited (raw) / exploited (strict)", "Downgrade reasons (a PoC can have several)"],
                    rows)
    if details:
        lines += ["", "Per PoC (Skeptic → final status of its finding):", ""]
        lines += _table(["Run", "Hypothesis", "Skeptic → final", "(a) in-scope call", "(b) no trivial assert",
                         "(c) no identical sides", "(d) assert after in-scope call", "Re-run", "Verdict",
                         "Trace"], details)
    if identical:
        lines += ["", "Assertions with identical sides (check c): " + "; ".join(identical) + "."]
    return lines + [""]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Before / after
# ─────────────────────────────────────────────────────────────────────────────

def _poc_bases(run: RunData) -> collections.Counter:
    """How many final PoCs inherit a forge Test-derived contract vs something else."""
    test_dir_files: list[Path] = []
    for sub in ("test", "tests"):
        root = run.contest.resolved_source_dir() / sub
        if root.is_dir():
            test_dir_files += list(root.rglob("*.sol"))
    test_bases = _test_contracts(test_dir_files) | {"Test"}
    c: collections.Counter[str] = collections.Counter()
    for e in run.state.get("exploits") or []:
        m = re.search(r"contract\s+\w+\s+is\s+([^{]+)\{", e.get("poc_code") or "")
        parents = _parents(m.group(1)) if m else []
        c["inherits a Test-derived contract" if any(p in test_bases for p in parents)
          else "no Test-derived parent"] += 1
    return c


def section_before_after(before: Sequence[RunData], after: Sequence[RunData]) -> list[str]:
    lines = ["## 5. Before / after", ""]
    if not before or not after:
        return lines + ["Not requested (use --before/--after).", ""]
    b, a = before[0], after[0]
    if b.contest.id != a.contest.id:
        lines += [f"**Warning:** different contests ({b.contest.id} vs {a.contest.id}).", ""]
    sb, sa = compile_stats(b), compile_stats(a)
    pb, pa = _poc_bases(b), _poc_bases(a)
    commits = [((r.meta.get("git") or {}).get("commit") or "?")[:10] for r in (b, a)]
    same = "Same contest" if b.contest.id == a.contest.id else "Different contests"
    lines += [f"Before: `{b.results_dir.name}` (commit `{commits[0]}`); "
              f"after: `{a.results_dir.name}` (commit `{commits[1]}`). {same}, one run each — "
              "LLM output differs between runs, so differences are indicative only.", ""]
    keys = sorted(set(sb.buckets) | set(sa.buckets))
    rows = [[f"Exploit: {k}", _n(sb.buckets[k], sb.pocs, "PoCs"), _n(sa.buckets[k], sa.pocs, "PoCs")] for k in keys]
    rows.append(["exploited (raw) / exploited (strict)", raw_vs_strict(b), raw_vs_strict(a)])
    rows += [[f"PoC {k}", _n(pb[k], sb.pocs, "PoCs"), _n(pa[k], sa.pocs, "PoCs")]
             for k in ("inherits a Test-derived contract", "no Test-derived parent")]
    nb, na = len(sb.compile_failed), len(sa.compile_failed)
    rows += [[f"Compile cause (auto): {c}", _n(sb.by_cause_poc[c], nb, "failed PoCs"),
              _n(sa.by_cause_poc[c], na, "failed PoCs")]
             for c in CAUSES if sb.by_cause_poc[c] or sa.by_cause_poc[c]]
    return lines + _table(["", "Before", "After"], rows) + [""]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Near misses (manual list)
# ─────────────────────────────────────────────────────────────────────────────

NEAR_MISS_FILE = "near_misses.csv"
NEAR_MISS_COLUMNS = ["results_dir", "contest", "run", "hypothesis_id", "gt_id", "criterion", "note"]


def section_near_misses(runs: Sequence[RunData], path: Optional[Path]) -> list[str]:
    lines = ["## 6. Near misses (manual list, not counted in recall)", ""]
    if path is None or not Path(path).exists():
        return lines + [f"No {NEAR_MISS_FILE}.", ""]
    with Path(path).open(newline="", encoding="utf-8") as fh:
        entries = list(csv.DictReader(fh))
    lines += [f"Hand-picked from `{Path(path).name}`: hypotheses that point at the code of a GT finding but "
              "name a different flaw. Under the labelling rules they are **not** matches; they show how "
              "close the pipeline got. The selection criterion is given per GT finding.", ""]
    groups: dict[tuple[str, str], list[list[str]]] = {}
    criteria: dict[tuple[str, str], str] = {}
    for run in runs:
        mine = [e for e in entries if e.get("results_dir") == run.results_dir.name
                and e.get("contest") == run.contest.id and int(e.get("run") or 0) == run.run_index]
        hyps = run.state.get("hypotheses") or []
        findings = {f.get("hypothesis_id"): f for f in run.state.get("validated_findings") or []}
        exploits = {e["hypothesis_id"]: ExploitResult.model_validate(e) for e in run.state.get("exploits") or []}
        for e in mine:
            key = (run.contest.id, e.get("gt_id", ""))
            criteria.setdefault(key, e.get("criterion", ""))
            h = next((h for h in hyps if h["id"].startswith(e["hypothesis_id"])), None)
            if h is None:
                groups.setdefault(key, []).append([run.label, f"`{e['hypothesis_id']}`", "**not found in run**",
                                                   "", "", "", e.get("note", "")])
                continue
            f = findings.get(h["id"])
            ex = exploits.get(h["id"])
            groups.setdefault(key, []).append([
                run.label, f"`{h['id'][:8]}`", h["title"].replace("|", "/"),
                "; ".join(h.get("affected_functions") or []),
                exploit_bucket(ex) if ex else "–",
                ("n/a (no Skeptic/Refuter)" if not run.skeptic_enabled else
                 f"{f.get('skeptic_status') or f['status']} → {f['status']}" if f else "no finding"),
                e.get("note", "").replace("|", "/")])
    if not groups:
        return lines + ["No entries for these runs.", ""]
    for (contest, gt), rows in groups.items():
        g = next((x for r in runs if r.contest.id == contest for x in r.contest.ground_truth() if x.id == gt), None)
        title = f" — {g.title}" if g else ""
        n_runs = len({r[0] for r in rows})
        lines += [f"### {contest} {gt}{title} (n = {len(rows)} hypotheses in {n_runs} runs)", ""]
        if criteria.get((contest, gt)):
            lines += [f"Criterion: {criteria[(contest, gt)]}.", ""]
        lines += _table(["Run", "Hypothesis", "Title", "Functions", "Exploit", "Skeptic → final", "Note"], rows)
        lines.append("")
    return lines


# ─────────────────────────────────────────────────────────────────────────────

def build_analysis(
    results_dirs: Sequence[Path],
    *,
    dataset_override: Optional[str] = None,
    before: Optional[Path] = None,
    after: Optional[Path] = None,
    seed: int = DEFAULT_SEED,
    near_misses: Optional[Path] = None,
) -> str:
    runs = load_runs(results_dirs, dataset_override)
    lines = ["# Pilot analysis", "",
             "Generated by `audit-agent benchmark analysis` from stored run files (no LLM calls). "
             "n = number of GT findings, findings or PoCs behind each number.", "",
             "Runs: " + ", ".join(f"`{r.label}` ({r.preset}, {len(r.gt_ids)} GT, {r.labels_progress()})"
                                  for r in runs), ""]
    lines += section_gt_paths(runs)
    lines += section_validators(runs)
    lines += section_compile(runs, seed)
    lines += section_strict(runs)
    lines += section_before_after(
        load_runs([before], dataset_override) if before else [],
        load_runs([after], dataset_override) if after else [])
    lines += section_near_misses(runs, near_misses)
    return "\n".join(lines) + "\n"
