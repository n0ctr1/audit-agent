"""`audit-agent benchmark` command group.

    audit-agent benchmark --dataset <name|path> [--split test|dev|smoke|all] [--contest <id>]...
        [--runs N] [--after-date YYYY-MM-DD] [--preset full] [--out results/]
    audit-agent benchmark rescore <results_dir> [--dataset <name|path>]
    audit-agent benchmark report <results_dir>... [--out results/RESULTS.md]
    audit-agent benchmark analysis <results_dir>... [--before DIR --after DIR] [--out results/ANALYSIS.md]
    audit-agent benchmark retrace <results_dir>...
    audit-agent benchmark labels <results_dir>...
    audit-agent benchmark matcher-agreement <results_dir>...
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console

benchmark_app = typer.Typer(
    help="Run the benchmark, recompute metrics from stored runs, build reports.",
    invoke_without_command=True,
    no_args_is_help=False,
)
console = Console()


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def _stat(s: Optional[dict[str, Any]], pct: bool = True) -> str:
    if not s or s.get("mean") is None:
        return "n/a"
    fmt = _pct if pct else (lambda x: f"{x:.3g}")
    if s.get("n", 0) > 1 and s["min"] != s["max"]:
        return f"{fmt(s['mean'])} ({fmt(s['min'])}–{fmt(s['max'])})"
    return fmt(s["mean"])


def print_summary(metrics: dict[str, Any]) -> None:
    for split, summ in metrics.get("splits", {}).items():
        s = summ["summary"]
        console.print(
            f"\n[bold]{split}[/bold]: {summ['num_contests']} contests, "
            f"{summ['num_runs_ok']} ok / {summ['num_runs_error']} failed runs"
        )
        console.print(f"  recall_hypotheses@30 : {_stat(s['recall_hypotheses_at_30'])}")
        console.print(f"  recall_final         : {_stat(s['recall_final'])}")
        console.print(f"  recall_validated     : {_stat(s['recall_validated'])}")
        console.print(f"  precision_gt         : {_stat(s['precision_gt'])}")
        console.print(f"  validator_loss (GT)  : {_stat(s['validator_loss_count'], pct=False)}")
        console.print(f"  cost_usd / contest   : {_stat(s['cost_usd'], pct=False)}")
    n = metrics.get("unlabelled_unmatched_findings", 0)
    if n:
        console.print(f"\n  [yellow]{n} unmatched validated findings are unlabelled[/yellow] "
                      "(unmatched_findings.csv) — precision_adjusted not computed for those runs")


@benchmark_app.callback()
def run(
    ctx: typer.Context,
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d",
                                          help="Dataset name in benchmarks/datasets/ or a JSON path"),
    split: str = typer.Option("test", "--split", help="test | dev | smoke | all"),
    contest: list[str] = typer.Option([], "--contest", help="Only these contest ids (repeatable)"),
    runs: int = typer.Option(1, "--runs", min=1, help="Independent runs per contest"),
    after_date: Optional[datetime] = typer.Option(
        None, "--after-date", formats=["%Y-%m-%d"],
        help="Only contests that ended after this date (model knowledge cutoff)"),
    preset: str = typer.Option("full", "--preset", help="Pipeline preset (see benchmarks/presets.py)"),
    out: Path = typer.Option(Path("results"), "--out", help="Parent directory for results"),
    include_needs_review: bool = typer.Option(False, "--include-needs-review",
                                              help="Also run unreviewed draft contests"),
    fuzz: bool = typer.Option(True, "--fuzz/--no-fuzz", help="Medusa fuzzing, as in `audit`"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run the pipeline on dataset contests and store raw states + metrics."""
    if ctx.invoked_subcommand is not None:
        return
    if not dataset:
        console.print(ctx.get_help())
        raise typer.Exit(2)

    import asyncio

    from audit_agent.observability.logging import configure_logging
    from benchmarks.dataset import load_dataset, select_contests
    from benchmarks.presets import preset_env
    from benchmarks.runner import METRICS_FILE, run_suite

    configure_logging(log_level)
    if split not in ("test", "dev", "smoke", "all"):
        console.print(f"[red]invalid --split {split!r}[/red]")
        raise typer.Exit(2)
    try:
        preset_env(preset)
        ds = load_dataset(dataset)
        cutoff: Optional[date] = after_date.date() if after_date else None
        selected = select_contests(ds, split=split, contest_ids=contest or None,
                                   after_date=cutoff, include_needs_review=include_needs_review)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if not selected:
        console.print("[yellow]No contests selected[/yellow] (check --split / --contest / "
                      "--after-date; drafts with needs_review are skipped by default)")
        raise typer.Exit(1)

    console.print(f"[bold]Benchmark[/bold] dataset={ds.name} split={split} preset={preset} "
                  f"runs={runs} contests={len(selected)}")
    results_dir = asyncio.run(run_suite(
        selected, dataset_ref=dataset, runs=runs, preset=preset, out_dir=out, fuzz=fuzz,
        selection={"split": split, "contests": contest, "after_date": str(cutoff) if cutoff else None,
                   "include_needs_review": include_needs_review},
        on_progress=lambda m: console.print(f"  {m}"),
    ))
    import json

    print_summary(json.loads((results_dir / METRICS_FILE).read_text()))
    console.print(f"\nResults: {results_dir}")


@benchmark_app.command("rescore")
def rescore_cmd(
    results_dir: Path = typer.Argument(..., help="results/<run> directory"),
    dataset: Optional[str] = typer.Option(
        None, "--dataset", help="Use GT from this dataset instead of the meta.json snapshot"),
) -> None:
    """Recompute metrics from stored raw states (no pipeline run)."""
    from benchmarks.runner import rescore

    try:
        metrics = rescore(results_dir, dataset_override=dataset)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    print_summary(metrics)
    console.print(f"\nWrote {results_dir / 'metrics.json'}")


@benchmark_app.command("report")
def report_cmd(
    results_dirs: list[Path] = typer.Argument(..., help="One or more results/<run> directories"),
    out: Path = typer.Option(Path("results/RESULTS.md"), "--out", help="Markdown file to write"),
    no_rescore: bool = typer.Option(False, "--no-rescore", help="Use existing metrics.json as is"),
) -> None:
    """Write a summary report (per contest, per preset, cost, metadata, limitations)."""
    from benchmarks.report import write_report

    try:
        path = write_report(results_dirs, out, do_rescore=not no_rescore)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"Wrote {path}")


@benchmark_app.command("analysis")
def analysis_cmd(
    results_dirs: list[Path] = typer.Argument(..., help="results/<run> directories"),
    dataset: Optional[str] = typer.Option(
        None, "--dataset", help="Use GT from this dataset instead of the meta.json snapshot"),
    before: Optional[Path] = typer.Option(None, "--before", help="Run before a change (before/after view)"),
    after: Optional[Path] = typer.Option(None, "--after", help="Run after the change"),
    seed: int = typer.Option(2026, "--seed", help="Seed for the manual-check PoC sample"),
    out: Path = typer.Option(Path("results/ANALYSIS.md"), "--out", help="Markdown file to write"),
    near_misses: Optional[Path] = typer.Option(
        None, "--near-misses", help="Manual near-miss list (default: near_misses.csv next to --out)"),
) -> None:
    """Pilot analysis: GT path by stage, Skeptic→Refuter, PoC compile failures, strict exploited,
    before/after, near misses."""
    from benchmarks.analysis import NEAR_MISS_FILE, build_analysis

    try:
        md = build_analysis(results_dirs, dataset_override=dataset, before=before, after=after, seed=seed,
                            near_misses=near_misses or out.parent / NEAR_MISS_FILE)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    console.print(f"Wrote {out}")


@benchmark_app.command("retrace")
def retrace_cmd(
    results_dirs: list[Path] = typer.Argument(..., help="results/<run> directories"),
    dataset: Optional[str] = typer.Option(
        None, "--dataset", help="Use contests from this dataset instead of the meta.json snapshot"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Re-run PoCs that already have a trace"),
) -> None:
    """Re-run stored `exploited` PoCs with `forge test -vvvv` in Docker (no LLM) for the strict check.

    Traces go to <results>/<contest>/run_<k>_retrace/<hypothesis_id>.json.
    """
    import asyncio

    from benchmarks.retrace import retrace_results

    for d in results_dirs:
        try:
            done = asyncio.run(retrace_results(d, dataset_override=dataset, overwrite=overwrite))
        except FileNotFoundError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
        for r in done:
            state = "mock" if r["mock"] else ("pass" if r["passed"] else "FAIL")
            console.print(f"  {d.name} {r['contest_id']} run {r['run_index']} {r['hypothesis_id'][:8]}: {state}")
        console.print(f"{d}: {len(done)} PoC(s) re-run")


@benchmark_app.command("labels")
def labels_cmd(
    results_dirs: list[Path] = typer.Argument(..., help="results/<run> directories"),
    dataset: Optional[str] = typer.Option(
        None, "--dataset", help="Use GT from this dataset instead of the meta.json snapshot"),
) -> None:
    """Create/refresh labels_<contest>_run<k>.csv (one row per hypothesis) for manual GT labels.

    Existing matches_gt / reviewed values are kept.
    """
    from benchmarks.runner import rescore

    for d in results_dirs:
        try:
            m = rescore(d, dataset_override=dataset)
        except FileNotFoundError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
        for r in m["runs"]:
            hl = r.get("human_labels") or {}
            if hl:
                console.print(f"{d / hl['file']}: {hl['rows']} rows, {hl['unreviewed']} unreviewed")


@benchmark_app.command("matcher-agreement")
def matcher_agreement_cmd(
    results_dirs: list[Path] = typer.Argument(..., help="results/<run> directories with labels files"),
    dataset: Optional[str] = typer.Option(
        None, "--dataset", help="Use GT from this dataset instead of the meta.json snapshot"),
    show: int = typer.Option(30, "--show", help="Max disagreements to print per run"),
) -> None:
    """Human recall and matcher agreement from labels_<contest>_run<k>.csv."""
    from benchmarks.dataset import BenchmarkContest, load_dataset
    from benchmarks.human_labels import evaluate_labels, labels_filename, read_rows
    from benchmarks.runner import iter_run_records, load_meta

    tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for d in results_dirs:
        try:
            meta = load_meta(d)
        except FileNotFoundError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
        contests = {c["id"]: BenchmarkContest.model_validate(c) for c in meta["contests"]}
        if dataset:
            contests.update({c.id: c for c in load_dataset(dataset).contests if c.id in contests})
        for _p, rec in iter_run_records(d):
            if rec.get("status") != "ok" or rec["contest_id"] not in contests:
                continue
            path = Path(d) / labels_filename(rec["contest_id"], rec["run_index"])
            if not path.exists():
                console.print(f"[yellow]{path} missing — run `benchmark labels` first[/yellow]")
                continue
            res = evaluate_labels(read_rows(path), rec.get("state") or {},
                                  contests[rec["contest_id"]].ground_truth())
            for k in tot:
                tot[k] += getattr(res, k)
            console.print(f"\n[bold]{Path(d).name} · {rec['contest_id']} run {rec['run_index']}[/bold]")
            console.print(f"  reviewed rows      : {res.rows - res.unreviewed}/{res.rows}")
            if res.unknown_gt_ids:
                console.print(f"  [red]unknown GT ids   : {', '.join(res.unknown_gt_ids)}[/red]")
            console.print(f"  human recall hyp@30: {_pct(res.human_recall_hypotheses)} {res.covered_hypotheses}")
            console.print(f"  human recall final : {_pct(res.human_recall_final)} {res.covered_final}")
            console.print(f"  matcher vs human   : TP={res.tp} FP={res.fp} FN={res.fn} TN={res.tn} "
                          f"precision={_pct(res.matcher_precision)} recall={_pct(res.matcher_recall)} "
                          f"kappa={'n/a' if res.cohen_kappa is None else f'{res.cohen_kappa:.3f}'}")
            for x in res.disagreements[:show]:
                console.print(f"    [{x.gt_id}] matcher={x.matcher} human={x.human} | {x.hyp_title[:70]}")
    n = sum(tot.values())
    if n:
        p = tot["tp"] / (tot["tp"] + tot["fp"]) if tot["tp"] + tot["fp"] else None
        r = tot["tp"] / (tot["tp"] + tot["fn"]) if tot["tp"] + tot["fn"] else None
        console.print(f"\n[bold]All runs[/bold]: {n} labelled pairs, matcher precision {_pct(p)}, recall {_pct(r)}")
