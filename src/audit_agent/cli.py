"""CLI entry point using Typer."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from audit_agent.observability.logging import configure_logging, get_logger

app = typer.Typer(
    name="audit-agent",
    help="Autonomous Solidity smart contract security auditor.",
    add_completion=False,
)
console = Console()


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


# ── audit command ─────────────────────────────────────────────────────────────

@app.command()
def audit(
    target: str = typer.Argument(..., help="Path to folder with Solidity contracts"),
    mode: str = typer.Option("audit", "--mode", "-m", help="audit | bugbounty | ctf | review"),
    docs: Optional[str] = typer.Option(None, "--docs", "-d", help="Path to documentation folder"),
    platform: str = typer.Option("generic", "--platform", "-p"),
    budget_usd: float = typer.Option(50.0, "--budget"),
    log_level: str = typer.Option("INFO", "--log-level"),
    metrics: bool = typer.Option(False, "--metrics", help="Start Prometheus metrics server"),
    reports_dir: Optional[str] = typer.Option(
        None, "--reports-dir", help="Directory to save reports (default: ./audit-reports)"
    ),
    fuzz: bool = typer.Option(True, "--fuzz/--no-fuzz", help="Enable Medusa fuzzing for fuzz-type hypotheses"),
    scope: Optional[Path] = typer.Option(
        None,
        "--scope",
        help=(
            "Path to scope.txt listing in-scope .sol files (one per line). "
            "If omitted, looks for scope.txt in target_dir. "
            "If neither exists, all non-library .sol files are in scope."
        ),
    ),
    no_log: bool = typer.Option(
        False, "--no-log",
        help="Disable detailed audit logging (audit_logs/ directory)"
    ),
    log_dir: Optional[Path] = typer.Option(
        None, "--log-dir",
        help="Directory for audit logs. Default: {target_dir}/audit_logs/{audit_id}/"
    ),
) -> None:
    """Run a full audit on a smart contract codebase."""
    configure_logging(log_level)
    log = get_logger("cli")

    target_path = Path(target).resolve()
    if not target_path.exists():
        console.print(f"[red]Error:[/red] target path does not exist: {target_path}")
        raise typer.Exit(1)

    if mode not in ("audit", "bugbounty", "ctf", "review"):
        console.print(f"[red]Error:[/red] invalid mode '{mode}'")
        raise typer.Exit(1)

    scope_path = scope.resolve() if scope else None
    if scope_path and not scope_path.exists():
        console.print(f"[red]Error:[/red] scope file not found: {scope_path}")
        raise typer.Exit(1)

    try:
        _run(_run_audit(
            str(target_path), mode, docs, platform, budget_usd, metrics, log,
            reports_dir_override=reports_dir, fuzz_enabled=fuzz,
            scope_file=str(scope_path) if scope_path else None,
            audit_log_enabled=not no_log,
            audit_log_dir=str(log_dir) if log_dir else None,
        ))
    except Exception as exc:
        from audit_agent.tools.errors import ToolError
        if isinstance(exc, ToolError):
            console.print(
                f"\n[red bold]Audit failed:[/red bold] tool '{exc.tool}' unavailable.\n"
                f"  {exc}\n\n"
                "Hint: set AUDIT_AGENT_TOOL_MODE=lenient to continue without the tool, "
                "or AUDIT_AGENT_TOOL_MODE=mock for testing without Docker."
            )
        else:
            console.print(f"\n[red bold]Audit failed:[/red bold] {exc}")
        raise typer.Exit(1)


async def _run_audit(
    target_dir: str,
    mode: str,
    docs_dir: Optional[str],
    platform: str,
    budget_usd: float,
    start_metrics: bool,
    log,  # type: ignore[type-arg]
    checkpointer=None,  # type: ignore[type-arg]
    audit_id: Optional[str] = None,
    reports_dir_override: Optional[str] = None,
    fuzz_enabled: bool = True,
    scope_file: Optional[str] = None,
    audit_log_enabled: bool = True,
    audit_log_dir: Optional[str] = None,
) -> dict:  # type: ignore[type-arg]
    from audit_agent.config import get_settings
    from audit_agent.graph.workflow import build_graph
    from audit_agent.observability.audit_logger import AuditLogger, set_audit_logger
    from audit_agent.observability.metrics import get_metrics

    settings = get_settings()

    if start_metrics:
        get_metrics().start_server(settings.prometheus_port)
        log.info("Metrics server started", port=settings.prometheus_port)

    if audit_id is None:
        audit_id = str(uuid.uuid4())

    if checkpointer is None:
        checkpointer = await _create_checkpointer(log)

    log.info("Starting audit", audit_id=audit_id, mode=mode, target=target_dir)

    # Create audit logger if enabled
    effective_log_enabled = audit_log_enabled and settings.audit_log_enabled
    if effective_log_enabled:
        log_base = Path(audit_log_dir) if audit_log_dir else Path(target_dir)
        audit_logger = AuditLogger(log_base, audit_id)
        set_audit_logger(audit_logger)
        log.info("Audit logging enabled", log_dir=str(audit_logger.log_dir))
    else:
        set_audit_logger(None)

    await _persist_audit_start(audit_id, mode, platform, target_dir, budget_usd)

    graph = build_graph(checkpointer=checkpointer)

    initial_state = {
        "audit_id": audit_id,
        "mode": mode,
        "target_dir": target_dir,
        "docs_dir": docs_dir,
        "known_issues": None,
        "platform": platform,
        "budget_tokens": settings.default_budget_tokens,
        "budget_usd": budget_usd,
        "fuzz_enabled": fuzz_enabled,
        "scope_file": scope_file,
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
    final_state = await graph.ainvoke(initial_state, config=config)

    # Write audit summary log
    from audit_agent.observability.audit_logger import get_audit_logger
    al = get_audit_logger()
    if al:
        al.write_summary(dict(final_state))
        set_audit_logger(None)

    findings = final_state.get("static_findings", [])
    degraded = final_state.get("degraded_tools", [])

    log.info("Slither: %d findings" % len(findings), findings=len(findings))
    if degraded:
        log.warning(
            "Audit completed in DEGRADED mode — findings may be incomplete",
            unavailable_tools=degraded,
        )
    log.info("Run persisted as", audit_id=audit_id)

    console.print(f"\n[green]Audit complete.[/green] audit_id={audit_id}")
    console.print(f"  Static findings : {len(findings)}")
    if degraded:
        console.print(
            f"  [yellow]DEGRADED MODE[/yellow]: tools unavailable → {', '.join(degraded)}\n"
            "  Findings may be incomplete."
        )
    if final_state.get("protocol_summary"):
        ps = final_state["protocol_summary"]
        console.print(f"  Protocol        : {ps.name} ({ps.category})")

    # Save reports to disk
    base_reports_dir = Path(reports_dir_override) if reports_dir_override else settings.reports_dir
    _save_reports(final_state, audit_id, target_dir, mode, platform, base_reports_dir, log)

    await _persist_audit_complete(audit_id, len(findings))
    return final_state


def _save_reports(
    state: dict,  # type: ignore[type-arg]
    audit_id: str,
    target_dir: str,
    mode: str,
    platform: str,
    base_dir: Path,
    log,  # type: ignore[type-arg]
) -> None:
    """Save all reports, debug artefacts, and a combined SUMMARY.md to disk."""
    reports: dict[str, str] = state.get("reports") or {}
    validated = state.get("validated_findings") or []
    hypotheses = state.get("hypotheses") or []
    exploits = state.get("exploits") or []
    invariants = state.get("invariants") or []
    chains = state.get("chains") or []

    run_dir = base_dir / audit_id
    run_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = run_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    # Individual report files
    for finding_id, report_md in reports.items():
        (run_dir / f"{finding_id[:8]}-finding.md").write_text(report_md, encoding="utf-8")

    # Summary counts
    tp = sum(1 for f in validated if f.status == "validated")
    nr = sum(1 for f in validated if f.status == "unverified")
    fp = sum(1 for f in validated if f.status == "false_positive")

    # Scope self-flag breakdown over reportable (validated + needs_review) findings.
    reportable = [f for f in validated if f.status in ("validated", "unverified")]
    from collections import Counter
    scope_counts = Counter(getattr(f, "scope_flag", None) or "unassessed" for f in reportable)
    in_scope_n = scope_counts.get("likely_in_scope", 0) + scope_counts.get("needs_scope_review", 0)
    out_n = (scope_counts.get("out_of_scope", 0) + scope_counts.get("known_issue", 0)
             + scope_counts.get("below_paid_tier", 0))

    # SUMMARY.md
    summary_parts = [
        f"# Audit Report: {audit_id}\n\n",
        f"**Date:** {datetime.now().isoformat()}\n",
        f"**Target:** {target_dir}\n",
        f"**Mode:** {mode}\n",
        f"**Platform:** {platform}\n\n",
        f"## Results\n\n",
        f"- True positives: {tp}\n",
        f"- Needs review: {nr}\n",
        f"- False positives: {fp}\n\n",
        f"## Scope (reportable findings)\n\n",
        f"- Likely in scope / needs review: {in_scope_n}\n",
        f"- Out of scope / known issue / below paid tier: {out_n}\n",
        *[f"  - {k}: {v}\n" for k, v in sorted(scope_counts.items())],
        "\n---\n\n",
    ]
    for report_md in reports.values():
        summary_parts.append(report_md)
        summary_parts.append("\n\n---\n\n")
    (run_dir / "SUMMARY.md").write_text("".join(summary_parts), encoding="utf-8")

    # Debug JSON files
    def _dump(obj):  # type: ignore[no-untyped-def]
        return obj.model_dump() if hasattr(obj, "model_dump") else obj

    (debug_dir / "hypotheses.json").write_text(
        json.dumps([_dump(h) for h in hypotheses], indent=2, default=str), encoding="utf-8"
    )
    (debug_dir / "exploits.json").write_text(
        json.dumps([_dump(e) for e in exploits], indent=2, default=str), encoding="utf-8"
    )
    (debug_dir / "validated_findings.json").write_text(
        json.dumps([_dump(v) for v in validated], indent=2, default=str), encoding="utf-8"
    )
    (debug_dir / "invariants.json").write_text(
        json.dumps([_dump(i) for i in invariants], indent=2, default=str), encoding="utf-8"
    )
    (debug_dir / "chains.json").write_text(
        json.dumps([_dump(c) for c in chains], indent=2, default=str), encoding="utf-8"
    )

    log.info("Reports saved", path=str(run_dir), count=len(reports))
    console.print(f"\n  Reports saved to: {run_dir}/")
    console.print(f"  Summary: {run_dir / 'SUMMARY.md'}")


async def _create_checkpointer(log):  # type: ignore[type-arg]
    """Try Postgres checkpointer; fall back to in-memory with a warning."""
    from audit_agent.config import get_settings

    settings = get_settings()
    # langgraph-checkpoint-postgres needs the psycopg3 URL format (no +asyncpg driver suffix)
    pg_url = settings.postgres_url_sync.replace("postgresql+psycopg2://", "postgresql://")

    try:
        from psycopg_pool import AsyncConnectionPool
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        # from_conn_string() returns an async context manager, not the saver directly.
        # Use psycopg_pool so we own the connection lifetime.
        pool = AsyncConnectionPool(pg_url, open=False, kwargs={"autocommit": True})
        await pool.open()
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        log.info("Checkpointer: using Postgres (audit is resumable)")
        return saver
    except Exception as exc:
        log.warning(
            "Checkpointing disabled — audit cannot be resumed if interrupted",
            reason=str(exc),
            hint="Start Postgres and run `audit-agent db-init` to enable checkpointing.",
        )
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


# ── ingest command ───────────────────────────────────────────────────────────

@app.command()
def ingest(
    target: str = typer.Argument(..., help="Path to folder with Solidity contracts"),
    docs: Optional[str] = typer.Option(None, "--docs", "-d", help="Path to documentation folder"),
    slither_fixture: Optional[str] = typer.Option(
        None, "--slither-fixture", help="Path to pre-computed slither_output.json (offline mode)"
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run Scout + Mapper on a codebase (Phase 2: intake + graph building)."""
    configure_logging(log_level)
    log = get_logger("cli.ingest")

    target_path = Path(target).resolve()
    if not target_path.exists():
        console.print(f"[red]Error:[/red] target path does not exist: {target_path}")
        raise typer.Exit(1)

    try:
        _run(_run_ingest(str(target_path), docs, slither_fixture, log))
    except Exception as exc:
        console.print(f"\n[red bold]Ingest failed:[/red bold] {exc}")
        raise typer.Exit(1)


async def _run_ingest(
    target_dir: str,
    docs_dir: Optional[str],
    slither_fixture: Optional[str],
    log,  # type: ignore[type-arg]
) -> None:
    import json
    import uuid

    from audit_agent.agents.mapper import MapperAgent
    from audit_agent.agents.scout import ScoutAgent, _enumerate_sol_files, _run_fork_detection, _build_scope_map, _count_loc

    audit_id = str(uuid.uuid4())[:8]

    # ── Scout phase ──────────────────────────────────────────────────────────
    in_scope, out_of_scope = _enumerate_sol_files(target_dir)
    console.print(f"\n[bold]Scout[/bold]: {len(in_scope)} in-scope files, "
                  f"{len(out_of_scope)} dependencies, {_count_loc(in_scope)} LOC")

    fork_analysis = _run_fork_detection(in_scope)
    if fork_analysis.is_fork:
        pct = int(fork_analysis.similarity_score * 100)
        mods = fork_analysis.modified_files[:5]
        console.print(
            f"[SCOUT] Detected {pct}% match with {fork_analysis.base_protocol} "
            f"→ modifications={[Path(m).stem for m in mods]}"
        )
    else:
        console.print("[SCOUT] No known fork detected")

    # ── Mapper phase ─────────────────────────────────────────────────────────
    slither_json = None
    if slither_fixture:
        fixture_path = Path(slither_fixture)
        if fixture_path.exists():
            slither_json = json.loads(fixture_path.read_text())
            log.info("Mapper: using slither fixture", path=str(fixture_path))
        else:
            console.print(f"[yellow]Warning:[/yellow] slither fixture not found: {slither_fixture}")

    mapper = MapperAgent(slither_json=slither_json)
    docs_text = ""
    if docs_dir:
        from audit_agent.agents.mapper import _read_docs
        docs_text = _read_docs(docs_dir)

    result = await mapper.run_standalone(target_dir, audit_id=audit_id, docs_text=docs_text)

    console.print(
        f"[MAPPER] Graph: {result.functions} functions, "
        f"{result.contracts} contracts, {result.relationships} relationships"
    )
    console.print(
        f"[MAPPER] Extracted {result.total_invariants} invariants "
        f"({result.invariants_natspec} natspec, "
        f"{result.invariants_docs} docs, "
        f"{result.invariants_inferred} inferred)"
    )

    if result.top_surface_functions:
        console.print("\n[bold]Top attack surface functions:[/bold]")
        for fn in result.top_surface_functions:
            console.print(
                f"  {fn['contract']}.{fn['name']}() "
                f"[{fn['visibility']}] score={fn['score']:.1f}"
            )

    if result.errors:
        for err in result.errors:
            console.print(f"[yellow]Warning:[/yellow] {err}")


# ── resume command ────────────────────────────────────────────────────────────

@app.command()
def resume(
    audit_id: str = typer.Argument(..., help="audit_id of a previously interrupted audit"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Resume a previously interrupted audit from its last checkpoint."""
    configure_logging(log_level)
    log = get_logger("cli.resume")

    try:
        _run(_resume_audit(audit_id, log))
    except Exception as exc:
        console.print(f"\n[red bold]Resume failed:[/red bold] {exc}")
        raise typer.Exit(1)


async def _resume_audit(audit_id: str, log) -> None:  # type: ignore[type-arg]
    from audit_agent.config import get_settings
    from audit_agent.graph.workflow import build_graph

    settings = get_settings()

    checkpointer = await _create_checkpointer(log)
    graph = build_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": audit_id}}

    # Fetch last saved state to find out where we left off
    state_snapshot = await graph.aget_state(config)
    if state_snapshot is None or not state_snapshot.values:
        console.print(f"[red]No checkpoint found for audit_id={audit_id}[/red]")
        raise typer.Exit(1)

    last_node = state_snapshot.next
    log.info(
        "Resuming audit from checkpoint",
        audit_id=audit_id,
        next_nodes=list(last_node) if last_node else [],
    )

    # Re-invoke from the checkpoint — LangGraph continues from where it stopped
    final_state = await graph.ainvoke(None, config=config)  # type: ignore[arg-type]

    findings = final_state.get("static_findings", []) if final_state else []
    console.print(f"\n[green]Audit resumed and completed.[/green] audit_id={audit_id}")
    console.print(f"  Static findings: {len(findings)}")

    # Save reports to disk (mirroring what the main audit command does)
    if final_state:
        target_dir = final_state.get("target_dir", ".")
        mode = final_state.get("mode", "audit")
        platform = final_state.get("platform", "generic")
        base_reports_dir = settings.reports_dir
        _save_reports(final_state, audit_id, target_dir, mode, platform, base_reports_dir, log)


# ── infrastructure commands ───────────────────────────────────────────────────

@app.command()
def db_init(log_level: str = typer.Option("INFO", "--log-level")) -> None:
    """Run Alembic migrations and set up LangGraph checkpoint schema."""
    configure_logging(log_level)
    log = get_logger("cli.db_init")
    log.info("Running database migrations")
    import subprocess

    result = subprocess.run(
        ["alembic", "upgrade", "head"], capture_output=True, text=True
    )
    if result.returncode != 0:
        log.error("Migration failed", stderr=result.stderr)
        raise typer.Exit(1)
    log.info("Migrations applied successfully")

    # Set up LangGraph checkpoint tables (AsyncPostgresSaver.setup())
    _run(_setup_langgraph_checkpoints(log))


async def _setup_langgraph_checkpoints(log) -> None:  # type: ignore[type-arg]
    """Call AsyncPostgresSaver.setup() to create checkpoint tables."""
    try:
        from audit_agent.config import get_settings
        from psycopg_pool import AsyncConnectionPool
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        settings = get_settings()
        pg_url = settings.postgres_url_sync.replace("postgresql+psycopg2://", "postgresql://")
        pool = AsyncConnectionPool(pg_url, open=False, kwargs={"autocommit": True})
        await pool.open()
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        await pool.close()
        log.info("LangGraph checkpoint schema created")
    except Exception as exc:
        log.warning("Could not set up LangGraph checkpoint schema", error=str(exc))


@app.command()
def neo4j_init(log_level: str = typer.Option("INFO", "--log-level")) -> None:
    """Set up Neo4j schema constraints and indexes."""
    configure_logging(log_level)
    _run(_neo4j_init())


async def _neo4j_init() -> None:
    from audit_agent.knowledge.graph.neo4j_client import Neo4jClient
    from audit_agent.knowledge.graph.schema import setup_schema

    client = Neo4jClient.from_settings()
    await setup_schema(client)
    await client.close()


@app.command()
def qdrant_init(log_level: str = typer.Option("INFO", "--log-level")) -> None:
    """Set up Qdrant collections."""
    configure_logging(log_level)
    _run(_qdrant_init())


async def _qdrant_init() -> None:
    from qdrant_client import AsyncQdrantClient

    from audit_agent.config import get_settings
    from audit_agent.knowledge.vectors.collections import setup_collections

    s = get_settings()
    client = AsyncQdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None)
    await setup_collections(client)
    await client.close()


# ── Postgres helpers ──────────────────────────────────────────────────────────

async def _persist_audit_start(
    audit_id: str, mode: str, platform: str, target_dir: str, budget_usd: float
) -> None:
    try:
        import uuid as _uuid

        from audit_agent.storage.postgres import AuditRun, get_session_factory

        session_factory = get_session_factory()
        async with session_factory() as session:
            run = AuditRun(
                id=_uuid.UUID(audit_id),
                name=f"audit-{audit_id[:8]}",
                mode=mode,
                platform=platform,
                target_dir=target_dir,
                status="running",
                budget_usd=budget_usd,
            )
            session.add(run)
            await session.commit()
    except Exception as exc:
        get_logger("cli").warning("Could not persist audit to Postgres", error=str(exc))


async def _persist_audit_complete(audit_id: str, finding_count: int) -> None:
    try:
        import uuid as _uuid
        from datetime import datetime

        from sqlalchemy import update

        from audit_agent.storage.postgres import AuditRun, get_session_factory

        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(
                update(AuditRun)
                .where(AuditRun.id == _uuid.UUID(audit_id))
                .values(status="completed", completed_at=datetime.utcnow())
            )
            await session.commit()
    except Exception:
        pass


# ── benchmark command group (benchmarks/cli.py) ───────────────────────────────

from benchmarks.cli import benchmark_app  # noqa: E402

app.add_typer(benchmark_app, name="benchmark")


# ── feedback command ──────────────────────────────────────────────────────────

@app.command()
def feedback(
    audit_id: str = typer.Argument(..., help="audit_id of the completed audit run"),
    feedback_file: str = typer.Argument(..., help="Path to JSON feedback file"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Process post-audit feedback for the learning loop.

    The feedback JSON file must have this structure:
    {
      "confirmed":      [{"finding_id": "...", "title": "...", "vulnerability_class": "...", ...}],
      "false_positives":[...],
      "missed":         [...]
    }
    """
    configure_logging(log_level)
    log = get_logger("cli.feedback")

    feedback_path = Path(feedback_file).resolve()
    if not feedback_path.exists():
        console.print(f"[red]Error:[/red] feedback file not found: {feedback_path}")
        raise typer.Exit(1)

    try:
        _run(_run_feedback(audit_id, str(feedback_path), log))
    except Exception as exc:
        console.print(f"\n[red bold]Feedback processing failed:[/red bold] {exc}")
        raise typer.Exit(1)


async def _run_feedback(audit_id: str, feedback_path: str, log) -> None:  # type: ignore[type-arg]
    import json as _json

    from audit_agent.learning.feedback import AuditFeedback, FeedbackEvent, LearningLoop

    raw = _json.loads(Path(feedback_path).read_text())

    def _parse_events(items: list) -> list[FeedbackEvent]:  # type: ignore[type-arg]
        events = []
        for item in items:
            events.append(FeedbackEvent(
                finding_id=item.get("finding_id", str(uuid.uuid4())),
                audit_id=audit_id,
                title=item.get("title", ""),
                vulnerability_class=item.get("vulnerability_class", ""),
                affected_functions=item.get("affected_functions", []),
                description=item.get("description", ""),
                severity=item.get("severity", "medium"),
                notes=item.get("notes", ""),
            ))
        return events

    fb = AuditFeedback(
        audit_id=audit_id,
        confirmed=_parse_events(raw.get("confirmed", [])),
        false_positives=_parse_events(raw.get("false_positives", [])),
        missed=_parse_events(raw.get("missed", [])),
    )

    loop = LearningLoop()
    await loop.process_feedback(audit_id, fb)

    console.print(f"\n[green]Feedback processed.[/green] audit_id={audit_id}")
    console.print(f"  Confirmed    : {len(fb.confirmed)}")
    console.print(f"  False positives: {len(fb.false_positives)}")
    console.print(f"  Missed       : {len(fb.missed)}")
    if loop.semgrep_rules:
        console.print(f"  Semgrep rules generated: {len(loop.semgrep_rules)}")


if __name__ == "__main__":
    app()
