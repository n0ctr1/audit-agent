"""Rich Slither analysis wrapper — runs the full Slither Python API in Docker.

This complements `tools.slither.run_slither` (which produces `StaticFinding`s
from detectors). Here we run `docker/slither/rich_analysis.py` inside the
slither image to extract a COMPLETE code model — every function, the real
internal call graph, accurate state reads/writes, external-call targets, and
taint — emitted as normalized JSON that `parse_slither_json` consumes directly.

Design goals:
  • Best-effort: any failure returns None and the caller falls back to the
    detector-only path (no regression vs the previous pipeline).
  • Mock-mode aware: returns MockRunResult under AUDIT_AGENT_TOOL_MODE=mock.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from audit_agent.observability.logging import get_logger
from audit_agent.sandbox.docker_runner import DockerRunner
from audit_agent.tools.errors import MockRunResult, ToolError

log = get_logger("tools.slither_rich")

RICH_OUTPUT_FILENAME = "slither_rich.json"
_CONTAINER_SCRIPT_PATH = "/opt/rich_analysis.py"


def _host_script_path() -> Optional[Path]:
    """Locate docker/slither/rich_analysis.py relative to the source tree."""
    # src/audit_agent/tools/slither_rich.py → parents[3] == repo root
    candidate = Path(__file__).resolve().parents[3] / "docker" / "slither" / "rich_analysis.py"
    if candidate.exists():
        return candidate
    return None


async def run_slither_rich(
    target_dir: str,
    audit_id: str = "",
    timeout: int = 240,
) -> Optional[dict] | MockRunResult:
    """Run the rich Slither analysis in Docker, return the parsed model dict.

    Returns:
        dict           — the rich code model (normalized format) on success
        MockRunResult  — when running in mock mode
        None           — on any failure (caller should fall back to detector JSON)
    """
    script = _host_script_path()
    if script is None:
        log.warning("slither_rich: analysis script not found on host — skipping rich analysis")
        return None

    runner = DockerRunner()
    abs_target = str(Path(target_dir).resolve())
    host_output_path = Path(target_dir) / RICH_OUTPUT_FILENAME
    if host_output_path.exists():
        try:
            host_output_path.unlink()
        except OSError:
            pass

    container_output = f"/target/{RICH_OUTPUT_FILENAME}"
    cmd = ["python3", _CONTAINER_SCRIPT_PATH, "/target", container_output]

    try:
        result = await runner.run(
            image="audit-agent/slither:latest",
            command=cmd,
            workdir="/target",
            volumes={
                abs_target: {"bind": "/target", "mode": "rw"},
                str(script): {"bind": _CONTAINER_SCRIPT_PATH, "mode": "ro"},
            },
            timeout=timeout,
        )
    except ToolError as exc:
        log.warning("slither_rich: docker run failed", error=str(exc))
        return None

    if isinstance(result, MockRunResult):
        return result

    if not host_output_path.exists():
        log.warning(
            "slither_rich: output file not produced",
            exit_code=getattr(result, "exit_code", None),
            stderr=(getattr(result, "error", "") or "")[:400],
        )
        return None

    try:
        data = json.loads(host_output_path.read_text())
    except Exception as exc:
        log.warning("slither_rich: failed to parse output", error=str(exc))
        return None

    contracts = data.get("contracts") or []
    n_fns = sum(len(c.get("functions", [])) for c in contracts)
    if data.get("error"):
        log.warning("slither_rich: analysis reported error", error=data["error"])
    log.info("slither_rich: complete", contracts=len(contracts), functions=n_fns)

    # Cache for offline/replay use, mirroring run_slither's .slither_analysis.json
    try:
        (Path(target_dir) / ".slither_rich.json").write_text(json.dumps(data))
    except OSError:
        pass

    return data if contracts else None
