"""Semgrep static-analysis tool wrapper.

Runs a Solidity-focused Semgrep ruleset (baked into the Docker image) over the
target and maps results into ``StaticFinding`` objects. Pattern-matching catches
classes Slither/Aderyn miss and feeds the learning loop, which can emit new
semgrep rules from confirmed/missed findings (``learning/feedback.py``).

Mirrors ``tools/slither.py``: Docker sandbox, JSON parse, mock-mode aware.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from audit_agent.models.schemas import Severity, StaticFinding
from audit_agent.observability.logging import get_logger
from audit_agent.sandbox.docker_runner import DockerRunner
from audit_agent.tools.errors import MockRunResult

log = get_logger("tools.semgrep")

# Semgrep emits ERROR / WARNING / INFO; map onto the audit severity scale.
_SEVERITY_MAP: dict[str, Severity] = {
    "ERROR": "high",
    "WARNING": "medium",
    "INFO": "info",
}

# Default ruleset path inside the image (Decurity smart-contract rules baked in).
DEFAULT_CONFIG = "/rules"


async def run_semgrep(
    target_dir: str,
    in_scope_files: list[str],
    audit_id: str = "",
    config: str = DEFAULT_CONFIG,
) -> list[StaticFinding] | MockRunResult:
    runner = DockerRunner()
    abs_target = str(Path(target_dir).resolve())

    report_file_name = "semgrep_report.json"
    host_report_path = Path(target_dir) / report_file_name
    if host_report_path.exists():
        host_report_path.unlink()

    cmd = [
        "semgrep",
        "--config", config,
        "--json",
        "--output", report_file_name,
        "--metrics", "off",
        "--quiet",
        ".",
    ]

    result = await runner.run(
        image="audit-agent/semgrep:latest",
        command=cmd,
        workdir="/target",
        volumes={abs_target: {"bind": "/target", "mode": "rw"}},
        timeout=180,
    )

    if isinstance(result, MockRunResult):
        return result

    # Semgrep exits 0 (no findings) or 1 (findings). Other codes = real error.
    if result.exit_code not in (0, 1):
        log.warning(
            "Semgrep: unexpected exit code",
            exit_code=result.exit_code,
            stderr=(result.error or "")[:300],
        )

    if not host_report_path.exists():
        log.error(
            "Semgrep: report file not found on host",
            exit_code=result.exit_code,
            stderr=(result.error or "")[:500],
        )
        return []

    try:
        data = json.loads(host_report_path.read_text())
    except Exception as exc:
        log.error("Semgrep: failed to parse report file", error=str(exc))
        return []

    findings = _parse_semgrep_report(data, audit_id)
    log.info("Semgrep: parsed findings", count=len(findings))
    return findings


def _parse_semgrep_report(data: dict, audit_id: str) -> list[StaticFinding]:
    findings: list[StaticFinding] = []
    for res in data.get("results", []):
        extra = res.get("extra", {}) or {}
        severity = _SEVERITY_MAP.get(str(extra.get("severity", "INFO")).upper(), "info")
        path = res.get("path")
        affected_contract = Path(path).stem if path else None
        line = (res.get("start", {}) or {}).get("line")
        check_id = res.get("check_id", "semgrep")
        findings.append(
            StaticFinding(
                id=str(uuid.uuid4()),
                audit_id=audit_id,
                tool="semgrep",
                title=str(check_id).split(".")[-1],
                severity=severity,
                vulnerability_class=check_id,
                description=extra.get("message", ""),
                affected_contract=affected_contract,
                affected_function=None,
                line=int(line) if isinstance(line, int) else None,
                raw={"check_id": check_id, "path": path, "metadata": extra.get("metadata", {})},
            )
        )
    return findings
