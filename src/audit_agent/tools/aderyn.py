"""Aderyn static-analysis tool wrapper.

Aderyn (Cyfrin) is a fast Rust/AST-based Solidity analyzer. It complements
Slither with a different detector set and is the analyzer Code4rena-style
automated tooling leans on. Mirrors ``tools/slither.py``: runs in Docker, parses
the JSON report into ``StaticFinding`` objects, and returns ``MockRunResult`` in
mock mode so the pipeline stays offline-runnable.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from audit_agent.models.schemas import Severity, StaticFinding
from audit_agent.observability.logging import get_logger
from audit_agent.sandbox.docker_runner import DockerRunner
from audit_agent.tools.errors import MockRunResult

log = get_logger("tools.aderyn")

# Aderyn buckets issues into severity sections in its JSON report.
_SECTIONS: list[tuple[str, Severity]] = [
    ("critical_issues", "critical"),
    ("high_issues", "high"),
    ("medium_issues", "medium"),
    ("low_issues", "low"),
    ("nc_issues", "info"),
]


async def run_aderyn(
    target_dir: str,
    in_scope_files: list[str],
    audit_id: str = "",
) -> list[StaticFinding] | MockRunResult:
    runner = DockerRunner()
    abs_target = str(Path(target_dir).resolve())

    report_file_name = "aderyn_report.json"
    host_report_path = Path(target_dir) / report_file_name
    if host_report_path.exists():
        host_report_path.unlink()

    # `-o <file>.json` makes aderyn emit JSON instead of markdown.
    cmd = ["aderyn", ".", "-o", report_file_name]

    result = await runner.run(
        image="audit-agent/aderyn:latest",
        command=cmd,
        workdir="/target",
        volumes={abs_target: {"bind": "/target", "mode": "rw"}},
        timeout=120,
    )

    if isinstance(result, MockRunResult):
        return result

    if result.exit_code != 0:
        log.warning(
            "Aderyn: non-zero exit",
            exit_code=result.exit_code,
            stderr=(result.error or "")[:300],
        )

    if not host_report_path.exists():
        log.error(
            "Aderyn: report file not found on host",
            exit_code=result.exit_code,
            stderr=(result.error or "")[:500],
        )
        return []

    try:
        data = json.loads(host_report_path.read_text())
    except Exception as exc:
        log.error("Aderyn: failed to parse report file", error=str(exc))
        return []

    findings = _parse_aderyn_report(data, audit_id)
    log.info("Aderyn: parsed findings", count=len(findings))
    return findings


def _parse_aderyn_report(data: dict, audit_id: str) -> list[StaticFinding]:
    findings: list[StaticFinding] = []
    for section, severity in _SECTIONS:
        block = data.get(section)
        if not isinstance(block, dict):
            continue
        for issue in block.get("issues", []):
            instances = issue.get("instances", []) or []
            first = instances[0] if instances else {}
            contract_path = first.get("contract_path") or first.get("path")
            affected_contract = (
                Path(contract_path).stem if contract_path else None
            )
            line = first.get("line_no") or first.get("line")
            findings.append(
                StaticFinding(
                    id=str(uuid.uuid4()),
                    audit_id=audit_id,
                    tool="aderyn",
                    title=issue.get("title", "unknown"),
                    severity=severity,
                    vulnerability_class=issue.get("detector_name") or issue.get("title"),
                    description=issue.get("description", ""),
                    affected_contract=affected_contract,
                    affected_function=None,
                    line=int(line) if isinstance(line, (int, str)) and str(line).isdigit() else None,
                    raw={"section": section, "issue": issue},
                )
            )
    return findings
