"""Slither static analysis tool wrapper."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from audit_agent.models.schemas import Severity, StaticFinding
from audit_agent.observability.logging import get_logger
from audit_agent.sandbox.docker_runner import DockerRunner
from audit_agent.tools.errors import MockRunResult, ToolExecutionError

log = get_logger("tools.slither")

_SEVERITY_MAP: dict[str, Severity] = {
    "High": "high",
    "Medium": "medium",
    "Low": "low",
    "Informational": "info",
    "Optimization": "info",
}


async def run_slither(
    target_dir: str,
    in_scope_files: list[str],
    audit_id: str = "",
) -> list[StaticFinding] | MockRunResult:
    runner = DockerRunner()
    abs_target = str(Path(target_dir).resolve())
    
    report_file_name = "slither_output.json"
    host_report_path = Path(target_dir) / report_file_name

    if host_report_path.exists():
        host_report_path.unlink()

    # Pass --solc solc so Slither does not invoke solc-select
    cmd = [
        "slither", ".",
        "--json", report_file_name,
        "--solc", "solc", 
        "--solc-args", "--base-path /target --allow-paths /target"
    ]

    result = await runner.run(
        image="audit-agent/slither:latest",
        command=cmd,
        workdir="/target",
        volumes={abs_target: {"bind": "/target", "mode": "rw"}},
        timeout=120,
    )

    if isinstance(result, MockRunResult):
        return result

    # exit_code semantics: 0 = no findings, 255 = findings detected (both are normal).
    # Any other code means Slither itself crashed; log a warning but still try to
    # read the output file — it may contain partial results.
    if result.exit_code not in (0, 255):
        log.warning(
            "Slither: unexpected exit code",
            exit_code=result.exit_code,
            stderr=(result.error or "")[:300],
        )

    # Read report from the host-side path where the container wrote it
    data: dict = {"results": {"detectors": []}}
    raw_json = "{}"

    if not host_report_path.exists():
        # Slither crashed before writing a report (e.g. crytic-compile failed).
        # Returning [] here used to hide the failure even in strict mode and left
        # Mapper with an empty graph; Scout decides strict vs lenient handling.
        log.error(
            "Slither: report file not found on host",
            exit_code=result.exit_code,
            stderr=(result.error or "")[:500],
        )
        raise ToolExecutionError(
            tool="slither",
            message=f"Slither produced no report (exit code {result.exit_code})",
            output=(result.output or "")[-2000:],
            stderr=(result.error or "")[-2000:],
        )
    else:
        try:
            raw_json = host_report_path.read_text()
            data = json.loads(raw_json)
            log.info(
                "Slither: loaded report from disk",
                detectors=len(data.get("results", {}).get("detectors", [])),
            )
        except Exception as exc:
            log.error("Slither: failed to parse report file", error=str(exc))

    # Cache raw JSON for Mapper's offline mode
    cache_path = Path(target_dir) / ".slither_analysis.json"
    cache_path.write_text(raw_json)

    # Parse detector entries into StaticFinding objects for Scout
    findings: list[StaticFinding] = []
    for detector in data.get("results", {}).get("detectors", []):
        severity = _SEVERITY_MAP.get(detector.get("impact", ""), "info")
        elements = detector.get("elements", [])
        affected_contract = None
        affected_function = None
        line = None

        for elem in elements:
            if elem.get("type") == "contract":
                affected_contract = elem.get("name")
            elif elem.get("type") == "function":
                affected_function = elem.get("name")
                src = elem.get("source_mapping", {})
                line = src.get("lines", [None])[0] if src.get("lines") else None

        findings.append(
            StaticFinding(
                id=str(uuid.uuid4()),
                audit_id=audit_id,
                tool="slither",
                title=detector.get("check", "unknown"),
                severity=severity,
                vulnerability_class=detector.get("check"),
                description=detector.get("description", ""),
                affected_contract=affected_contract,
                affected_function=affected_function,
                line=line,
                raw=detector,
            )
        )

    log.info("Slither: parsed findings", count=len(findings))
    return findings
