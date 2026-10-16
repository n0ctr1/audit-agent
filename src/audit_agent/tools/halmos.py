"""Halmos symbolic-execution tool wrapper.

Halmos runs Foundry-style `check_*` test functions with SYMBOLIC inputs and an
SMT solver, proving (or finding a counterexample to) properties over ALL inputs
rather than sampling like a fuzzer. This is the right tool for arithmetic
invariants (precision/overflow/rounding/sign) that fuzzing can miss.

Mirrors MedusaRunner: injectable, mock-mode aware, degrades gracefully when the
halmos Docker image is unavailable (returns status="error" rather than raising,
so the pipeline keeps going).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from audit_agent.observability.logging import get_logger
from audit_agent.sandbox.docker_runner import DockerRunner
from audit_agent.tools.errors import MockRunResult, ToolTimeoutError, ToolUnavailableError
from audit_agent.tools.poc_guard import find_unsafe_constructs, rejection_message

log = get_logger("tools.halmos")


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

class HalmosConfig(BaseModel):
    function_prefix: str = "check_"
    contract: Optional[str] = None
    solver_timeout_ms: int = 60_000
    loop_bound: int = 4
    timeout: int = 300


class BrokenSpec(BaseModel):
    name: str
    counterexample: str = ""


class HalmosResult(BaseModel):
    status: Literal["counterexamples_found", "all_proven", "error"]
    broken_specs: list[BrokenSpec] = Field(default_factory=list)
    passed_specs: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    raw_output: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Output parser
# ─────────────────────────────────────────────────────────────────────────────

# Halmos prints forge-style markers: "[PASS] check_foo()" / "[FAIL] check_foo()"
# and a "Counterexample:" block on failure.
_FAIL_RE = re.compile(r"\[(?:FAIL|ERROR)\]\s+(\w+)", re.IGNORECASE)
_PASS_RE = re.compile(r"\[PASS\]\s+(\w+)", re.IGNORECASE)
_COUNTEREX_RE = re.compile(r"Counterexample:\s*(.+)", re.IGNORECASE)


def parse_halmos_output(stdout: str, stderr: str = "") -> HalmosResult:
    combined = (stdout or "") + "\n" + (stderr or "")
    lines = combined.splitlines()

    broken: list[BrokenSpec] = []
    passed: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        fm = _FAIL_RE.search(line)
        if fm:
            name = fm.group(1)
            ce = ""
            # Look a few lines ahead for a counterexample block.
            for j in range(i, min(i + 8, len(lines))):
                cm = _COUNTEREX_RE.search(lines[j])
                if cm:
                    ce = cm.group(1).strip()[:500]
                    break
            broken.append(BrokenSpec(name=name, counterexample=ce))
            i += 1
            continue
        pm = _PASS_RE.search(line)
        if pm:
            passed.append(pm.group(1))
        i += 1

    if not broken and not passed:
        return HalmosResult(status="error", raw_output=combined[:3000])

    return HalmosResult(
        status="counterexamples_found" if broken else "all_proven",
        broken_specs=broken,
        passed_specs=passed,
        raw_output=combined[:3000],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

class HalmosRunner:
    """Runs Halmos symbolic checks inside Docker. Injectable for tests."""

    IMAGE = "audit-agent/halmos:latest"

    def __init__(self, tool_mode: Optional[str] = None) -> None:
        self._docker = DockerRunner(tool_mode=tool_mode)

    async def run_suite(
        self,
        test_contract: str,
        target_dir: Path,
        config: Optional[HalmosConfig] = None,
        test_filename: str = "AuditAgentHalmos.t.sol",
    ) -> HalmosResult | MockRunResult:
        """Write a symbolic test into the Foundry project and run halmos.

        The project must already be a Foundry project (foundry.toml present) so
        halmos can compile it. The test is written into the project's test/ dir
        and removed afterwards.
        """
        config = config or HalmosConfig()
        unsafe = find_unsafe_constructs(test_contract)
        if unsafe:
            log.warning("Halmos: test rejected by safety guard", reasons=unsafe)
            return HalmosResult(status="error", raw_output=rejection_message(unsafe))
        tdir = Path(target_dir)
        test_dir = tdir / "test"
        try:
            test_dir.mkdir(parents=True, exist_ok=True)
            test_path = test_dir / test_filename
            test_path.write_text(test_contract)
        except OSError as exc:
            log.warning("Halmos: cannot write test file", error=str(exc))
            return HalmosResult(status="error", raw_output=str(exc))

        command = ["halmos", "--function", config.function_prefix,
                   "--solver-timeout-assertion", str(config.solver_timeout_ms),
                   "--loop", str(config.loop_bound)]
        if config.contract:
            command += ["--contract", config.contract]

        t0 = time.monotonic()
        try:
            result = await self._docker.run(
                image=self.IMAGE,
                command=command,
                workdir="/project",
                volumes={str(tdir.resolve()): {"bind": "/project", "mode": "rw"}},
                timeout=config.timeout + 30,
            )
        except ToolTimeoutError:
            log.warning("Halmos: timed out")
            return HalmosResult(status="error",
                                raw_output=f"halmos timed out after {config.timeout}s",
                                duration_seconds=round(time.monotonic() - t0, 2))
        except ToolUnavailableError as exc:
            # No halmos image / Docker down → degrade, do not crash the audit.
            log.warning("Halmos: unavailable (degraded)", error=str(exc))
            return HalmosResult(status="error", raw_output=str(exc))
        finally:
            try:
                (test_dir / test_filename).unlink(missing_ok=True)
            except OSError:
                pass

        if isinstance(result, MockRunResult):
            log.info("Halmos: mock mode — skipping")
            return result

        parsed = parse_halmos_output(result.output or "", result.error or "")
        parsed.duration_seconds = round(time.monotonic() - t0, 2)
        log.info("Halmos: complete", status=parsed.status,
                 broken=len(parsed.broken_specs), passed=len(parsed.passed_specs))
        return parsed
