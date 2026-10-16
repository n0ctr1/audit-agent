"""Medusa fuzzing tool — the project's smart-contract fuzzer.

Medusa (Trail of Bits) is a parallelized, coverage-guided fuzzer. It consumes
property contracts with ``echidna_*`` / ``property_*`` test functions (generated
by ``EchidnaPropertyGenerator``) via crytic-compile.

This module also owns the generic fuzz-result model (``MedusaResult`` /
``BrokenProperty``) used across the exploit and invariant-prover agents.
Injectable for tests, mock-mode aware, degrades to status="error".
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from audit_agent.observability.logging import get_logger
from audit_agent.sandbox.docker_runner import DockerRunner
from audit_agent.tools.errors import MockRunResult, ToolTimeoutError, ToolUnavailableError
from audit_agent.tools.poc_guard import find_unsafe_constructs, rejection_message

log = get_logger("tools.medusa")

_PROP_PREFIX = r"(?:echidna|property)_\w+"


# ─────────────────────────────────────────────────────────────────────────────
# Result model (generic fuzz result — shared by exploit.py / invariant_prover.py)
# ─────────────────────────────────────────────────────────────────────────────

class BrokenProperty(BaseModel):
    name: str
    counterexample: str = ""
    shrunk: bool = False


class MedusaResult(BaseModel):
    status: Literal["properties_broken", "all_passed", "error"]
    broken_properties: list[BrokenProperty] = Field(default_factory=list)
    passed_properties: list[str] = Field(default_factory=list)
    total_tests: int = 0
    duration_seconds: float = 0.0
    raw_output: str = ""


def _parse_property_contract_name(property_contract: str) -> Optional[str]:
    """Extract the first contract name from a generated property .sol file."""
    m = re.search(r"\bcontract\s+(\w+)", property_contract)
    return m.group(1) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class MedusaConfig(BaseModel):
    test_limit: int = 50_000
    seq_len: int = 100
    timeout: int = 120
    workers: int = 4
    corpus_dir: str = "corpus"
    # Property functions Medusa should treat as invariants.
    test_prefixes: list[str] = ["echidna_", "property_"]

    def to_json(self, target_contract: str) -> str:
        """Render a medusa.json. ``target_contract`` is the property contract to
        deploy and fuzz (matches Echidna's ``--contract``)."""
        cfg = {
            "fuzzing": {
                "workers": self.workers,
                "testLimit": self.test_limit,
                "timeout": self.timeout,
                "callSequenceLength": self.seq_len,
                "corpusDirectory": self.corpus_dir,
                "deploymentOrder": [target_contract] if target_contract else [],
                "targetContracts": [target_contract] if target_contract else [],
                "testing": {
                    "propertyTesting": {
                        "enabled": True,
                        "testPrefixes": self.test_prefixes,
                    },
                    "assertionTesting": {"enabled": True},
                    "optimizationTesting": {"enabled": False},
                },
            },
            "compilation": {
                "platform": "crytic-compile",
                "platformConfig": {"target": "."},
            },
        }
        return json.dumps(cfg, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Output parser
# ─────────────────────────────────────────────────────────────────────────────

# Medusa prints forge-style markers and a "failed after N call(s)" line carrying
# the method name in quotes. Be tolerant across versions.
_FAIL_MARKER_RE = re.compile(rf"\[FAILED\][^\n]*?({_PROP_PREFIX})", re.IGNORECASE)
_PASS_MARKER_RE = re.compile(rf"\[PASSED\][^\n]*?({_PROP_PREFIX})", re.IGNORECASE)
_FAILED_AFTER_RE = re.compile(
    rf'(?:method|test)[^\n"]*"[^"]*?({_PROP_PREFIX})[^"]*"\s+failed', re.IGNORECASE
)


def parse_medusa_output(stdout: str, stderr: str = "") -> MedusaResult:
    """Parse Medusa console output into a (generic) fuzz result."""
    combined = (stdout or "") + "\n" + (stderr or "")
    lines = combined.splitlines()

    broken: dict[str, BrokenProperty] = {}
    passed: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        m = _FAIL_MARKER_RE.search(line) or _FAILED_AFTER_RE.search(line)
        if m:
            name = m.group(1)
            # Collect the following call-sequence block as the counterexample.
            ce_lines: list[str] = []
            j = i + 1
            while j < len(lines) and j < i + 20:
                nxt = lines[j].strip()
                if _FAIL_MARKER_RE.search(nxt) or _PASS_MARKER_RE.search(nxt):
                    break
                if nxt:
                    ce_lines.append(nxt)
                j += 1
            broken.setdefault(
                name,
                BrokenProperty(name=name, counterexample="\n".join(ce_lines[:15])),
            )
            i += 1
            continue

        pm = _PASS_MARKER_RE.search(line)
        if pm:
            passed.append(pm.group(1))
        i += 1

    if not broken and not passed:
        return MedusaResult(status="error", raw_output=combined[:3000])

    return MedusaResult(
        status="properties_broken" if broken else "all_passed",
        broken_properties=list(broken.values()),
        passed_properties=passed,
        raw_output=combined[:3000],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

class MedusaRunner:
    """Runs Medusa fuzzing campaigns inside Docker. Injectable for tests."""

    IMAGE = "audit-agent/medusa:latest"

    def __init__(self, tool_mode: Optional[str] = None) -> None:
        self._docker = DockerRunner(tool_mode=tool_mode)

    async def run_campaign(
        self,
        property_contract: str,
        target_contracts: list[Path],
        config: MedusaConfig,
        target_dir: Path,
    ) -> MedusaResult | MockRunResult:
        """Flat workspace: all target .sol files + the property contract +
        medusa.json in one directory, so crytic-compile resolves
        ``import "./Contract.sol"`` regardless of which file it compiles.
        """
        if not isinstance(config, MedusaConfig):
            # Tolerate a duck-typed config from older call sites.
            config = MedusaConfig(
                test_limit=getattr(config, "test_limit", 50_000),
                seq_len=getattr(config, "seq_len", 100),
                timeout=getattr(config, "timeout", 120),
            )

        unsafe = find_unsafe_constructs(property_contract)
        if unsafe:
            log.warning("Medusa: property contract rejected by safety guard", reasons=unsafe)
            return MedusaResult(status="error", raw_output=rejection_message(unsafe))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for tc in target_contracts:
                if tc.exists():
                    shutil.copy(tc, tmp / tc.name)

            (tmp / "MedusaProperties.sol").write_text(property_contract)
            prop_contract_name = _parse_property_contract_name(property_contract) or ""
            (tmp / "medusa.json").write_text(config.to_json(prop_contract_name))

            command = ["medusa", "fuzz", "--config", "/workspace/medusa.json"]

            log.info(
                "Medusa: starting campaign",
                contracts=[tc.name for tc in target_contracts],
                property_contract=prop_contract_name,
                test_limit=config.test_limit,
                timeout=config.timeout,
            )
            t0 = time.monotonic()

            try:
                result = await self._docker.run(
                    image=self.IMAGE,
                    command=command,
                    workdir="/workspace",
                    volumes={str(tmp): {"bind": "/workspace", "mode": "rw"}},
                    timeout=config.timeout + 60,
                )
            except ToolTimeoutError:
                log.warning("Medusa: campaign timed out")
                return MedusaResult(
                    status="error",
                    raw_output=f"Medusa campaign timed out after {config.timeout}s",
                    duration_seconds=round(time.monotonic() - t0, 2),
                )
            except ToolUnavailableError as exc:
                log.warning("Medusa: Docker unavailable", error=str(exc))
                return MedusaResult(status="error", raw_output=str(exc))

            duration = round(time.monotonic() - t0, 2)

            if isinstance(result, MockRunResult):
                log.info("Medusa: mock mode — skipping campaign")
                return result

            if result.exit_code not in (0, 1) and (result.error or result.output):
                log.error(
                    "Medusa: campaign exited with error",
                    exit_code=result.exit_code,
                    stderr=(result.error or "")[:500],
                    stdout=(result.output or "")[:500],
                )

            parsed = parse_medusa_output(result.output or "", result.error or "")
            parsed.duration_seconds = duration
            log.info(
                "Medusa: campaign complete",
                status=parsed.status,
                broken=len(parsed.broken_properties),
                passed=len(parsed.passed_properties),
                duration_s=duration,
            )
            return parsed
