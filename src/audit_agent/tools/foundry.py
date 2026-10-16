"""Foundry tool wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from audit_agent.observability.logging import get_logger
from audit_agent.sandbox.docker_runner import DockerRunner
from audit_agent.tools.errors import MockRunResult, ToolExecutionError
from audit_agent.tools.poc_guard import SAFE_FOUNDRY_ENV, rejection_message, scan_files

log = get_logger("tools.foundry")


@dataclass
class ForgeTestResult:
    passed: bool
    output: str
    error: str
    duration_ms: int


async def run_forge_test(
    project_dir: str,
    test_contract: str | None = None,
    match_path: str | None = None,
    timeout: int = 120,
    profile: str | None = None,
    verbosity: int = 2,
) -> ForgeTestResult | MockRunResult:
    """Run `forge test` in Docker.

    profile: when set, runs under FOUNDRY_PROFILE=<profile> so a project's
    mainnet-fork profile (eth_rpc_url, remappings) is used — required for fork
    PoCs that inherit the project's mainnet test harness.
    verbosity: number of -v flags (2 for the pipeline; 4 = call traces, used by
    ``benchmark retrace``).
    """
    runner = DockerRunner()
    abs_dir = str(Path(project_dir).resolve())

    # Safety guard: refuse generated tests that could reach a live network or
    # load real keys. With match_path only that file runs; otherwise scan the
    # test sources of the (temporary) project.
    root = Path(abs_dir)
    if match_path:
        to_scan = [root / match_path]
    else:
        to_scan = [p for p in root.rglob("*.sol") if "lib" not in p.relative_to(root).parts]
    violations = scan_files(to_scan)
    if violations:
        msg = rejection_message(violations)
        log.warning("Foundry: PoC rejected by safety guard", violations=violations)
        return ForgeTestResult(passed=False, output="", error=msg, duration_ms=0)

    # Use the forge-std pre-installed in the image (no network needed at runtime).
    # If the project already has lib/ (e.g. panoptic-v1.1 submodule), the -n flag
    # makes the cp a no-op so existing deps are preserved.
    # Plain text output — trace_parser expects [PASS]/[FAIL] markers, not JSON
    forge_test_cmd = "forge test -" + "v" * max(1, min(verbosity, 5))
    if test_contract:
        forge_test_cmd += f" --match-contract {test_contract}"
    if match_path:
        # Shell-quote the path to handle any special characters
        safe_path = match_path.replace("'", "'\\''")
        forge_test_cmd += f" --match-path '{safe_path}'"

    bash_cmd = (
        # Create minimal foundry.toml if missing (forge test needs it)
        "[ -f foundry.toml ] || cat > foundry.toml << 'EOF'\n"
        "[profile.default]\nsrc = \"src\"\nout = \"out\"\nlibs = [\"lib\"]\n"
        "EOF\n"
        # Copy pre-installed forge-std from image template only if not already present
        "cp -rn /forge-template/lib . 2>/dev/null || true; "
        # Create out/ dir so forge doesn't complain
        "mkdir -p out src; "
        f"{forge_test_cmd}; rc=$?; "
        # Fix permissions so the host process can clean up /tmp; keep forge's exit code
        "chmod -R 777 .; exit $rc"
    )

    # ENTRYPOINT is ["/bin/bash", "-c"] — pass the script as a single argument
    cmd = [bash_cmd]

    env = dict(SAFE_FOUNDRY_ENV)
    if profile:
        env["FOUNDRY_PROFILE"] = profile

    result = await runner.run(
        image="audit-agent/foundry:latest",
        command=cmd,
        workdir="/project",
        volumes={abs_dir: {"bind": "/project", "mode": "rw"}},
        env=env,
        timeout=timeout,
    )

    if isinstance(result, MockRunResult):
        log.info("Foundry: mock mode — skipping test execution")
        return result

    # Non-zero exit (including exit 2 for compile errors) sets passed=False so
    # ExploitAgent can read stderr and retry with a corrected PoC.
    passed = result.exit_code == 0
    
    log.info("Foundry: test run complete", passed=passed, exit_code=result.exit_code)
    
    return ForgeTestResult(
        passed=passed,
        output=result.output or "",
        error=result.error or "",
        duration_ms=result.duration_ms,
    )
