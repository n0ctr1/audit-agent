"""Docker-based sandboxed tool execution."""

from __future__ import annotations

import asyncio
import io
import tarfile
import time
from dataclasses import dataclass
from typing import Any

from audit_agent.observability.logging import get_logger
from audit_agent.tools.errors import (
    MockRunResult,
    ToolTimeoutError,
    ToolUnavailableError,
)

log = get_logger("sandbox.docker")


@dataclass
class RunResult:
    exit_code: int
    output: str
    error: str
    duration_ms: int
    output_file_content: str = ""


# Union type for what DockerRunner.run() can return.
# MockRunResult is returned ONLY in mock mode — never pretends to be a success.
RunResultType = RunResult | MockRunResult


class DockerRunner:
    """Runs tools inside isolated Docker containers.

    Health check happens once in __init__ (not per-call).
    Behaviour depends on AUDIT_AGENT_TOOL_MODE (from settings or explicit override):
      strict  — raise ToolUnavailableError if Docker is unreachable
      lenient — raise ToolUnavailableError but callers are expected to catch + degrade
      mock    — skip Docker entirely, return MockRunResult immediately
    """

    def __init__(self, tool_mode: str | None = None) -> None:
        from audit_agent.config import get_settings

        self._tool_mode = tool_mode or get_settings().tool_mode
        self._client: Any = None
        self._unavailable_error: str | None = None

        if self._tool_mode != "mock":
            self._initialize_client()

    def _initialize_client(self) -> None:
        """Attempt Docker connection once. Store result; raise only in run()."""
        try:
            import docker

            client = docker.from_env()
            client.ping()
            self._client = client
        except Exception as exc:
            self._unavailable_error = str(exc)
            log.debug("Docker: client init failed (will raise on run)", error=str(exc))

    async def run(
        self,
        image: str,
        command: list[str],
        workdir: str = "/work",
        volumes: dict[str, dict[str, str]] | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 120,
        output_file: str | None = None,
        mem_limit: str = "2g",
        cpu_quota: int = 100000,
    ) -> RunResultType:
        # ── mock mode: skip Docker entirely ──────────────────────────────────
        if self._tool_mode == "mock":
            log.info("DockerRunner: mock mode — skipping execution", image=image)
            return MockRunResult()

        # ── Docker unavailable ────────────────────────────────────────────────
        if self._client is None:
            raise ToolUnavailableError(
                tool="docker",
                message=f"Docker daemon unreachable: {self._unavailable_error}",
            )

        log.info("Docker: running", image=image, cmd=" ".join(command[:3]))
        t0 = time.monotonic()
        loop = asyncio.get_event_loop()

        try:
            container = await loop.run_in_executor(
                None,
                lambda: self._client.containers.run(
                    image=image,
                    command=command,
                    working_dir=workdir,
                    volumes=volumes or {},
                    environment=env or {},
                    mem_limit=mem_limit,
                    cpu_quota=cpu_quota,
                    network_disabled=False,
                    remove=False,
                    detach=True,
                ),
            )

            try:
                exit_result = await asyncio.wait_for(
                    loop.run_in_executor(None, container.wait),
                    timeout=float(timeout),
                )
                exit_code: int = exit_result.get("StatusCode", 1)
            except asyncio.TimeoutError:
                await loop.run_in_executor(None, container.kill)
                await loop.run_in_executor(None, container.remove)
                raise ToolTimeoutError(
                    tool=image,
                    message=f"Container exceeded {timeout}s wall-clock limit",
                )

            stdout: str = await loop.run_in_executor(
                None,
                lambda: container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace"),
            )
            stderr: str = await loop.run_in_executor(
                None,
                lambda: container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace"),
            )

            output_file_content = ""
            if output_file:
                try:
                    bits, _ = await loop.run_in_executor(
                        None, lambda: container.get_archive(output_file)
                    )
                    buf = io.BytesIO(b"".join(bits))
                    with tarfile.open(fileobj=buf) as tar:
                        member = tar.getmembers()[0]
                        extracted = tar.extractfile(member)
                        if extracted:
                            output_file_content = extracted.read().decode("utf-8", errors="replace")
                except Exception as exc:
                    log.debug("Docker: could not extract output file", error=str(exc))

            await loop.run_in_executor(None, container.remove)

            duration_ms = int((time.monotonic() - t0) * 1000)
            log.info(
                "Docker: run complete",
                image=image,
                exit_code=exit_code,
                duration_ms=duration_ms,
            )
            return RunResult(
                exit_code=exit_code,
                output=output_file_content or stdout,
                error=stderr,
                duration_ms=duration_ms,
                output_file_content=output_file_content,
            )

        except (ToolTimeoutError, ToolUnavailableError):
            raise
        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            log.error("Docker: unexpected error", image=image, error=str(exc))
            raise ToolUnavailableError(
                tool=image,
                message=f"Unexpected Docker error: {exc}",
            ) from exc
