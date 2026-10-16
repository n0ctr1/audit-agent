"""Long-lived, network-less Docker shell for the baseline agent.

The baseline agent works on a *copy* of the target project mounted at
``/project`` inside the Foundry image. The container has networking disabled,
so nothing run inside it can reach a live chain; it is removed afterwards.
"""

from __future__ import annotations

import asyncio
import shlex
from typing import Optional

from audit_agent.observability.logging import get_logger
from audit_agent.tools.errors import ToolUnavailableError

log = get_logger("tools.shell_session")


class DockerShellSession:
    def __init__(
        self,
        project_dir: str,
        image: str = "audit-agent/foundry:latest",
        mem_limit: str = "4g",
        command_timeout: int = 120,
    ) -> None:
        self.project_dir = project_dir
        self.image = image
        self.mem_limit = mem_limit
        self.command_timeout = command_timeout
        self._container = None

    async def start(self) -> None:
        import docker

        loop = asyncio.get_running_loop()
        try:
            client = docker.from_env()
            self._container = await loop.run_in_executor(None, lambda: client.containers.run(
                image=self.image,
                entrypoint=["sleep", "infinity"],
                working_dir="/project",
                volumes={self.project_dir: {"bind": "/project", "mode": "rw"}},
                network_disabled=True,
                mem_limit=self.mem_limit,
                detach=True,
                remove=False,
            ))
        except Exception as exc:
            raise ToolUnavailableError(tool=self.image, message=f"cannot start shell: {exc}") from exc
        # forge-std from the image template, as tools/foundry.py does.
        await self.run("cp -rn /forge-template/lib . 2>/dev/null || true")

    async def run(self, command: str, timeout: Optional[int] = None) -> tuple[int, str]:
        if self._container is None:
            raise RuntimeError("shell session not started")
        t = int(timeout or self.command_timeout)
        argv = ["timeout", str(t), "bash", "-lc", command]
        loop = asyncio.get_running_loop()
        container = self._container
        res = await loop.run_in_executor(
            None, lambda: container.exec_run(argv, workdir="/project", demux=False))
        out = (res.output or b"").decode("utf-8", errors="replace")
        if res.exit_code == 124:
            out += f"\n[command timed out after {t}s]"
        return int(res.exit_code or 0), out

    async def close(self) -> None:
        if self._container is None:
            return
        container, self._container = self._container, None
        loop = asyncio.get_running_loop()
        try:
            # Files created as root inside the container must stay removable on the host.
            await loop.run_in_executor(None, lambda: container.exec_run(
                ["bash", "-lc", "chmod -R a+rwX /project"], workdir="/project"))
        except Exception:  # noqa: BLE001
            pass
        try:
            await loop.run_in_executor(None, lambda: container.remove(force=True))
        except Exception as exc:  # noqa: BLE001
            log.warning("shell_session.remove_failed", error=str(exc))


class MockShellSession:
    """Mock-mode / test shell: records commands, executes nothing."""

    def __init__(self, project_dir: str = "", responses: Optional[dict[str, str]] = None) -> None:
        self.project_dir = project_dir
        self.commands: list[str] = []
        self._responses = responses or {}

    async def start(self) -> None:
        return None

    async def run(self, command: str, timeout: Optional[int] = None) -> tuple[int, str]:
        self.commands.append(command)
        for prefix, out in self._responses.items():
            if command.startswith(prefix):
                return 0, out
        return 0, f"[mock mode: not executed] {shlex.quote(command)[:200]}"

    async def close(self) -> None:
        return None
