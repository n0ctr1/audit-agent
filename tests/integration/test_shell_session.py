"""DockerShellSession against a real Docker daemon (skipped without Docker or the image).

Uses a small local image with bash (BASELINE_TEST_IMAGE, default debian:13);
it is never pulled.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

IMAGE = os.environ.get("BASELINE_TEST_IMAGE", "debian:13")


@pytest.fixture
def docker_image():
    try:
        import docker

        client = docker.from_env()
        client.ping()
        client.images.get(IMAGE)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docker or image {IMAGE} unavailable: {exc}")
    return IMAGE


@pytest.mark.integration
async def test_shell_has_no_network_and_cleans_up(docker_image, tmp_path: Path):
    import docker

    from audit_agent.tools.shell_session import DockerShellSession

    s = DockerShellSession(str(tmp_path), image=docker_image, command_timeout=5)
    await s.start()
    cid = s._container.id
    try:
        code, out = await s.run("echo hi > made.txt && cat made.txt")
        assert (code, out.strip()) == (0, "hi")
        _, out = await s.run("(exec 3<>/dev/tcp/1.1.1.1/80 && echo NET_OK) 2>&1 | tail -1")
        assert "NET_OK" not in out
        code, out = await s.run("sleep 10")
        assert code == 124 and "timed out" in out
    finally:
        await s.close()
    assert (tmp_path / "made.txt").read_text().strip() == "hi"
    with pytest.raises(docker.errors.NotFound):
        docker.from_env().containers.get(cid)
