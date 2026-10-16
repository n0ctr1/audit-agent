"""Tool images that compile Foundry projects must ship a working `forge`.

Regression for images whose Foundry install was moved after foundryup, leaving
dangling symlinks (Slither then silently produced nothing on Foundry projects).
Skipped when Docker or an image is unavailable; images are never pulled.
"""

from __future__ import annotations

import pytest

IMAGES = ["audit-agent/slither:latest", "audit-agent/medusa:latest",
          "audit-agent/halmos:latest", "audit-agent/foundry:latest"]


@pytest.mark.integration
@pytest.mark.parametrize("image", IMAGES)
def test_image_has_forge(image):
    try:
        import docker

        client = docker.from_env()
        client.ping()
        client.images.get(image)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docker or {image} unavailable: {exc}")
    out = client.containers.run(image, entrypoint=["bash", "-c"],
                                command=["forge --version"], remove=True)
    assert b"forge Version" in out
