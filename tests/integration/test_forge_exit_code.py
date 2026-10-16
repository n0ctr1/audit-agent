"""run_forge_test reports forge's own exit code (it used to report chmod's)."""

from __future__ import annotations

from pathlib import Path

import pytest

IMAGE = "audit-agent/foundry:latest"

TEST = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "forge-std/Test.sol";
contract T is Test {
    function test_ok() public { assertEq(uint256(1), 1); }
    function test_bad() public { assertEq(uint256(1), 2); }
}
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    try:
        import docker

        c = docker.from_env()
        c.ping()
        c.images.get(IMAGE)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docker or {IMAGE} unavailable: {exc}")
    (tmp_path / "test").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "test" / "T.t.sol").write_text(TEST)
    return tmp_path


@pytest.mark.integration
async def test_exit_code_reflects_forge(project: Path, monkeypatch):
    from audit_agent.tools.foundry import run_forge_test

    monkeypatch.setenv("AUDIT_AGENT_TOOL_MODE", "strict")
    ok = await run_forge_test(str(project), match_path="test/T.t.sol", test_contract="T",
                              timeout=300)
    assert ok.passed is False  # test_bad fails → forge exits non-zero
    assert "[FAIL" in ok.output
