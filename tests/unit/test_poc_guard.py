"""Execution-safety guard: generated code must never reach a live network or real keys."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from audit_agent.tools.poc_guard import SAFE_FOUNDRY_ENV, find_unsafe_constructs

SAFE_POC = """
import "forge-std/Test.sol";
contract PoC is Test {
    function setUp() public { vm.createSelectFork(vm.envString("MAINNET_RPC_URL"), 19_000_000); }
    function testExploit() public {
        (address alice, uint256 pk) = makeAddrAndKey("alice");
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(pk, keccak256("x"));
        vm.prank(alice);
        // vm.ffi(cmd) in a comment is fine
        assertTrue(v != 0 && r != 0 && s != 0);
    }
}
"""


@pytest.mark.parametrize("snippet", [
    'vm.ffi(cmd);',
    'vm.rpc("eth_blockNumber", "[]");',
    'bytes memory r = vm.rpc("eth_sendRawTransaction", params);',
    'string memory m = "eth_sendTransaction";',
    'vm.startBroadcast();',
    'vm.broadcast(attacker);',
    'uint256 k = vm.envUint("PRIVATE_KEY");',
    'uint256 k = vm.envOr("DEPLOYER_PRIVATE_KEY", uint256(1));',
    'string memory m = vm.envString("MNEMONIC");',
    'vm.rememberKey(k);',
])
def test_unsafe_constructs_detected(snippet):
    assert find_unsafe_constructs(f"contract X {{ function t() public {{ {snippet} }} }}")


def test_normal_fork_poc_is_allowed():
    assert find_unsafe_constructs(SAFE_POC) == []


async def test_forge_runner_rejects_before_docker(tmp_path: Path, monkeypatch):
    from audit_agent.tools import foundry

    monkeypatch.setenv("AUDIT_AGENT_TOOL_MODE", "strict")
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "PoC.t.sol").write_text('contract P { function t() public { vm.ffi(c); } }')
    with patch.object(foundry.DockerRunner, "run", new=AsyncMock()) as run:
        res = await foundry.run_forge_test(str(tmp_path), match_path="test/PoC.t.sol")
    run.assert_not_called()
    assert res.passed is False
    assert "safety guard" in res.error and "vm.ffi" in res.error


async def test_forge_runner_rejects_without_match_path(tmp_path: Path):
    from audit_agent.tools import foundry

    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "ExploitTest.t.sol").write_text('contract P { function t() public { vm.rpc("eth_sendRawTransaction", p); } }')
    with patch.object(foundry.DockerRunner, "run", new=AsyncMock()) as run:
        res = await foundry.run_forge_test(str(tmp_path))
    run.assert_not_called()
    assert res.passed is False


async def test_forge_runner_forces_ffi_off(tmp_path: Path):
    from audit_agent.tools import foundry
    from audit_agent.tools.errors import MockRunResult

    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "PoC.t.sol").write_text(SAFE_POC)
    with patch.object(foundry.DockerRunner, "run", new=AsyncMock(return_value=MockRunResult())) as run:
        await foundry.run_forge_test(str(tmp_path), match_path="test/PoC.t.sol", profile="mainnet")
    env = run.call_args.kwargs["env"]
    assert env["FOUNDRY_FFI"] == "false" == SAFE_FOUNDRY_ENV["FOUNDRY_FFI"]
    assert env["FOUNDRY_PROFILE"] == "mainnet"


async def test_medusa_and_halmos_reject(tmp_path: Path):
    from audit_agent.tools.halmos import HalmosRunner
    from audit_agent.tools.medusa import MedusaConfig, MedusaRunner

    bad = 'contract P { function property_x() public returns (bool) { vm.ffi(c); return true; } }'
    m = await MedusaRunner(tool_mode="strict").run_campaign(bad, [], MedusaConfig(), tmp_path)
    assert m.status == "error" and "safety guard" in m.raw_output
    h = await HalmosRunner(tool_mode="strict").run_suite(bad, tmp_path)
    assert h.status == "error" and "safety guard" in h.raw_output
    assert not (tmp_path / "test").exists()  # nothing written


def test_no_broadcast_paths_in_source():
    """No code path sends transactions: no `forge script --broadcast`, `cast send`, or raw-tx RPC."""
    src = Path(__file__).resolve().parents[2] / "src" / "audit_agent"
    offenders = []
    for p in src.rglob("*.py"):
        if p.name == "poc_guard.py":
            continue
        text = p.read_text()
        for needle in ("--broadcast", "cast send", "eth_sendRawTransaction", "eth_sendTransaction"):
            if needle in text:
                offenders.append(f"{p.name}: {needle}")
    assert offenders == []
