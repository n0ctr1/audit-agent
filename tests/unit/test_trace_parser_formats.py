"""parse_forge_output on real forge 1.x output lines (FAIL format changed in forge 1.0)."""

from __future__ import annotations

import pytest

from audit_agent.agents.exploit import _map_status
from audit_agent.exploit.trace_parser import parse_forge_output

FORGE_1X_FAIL = """No files changed, compilation skipped

Ran 1 test for test/PoC.t.sol:PoC
[FAIL: revert: Ownable: caller is not the owner] test_drain() (gas: 23456)
Suite result: FAILED. 0 passed; 1 failed; 0 skipped; finished in 1.2ms
"""


@pytest.mark.parametrize("line,reason,name", [
    ("[FAIL: revert: Ownable: caller is not the owner] test_drain() (gas: 23456)",
     "revert: Ownable: caller is not the owner", "test_drain()"),
    ("[FAIL: assertion failed: 1 != 2] test_y() (gas: 99)", "assertion failed: 1 != 2", "test_y()"),
    ("[FAIL: panic: arithmetic underflow (0x11); counterexample: calldata=0x12 args=[1]] testFuzz(uint256) (runs: 3, μ: 1, ~: 1)",
     "panic: arithmetic underflow (0x11); counterexample: calldata=0x12 args=[1]", "testFuzz(uint256)"),
    ("[FAIL. Reason: revert: paused] test_old() (gas: 1)", "revert: paused", "test_old()"),
    ("[FAIL] test_plain() (gas: 1)", None, "test_plain()"),
])
def test_fail_formats(line, reason, name):
    p = parse_forge_output(line, "")
    assert p.passed is False and not p.is_compile_error
    assert p.test_name == name
    if reason is not None:
        assert p.revert_reason == reason
    assert p.revert_reason != "No test output detected"


def test_guard_revert_maps_to_blocked_with_forge_1x():
    p = parse_forge_output(FORGE_1X_FAIL.replace("Ownable: caller is not the owner", "Pausable: paused"), "")
    assert _map_status(p) == "blocked"


def test_real_fork_failure_line_is_partial():
    line = ("[FAIL: EVM error; database error: failed to get account for 0x9522: server returned "
            "an error response: error code 4444: pruned history unavailable] test_fork_reads_mainnet() (gas: 0)")
    p = parse_forge_output(line, "")
    assert p.revert_reason.startswith("EVM error")
    assert _map_status(p) == "partial"


def test_pass_still_exploited():
    p = parse_forge_output("[PASS] test_reentrancy() (gas: 421071)", "")
    assert p.passed and _map_status(p) == "exploited"
