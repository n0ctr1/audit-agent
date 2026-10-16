"""Base-harness selection for PoC generation (exploit/harness_context.py).

Regression: on a test contest the generator picked `ToL1WeiHarness` — a plain helper
contract in a test file — as the base, so PoCs had no `vm`/`assert*` and 9/14 failed
to compile. Only contracts inheriting forge-std `Test` (directly or via parents) qualify.
"""

from __future__ import annotations

from pathlib import Path

from audit_agent.exploit.harness_context import _parents, _pick_base, _test_contracts


def _w(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


HELPER = """
import "forge-std/Test.sol";
contract ToL1WeiHarness {
    function call(uint256 a) external pure returns (uint64) { return uint64(a); }
}
contract BoundCheckTest is Test {
    ToL1WeiHarness h;
    function setUp() public { h = new ToL1WeiHarness(); }
}
"""


def test_parents_strips_constructor_args():
    assert _parents("A, B(1, f(x)), C ") == ["A", "B", "C"]


def test_transitive_test_inheritance(tmp_path: Path):
    files = [
        _w(tmp_path, "test/utils/Adv.sol", "import 'forge-std/Test.sol'; contract AdvTest is Test {}"),
        _w(tmp_path, "test/SessionTestBase.sol", "abstract contract SessionTestBase is AdvTest { function setUp() public {} }"),
        _w(tmp_path, "test/Helpers.t.sol", HELPER),
        _w(tmp_path, "test/Wrapped.sol", "contract FooHarness is Foo {} // contract Fake is Test {}"),
    ]
    bases = _test_contracts(files)
    assert {"AdvTest", "SessionTestBase", "BoundCheckTest"} <= bases
    assert "ToL1WeiHarness" not in bases
    assert "FooHarness" not in bases  # parent is a production contract, not Test
    assert "Fake" not in bases        # declared only inside a comment


def test_helper_harness_never_picked(tmp_path: Path):
    _w(tmp_path, "test/ActionEncoderBoundCheck.t.sol", HELPER)
    assert _pick_base(str(tmp_path), ["Vault.deposit"]) is None


def test_test_base_picked_over_helper(tmp_path: Path):
    _w(tmp_path, "test/ActionEncoderBoundCheck.t.sol", HELPER)
    _w(tmp_path, "test/utils/Adv.sol", "contract AdvTest is Test {}")
    base = _w(tmp_path, "test/VaultTestBase.sol",
              "abstract contract VaultTestBase is AdvTest { function setUp() public virtual {} }")
    _w(tmp_path, "test/SessionSig.t.sol",
       "contract SessionSigHarness { function setUp() public {} }")  # name matches affected contract
    picked = _pick_base(str(tmp_path), ["SessionSig.recover", "Vault.deposit"])
    assert picked == ("VaultTestBase", base)
