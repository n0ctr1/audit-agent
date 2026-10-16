"""Unit tests for attack_surface_score() — pure function, fully offline."""

from __future__ import annotations

from audit_agent.agents.scoring import ExternalCallInfo, ScoredFunction, attack_surface_score


def _fn(**kwargs) -> ScoredFunction:
    defaults = dict(
        id="test.fn()",
        name="fn",
        visibility="external",
        state_mutability="nonpayable",
        modifiers=[],
        external_calls=[],
        writes_state_vars=[],
        handles_value=False,
        reads_oracle=False,
        is_privileged=False,
        is_modified_in_fork=False,
        is_new_in_fork=False,
    )
    defaults.update(kwargs)
    return ScoredFunction(**defaults)


def _ext_call(call_type="call", target_type="user_controlled") -> ExternalCallInfo:
    return ExternalCallInfo(id="ec", call_type=call_type, target_type=target_type)


# ── Accessibility ──────────────────────────────────────────────────────────────

def test_external_visibility_adds_3():
    fn = _fn(visibility="external")
    assert attack_surface_score(fn) == 3.0


def test_public_visibility_adds_2():
    fn = _fn(visibility="public")
    assert attack_surface_score(fn) == 2.0


def test_internal_visibility_zero():
    fn = _fn(visibility="internal")
    assert attack_surface_score(fn) == 0.0


def test_private_visibility_zero():
    fn = _fn(visibility="private")
    assert attack_surface_score(fn) == 0.0


# ── Value handling ─────────────────────────────────────────────────────────────

def test_handles_value_adds_4():
    fn = _fn(visibility="internal", handles_value=True)
    assert attack_surface_score(fn) == 4.0


def test_payable_adds_2():
    fn = _fn(visibility="internal", state_mutability="payable")
    assert attack_surface_score(fn) == 2.0


def test_payable_and_handles_value():
    fn = _fn(visibility="external", state_mutability="payable", handles_value=True)
    # external=3, payable=2, handles_value=4 = 9
    assert attack_surface_score(fn) == 9.0


# ── External calls ─────────────────────────────────────────────────────────────

def test_one_external_call_adds_2_plus_3():
    fn = _fn(
        visibility="internal",
        external_calls=[_ext_call(call_type="call", target_type="user_controlled")],
    )
    # 1 call × 2 + user_controlled × 3 + no guard = 2 + 3 + 6 = 11
    assert attack_surface_score(fn) == 11.0


def test_delegatecall_adds_5():
    fn = _fn(
        visibility="internal",
        external_calls=[_ext_call(call_type="delegatecall", target_type="known_contract")],
    )
    # 1 call × 2 + delegatecall × 5 + no guard = 2 + 5 + 6 = 13
    assert attack_surface_score(fn) == 13.0


def test_reentrancy_guard_removes_plus6():
    fn = _fn(
        visibility="external",
        external_calls=[_ext_call()],
        modifiers=["nonReentrant"],
    )
    # external=3, 1 call×2=2, user_controlled×3=3, guard present → no +6
    assert attack_surface_score(fn) == 8.0  # 3 + 2 + 3 = 8


def test_no_guard_adds_6():
    fn = _fn(
        visibility="external",
        external_calls=[_ext_call()],
        modifiers=[],
    )
    # external=3, 1 call×2=2, user_controlled=3, no guard=+6 → 14
    assert attack_surface_score(fn) == 14.0


# ── Oracle and privilege ────────────────────────────────────────────────────────

def test_oracle_read_adds_5():
    fn = _fn(visibility="internal", reads_oracle=True)
    assert attack_surface_score(fn) == 5.0


def test_privileged_adds_3():
    fn = _fn(visibility="internal", is_privileged=True)
    assert attack_surface_score(fn) == 3.0


# ── State variables ────────────────────────────────────────────────────────────

def test_one_write_adds_1_5():
    fn = _fn(visibility="internal", writes_state_vars=["balances"])
    assert attack_surface_score(fn) == 1.5


def test_two_writes_adds_3():
    fn = _fn(visibility="internal", writes_state_vars=["balances", "totalSupply"])
    assert attack_surface_score(fn) == 3.0


# ── Fork multipliers ────────────────────────────────────────────────────────────

def test_fork_modified_doubles():
    base = _fn(visibility="external")
    base_score = attack_surface_score(base)
    modified = _fn(visibility="external", is_modified_in_fork=True)
    assert attack_surface_score(modified) == base_score * 2.0


def test_fork_new_triples():
    base = _fn(visibility="external")
    base_score = attack_surface_score(base)
    new_fn = _fn(visibility="external", is_new_in_fork=True)
    assert attack_surface_score(new_fn) == base_score * 3.0


def test_fork_modified_and_new_only_new_applies():
    # Both flags: only is_new_in_fork=True multiplier applies (it's checked second)
    fn = _fn(visibility="external", is_modified_in_fork=True, is_new_in_fork=True)
    base = attack_surface_score(_fn(visibility="external"))
    # 2.0 × 3.0 = 6.0 total multiplier (both apply sequentially)
    assert attack_surface_score(fn) == base * 6.0


# ── VulnerableBank.withdraw() expected score ──────────────────────────────────

def test_vulnerable_bank_withdraw_score():
    """
    From fixture VulnerableBank.withdraw(uint256):
    - visibility=external: +3
    - 1 external call (call, user_controlled): +2 (per call) + 3 (user_controlled)
    - writes_state_vars=["balances"]: +1.5
    - no nonReentrant modifier: +6
    Total: 3 + 2 + 3 + 1.5 + 6 = 15.5
    """
    fn = _fn(
        id="vulnerablebank.withdraw(uint256)",
        name="withdraw",
        visibility="external",
        state_mutability="nonpayable",
        modifiers=[],
        external_calls=[_ext_call(call_type="call", target_type="user_controlled")],
        writes_state_vars=["vulnerablebank.balances"],
        handles_value=False,
        reads_oracle=False,
        is_privileged=False,
    )
    assert attack_surface_score(fn) == 15.5


def test_vulnerable_bank_emergency_drain_score():
    """
    emergencyDrain():
    - visibility=external: +3
    - 1 external call (transfer, user_controlled): +2 + 3
    - is_privileged=True: +3
    - no guard: +6
    Total: 3 + 2 + 3 + 3 + 6 = 17
    """
    fn = _fn(
        id="vulnerablebank.emergencyDrain()",
        name="emergencyDrain",
        visibility="external",
        state_mutability="nonpayable",
        modifiers=[],
        external_calls=[_ext_call(call_type="transfer", target_type="user_controlled")],
        writes_state_vars=[],
        is_privileged=True,
    )
    assert attack_surface_score(fn) == 17.0


def test_view_function_has_low_score():
    fn = _fn(
        visibility="public",
        state_mutability="view",
        external_calls=[],
        writes_state_vars=[],
    )
    assert attack_surface_score(fn) <= 2.0


def test_pure_function_zero():
    fn = _fn(visibility="public", state_mutability="pure")
    assert attack_surface_score(fn) == 2.0  # only public visibility points


# ── Precision loss surface ─────────────────────────────────────────────────────

def test_scoring_fee_functions_boosted():
    """Functions named with fee/rate keywords get +3 bonus."""
    base = attack_surface_score(_fn(name="doSomething"))
    for name in ("collectFee", "setFeeRate", "calcReward", "getYield", "updateInterest", "sharePrice"):
        fn = _fn(name=name)
        assert attack_surface_score(fn) > base, f"expected boost for {name}"


def test_scoring_division_boosted():
    """has_division=True adds +2 to the score."""
    without = attack_surface_score(_fn(name="fn", has_division=False))
    with_div = attack_surface_score(_fn(name="fn", has_division=True))
    assert with_div == without + 2.0


def test_scoring_multiplication_boosted():
    """has_multiplication=True adds +2 to the score."""
    without = attack_surface_score(_fn(name="fn", has_multiplication=False))
    with_mul = attack_surface_score(_fn(name="fn", has_multiplication=True))
    assert with_mul == without + 2.0


def test_scoring_fee_function_with_division_stacks():
    """Fee keyword + has_division both apply (total +5 above baseline)."""
    base = attack_surface_score(_fn(name="fn", has_division=False))
    boosted = attack_surface_score(_fn(name="collectFee", has_division=True))
    assert boosted == base + 3.0 + 2.0


def test_scoring_collect_fee_beats_plain_view():
    """collectFee() should score higher than a plain view function."""
    fee_fn = _fn(
        name="collectFee",
        visibility="external",
        state_mutability="nonpayable",
        writes_state_vars=["feesCollected"],
        has_division=True,
    )
    plain = _fn(visibility="external", state_mutability="view")
    assert attack_surface_score(fee_fn) > attack_surface_score(plain)
