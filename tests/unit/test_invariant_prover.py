"""Tests for the protocol-wide invariant-prover stage (roadmap #4).

Covers the Halmos output parser, the deterministic Halmos spec generator, and
the InvariantProverAgent with injected mock runners (offline, no Docker/LLM).
"""

from __future__ import annotations

from audit_agent.agents.invariant_prover import InvariantProverAgent
from audit_agent.exploit.halmos_gen import HalmosSpecGenerator, generate_halmos_spec
from audit_agent.models.schemas import Hypothesis, Invariant
from audit_agent.tools.medusa import BrokenProperty, MedusaResult as EchidnaResult
from audit_agent.tools.errors import MockRunResult
from audit_agent.tools.halmos import BrokenSpec, HalmosResult, parse_halmos_output


# ── Halmos parser ─────────────────────────────────────────────────────────────

def test_parse_halmos_pass_and_fail():
    out = (
        "[PASS] check_fee_nonzero(uint256)\n"
        "[FAIL] check_share_conservation(uint256)\n"
        "Counterexample: amount = 1, total = 0\n"
    )
    r = parse_halmos_output(out)
    assert r.status == "counterexamples_found"
    assert r.passed_specs == ["check_fee_nonzero"]
    assert len(r.broken_specs) == 1
    assert r.broken_specs[0].name == "check_share_conservation"
    assert "amount = 1" in r.broken_specs[0].counterexample


def test_parse_halmos_all_proven():
    r = parse_halmos_output("[PASS] check_a()\n[PASS] check_b()\n")
    assert r.status == "all_proven"
    assert set(r.passed_specs) == {"check_a", "check_b"}


def test_parse_halmos_unrecognized_is_error():
    assert parse_halmos_output("compilation failed").status == "error"


# ── Halmos spec generator ─────────────────────────────────────────────────────

def _arith_inv() -> Invariant:
    return Invariant(
        audit_id="a", invariant_type="economic",
        natural_language="Fee calculation must never truncate to zero (precision loss)",
        affected_functions=["VulnerableBank.calculateFee(uint256)"],
        source="inferred",
    )


def test_generate_halmos_spec_for_arithmetic():
    spec = generate_halmos_spec(_arith_inv())
    assert spec is not None
    assert "function check_" in spec
    assert "target.calculateFee(amount)" in spec
    assert "assert(result > 0)" in spec


def test_generate_halmos_spec_skips_non_arithmetic():
    inv = Invariant(audit_id="a", invariant_type="global",
                    natural_language="Only the owner may pause the contract",
                    affected_functions=["C.pause()"], source="inferred")
    assert generate_halmos_spec(inv) is None


def test_suite_revives_halmos_spec_field():
    inv = _arith_inv()
    assert inv.halmos_spec is None
    suite, mapping = HalmosSpecGenerator().generate_suite(
        [inv], "contract VulnerableBank { constructor(); }", "../src/VulnerableBank.sol"
    )
    assert "contract AuditAgentHalmos is Test" in suite
    assert len(mapping) == 1
    # The previously-unused field is now populated.
    assert inv.halmos_spec is not None and "check_" in inv.halmos_spec


# ── Agent: injected runners ────────────────────────────────────────────────────

class _FakeEchidna:
    def __init__(self, result):
        self._r = result
        self.called = False

    async def run_campaign(self, **kwargs):
        self.called = True
        return self._r


class _FakeHalmos:
    def __init__(self, result):
        self._r = result

    async def run_suite(self, **kwargs):
        return self._r


def _state(invariants, hypotheses=None):
    return {
        "audit_id": "audit-1",
        "invariants": invariants,
        "hypotheses": hypotheses or [],
        "target_dir": "",
        "scope_in_scope": [],
        "fuzz_enabled": True,
    }


async def test_no_invariants_is_noop():
    agent = InvariantProverAgent(
        echidna_runner=_FakeEchidna(EchidnaResult(status="all_passed")),
        halmos_runner=_FakeHalmos(HalmosResult(status="all_proven")),
    )
    out = await agent.run(_state([], hypotheses=[]))
    assert out["hypotheses"] == []


async def test_echidna_violation_becomes_hypothesis():
    inv = _arith_inv()
    from audit_agent.exploit.echidna_gen import _safe_identifier
    prop_name = f"echidna_{_safe_identifier(inv.natural_language)}"
    echidna = _FakeEchidna(EchidnaResult(
        status="properties_broken",
        broken_properties=[BrokenProperty(name=prop_name, counterexample="seq: deposit(1); withdraw()")],
    ))
    agent = InvariantProverAgent(
        echidna_runner=echidna,
        halmos_runner=_FakeHalmos(MockRunResult()),  # halmos no-op
    )
    out = await agent.run(_state([inv]))
    assert echidna.called
    hyps = out["hypotheses"]
    assert len(hyps) == 1
    h = hyps[0]
    assert h.vulnerability_class == "Invariant Violation"
    assert h.priority_score >= 88.0
    assert "truncate to zero" in h.title or "Fee calculation" in h.title
    assert "deposit(1)" in h.attack_scenario  # counterexample carried through


async def test_halmos_counterexample_becomes_hypothesis():
    inv = _arith_inv()
    halmos = _FakeHalmos(HalmosResult(
        status="counterexamples_found",
        broken_specs=[BrokenSpec(name="check_calculatefee_fee_calculation_must_neve",
                                 counterexample="amount = 1")],
    ))
    # Echidna passes → only halmos contributes.
    agent = InvariantProverAgent(
        echidna_runner=_FakeEchidna(EchidnaResult(status="all_passed")),
        halmos_runner=halmos,
    )
    out = await agent.run(_state([inv]))
    hyps = out["hypotheses"]
    assert len(hyps) == 1
    assert hyps[0].vulnerability_class == "Invariant Violation"
    # halmos_spec field was populated during suite generation
    assert inv.halmos_spec is not None


async def test_mock_mode_is_graceful():
    agent = InvariantProverAgent(
        echidna_runner=_FakeEchidna(MockRunResult()),
        halmos_runner=_FakeHalmos(MockRunResult()),
    )
    out = await agent.run(_state([_arith_inv()]))
    assert out["hypotheses"] == []


async def test_existing_hypotheses_preserved():
    existing = Hypothesis(id="h0", audit_id="audit-1", title="pre-existing",
                          attack_scenario="x", estimated_severity="medium")
    echidna = _FakeEchidna(EchidnaResult(
        status="properties_broken",
        broken_properties=[BrokenProperty(name="echidna_x")],
    ))
    agent = InvariantProverAgent(
        echidna_runner=echidna, halmos_runner=_FakeHalmos(MockRunResult()),
    )
    out = await agent.run(_state([_arith_inv()], hypotheses=[existing]))
    titles = [h.title for h in out["hypotheses"]]
    assert "pre-existing" in titles
    assert len(out["hypotheses"]) == 2  # original + 1 violation


async def test_fuzz_disabled_skips_echidna():
    inv = _arith_inv()
    echidna = _FakeEchidna(EchidnaResult(
        status="properties_broken",
        broken_properties=[BrokenProperty(name="echidna_x")],
    ))
    agent = InvariantProverAgent(
        echidna_runner=echidna, halmos_runner=_FakeHalmos(HalmosResult(status="all_proven")),
    )
    state = _state([inv])
    state["fuzz_enabled"] = False
    out = await agent.run(state)
    assert not echidna.called
    assert out["hypotheses"] == []
