"""Tests for the benchmark matcher (benchmarks/matching.py) and taxonomy."""

from __future__ import annotations

import itertools

import pytest
from benchmarks.matching import (
    TITLE_JACCARD_MIN,
    FunctionRef,
    GroundTruthFinding,
    evaluate_pair,
    finding_matches,
    match_hypotheses_to_ground_truth,
    match_one_to_one,
    parse_function_ref,
    title_jaccard,
)
from benchmarks.taxonomy import canonical_classes, class_compatible, class_relation

from audit_agent.models.schemas import Hypothesis


def _hyp(title: str, vc: str | None, fns: list[str], hid: str | None = None) -> Hypothesis:
    kwargs = {"id": hid} if hid else {}
    return Hypothesis(
        audit_id="t",
        title=title,
        vulnerability_class=vc,
        affected_functions=fns,
        attack_scenario="x",
        estimated_severity="high",
        **kwargs,
    )


def _gt(title: str, vc: str, fns: list[str], gid: str = "H-01") -> GroundTruthFinding:
    return GroundTruthFinding(id=gid, title=title, severity="high", vulnerability_class=vc,
                              affected_functions=fns)


# ── function identifier parsing ──────────────────────────────────────────────

class TestParseFunctionRef:
    @pytest.mark.parametrize("raw,expected", [
        ("withdraw", FunctionRef(None, "withdraw")),
        ("Vault.withdraw(uint256)", FunctionRef("vault", "withdraw")),
        ("Vault::withdraw", FunctionRef("vault", "withdraw")),
        ("Vault#withdraw", FunctionRef("vault", "withdraw")),
        ("src/core/Vault.sol:Vault.withdraw(uint256,address)", FunctionRef("vault", "withdraw")),
        ("Vault.sol:withdraw", FunctionRef("vault", "withdraw")),  # file name taken as contract
        ("AtomWallet._validateSignature", FunctionRef("atomwallet", "validatesignature")),
        ("function _validate_sig(bytes)", FunctionRef(None, "validate_sig")),
    ])
    def test_parse(self, raw, expected):
        assert parse_function_ref(raw) == expected

    def test_empty(self):
        assert parse_function_ref("") is None
        assert parse_function_ref("()") is None

    def test_underscores_do_not_split_function_names(self):
        # Old matcher split "_validate_sig" into {"validate", "sig"} tokens.
        g = _gt("Sig bug", "signature", ["Wallet._validate_sig"])
        h = _hyp("Sig bug", "signature", ["Wallet.validate"])
        assert evaluate_pair(h, g).function_match is False


# ── taxonomy ─────────────────────────────────────────────────────────────────

class TestTaxonomy:
    @pytest.mark.parametrize("label,cls", [
        ("SWC-107", "reentrancy"),
        ("swc107", "reentrancy"),
        ("Cross-function Reentrancy", "reentrancy"),
        ("Missing access control", "access-control"),
        ("SWC-115", "access-control"),
        ("Signature Replay", "signature-replay"),
        ("Unsigned Validity Window", "signature-replay"),
        ("SWC-121", "signature-replay"),
        ("oracle-manipulation", "oracle-manipulation"),
        ("Precision loss / rounding", "rounding-precision"),
        ("SWC-101", "arithmetic"),
        ("Integer Underflow", "arithmetic"),
        ("denial-of-service", "dos"),
        ("Epoch boundary off-by-one", "timestamp-epoch"),
        ("Unprotected initializer", "upgradeability"),
        ("Front-running", "front-running"),
        ("SWC-120", "randomness"),
        ("arbitrary-call", "arbitrary-call"),
        ("flash-loan", "flash-loan"),
    ])
    def test_canonical(self, label, cls):
        assert cls in canonical_classes(label)

    def test_unsigned_integer_is_not_signature(self):
        assert "signature-replay" not in canonical_classes("Unsigned integer underflow")

    def test_unknown_swc_code_maps_to_nothing(self):
        # SWC-207 does not exist in the SWC registry.
        assert canonical_classes("SWC-207") == frozenset()

    def test_relations(self):
        assert class_relation("SWC-107", "reentrancy") == "exact"
        assert class_relation("Flash loan", "Oracle manipulation") == "related"
        assert class_relation("SWC-107", "SWC-115") == "none"
        assert class_relation("Weird Custom Label", "weird custom label") == "exact"
        assert class_relation(None, "reentrancy") == "none"
        assert class_compatible("Signature replay", "Missing access control")


# ── acceptance cases from the spec ───────────────────────────────────────────

class TestMatchRule:
    def test_same_function_compatible_classes_match(self):
        g = _gt("Signature metadata not validated", "Unsigned Validity Window",
                ["AtomWallet._validateSignature"])
        h = _hyp("Replay of user operations", "Signature Replay", ["AtomWallet.validateSignature"])
        res = evaluate_pair(h, g)
        assert res.class_compatible and res.function_match
        assert res.is_match

    def test_same_function_incompatible_class_similar_title_match(self):
        g = _gt("Epoch end timestamp overlaps next epoch start", "logic-error",
                ["TrustBonding.epochTimestampEnd"])
        h = _hyp("epochTimestampEnd overlaps next epoch start", "SWC-101",
                 ["TrustBonding.epochTimestampEnd"])
        res = evaluate_pair(h, g)
        assert res.class_relation == "none"
        assert res.title_jaccard >= TITLE_JACCARD_MIN
        assert res.is_match

    def test_same_function_incompatible_class_unrelated_title_no_match(self):
        g = _gt("Reentrancy drains vault", "SWC-107", ["Vault.withdraw"])
        h = _hyp("Missing event emission", "events", ["Vault.withdraw"])
        assert finding_matches(h, g) is False

    def test_same_contract_different_functions_no_match(self):
        g = _gt("Rewards can be claimed twice", "logic", ["TrustBonding.claimRewards"])
        h = _hyp("Rewards can be claimed twice", "logic", ["TrustBonding.setRewards"])
        res = evaluate_pair(h, g)
        assert res.function_match is False
        assert res.is_match is False

    def test_exact_class_and_shared_title_word_without_function_no_match(self):
        g = _gt("Reentrancy in withdraw", "SWC-107", ["Vault.withdraw"])
        h = _hyp("Reentrancy in deposit", "SWC-107", ["Vault.deposit"])
        assert finding_matches(h, g) is False

    def test_contract_compared_only_when_both_sides_name_it(self):
        g = _gt("Reentrancy", "SWC-107", ["Vault.withdraw"])
        assert finding_matches(_hyp("Reentrancy", "SWC-107", ["withdraw"]), g)
        assert not finding_matches(_hyp("Reentrancy", "SWC-107", ["Pool.withdraw"]), g)

    def test_case_and_leading_underscore_ignored(self):
        g = _gt("Reentrancy", "SWC-107", ["Vault._Withdraw"])
        assert finding_matches(_hyp("Reentrancy", "SWC-107", ["vault.withdraw()"]), g)

    def test_no_functions_never_matches(self):
        g = _gt("Reentrancy", "SWC-107", ["Vault.withdraw"])
        assert not finding_matches(_hyp("Reentrancy", "SWC-107", []), g)
        g2 = _gt("Reentrancy", "SWC-107", [])
        assert not finding_matches(_hyp("Reentrancy", "SWC-107", ["Vault.withdraw"]), g2)

    def test_score_reported_for_non_matches(self):
        g = _gt("Reentrancy", "SWC-107", ["Vault.withdraw"])
        res = evaluate_pair(_hyp("Reentrancy", "SWC-107", ["Vault.deposit"]), g)
        assert not res.is_match
        assert res.score > 0


class TestTitleJaccard:
    def test_identical(self):
        assert title_jaccard("Reentrancy in withdraw", "reentrancy in withdraw") == 1.0

    def test_empty(self):
        assert title_jaccard("", "x") == 0.0


# ── one-to-one assignment ────────────────────────────────────────────────────

class TestOneToOne:
    def test_two_hypotheses_one_gt_closes_exactly_one(self):
        g = [_gt("Reentrancy in withdraw drains vault", "SWC-107", ["Vault.withdraw"])]
        weak = _hyp("Withdraw issue", "logic", ["Vault.withdraw", "Vault.deposit"], hid="weak")
        weak.title = "Reentrancy issue"  # compatible via title only
        strong = _hyp("Reentrancy in withdraw drains vault", "reentrancy", ["Vault.withdraw"],
                      hid="strong")
        for order in ([weak, strong], [strong, weak]):
            res = match_one_to_one(order, g)
            assert len(res.assignments) == 1
            assert res.assignments[0].candidate_id == "strong"
            assert res.assignments[0].gt_id == "H-01"

    def test_result_independent_of_hypothesis_order(self):
        gts = [
            _gt("Reentrancy in withdraw", "SWC-107", ["Vault.withdraw"], "H-01"),
            _gt("Oracle price manipulation", "oracle", ["Oracle.getPrice", "Pool.borrow"], "H-02"),
            _gt("Missing access control on setFee", "access-control", ["Pool.setFee"], "M-01"),
        ]
        hyps = [
            _hyp("Reentrancy", "reentrancy", ["Vault.withdraw"], "a"),
            _hyp("Withdraw CEI violation", "SWC-107", ["withdraw"], "b"),
            _hyp("Spot price used for borrow", "price manipulation", ["Pool.borrow"], "c"),
            _hyp("Anyone can set fee", "unauthorized", ["Pool.setFee"], "d"),
            _hyp("Unrelated", "gas", ["Pool.other"], "e"),
        ]
        baseline = None
        for perm in itertools.permutations(hyps):
            res = match_one_to_one(list(perm), gts)
            got = sorted((a.gt_id, a.candidate_id) for a in res.assignments)
            if baseline is None:
                baseline = got
            assert got == baseline
        assert {g for g, _ in baseline} == {"H-01", "H-02", "M-01"}

    def test_early_weak_hypothesis_does_not_steal_gt(self):
        # The weak hypothesis matches both GTs; the strong one matches only H-01.
        gts = [
            _gt("Reentrancy in withdraw", "SWC-107", ["Vault.withdraw"], "H-01"),
            _gt("Withdraw fee rounding", "rounding", ["Vault.withdraw", "Vault.fee"], "M-01"),
        ]
        weak = _hyp("Withdraw rounding", "rounding", ["Vault.withdraw", "Vault.fee"], "weak")
        strong = _hyp("Reentrancy in withdraw", "SWC-107", ["Vault.withdraw"], "strong")
        res = match_one_to_one([weak, strong], gts)
        got = {a.gt_id: a.candidate_id for a in res.assignments}
        assert got == {"H-01": "strong", "M-01": "weak"}

    def test_assignment_records_ids_and_flags(self):
        gts = [_gt("Reentrancy", "SWC-107", ["Vault.withdraw"], "H-07")]
        hyps = [_hyp("Gas", "gas", ["X.y"], "h0"), _hyp("Reentrancy", "SWC-107", ["withdraw"], "h1")]
        res = match_one_to_one(hyps, gts)
        assert res.assignments[0].candidate_id == "h1"
        assert res.assignments[0].candidate_index == 1
        assert res.assignments[0].gt_id == "H-07"
        assert res.candidate_flags == [False, True]
        assert res.gt_for_candidate(1) == 0
        assert match_hypotheses_to_ground_truth(hyps, gts) == [False, True]

    def test_gt_without_id_uses_index(self):
        gts = [GroundTruthFinding(title="Reentrancy", severity="high", vulnerability_class="SWC-107",
                                  affected_functions=["withdraw"])]
        res = match_one_to_one([_hyp("Reentrancy", "SWC-107", ["withdraw"], "h")], gts)
        assert res.assignments[0].gt_id == "0"
