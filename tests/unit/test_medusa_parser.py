"""Tests for the Medusa output parser and medusa.json config rendering."""
from __future__ import annotations

import json

from audit_agent.tools.medusa import MedusaConfig, MedusaResult, parse_medusa_output


class TestParseMedusaOutput:
    def test_failed_property_with_call_sequence(self):
        out = (
            "⇾ [PASSED] Property Test: MedusaProperties.echidna_solvency()\n"
            "⇾ [FAILED] Property Test: MedusaProperties.echidna_fee_nonzero()\n"
            "Call Sequence:\n"
            "1) deposit(1)\n"
            "2) withdraw()\n"
        )
        r = parse_medusa_output(out)
        assert r.status == "properties_broken"
        assert r.broken_properties[0].name == "echidna_fee_nonzero"
        assert "withdraw" in r.broken_properties[0].counterexample
        assert "echidna_solvency" in r.passed_properties

    def test_failed_after_n_calls_variant(self):
        out = 'Test for method "Counter.property_balance()" failed after 12 call(s)'
        r = parse_medusa_output(out)
        assert r.status == "properties_broken"
        assert r.broken_properties[0].name == "property_balance"

    def test_all_passed(self):
        out = (
            "⇾ [PASSED] Property Test: MedusaProperties.echidna_a()\n"
            "⇾ [PASSED] Property Test: MedusaProperties.echidna_b()\n"
        )
        r = parse_medusa_output(out)
        assert r.status == "all_passed"
        assert set(r.passed_properties) == {"echidna_a", "echidna_b"}

    def test_unrecognised_output_is_error(self):
        r = parse_medusa_output("medusa: compilation failed")
        assert r.status == "error"

    def test_result_model_shape(self):
        # MedusaResult is the canonical generic fuzz-result model.
        from audit_agent.tools.medusa import BrokenProperty
        fields = set(MedusaResult.model_fields.keys())
        assert {"status", "broken_properties", "passed_properties"} <= fields
        assert set(BrokenProperty.model_fields.keys()) == {"name", "counterexample", "shrunk"}


class TestMedusaConfig:
    def test_to_json_targets_property_contract(self):
        cfg = MedusaConfig(test_limit=1234, timeout=99)
        data = json.loads(cfg.to_json("MyProps"))
        fuzz = data["fuzzing"]
        assert fuzz["testLimit"] == 1234
        assert fuzz["timeout"] == 99
        assert fuzz["targetContracts"] == ["MyProps"]
        assert fuzz["deploymentOrder"] == ["MyProps"]
        prefixes = fuzz["testing"]["propertyTesting"]["testPrefixes"]
        assert "echidna_" in prefixes and "property_" in prefixes
        assert data["compilation"]["platform"] == "crytic-compile"
