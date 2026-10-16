"""INVARIANT PROVER — protocol-wide invariant testing stage (roadmap #4).

The base pipeline only ran Echidna for individual `fuzz`-type hypotheses. This
stage instead takes ALL invariants mined by the Mapper and subjects the whole
protocol to:

  • a stateful Echidna fuzzing campaign (samples many call sequences), and
  • Halmos symbolic checks of arithmetic invariants (proves over ALL inputs).

A broken property / counterexample is a *demonstrated* violation, so each one is
converted into a high-priority Hypothesis appended to the pipeline — it then
flows through Exploit → Skeptic → Scribe like any other finding. This catches
multi-transaction economic bugs that no single per-function prompt reveals.

Runs as a graph node between Hunter and Exploit. All tool runners are injectable
and mock-aware, so the stage is fully testable offline and degrades gracefully
when Docker / the halmos image is unavailable.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from audit_agent.agents.base import BaseAgent
from audit_agent.config import get_settings
from audit_agent.exploit.echidna_gen import EchidnaPropertyGenerator, _safe_identifier
from audit_agent.exploit.halmos_gen import HalmosSpecGenerator
from audit_agent.graph.state import AuditState
from audit_agent.models.schemas import Hypothesis, Invariant
from audit_agent.observability.audit_logger import get_audit_logger
from audit_agent.tools.errors import MockRunResult
from audit_agent.tools.halmos import HalmosConfig, HalmosResult, HalmosRunner
from audit_agent.tools.medusa import MedusaConfig, MedusaResult, MedusaRunner


class InvariantProverAgent(BaseAgent):
    """Runs a protocol-wide invariant campaign and converts violations to hypotheses."""

    def __init__(
        self,
        echidna_gen: Optional[EchidnaPropertyGenerator] = None,
        echidna_runner: Optional[MedusaRunner] = None,
        halmos_gen: Optional[HalmosSpecGenerator] = None,
        halmos_runner: Optional[HalmosRunner] = None,
    ) -> None:
        super().__init__()
        self._echidna_gen = echidna_gen or EchidnaPropertyGenerator()
        # Fuzz engine is Medusa (replaces Echidna); ``echidna_runner`` kept as the
        # test injection point and consumes the same property contracts.
        self._echidna_runner = echidna_runner or MedusaRunner()
        self._halmos_gen = halmos_gen or HalmosSpecGenerator()
        self._halmos_runner = halmos_runner or HalmosRunner()

    # ── LangGraph node ────────────────────────────────────────────────────
    async def run(self, state: AuditState) -> AuditState:
        audit_id = state.get("audit_id", "unknown")
        invariants: list[Invariant] = list(state.get("invariants") or [])
        hypotheses: list[Hypothesis] = list(state.get("hypotheses") or [])

        if not invariants:
            self.log.info("InvariantProver: no invariants — skipping")
            return state

        settings = get_settings()
        if not getattr(settings, "invariant_prover_enabled", True):
            self.log.info("InvariantProver: disabled via settings")
            return state

        target_dir = state.get("target_dir", "")
        in_scope_files: list[str] = list(state.get("scope_in_scope") or [])
        fuzz_enabled = state.get("fuzz_enabled", True)

        self.log.info("InvariantProver: starting", audit_id=audit_id,
                      invariants=len(invariants), fuzz_enabled=fuzz_enabled)
        al = get_audit_logger()
        ar = al.log_agent_start("invariant_prover", 4, {
            "audit_id": audit_id, "invariants": len(invariants),
            "fuzz_enabled": fuzz_enabled,
        }) if al else None

        target_contracts = self._resolve_targets(in_scope_files, target_dir)
        contract_metadata = self._contract_metadata(in_scope_files, target_dir)

        new_hyps: list[Hypothesis] = []

        if fuzz_enabled:
            try:
                new_hyps += await self._run_echidna(
                    invariants, contract_metadata, target_contracts, target_dir, audit_id
                )
            except Exception as exc:
                self.log.warning("InvariantProver: echidna stage failed", error=str(exc))

        try:
            new_hyps += await self._run_halmos(
                invariants, contract_metadata, in_scope_files, target_dir, audit_id
            )
        except Exception as exc:
            self.log.warning("InvariantProver: halmos stage failed", error=str(exc))

        self.log.info("InvariantProver: complete", new_hypotheses=len(new_hyps))
        if al and ar:
            al.log_agent_end(ar, {
                "new_hypotheses": len(new_hyps),
                "titles": [h.title for h in new_hyps[:10]],
            })

        return {**state, "hypotheses": hypotheses + new_hyps, "invariants": invariants}

    # ── Echidna campaign ──────────────────────────────────────────────────
    async def _run_echidna(
        self,
        invariants: list[Invariant],
        contract_metadata: str,
        target_contracts: list[Path],
        target_dir: str,
        audit_id: str,
    ) -> list[Hypothesis]:
        property_contract = self._echidna_gen.generate_from_invariants(
            invariants, contract_metadata, target_contracts
        )
        result = await self._echidna_runner.run_campaign(
            property_contract=property_contract,
            target_contracts=target_contracts,
            config=MedusaConfig(timeout=120, test_limit=50_000),
            target_dir=Path(target_dir) if target_dir else Path("."),
        )
        if isinstance(result, MockRunResult) or not isinstance(result, MedusaResult):
            return []
        if result.status != "properties_broken":
            return []

        al = get_audit_logger()
        hyps: list[Hypothesis] = []
        for prop in result.broken_properties:
            inv = self._match_invariant(prop.name, invariants)
            hyps.append(self._violation_hypothesis(
                audit_id, inv, prop.name, prop.counterexample, tool="Echidna fuzzing"
            ))
            if al:
                al.log_decision("invariant_prover", "echidna_broken",
                    f"{prop.name}: {(inv.natural_language[:70]) if inv else 'unknown'}",
                    {"property": prop.name})
        return hyps

    # ── Halmos symbolic checks ────────────────────────────────────────────
    async def _run_halmos(
        self,
        invariants: list[Invariant],
        contract_metadata: str,
        in_scope_files: list[str],
        target_dir: str,
        audit_id: str,
    ) -> list[Hypothesis]:
        target_import = self._halmos_import(in_scope_files, target_dir)
        suite, mapping = self._halmos_gen.generate_suite(
            invariants, contract_metadata, target_import
        )
        if not suite or not mapping:
            return []

        contract_name = None
        result = await self._halmos_runner.run_suite(
            test_contract=suite,
            target_dir=Path(target_dir) if target_dir else Path("."),
            config=HalmosConfig(contract="AuditAgentHalmos"),
        )
        if isinstance(result, MockRunResult) or not isinstance(result, HalmosResult):
            return []
        if result.status != "counterexamples_found":
            return []

        by_name = {name: inv for name, inv in mapping}
        al = get_audit_logger()
        hyps: list[Hypothesis] = []
        for spec in result.broken_specs:
            inv = by_name.get(spec.name)
            hyps.append(self._violation_hypothesis(
                audit_id, inv, spec.name, spec.counterexample, tool="Halmos symbolic execution"
            ))
            if al:
                al.log_decision("invariant_prover", "halmos_counterexample",
                    f"{spec.name}: {(inv.natural_language[:70]) if inv else 'unknown'}",
                    {"spec": spec.name})
        return hyps

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _violation_hypothesis(
        audit_id: str,
        inv: Optional[Invariant],
        name: str,
        counterexample: str,
        tool: str,
    ) -> Hypothesis:
        nl = inv.natural_language if inv else name
        affected = list(inv.affected_functions) if inv else []
        ce = (counterexample or "").strip()
        scenario = (
            f"{tool} found a concrete counterexample violating the invariant "
            f"\"{nl[:160]}\"."
        )
        if ce:
            scenario += f"\n\nCounterexample / call sequence:\n{ce[:800]}"
        return Hypothesis(
            id=str(uuid.uuid4()),
            audit_id=audit_id,
            title=f"Invariant violation: {nl[:80]}",
            vulnerability_class="Invariant Violation",
            affected_functions=affected,
            attack_scenario=scenario,
            required_preconditions=[],
            estimated_severity="high",
            similar_historical_cases=[],
            # Demonstrated by a tool → very high priority.
            priority_score=90.0,
            confidence=0.9,
        )

    @staticmethod
    def _match_invariant(prop_name: str, invariants: list[Invariant]) -> Optional[Invariant]:
        """Best-effort map an echidna_<id> property name back to its invariant."""
        stripped = prop_name
        for pre in ("echidna_", "property_"):
            if stripped.lower().startswith(pre):
                stripped = stripped[len(pre):]
                break
        for inv in invariants:
            sid = _safe_identifier(inv.natural_language)
            if stripped == sid or stripped.startswith(sid) or sid.startswith(stripped):
                return inv
        return None

    @staticmethod
    def _resolve_targets(in_scope_files: list[str], target_dir: str) -> list[Path]:
        out: list[Path] = []
        for f in in_scope_files:
            p = Path(f)
            if not p.exists() and target_dir:
                p = Path(target_dir) / f.lstrip("./")
            if p.exists():
                out.append(p)
        return out

    @staticmethod
    def _contract_metadata(in_scope_files: list[str], target_dir: str) -> str:
        try:
            from audit_agent.agents.exploit import _build_contract_metadata
            return _build_contract_metadata(target_dir, include_files=in_scope_files or None)
        except Exception:
            return ""

    @staticmethod
    def _halmos_import(in_scope_files: list[str], target_dir: str) -> str:
        """Build the import path for the primary in-scope contract from test/."""
        if not in_scope_files:
            return "../src/Target.sol"
        primary = in_scope_files[0]
        if target_dir:
            try:
                rel = Path(primary).resolve().relative_to(Path(target_dir).resolve())
                return f"../{rel}"
            except ValueError:
                pass
        return f"../{Path(primary).name}"
