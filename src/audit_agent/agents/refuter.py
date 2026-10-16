"""REFUTER — adversarial false-positive filter (runs after Skeptic).

Skeptic is structurally validate-biased ("reject only with an explicit guard"),
which lets through false positives whose attack path is actually unreachable or
which rest on a wrong assumption about an EXTERNAL contract's behavior. This
stage is the inverse: for every still-standing finding it ASSUMES the finding is
a false positive and tries to prove it, using a checklist drawn from the most
common FP categories observed in real runs:

  1. Reachability/control — can an attacker actually reach this path AND control
     the input claimed to be malicious? Or is that input set by the contract
     itself, a verified caller, or a governance/admin-gated path?
  2. External-contract behavior — does the finding ASSUME a flashloan provider /
     oracle / token / factory behaves a certain way? Is that assumption the known
     real behavior, or does the real behavior contradict the claim? (e.g. a
     flashloan callback is delivered to the *caller*, not an arbitrary address.)
  3. Intentional same-unit design — is a claimed "unit/decimal mismatch" actually
     an intentional same-unit comparison the rest of the code relies on?
  4. Guard elsewhere in the call path — is there a check in another function/
     contract on the path that prevents the attack?

The Refuter only ever DOWNGRADES (validated → unverified/false_positive); it never
promotes. A finding survives unchanged only if the refutation attempt fails.
Disabled with REFUTER_ENABLED=false; degrades gracefully (keeps Skeptic's verdict).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from audit_agent.agents.base import BaseAgent
from audit_agent.agents.skeptic import _read_sources
from audit_agent.config import get_settings
from audit_agent.knowledge.test_suite import find_relevant_tests
from audit_agent.graph.state import AuditState
from audit_agent.llm import get_provider
from audit_agent.llm.base import LLMProvider, ToolDefinition
from audit_agent.models.schemas import Hypothesis, ValidatedFinding
from audit_agent.observability.audit_logger import get_audit_logger
from audit_agent.observability.logging import get_logger

log = get_logger("agents.refuter")


_REFUTE_TOOL = ToolDefinition(
    name="refute_finding",
    description=(
        "Attempt to prove the finding is a FALSE POSITIVE. Report the strongest "
        "refutation and a verdict. Call exactly once."
    ),
    parameters={
        "type": "object",
        "required": ["verdict", "confidence", "reachability", "external_assumption", "strongest_refutation"],
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["upheld", "needs_review", "false_positive"],
                "description": (
                    "upheld: refutation FAILED — the exploit path is reachable, attacker-controlled, "
                    "and consistent with real external behavior. "
                    "needs_review: refutation raised serious doubt but is not conclusive. "
                    "false_positive: a concrete reason the attack cannot work (cite it)."
                ),
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                           "description": "Confidence in the verdict."},
            "reachability": {
                "type": "string",
                "description": (
                    "Can an attacker reach this path AND control the input claimed malicious? "
                    "Or is the input set by the contract / a verified caller / governance-gated? Be specific."
                ),
            },
            "external_assumption": {
                "type": "string",
                "description": (
                    "Does the finding assume an external contract (flashloan/oracle/token/factory) "
                    "behaves a certain way? State the REAL behavior and whether it contradicts the claim. "
                    "Write 'none' if the finding is self-contained in in-scope code."
                ),
            },
            "strongest_refutation": {
                "type": "string",
                "description": "The single strongest concrete reason this might be a false positive (or 'none found').",
            },
        },
    },
)

_REFUTER_SYSTEM = (
    "You are a RED-TEAM reviewer whose ONLY job is to DISPROVE a security finding that a "
    "previous (validation-biased) reviewer accepted. Assume it is a FALSE POSITIVE until "
    "proven otherwise. Be skeptical of the CLAIM, not of the code.\n\n"
    "Work through this checklist against the ACTUAL source code:\n\n"
    "1. REACHABILITY & CONTROL. Trace who can call the entry point and who sets the input the "
    "attack relies on. A finding is a FALSE POSITIVE if:\n"
    "   - the 'attacker-controlled' input is actually set by the contract itself, encoded by a "
    "     trusted caller, or only writable by a verified/whitelisted/governance (onlyAdmin / "
    "     onlyOwner / onlyRole) path;\n"
    "   - the malicious value cannot be injected because an upstream require() pins it to a trusted value.\n\n"
    "2. EXTERNAL-CONTRACT BEHAVIOR. If the attack depends on how an external contract behaves, "
    "state the REAL, well-known behavior and check it against the claim. Classic false positives:\n"
    "   - 'Anyone can trigger this flashloan callback with arbitrary data' — flashloan providers "
    "     (Morpho/Aave/Balancer/Uniswap) deliver the callback to the CALLER of flashLoan, not to an "
    "     arbitrary contract; a victim contract's callback only runs when IT initiated the loan.\n"
    "   - 'scaledBalanceOf / shares vs assets is a unit mismatch' — Aave scaled balances and "
    "     ERC4626 shares are deliberately tracked in their own unit; subtracting them only proves a "
    "     bug if the surrounding code does NOT treat them as the same unit.\n"
    "   - oracle/price assumptions that ignore the protocol's actual decimals or update model.\n\n"
    "3. INTENTIONAL DESIGN. Is the flagged 'mismatch' or 'missing check' actually relied upon "
    "consistently elsewhere (same unit on both sides, a check done in a sibling function, an invariant "
    "the protocol maintains)? If so → FALSE POSITIVE.\n\n"
    "4. GUARD ELSEWHERE. Look beyond the single function — a require()/modifier in a caller, factory, "
    "or the called contract may make the attack impossible.\n"
    "5. PROJECT TEST SUITE (strong signal). If the section '## PROJECT'S OWN TEST SUITE' is present and a "
    "test there CALLS the flagged function and ASSERTS the behavior the finding calls a bug — e.g. "
    "`vm.expectRevert(SomeError.selector); foo();` (the revert is deliberate), or a passing happy-path that "
    "contradicts the claim — then the behavior is INTENDED/known and the finding is a FALSE POSITIVE. Cite "
    "the test name in your refutation.\n\n"
    "RULES:\n"
    "- Only return verdict=false_positive when you can name a CONCRETE reason (cite function/line/known "
    "external behavior). Vague doubt → needs_review.\n"
    "- If you trace the path and the attack genuinely works (reachable + attacker-controlled + "
    "external behavior matches the claim) → verdict=upheld. Do not refute real bugs.\n"
    "- A PoC that compiled, RAN, and was reverted/blocked is real disconfirming evidence; a PoC that "
    "failed to COMPILE is noise — judge from source.\n"
    "Call refute_finding exactly once."
)


# verdict → how to mutate the finding's status (downgrade-only)
def _apply_verdict(status: str, verdict: str, confidence: float) -> tuple[str, str]:
    """Return (new_status, action). Never promotes; only the surviving finding can drop."""
    if verdict == "false_positive" and confidence >= 0.6:
        return "false_positive", "refuted"
    if verdict in ("false_positive", "needs_review"):
        # weak FP or genuine doubt → downgrade a 'validated' to needs-review; leave others
        if status == "validated":
            return "unverified", "downgraded"
        return status, "kept"
    return status, "upheld"  # verdict == "upheld"


class RefuterAgent(BaseAgent):
    """Adversarial second opinion that filters false positives out of Skeptic's output."""

    # 240s was too tight for qwen3.7-max (thinking model + up to 120k-char source
    # context): most calls timed out on a large real-world run, so the Refuter barely ran.
    # Refuter calls are sequential (full single-stream throughput), so 600s is ample.
    _LLM_TIMEOUT_SECS = 600.0

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__()
        settings = get_settings()
        self._provider = provider or get_provider()
        # Reuse the strongest model (same as Skeptic) — refutation needs real reasoning.
        self._model = model or settings.skeptic_model
        self._enabled = getattr(settings, "refuter_enabled", True)

    # ── LangGraph node ────────────────────────────────────────────────────
    async def run(self, state: AuditState) -> AuditState:
        audit_id = state.get("audit_id", "unknown")
        findings: list[ValidatedFinding] = list(state.get("validated_findings") or [])

        if not self._enabled:
            self.log.info("Refuter: disabled via settings")
            return state
        # Only re-examine still-standing findings; FPs are already out.
        targets = [f for f in findings if f.status in ("validated", "unverified")]
        if not targets:
            self.log.info("Refuter: no validated/unverified findings to challenge")
            return state

        self.log.info("Refuter: starting", audit_id=audit_id, candidates=len(targets))
        al = get_audit_logger()
        ar = al.log_agent_start("refuter", 6, {"audit_id": audit_id,
                                               "candidates": len(targets)}) if al else None

        hyp_by_id: dict[str, Hypothesis] = {
            h.id: h for h in (state.get("hypotheses") or [])
        }
        target_dir = state.get("target_dir", "")
        in_scope = list(state.get("scope_in_scope") or [])
        context_files = list(state.get("scope_context_files") or [])

        refuted = downgraded = upheld = 0
        for f in findings:
            if f.status not in ("validated", "unverified"):
                continue
            try:
                result = await self._challenge(
                    f, hyp_by_id.get(f.hypothesis_id or ""),
                    target_dir, in_scope, context_files,
                )
            except Exception as exc:  # never let one finding crash the stage
                self.log.warning("Refuter: challenge failed (keeping verdict)",
                                 finding=f.title[:50], error=str(exc))
                continue
            if result is None:
                continue

            verdict = str(result.get("verdict", "upheld"))
            conf = float(result.get("confidence", 0.5))
            new_status, action = _apply_verdict(f.status, verdict, conf)
            if action == "refuted":
                refuted += 1
            elif action == "downgraded":
                downgraded += 1
            else:
                upheld += 1

            if new_status != f.status:
                note = (
                    f"[REFUTER {verdict} c={conf:.2f}] "
                    f"reachability: {str(result.get('reachability',''))[:300]} | "
                    f"external: {str(result.get('external_assumption',''))[:200]} | "
                    f"refutation: {str(result.get('strongest_refutation',''))[:300]}"
                )
                f.status = new_status  # type: ignore[assignment]
                f.confidence = round(min(f.confidence, 1.0 - conf if verdict != "upheld" else f.confidence), 3)
                f.skeptic_notes = (f.skeptic_notes or "") + "\n\n" + note
            if al:
                al.log_decision("refuter", "challenge",
                    f"{f.title[:50]}: {verdict} → {action}",
                    {"verdict": verdict, "confidence": conf, "action": action,
                     "new_status": new_status})

        self.log.info("Refuter: complete", refuted_to_fp=refuted,
                      downgraded_to_review=downgraded, upheld=upheld)
        if al and ar:
            al.log_agent_end(ar, {"refuted_to_fp": refuted,
                                  "downgraded_to_review": downgraded, "upheld": upheld})
        return {**state, "validated_findings": findings}

    # ── per-finding adversarial challenge ─────────────────────────────────
    async def _challenge(
        self,
        finding: ValidatedFinding,
        hypothesis: Optional[Hypothesis],
        target_dir: str,
        in_scope: list[str],
        context_files: list[str],
    ) -> Optional[dict[str, Any]]:
        source_code = _read_sources(
            target_dir, in_scope, context_files,
            prioritize_functions=list(finding.affected_functions or []),
        )
        # Cross-check the project's own test suite — a passing test that asserts the
        # flagged behavior (e.g. expectRevert on it) is the strongest FP signal.
        relevant_tests = find_relevant_tests(target_dir, list(finding.affected_functions or []))
        prompt = self._build_prompt(finding, hypothesis, source_code, relevant_tests)

        al = get_audit_logger()
        rec = al.log_llm_call_start(
            agent="refuter", purpose=f"refute_{finding.id[:8]}", model=self._model,
            system_prompt=_REFUTER_SYSTEM, user_message=prompt,
            tools=[_REFUTE_TOOL], max_tokens=4096,
        ) if al else None

        t0 = time.time()
        try:
            resp = await asyncio.wait_for(
                self._provider.complete(
                    model=self._model, max_tokens=4096, system=_REFUTER_SYSTEM,
                    tools=[_REFUTE_TOOL], tool_choice={"name": "refute_finding"},
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=self._LLM_TIMEOUT_SECS,
            )
        except Exception as exc:
            if al and rec:
                al.log_llm_call_end(rec, error=str(exc), duration_seconds=time.time() - t0)
            log.warning("refuter.llm_unavailable", error=str(exc))
            return None

        args: Optional[dict[str, Any]] = None
        for tc in resp.tool_calls:
            if tc.name == "refute_finding":
                args = tc.arguments
        if al and rec:
            al.log_llm_call_end(
                rec, tool_calls=[{"name": "refute_finding",
                                  "verdict": (args or {}).get("verdict")}] if args else [],
                input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                stop_reason=resp.stop_reason, duration_seconds=time.time() - t0,
            )
        return args

    @staticmethod
    def _build_prompt(
        finding: ValidatedFinding,
        hypothesis: Optional[Hypothesis],
        source_code: str,
        relevant_tests: str = "",
    ) -> str:
        parts = [
            "## Finding accepted by the previous reviewer (your job: try to DISPROVE it)",
            f"Title: {finding.title}",
            f"Class: {finding.vulnerability_class}",
            f"Severity: {finding.severity}",
            f"Affected functions: {finding.affected_functions}",
            f"Claimed impact: {finding.description}",
        ]
        if hypothesis:
            parts.append(f"Attack scenario: {hypothesis.attack_scenario}")
            parts.append(f"Claimed preconditions: {hypothesis.required_preconditions}")
        if finding.skeptic_notes:
            parts.append(f"Prior reviewer's reasoning: {finding.skeptic_notes[:800]}")
        if finding.execution_trace:
            parts.append(f"PoC execution trace (if PoC ran): {finding.execution_trace[:800]}")
        if source_code:
            parts.append(
                "\n## Contract source code (authoritative — trace reachability here)\n"
                f"```solidity\n{source_code}\n```"
            )
        if relevant_tests:
            parts.append("\n" + relevant_tests)
        parts.append(
            "\nRun the refutation checklist. Only uphold if the attack genuinely works."
        )
        return "\n".join(parts)
