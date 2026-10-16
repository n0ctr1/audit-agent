"""Skeptic agent — adversarial validator of ExploitResults (§3.5).

Pipeline per ExploitResult:
  1. Find matching Hypothesis (by hypothesis_id)
  2. Call LLM via validate_exploit tool to get verdict
  3. On LLM failure: apply deterministic heuristic based on exploit status
  4. Produce ValidatedFinding with appropriate status / confidence
  5. Aggregate into validated_findings list in state
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from audit_agent.agents.base import BaseAgent
from audit_agent.graph.state import AuditState
from audit_agent.models.schemas import (
    ExploitResult,
    Hypothesis,
    Severity,
    ValidatedFinding,
)
from audit_agent.llm import get_provider
from audit_agent.llm.base import LLMProvider, ToolDefinition
from audit_agent.observability.audit_logger import get_audit_logger
from audit_agent.observability.logging import get_logger

log = get_logger("agents.skeptic")

# ─────────────────────────────────────────────────────────────────────────────
# LLM tool definition
# ─────────────────────────────────────────────────────────────────────────────

_VALIDATION_TOOL = ToolDefinition(
    name="validate_exploit",
    description=(
        "Adversarially validate whether a PoC exploit demonstrates a real vulnerability "
        "or is a false positive due to test setup errors, straw-man contracts, or "
        "insufficient proof of economic impact. Call this exactly once per exploit."
    ),
    parameters={
        "type": "object",
        "required": ["verdict", "confidence", "reasoning", "real_impact", "suggested_severity"],
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["true_positive", "false_positive", "needs_review"],
                "description": (
                    "true_positive: exploit succeeds and demonstrates a real vulnerability. "
                    "false_positive: test passed due to a setup artifact, not a real bug. "
                    "needs_review: ambiguous — human auditor must decide."
                ),
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in the verdict (0.0 = none, 1.0 = certain).",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed adversarial reasoning for the verdict.",
            },
            "real_impact": {
                "type": "string",
                "description": "Concrete description of real-world impact (if true positive).",
            },
            "suggested_severity": {
                "type": "string",
                "enum": ["critical", "high", "medium", "low", "info"],
                "description": "Severity based on actual demonstrated impact.",
            },
        },
    },
)

_SKEPTIC_SYSTEM = (
    "You are a senior smart-contract security auditor performing final triage. "
    "For each claim you must decide: true_positive, false_positive, or needs_review.\n\n"
    "IMPORTANT: You verify TECHNICAL CORRECTNESS only, not severity. Your job here is RECALL — keep "
    "anything that could plausibly be real; a dedicated adversarial 'refuter' stage runs AFTER you and "
    "makes the final false-positive call, so DO NOT over-reject. Reject as false_positive ONLY if ANY hold:\n"
    "1. The PoC uses a mock/straw-man contract instead of the real target contract.\n"
    "2. There is a SPECIFIC concrete protection in the code that prevents the attack "
    "(not just 'it seems unlikely' — show the exact guard).\n"
    "3. The preconditions are IMPOSSIBLE in reality (not merely unlikely — e.g., requires "
    "negative balance, or two contradictory state invariants simultaneously).\n"
    "4. The PoC test does not actually demonstrate the claimed vulnerability "
    "(it tests something different).\n\n"
    "When you have DOUBT about reachability/control (is the malicious input attacker-controlled or is it "
    "governance/contract-set?) or about EXTERNAL-contract behavior (e.g. a flashloan callback is delivered "
    "to the CALLER of flashLoan, not an arbitrary contract; Aave scaledBalanceOf / ERC4626 shares are "
    "intentional units) — do NOT hard-reject. Return needs_review and note the concern; the refuter will "
    "resolve it. Only return false_positive when you can name the concrete blocker per rules 1-4.\n\n"
    "Key rules:\n"
    "• A PoC that failed to COMPILE proves nothing either way — evaluate the "
    "vulnerability from the SOURCE CODE independently.\n"
    "• status=partial means PoC compilation/execution failed, NOT that the bug is fake. "
    "Evaluate source code directly for partial results.\n"
    "• status=exploited means a forge test passed — this is STRONG evidence of true_positive.\n"
    "• status=blocked means a guard actively blocked it — verify the guard is always in effect.\n"
    "• Do NOT dismiss a finding because impact seems low — that is a severity judgment, not yours.\n"
    "• If source code contains the vulnerability pattern described, prefer needs_review "
    "over false_positive when uncertain.\n"
    "Call validate_exploit exactly once.\n\n"
    "## SOURCE CODE IS YOUR PRIMARY EVIDENCE\n\n"
    "The PoC may have failed to compile or run. That is IRRELEVANT to your verdict.\n"
    "Your ONLY source of truth is the CONTRACT SOURCE CODE in \"## Contract Source Code\".\n\n"
    "### When source code clearly shows the vulnerability → true_positive:\n"
    "- Code does X before Y, and the hypothesis says \"X before Y causes bug\" → CONFIRMED\n"
    "- A signed hash omits a field that is parsed from the signature → TRUE POSITIVE\n"
    "  ↳ CRITICAL DISTINCTION: 'data is USED in logic' ≠ 'data is AUTHENTICATED by signature'.\n"
    "    Being passed to _packValidationData or another function does NOT make it authenticated.\n"
    "    Authentication requires the data to appear in the hash fed to ecrecover/ECDSA.recover.\n"
    "    Example: validUntil extracted from sig bytes but only `userOpHash` is hashed → UNSIGNED.\n"
    "- Signature fallback returns sentinel values (0, empty): if a parsing function returns "
    "validUntil=0 for unrecognized signature formats, and 0 means 'never expires' per the spec "
    "(e.g. EIP-4337), then ANY valid signature of that format bypasses the time restriction → "
    "TRUE POSITIVE regardless of whether ECDSA verification itself passes.\n"
    "- epochEnd(N) uses `start + epoch * length + length` while currentEpoch uses\n"
    "  `(ts - start) / length` → CHECK if epochEnd(N) == first moment of N+1 → CONFIRMED\n"
    "  if the hypothesis describes exactly this\n"
    "- Rounding direction in initialize() differs from previewMint() → CONFIRMED\n"
    "- Two parallel values computed for token0/token1 use OPPOSITE subtraction order\n"
    "  (e.g., token0: shortPremium - longPremium but token1: longPremium - shortPremium)\n"
    "  → SIGN INVERSION BUG — both slots must use the same order → TRUE POSITIVE\n\n"
    "### When evaluating from source code:\n"
    "1. TRACE the exact execution path described in the hypothesis step by step\n"
    "2. CHECK each precondition against the actual code (not against the PoC)\n"
    "3. VERIFY arithmetic: substitute concrete values and compute results\n"
    "4. If the vulnerability IS VISIBLE in source code and plausibly reachable → true_positive (confidence ≥ 0.80)\n"
    "5. If reachability/control or external-contract behavior is UNCLEAR → needs_review (0.4-0.7); the "
    "refuter stage will adjudicate. Bias toward needs_review over false_positive when uncertain.\n"
    "6. Only false_positive when source EXPLICITLY prevents the attack — cite the EXACT guard "
    "(line/function/condition). Reachability/external-assumption doubts are the refuter's call, not a "
    "reason for YOU to hard-reject.\n\n"
    "### PoC status interpretation:\n"
    "- status=exploited (forge test passed) → STRONG evidence of true_positive\n"
    "- status=partial/error (PoC failed) → IGNORE for verdict, evaluate source code only\n"
    "- status=blocked (guard caught it) → verify guard is ALWAYS in effect, not bypassable\n\n"
    "### Common mistakes to avoid:\n"
    "- Do NOT say \"cannot confirm without seeing source code\" — you HAVE the source code\n"
    "- Do NOT say \"PoC didn't compile so cannot verify\" — use source code\n"
    "- Do NOT dismiss a finding just because impact seems low — that's severity, not validity\n"
    "- Do NOT require the PoC to succeed to confirm a source-code-visible vulnerability\n"
)

# Append the curated external-behavior reference so cross-protocol assumptions can be checked.
from audit_agent.knowledge.external_behaviors import EXTERNAL_PROTOCOL_FACTS as _EXT_FACTS
_SKEPTIC_SYSTEM = _SKEPTIC_SYSTEM + "\n\n" + _EXT_FACTS


# ─────────────────────────────────────────────────────────────────────────────
# Verdict → FindingStatus mapping
# ─────────────────────────────────────────────────────────────────────────────

_VERDICT_TO_STATUS = {
    "true_positive": "validated",
    "false_positive": "false_positive",
    "needs_review": "unverified",
}


# ─────────────────────────────────────────────────────────────────────────────
# SkepticAgent
# ─────────────────────────────────────────────────────────────────────────────

class SkepticAgent(BaseAgent):
    """Validates ExploitResults adversarially, producing ValidatedFindings."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: str | None = None,
    ) -> None:
        super().__init__()
        self._provider = provider or get_provider()
        if model is None:
            from audit_agent.config import get_settings
            model = get_settings().skeptic_model
        self._model = model

    # ── LangGraph node ────────────────────────────────────────────────────

    async def run(self, state: AuditState) -> AuditState:
        audit_id = state.get("audit_id", "unknown")
        exploits: list[ExploitResult] = list(state.get("exploits") or [])
        hypotheses: list[Hypothesis] = list(state.get("hypotheses") or [])

        self.log.info("Skeptic: starting", audit_id=audit_id, exploits=len(exploits))

        al = get_audit_logger()
        ar = al.log_agent_start("skeptic", 5, {
            "audit_id": audit_id,
            "exploits_count": len(exploits),
            "hypotheses_count": len(hypotheses),
        }) if al else None

        hyp_by_id: dict[str, Hypothesis] = {h.id: h for h in hypotheses}

        target_dir = state.get("target_dir", "")
        in_scope_files: list[str] = list(state.get("scope_in_scope") or [])
        context_files: list[str] = list(state.get("scope_context_files") or [])
        fallback_source = _read_sources(target_dir, in_scope_files, context_files)
        if not fallback_source or len(fallback_source) < 50:
            fallback_source = self._read_scope_sources(state)

        validated: list[ValidatedFinding] = []
        for exploit in exploits:
            hyp = hyp_by_id.get(exploit.hypothesis_id)
            # Build per-hypothesis source: files containing affected functions come first
            affected = list(hyp.affected_functions) if hyp and hyp.affected_functions else []
            source_code = _read_sources(
                target_dir, in_scope_files, context_files,
                prioritize_functions=affected,
            ) or fallback_source
            finding = await self.validate(exploit, hyp, audit_id, source_code=source_code)
            validated.append(finding)
            self.log.info(
                "Skeptic: verdict",
                hypothesis_id=exploit.hypothesis_id,
                status=finding.status,
                confidence=finding.confidence,
            )
            if al:
                al.log_decision("skeptic", "verdict",
                    f"hyp={exploit.hypothesis_id}: {finding.status} "
                    f"(confidence={finding.confidence:.2f})",
                    {"hypothesis_id": exploit.hypothesis_id, "verdict": finding.status,
                     "confidence": finding.confidence,
                     "exploit_status": exploit.status})

        tp = sum(1 for f in validated if f.status == "validated")
        fp = sum(1 for f in validated if f.status == "false_positive")
        self.log.info(
            "Skeptic: complete",
            total=len(validated), true_positive=tp, false_positive=fp,
            needs_review=len(validated) - tp - fp,
        )

        if al and ar:
            al.log_agent_end(ar, {
                "total": len(validated),
                "true_positive": tp,
                "false_positive": fp,
                "needs_review": len(validated) - tp - fp,
            })

        return {**state, "validated_findings": validated}

    # ── Per-exploit validation ────────────────────────────────────────────

    async def validate(
        self,
        exploit: ExploitResult,
        hypothesis: Optional[Hypothesis],
        audit_id: str = "unknown",
        source_code: str = "",
    ) -> ValidatedFinding:
        """Validate a single ExploitResult, returning a ValidatedFinding."""
        # ── Anti-hallucination guard: detect stub / mock PoC ─────────────
        # A stub PoC (generation failure) proves nothing, but the hypothesis may still be
        # real and visible in source — so we do NOT short-circuit to unverified. We let the
        # LLM adjudicate from source code (as with partial/error PoCs) and, below, forbid the
        # stub from counting as positive exploit evidence.
        poc_issue = _detect_mock_poc(exploit.poc_code or "")

        llm_result = await self._llm_validate(exploit, hypothesis, source_code)

        if llm_result is None:
            # Tool-call failed (timeout / model didn't return tool call).
            # Retry with a simpler free-text prompt so we always get code analysis,
            # never a blind heuristic based solely on exploit status.
            log.info("skeptic.retrying_text_fallback",
                     hypothesis_id=exploit.hypothesis_id, exploit_status=exploit.status)
            llm_result = await self._llm_validate_text(exploit, hypothesis, source_code)

        if llm_result is not None:
            verdict = llm_result.get("verdict", "needs_review")
            confidence = float(llm_result.get("confidence", 0.5))
            reasoning = llm_result.get("reasoning") or llm_result.get("analysis", "")
            real_impact = llm_result.get("real_impact", "")
            _sev_raw = str(llm_result.get("suggested_severity", "medium")).lower().strip()
            severity: Severity = _sev_raw if _sev_raw in ("critical", "high", "medium", "low", "info") else "medium"  # type: ignore[assignment]
            # Mock contract penalty: PoC tests a fake contract, not the real one
            if poc_issue == "mock":
                confidence = max(0.0, confidence - 0.20)
                reasoning = (
                    "[WARNING: PoC uses a mock/redefined contract — result may not reflect "
                    "real protocol behavior. Confidence penalized.]\n\n" + reasoning
                )
                if verdict == "true_positive" and confidence < 0.5:
                    verdict = "needs_review"
            elif poc_issue == "stub":
                # PoC was a non-functional generation stub: it can back neither a positive
                # nor a negative verdict. Keep the source-based verdict but cap confidence so
                # a stub never masquerades as exploit-backed proof.
                reasoning = (
                    "[NOTE: PoC was a non-functional generation stub — verdict rests on "
                    "source-code analysis only, not on any exploit run.]\n\n" + reasoning
                )
                confidence = min(confidence, 0.80)
        else:
            # Both LLM paths failed (e.g. total infra outage).  Use heuristic as last resort.
            verdict, confidence = _heuristic_verdict(exploit)
            reasoning = f"Heuristic (both LLM attempts failed) — exploit status={exploit.status!r}"
            real_impact = _heuristic_impact(exploit, hypothesis)
            severity = hypothesis.estimated_severity if hypothesis else "medium"  # type: ignore[assignment]

        title = hypothesis.title if hypothesis else f"Finding for {exploit.hypothesis_id}"
        vuln_class = hypothesis.vulnerability_class if hypothesis else None
        affected = hypothesis.affected_functions if hypothesis else []

        return ValidatedFinding(
            audit_id=audit_id,
            hypothesis_id=exploit.hypothesis_id,
            title=title,
            vulnerability_class=vuln_class,
            severity=severity,
            confidence=confidence,
            status=_VERDICT_TO_STATUS[verdict],  # type: ignore[arg-type]
            affected_functions=affected,
            description=real_impact,
            poc_code=exploit.poc_code or None,
            execution_trace=exploit.execution_trace or None,
            skeptic_notes=reasoning,
        )

    # ── Source code helpers ───────────────────────────────────────────────

    def _read_scope_sources(self, state: dict) -> str:
        """Fallback: read source code directly from scope file paths."""
        from pathlib import Path

        target_dir = state.get("target_dir", "")
        scope_files: list[str] = list(state.get("scope_in_scope") or [])
        context_files: list[str] = list(state.get("scope_context_files") or [])
        parts: list[str] = []

        for sol_path in scope_files:
            p = Path(sol_path)
            if not p.exists() and target_dir:
                p = Path(target_dir) / sol_path.lstrip("./")
            if p.exists():
                try:
                    content = p.read_text(errors="replace")
                    parts.append(f"// === {p.name} [IN SCOPE] ===\n{content}")
                except OSError:
                    pass

        for sol_path in context_files[:5]:
            p = Path(sol_path)
            if not p.exists() and target_dir:
                p = Path(target_dir) / sol_path.lstrip("./")
            if p.exists():
                try:
                    content = p.read_text(errors="replace")
                    parts.append(f"// === {p.name} [DEPENDENCY] ===\n{content[:3000]}")
                except OSError:
                    pass

        return "\n\n".join(parts)[:15000]

    # ── LLM call ─────────────────────────────────────────────────────────

    async def _llm_validate(
        self,
        exploit: ExploitResult,
        hypothesis: Optional[Hypothesis],
        source_code: str = "",
    ) -> Optional[dict[str, Any]]:
        """Call LLM for adversarial validation. Returns parsed tool input or None."""
        prompt = _build_validation_prompt(exploit, hypothesis, source_code)
        al = get_audit_logger()
        llm_record = None
        if al:
            llm_record = al.log_llm_call_start(
                agent="skeptic",
                purpose=f"validate_{exploit.hypothesis_id}",
                model=self._model,
                system_prompt=_SKEPTIC_SYSTEM,
                user_message=prompt,
                tools=[_VALIDATION_TOOL],
                max_tokens=4096,
            )
        t0 = time.time()
        try:
            response = await asyncio.wait_for(
                self._provider.complete(
                    model=self._model,
                    max_tokens=4096,
                    system=_SKEPTIC_SYSTEM,
                    tools=[_VALIDATION_TOOL],
                    tool_choice={"name": "validate_exploit"},
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=240.0,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            duration = time.time() - t0
            err = "timeout" if isinstance(exc, asyncio.TimeoutError) else str(exc)
            if al and llm_record:
                al.log_llm_call_end(llm_record, error=err, duration_seconds=duration)
            log.warning("skeptic.llm_unavailable", error=err)
            return None

        duration = response.duration_seconds or (time.time() - t0)
        result_data = None
        for tc in response.tool_calls:
            if tc.name == "validate_exploit":
                result_data = tc.arguments

        if al and llm_record:
            tool_calls_logged = []
            if result_data:
                tool_calls_logged = [{"name": "validate_exploit",
                                      "verdict": result_data.get("verdict"),
                                      "confidence": result_data.get("confidence")}]
            al.log_llm_call_end(
                llm_record,
                tool_calls=tool_calls_logged,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                stop_reason=response.stop_reason,
                duration_seconds=duration,
            )
        return result_data

    async def _llm_validate_text(
        self,
        exploit: ExploitResult,
        hypothesis: Optional[Hypothesis],
        source_code: str = "",
    ) -> Optional[dict[str, Any]]:
        """Fallback: ask for a free-text verdict and parse it.

        Used when the tool-call path times out or returns no tool call.
        Shorter prompt, no tool forcing — just asks for a structured text block.
        """
        prompt = _build_validation_prompt(exploit, hypothesis, source_code)
        text_system = (
            _SKEPTIC_SYSTEM
            + "\n\nRespond in this exact format (no other text):\n"
            "VERDICT: <true_positive|false_positive|needs_review>\n"
            "CONFIDENCE: <0.0-1.0>\n"
            "SEVERITY: <critical|high|medium|low|info>\n"
            "REASONING: <one paragraph — cite specific source code lines>\n"
            "IMPACT: <one sentence on real-world effect>\n"
        )
        try:
            response = await asyncio.wait_for(
                self._provider.complete_simple(
                    model=self._model,
                    system=text_system,
                    user_message=prompt,
                    max_tokens=1024,
                ),
                timeout=180.0,
            )
        except Exception as exc:
            log.warning("skeptic.text_fallback_failed", error=str(exc))
            return None

        return _parse_text_verdict(response)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_validation_prompt(
    exploit: ExploitResult,
    hypothesis: Optional[Hypothesis],
    source_code: str = "",
) -> str:
    parts: list[str] = []

    if hypothesis:
        parts.append(
            f"## Hypothesis\n"
            f"Title: {hypothesis.title}\n"
            f"Vulnerability class: {hypothesis.vulnerability_class}\n"
            f"Severity: {hypothesis.estimated_severity}\n"
            f"Affected functions: {hypothesis.affected_functions}\n"
            f"Attack scenario: {hypothesis.attack_scenario}\n"
            f"Preconditions: {hypothesis.required_preconditions}"
        )
    else:
        parts.append(f"## Hypothesis\nNo matching hypothesis found for id={exploit.hypothesis_id}")

    parts.append(
        f"\n## Exploit Result\n"
        f"Status: {exploit.status}\n"
        f"NOTE: status=partial/error means the PoC SCRIPT failed — it does NOT mean "
        f"the vulnerability does not exist. Evaluate from source code independently.\n"
        f"Test type: {exploit.test_type}\n"
        f"Gas used: {exploit.gas_used}\n"
        f"Profit USD: {exploit.profit_usd}"
    )

    if exploit.reproduction_steps:
        steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(exploit.reproduction_steps[:10]))
        parts.append(f"\n## Reproduction Steps\n{steps}")

    if exploit.execution_trace:
        parts.append(f"\n## Execution Trace\n{exploit.execution_trace[:1500]}")

    if source_code:
        parts.append(
            f"\n## Contract Source Code (authoritative — use this for independent analysis)\n"
            f"```solidity\n{source_code}\n```"
        )

    if exploit.poc_code:
        parts.append(f"\n## PoC Code\n```solidity\n{exploit.poc_code[:1500]}\n```")

    return "\n".join(parts)


def _detect_mock_poc(poc_code: str) -> str:
    """Classify PoC code quality.

    Returns:
        "stub"  — generation failed, assertTrue(false) placeholder
        "mock"  — PoC redefines a target contract instead of importing it
        ""      — PoC looks legitimate
    """
    if not poc_code:
        return "stub"
    # Explicit generation stubs
    if 'assertTrue(false, "PoC generation failed' in poc_code:
        return "stub"
    if "/// @notice Stub PoC" in poc_code or "/// Stub PoC" in poc_code:
        return "stub"
    # Mock contracts: LLM redefined the target instead of importing it
    import re as _re
    mock_patterns = [
        r"\bcontract\s+Mock\w+\s*\{",
        r"\bcontract\s+Fake\w+\s*\{",
        r"\bcontract\s+\w+Mock\s*\{",
    ]
    for pat in mock_patterns:
        if _re.search(pat, poc_code):
            return "mock"
    return ""


def _read_sources(
    target_dir: str,
    in_scope_files: Optional[list[str]] = None,
    context_files: Optional[list[str]] = None,
    prioritize_functions: Optional[list[str]] = None,
) -> str:
    """Read in-scope and context .sol files for the Skeptic prompt.

    Reads directly from the provided file paths (not via glob) so that
    files in subdirectories (e.g. src/accountants/Foo.sol) are included.
    Context files are appended up to a separate budget to avoid overflowing
    the LLM context window.

    If prioritize_functions is given, files that contain any of those function
    names are sorted first so the relevant code always appears in the prompt.
    """
    from pathlib import Path

    in_scope_paths = list(in_scope_files or [])
    context_paths = list(context_files or [])

    # ── Load all in-scope files ───────────────────────────────────────────
    loaded: list[tuple[str, str]] = []  # (label, content)
    for sol_path in in_scope_paths:
        p = Path(sol_path)
        if not p.exists() and target_dir:
            p = Path(target_dir) / sol_path.lstrip("./")
        if p.exists():
            try:
                content = p.read_text(errors="replace")
                loaded.append((f"// === {p.name} [IN SCOPE] ===\n{content}", content))
            except OSError:
                pass

    # ── Reorder: files containing any affected function name come first ───
    if prioritize_functions:
        fn_names = [f.split("(")[0].strip() for f in prioritize_functions]
        def _priority(item: tuple[str, str]) -> int:
            label, content = item
            return 0 if any(f"function {fn}" in content for fn in fn_names) else 1
        loaded.sort(key=_priority)

    parts: list[str] = [label for label, _ in loaded]

    # ── Context (dependency) files: include up to a char budget ──────────
    context_budget = 80_000
    for sol_path in context_paths:
        if context_budget <= 0:
            break
        p = Path(sol_path)
        if not p.exists() and target_dir:
            p = Path(target_dir) / sol_path.lstrip("./")
        if p.exists():
            try:
                content = p.read_text(errors="replace")[:context_budget]
                if content:
                    parts.append(f"// === {p.name} [context dependency] ===\n{content}")
                    context_budget -= len(content)
            except OSError:
                pass

    # ── Fallback when no explicit scope paths given ───────────────────────
    if not parts and target_dir:
        _tp = Path(target_dir)
        _sd = _tp / "src"
        _source_dir = _sd if _sd.is_dir() else _tp
        for sol in sorted(_source_dir.rglob("*.sol")):
            try:
                content = sol.read_text(errors="replace")
                parts.append(f"// === {sol.name} ===\n{content}")
            except OSError:
                pass

    # Qwen3.7 Max has 1M context — allow much larger source code window
    return "\n\n".join(parts)[:120_000]


def _parse_text_verdict(text: str) -> Optional[dict[str, Any]]:
    """Parse the structured free-text response from _llm_validate_text."""
    import re as _re
    result: dict[str, Any] = {}

    m = _re.search(r"VERDICT:\s*(true_positive|false_positive|needs_review)", text, _re.I)
    if not m:
        return None
    result["verdict"] = m.group(1).lower()

    m = _re.search(r"CONFIDENCE:\s*([0-9.]+)", text)
    result["confidence"] = float(m.group(1)) if m else 0.5

    m = _re.search(r"SEVERITY:\s*(critical|high|medium|low|info)", text, _re.I)
    result["suggested_severity"] = m.group(1).lower() if m else "medium"

    m = _re.search(r"REASONING:\s*(.+?)(?=\nIMPACT:|\Z)", text, _re.S)
    result["reasoning"] = m.group(1).strip() if m else text[:300]

    m = _re.search(r"IMPACT:\s*(.+)", text, _re.S)
    result["real_impact"] = m.group(1).strip()[:300] if m else ""

    return result


def _heuristic_verdict(exploit: ExploitResult) -> tuple[str, float]:
    """Deterministic verdict when LLM is unavailable."""
    if exploit.status == "exploited":
        return "true_positive", 0.70
    if exploit.status == "blocked":
        return "false_positive", 0.80
    if exploit.status == "partial":
        return "needs_review", 0.50
    # status == "error"
    return "needs_review", 0.30


def _heuristic_impact(
    exploit: ExploitResult,
    hypothesis: Optional[Hypothesis],
) -> str:
    if exploit.status == "exploited":
        base = "PoC test passed — exploit may be viable."
    elif exploit.status == "blocked":
        base = "Exploit was blocked by an access control guard."
    elif exploit.status == "partial":
        base = "PoC partially succeeded — requires further investigation."
    else:
        base = "PoC did not run successfully due to an error."

    if hypothesis:
        return f"{base} Scenario: {hypothesis.attack_scenario[:200]}"
    return base
