"""Invariant miner — extracts testable invariants from three sources (§8.3).

Source 1: NatSpec — parsed from SlitherFunction.natspec fields (no LLM).
Source 2: Docs    — LLM extraction from documentation text.
Source 3: Inferred — LLM inference from code patterns.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any

from audit_agent.knowledge.graph.builders import SlitherAnalysis, SlitherContract
from audit_agent.llm import get_provider
from audit_agent.llm.base import LLMProvider, ToolDefinition
from audit_agent.models.schemas import Invariant, InvariantType, InvariantSource
from audit_agent.observability.logging import get_logger

log = get_logger("agents.invariant_miner")

# ─────────────────────────────────────────────────────────────────────────────
# Source 1: NatSpec parser (pure, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that signal an invariant in NatSpec
_INVARIANT_KEYWORDS = re.compile(
    r"(?:@invariant|invariant:|must\s+(?:always|never)|"
    r"always\s+(?:true|hold|equal|be)|never\s+(?:exceed|drop|be)|"
    r"sum\s+of|total(?:\s+\w+)?\s+(?:must|==|<=|>=)|"
    r"cannot\s+(?:exceed|decrease|be\s+negative))",
    re.IGNORECASE,
)

_VULNERABILITY_SIGNALS = re.compile(
    r"VULNERABLE|BUG:|SECURITY:|AUDIT:|HACK:|EXPLOIT:", re.IGNORECASE
)


def extract_natspec_invariants(analysis: SlitherAnalysis, audit_id: str) -> list[Invariant]:
    """Extract invariants from NatSpec @notice / @dev fields.

    Only treats text as an invariant candidate if it contains invariant-like language.
    Vulnerability notices (VULNERABLE:, BUG:) are skipped — those are for Hunter.
    """
    results: list[Invariant] = []

    for contract in analysis.contracts:
        # Contract-level NatSpec
        _extract_from_text(
            text=contract.natspec.notice + " " + contract.natspec.dev,
            affected_functions=[f.id for f in contract.functions],
            source="natspec",
            audit_id=audit_id,
            results=results,
        )
        # Function-level NatSpec
        for fn in contract.functions:
            combined = (fn.natspec.notice + " " + fn.natspec.dev).strip()
            if combined and not _VULNERABILITY_SIGNALS.search(combined):
                _extract_from_text(
                    text=combined,
                    affected_functions=[fn.id],
                    source="natspec",
                    audit_id=audit_id,
                    results=results,
                )

    log.debug("natspec_invariants", count=len(results))
    return results


def _extract_from_text(
    text: str,
    affected_functions: list[str],
    source: InvariantSource,
    audit_id: str,
    results: list[Invariant],
) -> None:
    text = text.strip()
    if not text or len(text) < 15:
        return
    if not _INVARIANT_KEYWORDS.search(text):
        return
    if _VULNERABILITY_SIGNALS.search(text):
        return

    inv = Invariant(
        id=str(uuid.uuid4()),
        audit_id=audit_id,
        invariant_type=_classify_invariant_type(text),
        natural_language=text,
        affected_functions=affected_functions,
        solidity_assert=_generate_assert_stub(text),
        echidna_property=_generate_echidna_stub(text),
        source=source,
    )
    results.append(inv)


def _classify_invariant_type(text: str) -> InvariantType:
    lower = text.lower()
    if any(w in lower for w in ("profit", "loss", "value", "price", "cost", "fee", "balance sum")):
        return "economic"
    if any(w in lower for w in ("after ", "before ", "when ", "deposit", "withdraw", "transfer")):
        return "transition"
    return "global"


def _generate_assert_stub(text: str) -> str:
    # Best-effort: wrap text in a require comment for Foundry
    safe = re.sub(r'["\n]', " ", text[:120])
    return f'// assert: {safe}'


def _generate_echidna_stub(text: str) -> str:
    safe = re.sub(r"\W+", "_", text[:40]).strip("_").lower()
    return f"function echidna_{safe}() public returns (bool) {{ return true; /* TODO */ }}"


# ─────────────────────────────────────────────────────────────────────────────
# LLM extraction tool definition
# ─────────────────────────────────────────────────────────────────────────────

_INVARIANT_TOOL = ToolDefinition(
    name="extract_invariants",
    description="Extract testable invariants from protocol documentation and code.",
    parameters={
        "type": "object",
        "required": ["invariants"],
        "properties": {
            "invariants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["natural_language", "invariant_type", "affected_functions",
                                 "solidity_assert", "echidna_property", "confidence"],
                    "properties": {
                        "natural_language": {"type": "string"},
                        "invariant_type": {
                            "type": "string",
                            "enum": ["global", "transition", "economic"],
                        },
                        "affected_functions": {
                            "type": "array", "items": {"type": "string"},
                        },
                        "solidity_assert": {"type": "string"},
                        "echidna_property": {"type": "string"},
                        "halmos_spec": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
            },
        },
    },
)

_DOCS_SYSTEM = """\
You are a formal verification specialist extracting testable invariants from protocol documentation.
Extract only invariants that are objectively checkable in code — not design goals or aspirations.
"""

_INFERRED_SYSTEM = """\
You are a formal verification specialist. Analyze the following Solidity contract code and infer
implicit invariants — properties that the code appears to assume are always true, even if not
documented. Mark your confidence: 0.3-0.6 for weak inferences, 0.7-0.9 for strong ones.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 + 3: LLM-based extraction
# ─────────────────────────────────────────────────────────────────────────────

class InvariantMiner:
    """Three-source invariant extractor."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        model: str | None = None,
    ) -> None:
        self._provider = provider or get_provider()
        if model is None:
            from audit_agent.config import get_settings
            model = get_settings().scout_model
        self._model = model

    async def mine_all(
        self,
        analysis: SlitherAnalysis,
        audit_id: str,
        docs_text: str = "",
        target_dir: str = "",
    ) -> list[Invariant]:
        """Run all extraction passes and merge results."""
        results: list[Invariant] = []

        # Source 1: NatSpec (no LLM)
        natspec = extract_natspec_invariants(analysis, audit_id)
        results.extend(natspec)
        log.info("invariant_miner.natspec", count=len(natspec))

        # Source 2: Deterministic arithmetic invariants (no LLM)
        all_fns = [fn for c in analysis.contracts for fn in c.functions]
        arithmetic = mine_arithmetic_invariants(all_fns, audit_id, target_dir=target_dir)
        results.extend(arithmetic)
        log.info("invariant_miner.arithmetic", count=len(arithmetic))

        # Source 3: Documentation (LLM)
        if docs_text.strip():
            docs_invs = await self._extract_from_docs(analysis, docs_text, audit_id)
            results.extend(docs_invs)
            log.info("invariant_miner.docs", count=len(docs_invs))

        # Source 4: Inferred from code (LLM)
        inferred = await self._extract_inferred(analysis, audit_id)
        results.extend(inferred)
        log.info("invariant_miner.inferred", count=len(inferred))

        return results

    async def extract_from_docs(
        self, analysis: SlitherAnalysis, docs_text: str, audit_id: str
    ) -> list[Invariant]:
        return await self._extract_from_docs(analysis, docs_text, audit_id)

    async def extract_inferred(
        self, analysis: SlitherAnalysis, audit_id: str
    ) -> list[Invariant]:
        return await self._extract_inferred(analysis, audit_id)

    async def _extract_from_docs(
        self, analysis: SlitherAnalysis, docs_text: str, audit_id: str
    ) -> list[Invariant]:
        fn_names = [fn.name for c in analysis.contracts for fn in c.functions]
        prompt = (
            f"Documentation:\n{docs_text[:6000]}\n\n"
            f"Available functions: {', '.join(fn_names[:30])}\n\n"
            "Extract all testable invariants from this documentation."
        )
        raw = await self._llm_extract(prompt, _DOCS_SYSTEM)
        return self._build_invariants(raw, audit_id, "docs")

    async def _extract_inferred(
        self, analysis: SlitherAnalysis, audit_id: str
    ) -> list[Invariant]:
        # Build a compact code summary for the LLM
        code_summary = _summarize_for_llm(analysis)
        prompt = (
            f"Contract code summary:\n{code_summary}\n\n"
            "Infer implicit invariants from these state variables and functions."
        )
        raw = await self._llm_extract(prompt, _INFERRED_SYSTEM)
        return self._build_invariants(raw, audit_id, "inferred")

    async def _llm_extract(self, user_prompt: str, system: str) -> list[dict[str, Any]]:
        try:
            response = await asyncio.wait_for(
                self._provider.complete(
                    model=self._model,
                    max_tokens=2048,
                    system=system,
                    tools=[_INVARIANT_TOOL],
                    tool_choice={"name": "extract_invariants"},
                    messages=[{"role": "user", "content": user_prompt}],
                ),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            log.warning("invariant_miner.llm_timeout")
            return []
        except Exception as exc:
            log.warning("invariant_miner.llm_unavailable", error=str(exc))
            return []
        for tc in response.tool_calls:
            if tc.name == "extract_invariants":
                return tc.arguments.get("invariants", [])
        return []

    def _build_invariants(
        self,
        raw: list[dict[str, Any]],
        audit_id: str,
        source: InvariantSource,
    ) -> list[Invariant]:
        results: list[Invariant] = []
        for item in raw:
            if not item.get("natural_language"):
                continue
            results.append(Invariant(
                id=str(uuid.uuid4()),
                audit_id=audit_id,
                invariant_type=item.get("invariant_type", "global"),  # type: ignore[arg-type]
                natural_language=item["natural_language"],
                affected_functions=item.get("affected_functions", []),
                solidity_assert=item.get("solidity_assert", ""),
                echidna_property=item.get("echidna_property", ""),
                halmos_spec=item.get("halmos_spec"),
                source=source,
            ))
        return results


_FEE_KEYWORDS = ("fee", "rate", "reward", "yield", "price", "share", "ratio", "interest")
_DIV_LARGE = re.compile(r"/\s*(\d+)")
_FN_BODY_RE = re.compile(r"function\s+{name}\s*\(", re.DOTALL)


def _read_fn_source(fn, target_dir: str) -> str:  # type: ignore[type-arg]
    """Return source text for a SlitherFunction, with three fallbacks.

    Priority:
    1. fn.source_code (populated from fixture JSON)
    2. fn.source_location.file + line range (real Slither output)
    3. Scan all *.sol files for `function <name>(` and extract body
    """
    from pathlib import Path as _Path

    src = (getattr(fn, "source_code", "") or "").strip()
    if src:
        return src

    if not target_dir:
        return ""

    target = _Path(target_dir)

    # Fallback 2: source_location line range
    loc = getattr(fn, "source_location", None)
    if loc:
        raw_file = getattr(loc, "file", "") or ""
        start_line = getattr(loc, "start_line", 0)
        end_line = getattr(loc, "end_line", 0)
        if raw_file and start_line and end_line:
            # source_location.file may be relative ("contracts/Foo.sol") or absolute
            candidates = [_Path(raw_file), target / raw_file, target / _Path(raw_file).name]
            for candidate in candidates:
                try:
                    if candidate.exists():
                        lines = candidate.read_text().splitlines()
                        src = "\n".join(lines[start_line - 1: end_line])
                        if src.strip():
                            return src
                except OSError:
                    pass

    # Fallback 3: scan *.sol files for the function by name
    fn_name: str = fn.name
    pattern = re.compile(r"function\s+" + re.escape(fn_name) + r"\s*\(")
    for sol_file in sorted(target.rglob("*.sol")):
        try:
            content = sol_file.read_text()
        except OSError:
            continue
        m = pattern.search(content)
        if not m:
            continue
        # Extract balanced braces → function body
        start_pos = m.start()
        brace_count = 0
        for i, ch in enumerate(content[start_pos:], start_pos):
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    return content[start_pos: i + 1]
        # If braces never closed, return what we found so far
        return content[start_pos: start_pos + 500]

    return ""


def mine_arithmetic_invariants(
    functions: list,  # list[SlitherFunction]
    audit_id: str,
    target_dir: str = "",
) -> list[Invariant]:
    """Deterministically detect integer-division precision-loss risks.

    For every function whose source contains division:
    - If the function name is fee/rate/reward-related → emit a non-zero economic invariant.
    - For every large divisor (≥100) found in source → emit a truncation-warning invariant.

    Works even when fn.source_code is empty: reads from source_location or .sol files.
    Does not call the LLM.
    """
    has_source = [bool((getattr(fn, "source_code", "") or "").strip()) for fn in functions]
    log.info(
        "arithmetic_miner.input",
        num_functions=len(functions),
        function_names=[fn.name for fn in functions],
        has_source=has_source,
        target_dir=target_dir or "(none)",
    )

    results: list[Invariant] = []
    for fn in functions:
        src = _read_fn_source(fn, target_dir)
        if not src or "/" not in src:
            continue

        fn_name: str = fn.name
        fn_id: str = fn.id

        log.debug("arithmetic_miner.source_found", fn=fn_name, src_len=len(src))

        # --- Fee/rate function → must-be-nonzero invariant ---
        if any(kw in fn_name.lower() for kw in _FEE_KEYWORDS):
            results.append(Invariant(
                id=str(uuid.uuid4()),
                audit_id=audit_id,
                invariant_type="economic",
                natural_language=(
                    f"{fn_name}() must return a non-zero fee for any non-zero input "
                    f"when the rate/divisor is non-zero. "
                    f"Integer division truncation to 0 is a precision loss vulnerability."
                ),
                affected_functions=[fn_id],
                solidity_assert=(
                    f"// PRECISION CHECK: assert({fn_name}(amount) > 0 || amount == 0 || rate == 0)"
                ),
                echidna_property=(
                    f"function echidna_{fn_name}_nonzero() public returns (bool) {{\n"
                    f"    return {fn_name}(1000) > 0;\n}}"
                ),
                source="inferred",
                confidence=0.85,
            ))

        # --- Any large divisor → truncation-warning invariant ---
        for dm in _DIV_LARGE.finditer(src):
            divisor = int(dm.group(1))
            if divisor < 100:
                continue
            results.append(Invariant(
                id=str(uuid.uuid4()),
                audit_id=audit_id,
                invariant_type="economic",
                natural_language=(
                    f"{fn_name}() divides by {divisor}. "
                    f"For inputs smaller than {divisor}, integer division truncates to 0 — "
                    f"fee/reward/share is silently lost. "
                    f"Check: (small_amount * rate) / {divisor} == 0 when small_amount < {divisor} / rate."
                ),
                affected_functions=[fn_id],
                solidity_assert=(
                    f"// TRUNCATION RISK: ({fn_name} input * rate) / {divisor} "
                    f"truncates to 0 for small inputs"
                ),
                echidna_property="",
                source="inferred",
                confidence=0.90,
            ))

    log.info("arithmetic_miner.output", count=len(results))
    return results


def _summarize_for_llm(analysis: SlitherAnalysis, max_chars: int = 5000) -> str:
    """Build a compact text summary of the contracts for LLM context."""
    parts: list[str] = []
    for c in analysis.contracts:
        parts.append(f"Contract: {c.name}")
        svs = [f"  var {sv.name}: {sv.type}" for sv in c.state_variables[:10]]
        parts.extend(svs)
        fns = [
            f"  fn {fn.name}({', '.join(p.get('type','') for p in fn.parameters)}) "
            f"[{fn.visibility}/{fn.state_mutability}]"
            + (f" writes=[{','.join(fn.writes[:3])}]" if fn.writes else "")
            + (f" external_calls={len(fn.external_calls)}" if fn.external_calls else "")
            for fn in c.functions[:15]
        ]
        parts.extend(fns)
    summary = "\n".join(parts)
    return summary[:max_chars]
