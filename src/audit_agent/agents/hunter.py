"""Hunter agent — vulnerability hypothesis generator (§3.3).

Workflow per function (sorted by attack_surface_score desc):
  1. build_context()     → SubgraphContext (minimal code slice via BFS)
  2. _llm_call()         → raw hypothesis dicts via tool_use
  3. _deduplicate()      → remove overlap with static findings
  4. sort by priority    → top-N returned

All external dependencies (LLM, Qdrant, graph) are injected → fully testable offline.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Optional

from audit_agent.agents.base import BaseAgent
from audit_agent.graph.state import AuditState
from audit_agent.knowledge.graph.builders import SlitherAnalysis, parse_slither_json
from audit_agent.knowledge.graph.context import (
    GraphAdapter,
    InMemoryGraph,
    NullGraphAdapter,
    SubgraphContext,
    SubgraphContextBuilder,
)
from audit_agent.config import get_settings
from audit_agent.llm import get_provider
from audit_agent.llm.base import LLMProvider, ToolDefinition
from audit_agent.models.schemas import Hypothesis, StaticFinding
from audit_agent.observability.audit_logger import get_audit_logger
from audit_agent.observability.logging import get_logger

log = get_logger("agents.hunter")


async def _gather_limited(coros: list, sem: "asyncio.Semaphore") -> list:
    """Run coroutines concurrently, gated by a SHARED semaphore.

    Passing one semaphore across all Hunter passes makes max_concurrency a true
    global ceiling on simultaneous in-flight LLM calls (not per-group). Results
    preserve input order; exceptions are captured so one failing pass never
    aborts the others — callers filter non-list results.
    """
    if not coros:
        return []

    async def _run(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*(_run(c) for c in coros), return_exceptions=True)


def _threat_targets_and_summary(threat_model: Any) -> tuple[set[str], str]:
    """From a ThreatModel, return (prioritized function base-names, prompt summary)."""
    if threat_model is None:
        return set(), ""
    targets: set[str] = set()
    for t in getattr(threat_model, "prioritized_targets", []) or []:
        fn = getattr(t, "function", "") or ""
        base = fn.split("(")[0].split(".")[-1].strip().lower()
        if base:
            targets.add(base)
    return targets, _format_threat_summary(threat_model)


def _format_threat_summary(threat_model: Any) -> str:
    """Compact, prompt-friendly rendering of the threat model."""
    if threat_model is None:
        return ""
    parts: list[str] = []
    actors = getattr(threat_model, "actors", []) or []
    if actors:
        parts.append("Actors: " + "; ".join(
            f"{getattr(a, 'name', '')}" + (f" ({getattr(a, 'capability', '')})" if getattr(a, 'capability', '') else "")
            for a in actors[:6]
        ))
    assets = getattr(threat_model, "assets", []) or []
    if assets:
        parts.append("Value/assets: " + ", ".join(map(str, assets[:10])))
    deps = getattr(threat_model, "external_dependencies", []) or []
    if deps:
        parts.append("Trusted external deps: " + ", ".join(map(str, deps[:10])))
    targets = getattr(threat_model, "prioritized_targets", []) or []
    if targets:
        parts.append("High-value targets: " + "; ".join(
            f"{getattr(t, 'function', '')} [{getattr(t, 'value_at_risk', '')}]"
            + (f" — {getattr(t, 'reason', '')}" if getattr(t, 'reason', '') else "")
            for t in targets[:10]
        ))
    return "\n".join(parts)


def _navigator_lead_prompt(
    fn: Any,
    contract_name: str,
    navigator: Any,
    derived_invariants: list[str],
    documentation: str,
    threat_summary: str = "",
) -> str:
    """Seed prompt for one navigator investigation, anchored on a lead function."""
    facts = navigator.get_function_facts(fn.name)
    inv_section = ""
    if derived_invariants:
        inv_section = "\n## Protocol invariants (must always hold)\n" + "\n".join(
            f"- {i}" for i in derived_invariants[:8]
        )
    doc = (documentation or "").strip()
    doc_section = f"\n## Project documentation (excerpt)\n{doc[:1500]}" if doc else ""
    threat_section = f"\n## Threat model (where value concentrates)\n{threat_summary}" if threat_summary else ""
    return (
        f"LEAD: Investigate `{contract_name}.{fn.name}` and everything it touches for "
        f"security vulnerabilities — especially multi-step / cross-function bugs.\n\n"
        f"## Starting function facts\n{facts}\n"
        f"{inv_section}{threat_section}{doc_section}\n\n"
        "Begin. Pull whatever code you need with the tools, follow the data and call "
        "edges (callers, state writers, callees), and report concrete, exploitable "
        "hypotheses when ready."
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM tool definition
# ─────────────────────────────────────────────────────────────────────────────

_HYPOTHESIS_TOOL = ToolDefinition(
    name="report_hypotheses",
    description=(
        "Report vulnerability hypotheses found in the analyzed code slice. "
        "Call this exactly once with all hypotheses for this function."
    ),
    parameters={
        "type": "object",
        "required": ["hypotheses"],
        "properties": {
            "hypotheses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "title",
                        "vulnerability_class",
                        "affected_functions",
                        "attack_scenario",
                        "required_preconditions",
                        "estimated_severity",
                        "priority_score",
                    ],
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Specific title, e.g. 'Reentrancy in withdraw() via msg.sender.call'",
                        },
                        "vulnerability_class": {
                            "type": "string",
                            "description": "SWC-ID or class name, e.g. 'SWC-107'",
                        },
                        "affected_functions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Function IDs from the code slice",
                        },
                        "attack_scenario": {
                            "type": "string",
                            "description": "Step-by-step attack description",
                        },
                        "required_preconditions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "estimated_severity": {
                            "type": "string",
                            "enum": ["critical", "high", "medium", "low"],
                        },
                        "similar_historical_cases": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "priority_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 100.0,
                            "description": "0-100 priority; higher = analyze first",
                        },
                    },
                },
            },
        },
    },
)

_HUNTER_SYSTEM = """You are a world-class smart contract security auditor with 10+ years of DeFi bug bounty experience.

## PHASE 1: UNDERSTAND INTENT (do this FIRST, before looking for bugs)

For each function in the code slice:
1. What is this function SUPPOSED to do? (based on name, NatSpec, documentation, context)
2. What INVARIANTS should hold? (e.g., "rewards only for users who participated during the epoch")
3. What are the TRUST BOUNDARIES? (who calls this, with what authority?)
4. What are the IMPLICIT ASSUMPTIONS? (e.g., "epoch boundaries are exclusive",
   "balances are snapshotted at epoch end", "oracle prices are fresh")

## PHASE 2: CHALLENGE EVERY ASSUMPTION

Now systematically try to break each assumption from Phase 1:

### Temporal attacks (CHECK THESE EXPLICITLY for ANY time-related code):
- What happens at EXACT epoch/period boundaries? (first second, last second, transition moment)
- Can an action in epoch N affect rewards/balances for epoch N-1?
- Does the function use block.timestamp? What if it equals a boundary exactly?
- Can a user perform action X, wait for a state transition, then perform action Y
  to get benefits from both states?
- Are snapshots taken at the right moment, or can they be manipulated by acting
  right before/after the snapshot?
- Does epochEnd(N) == epochStart(N+1)? If yes, which epoch "owns" that moment?

### Economic/Logic attacks:
- Can a user get rewards without actually providing value during the relevant period?
- Can first/last depositor manipulate share prices or reward distribution?
- Does the function assume values are denominated in the same units? Verify.
- Can rounding direction differences between initialization and runtime be exploited?
  (e.g., initialize() rounds down but previewMint() rounds up → boundary unreachable)
- Is there a path where a view function returns stale data used by a state-changing function?
- Are there feedback loops? (e.g., claiming rewards affects next epoch's target)

### Invariant-violation analysis (CHECK WHEN Protocol Invariants are provided):
For each listed invariant, ask: can this function violate it with any input sequence?

**Aggregation + floor bugs** (common in vault/NAV functions):
- Does this function sum components with `total += component_i`?
- Is a floor applied after aggregation: `max(total, 0)` or `uint256(int256(total))`?
- Within the floored expression, can ALWAYS-POSITIVE components (e.g. free token balance)
  be mixed with POSSIBLY-NEGATIVE ones (e.g. unrealized losses)?
- When the sum goes below zero, the floor zeros out the ENTIRE sum, discarding the positives.
- Key: `max(asset + liability, 0)` loses `asset` when `liability` dominates. This is wrong
  if `asset` (e.g. cash in wallet) should always be counted regardless of P&L.

**Deduplication skip-flag bugs**:
- Is there a "skip" or "seen" mechanism (array, bool, mapping) that prevents double-counting?
- Was the skip flag set because the value was SUPPOSEDLY counted in an aggregation above?
- If that aggregation applied a floor and discarded the value, the flag is now incorrect:
  it says "counted" when the value was actually zeroed out.
- The fallback path that would recover the value is now blocked by the flag.
- Result: value counted zero times, not once.

**Conditional inclusion bugs**:
- For any `if (!skip) { nav += balance }` or similar guard:
  is there a scenario where `skip = true` but the balance was never actually included?

### Cross-contract attacks:
- If function A reads state set by contract B, can B's state change between
  A's read and A's use of that value?
- Are external calls to other contracts trusting their return values?
- If contract X snapshots at time T, does contract Y also snapshot at time T,
  or is there a gap an attacker can exploit?
- Do two contracts define the same concept (epoch, balance, price) differently?

### Cryptographic / Encoding attacks:
- Are ALL fields that are parsed from a signature/message also included in the signed hash?
  If ANY parsed field is NOT in the hash → attacker can modify it without invalidating signature
- Is abi.encodePacked used with variable-length types? → hash collision risk
- Is the hash domain-separated (EIP-712 / chain-specific)?
- Can the same signature be replayed on a different chain/contract?
- Is there a nonce that increments? Is deadline checked AND included in the signed hash?

## VULNERABILITY CLASSIFICATION — use the MOST SPECIFIC SWC ID

When filling `vulnerability_class`, pick the SWC ID that matches the ROOT CAUSE:

| Root cause | Correct class | WRONG class |
|---|---|---|
| External call before state update; malicious receiver re-enters | SWC-107 (Reentrancy) | — |
| Missing/wrong access restriction (onlyOwner, modifier) | SWC-115 (Authorization) | SWC-107 |
| No zero-address / range check on input parameter | Missing Input Validation | SWC-107 |
| block.timestamp / blockhash as RNG or for logic | SWC-116 (Timestamp Dep.) | SWC-107 |
| Integer overflow/underflow without SafeMath | SWC-101 (Integer Overflow) | SWC-107 |
| Division truncation / precision loss | Precision Loss | SWC-107 |
| Griefing via revert in fallback | SWC-113 (DoS) | SWC-107 |

**SWC-107 (Reentrancy) REQUIRES ALL of these**:
1. An external call (`.call`, `.transfer`, `token.transfer`, interface call, or ERC-777 hook)
2. The call happens BEFORE a critical state update (checks-effects-interactions violation)
3. A malicious contract can re-enter and the second entry finds inconsistent state

If ANY of the three is missing, DO NOT use SWC-107. Use the most specific alternative above.

## OUTPUT RULES
- Each hypothesis MUST state which assumption from Phase 1 it violates
- Each hypothesis MUST include a concrete TIMELINE of attacker actions (step-by-step)
- Priority 90-100: Proven violation of a stated invariant with clear profit/loss
- Priority 70-89: Likely violation requiring specific timing or state conditions
- Priority 50-69: Possible design concern requiring manual verification
- Do NOT report anything below priority 50
- Do NOT report generic issues (gas optimization, style, informational)
- Do NOT report vulnerabilities in OpenZeppelin/Solady — focus on custom protocol logic
- NEVER generate hypotheses that require admin/owner cooperation
- Generate 2-5 hypotheses per function call

"""

# Appended so Hunter avoids generating cross-protocol false positives at the source.
from audit_agent.knowledge.external_behaviors import EXTERNAL_PROTOCOL_FACTS as _EXT_FACTS
_HUNTER_SYSTEM = _HUNTER_SYSTEM + "\n" + _EXT_FACTS

# Minimal system prompt used only in timeout retries where the full prompt
# would again exceed the model's generation budget.
_HUNTER_SYSTEM_TRIAGE = (
    "You are a smart contract security auditor. "
    "Find vulnerabilities and report them by calling the report_hypotheses tool. "
    "Output ONLY the tool call — no prose."
)

_FIN_RETRY_HINT = (
    "\n\nCRITICAL — check sign conventions: for every subtraction A-B, verify the order. "
    "Example: shortPremium-longPremium = net owed to protocol (CORRECT); "
    "longPremium-shortPremium inverts the sign and overstates NAV (BUG). "
    "Also: if token0 uses (short-long) but token1 uses (long-short), that slot mismatch "
    "is a sign inversion bug even if each line looks locally reasonable. "
    "Report any sign inversion or double-count as HIGH/CRITICAL."
)
_FIN_RETRY_KWS = frozenset({
    "premium", "exposure", "longamount", "shortamount", "nav", "totalnav",
    "poolexposure", "computenav", "longpremium", "shortpremium", "haircut",
})


# ─────────────────────────────────────────────────────────────────────────────
# MoE: expert system prompts + rule-based router
# ─────────────────────────────────────────────────────────────────────────────

_EXPERT_REGISTRY: dict[str, str] = {
    "reentrancy": (
        "You are a reentrancy and CEI violation specialist for smart contracts.\n\n"
        "FOCUS EXCLUSIVELY on:\n"
        "- Classic reentrancy (SWC-107): external call BEFORE state update; re-entry sees inconsistent state.\n"
        "  Requires ALL THREE: (1) external call (.call, .transfer, token.transfer, ERC-777/1155 hook),\n"
        "  (2) call happens before critical state update, (3) re-entry exploits the inconsistency.\n"
        "- Cross-function reentrancy: callback re-enters a DIFFERENT function that reads stale state.\n"
        "- Read-only reentrancy: view function called by external protocol during mid-update state.\n\n"
        "DO NOT report access control, arithmetic, oracle, or MEV issues.\n"
        "Output ONLY the tool call with 1-2 high-confidence findings. "
        "If none found, call report_hypotheses with an empty list."
    ),
    "arithmetic": (
        "You are an arithmetic and precision loss specialist for smart contracts.\n\n"
        "FOCUS EXCLUSIVELY on:\n"
        "- Divide-before-multiply: precision lost because division truncates before multiplication.\n"
        "- Integer overflow/underflow in unchecked blocks or pre-0.8 code (SWC-101).\n"
        "- Rounding direction inconsistency: deposit rounds down, withdraw rounds up → boundary unreachable.\n"
        "- Decimal mismatch: different token decimals (6 vs 18) mixed in the same arithmetic.\n"
        "- WAD/RAY confusion: 1e18 and 1e27 denominators mixed without conversion.\n"
        "- First depositor inflation: totalSupply=0 edge case lets attacker manipulate share price.\n\n"
        "DO NOT report reentrancy, access control, oracle manipulation, or signature issues.\n"
        "Output ONLY the tool call with 1-2 high-confidence findings. "
        "If none found, call report_hypotheses with an empty list."
    ),
    "access_control": (
        "You are an access control and authorization specialist for smart contracts.\n\n"
        "FOCUS EXCLUSIVELY on:\n"
        "- Missing or wrong access modifier on a sensitive function (SWC-115).\n"
        "- Incorrect role: function uses roleA but should require roleB.\n"
        "- tx.origin authentication (bypassable by any intermediate contract).\n"
        "- Modifier bypass: modifier applied to wrong function, or has a path that skips the check.\n"
        "- Unauthorized initialization: initialize() callable by anyone post-deployment.\n"
        "- Role escalation: a role can grant itself or others higher privileges.\n\n"
        "DO NOT report reentrancy, arithmetic, oracle, or signature issues.\n"
        "Output ONLY the tool call with 1-2 high-confidence findings. "
        "If none found, call report_hypotheses with an empty list."
    ),
    "cryptography": (
        "You are a cryptographic vulnerability specialist for smart contracts.\n\n"
        "FOCUS EXCLUSIVELY on:\n"
        "- Unsigned field: a field is parsed from the message/signature bytes but NOT included\n"
        "  in the hash passed to ecrecover/ECDSA.recover/tryRecover. The attacker can modify\n"
        "  this field without invalidating the signature. KEY: 'field is USED in logic' is NOT\n"
        "  the same as 'field is AUTHENTICATED'. Check every field extracted from sig bytes.\n"
        "- Signature fallback sentinel: a parsing function returns a default/zero value "
        "(e.g. validUntil=0, nonce=0) for unrecognized or short signature formats. "
        "If 0 has a special permissive meaning in the protocol (e.g. EIP-4337: validUntil=0 "
        "means 'never expires'), then every standard-format signature effectively has that "
        "permissive behaviour — report as unsigned-field / bypass.\n"
        "- Signature malleability: raw ecrecover without checking s ≤ secp256k1n/2.\n"
        "- Hash collision via abi.encodePacked with variable-length types (bytes, string, arrays).\n"
        "- Missing domain separator: hash not chain-specific or contract-specific → replay attacks.\n"
        "- Deadline not in hash: deadline checked but not signed → attacker extends existing signatures.\n"
        "- Nonce not incremented or absent: signature reuse attack.\n\n"
        "DO NOT report reentrancy, arithmetic, access control, or oracle issues.\n"
        "Output ONLY the tool call with 1-2 high-confidence findings. "
        "If none found, call report_hypotheses with an empty list."
    ),
    "oracle_price": (
        "You are a price oracle and external data feed specialist for smart contracts.\n\n"
        "FOCUS EXCLUSIVELY on:\n"
        "- Spot price manipulation: protocol reads Uniswap/Balancer spot price in same block it's used.\n"
        "- TWAP window too short: time-weighted average manipulatable with sufficient capital.\n"
        "- Stale price: updatedAt timestamp from Chainlink not validated.\n"
        "- No L2 sequencer uptime check: price may be stale after sequencer downtime.\n"
        "- Price as balanceOf: using token.balanceOf() or getReserves() as price → flash loan attack.\n"
        "- Unchecked oracle return: latestRoundData() answer not validated for zero or negative.\n\n"
        "DO NOT report reentrancy, signature, access control, or arithmetic issues.\n"
        "Output ONLY the tool call with 1-2 high-confidence findings. "
        "If none found, call report_hypotheses with an empty list."
    ),
    "economic_mev": (
        "You are an economic security and MEV specialist for smart contracts.\n\n"
        "FOCUS EXCLUSIVELY on:\n"
        "- Sandwich attack: transaction predictably moves a price; attacker front- and back-runs.\n"
        "- Donation/inflation attack: direct token.transfer() to vault inflates share price.\n"
        "- Missing slippage protection: no minimum output amount on swaps.\n"
        "- Flash loan attack: attacker manipulates protocol state within one transaction.\n"
        "- Reward front-running: depositing right before snapshot to claim share without contributing.\n"
        "- Liquidation race: front-running under-collateralized positions visible in mempool.\n\n"
        "DO NOT report reentrancy, signature, oracle staleness, access control, or arithmetic precision.\n"
        "Output ONLY the tool call with 1-2 high-confidence findings. "
        "If none found, call report_hypotheses with an empty list."
    ),
    "time_epoch": (
        "You are a temporal logic and epoch boundary specialist for smart contracts.\n\n"
        "FOCUS EXCLUSIVELY on:\n"
        "- Epoch boundary ownership: epochEnd(N) == epochStart(N+1) — an action at that timestamp\n"
        "  affects TWO epochs; which one 'owns' it?\n"
        "- Off-by-one in period definitions: >= vs > in boundary comparisons.\n"
        "- Snapshot timing: snapshot at block T but underlying value changes at block T in same tx.\n"
        "- Retroactive modification: action in epoch N+1 changes accounting for completed epoch N.\n"
        "- block.timestamp dependence: miner-adjustable within ~15 seconds (SWC-116).\n"
        "- Cooldown bypass: timestamp arithmetic wraps, or user acts twice within the same block.\n\n"
        "DO NOT report reentrancy, arithmetic, access control, oracle, or MEV issues.\n"
        "Output ONLY the tool call with 1-2 high-confidence findings. "
        "If none found, call report_hypotheses with an empty list."
    ),
    "cross_contract": (
        "You are a cross-contract interaction and composability specialist for smart contracts.\n\n"
        "FOCUS EXCLUSIVELY on:\n"
        "- Delegatecall to untrusted contract: storage layout mismatch, malicious implementation.\n"
        "- Callback injection: external contract called mid-update manipulates state before return.\n"
        "- Interface trust: protocol trusts external return value without validation.\n"
        "- Upgrade hazard: implementation slot overwritable, or storage layout breaks on upgrade.\n"
        "- Cross-contract invariant gap: contract A snapshots at T, contract B at T+1 → exploitable.\n"
        "- Re-initialization after upgrade: initialize() callable again on new implementation.\n\n"
        "DO NOT report direct reentrancy (SWC-107), arithmetic, access control, or oracle issues.\n"
        "Output ONLY the tool call with 1-2 high-confidence findings. "
        "If none found, call report_hypotheses with an empty list."
    ),
}


def _select_experts(code_slice: str) -> list[str]:
    """Rule-based router: return relevant expert names for the given code slice.

    Pure keyword matching — no LLM cost. Designed to have low false-negative rate
    (better to run an expert unnecessarily than to miss a finding class).
    """
    code = code_slice.lower()
    experts: list[str] = []

    # Reentrancy: low-level call patterns
    if any(kw in code for kw in (".call{", ".call(", "delegatecall(", "staticcall(")):
        experts.append("reentrancy")

    # Arithmetic: explicit precision/scale markers
    if any(kw in code for kw in ("1e18", "1e6", "1e8", "wad", " ray", "_wad", "_ray",
                                  "decimals()", "pricescale", "pricedecimals", "scaledby")):
        experts.append("arithmetic")

    # Access control: role/modifier/owner patterns
    if any(kw in code for kw in ("onlyowner", "onlyrole(", "hasrole(", "require(msg.sender",
                                  "accesscontrol", "_checkrole(", "tx.origin")):
        experts.append("access_control")

    # Cryptography: signature/hash patterns
    if any(kw in code for kw in ("ecrecover(", "ecdsa.", ".recover(", "keccak256(abi.encode",
                                  "encodepacked(", "eip712", "domainseparator", "validuntil",
                                  "validafter", "packvalidationdata")):
        experts.append("cryptography")

    # Oracle/price: price feed interactions
    if any(kw in code for kw in ("latestrounddata(", "latestanswer(", "getrounddata(",
                                  "getprice(", "twap", "getreserves(", "pricefeed",
                                  "ioracle", "ipricefeed")):
        experts.append("oracle_price")

    # Economic/MEV: DeFi swap/borrow patterns
    if any(kw in code for kw in ("flashloan(", "swap(", "addliquidity(", "removeliquidity(",
                                  "borrow(", "liquidat", "minamountout", "minout",
                                  "iuniswap", "ibalancer")):
        experts.append("economic_mev")

    # Time/epoch: temporal patterns
    if any(kw in code for kw in ("epoch", "block.timestamp", "snapshot", "checkpoint",
                                  "cooldown", "deadline", "vest", "cliff", "epochstart",
                                  "epochend", "epochlength")):
        experts.append("time_epoch")

    # Cross-contract: proxy/delegate/upgrade patterns
    if any(kw in code for kw in ("delegatecall(", "upgradeable", "implementation(",
                                  "_proxy", "initialize(", "upgradeability", "beacon")):
        experts.append("cross_contract")

    return experts


# ─────────────────────────────────────────────────────────────────────────────
# HunterAgent
# ─────────────────────────────────────────────────────────────────────────────

class HunterAgent(BaseAgent):
    """Generates prioritized vulnerability hypotheses for all high-surface functions."""

    def __init__(
        self,
        graph_adapter: Optional[GraphAdapter] = None,
        qdrant_search=None,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        max_hypotheses: Optional[int] = None,
        top_functions: Optional[int] = None,
        max_concurrency: Optional[int] = None,
        triager: Optional[Any] = None,
    ) -> None:
        super().__init__()
        settings = get_settings()
        self._graph: GraphAdapter = graph_adapter or NullGraphAdapter()
        self._qdrant_search = qdrant_search
        self._provider = provider or get_provider()
        self._model = model or settings.hunter_model
        # Explicit args win (tests pass ints); otherwise fall back to settings.
        self._max_hypotheses = max_hypotheses if max_hypotheses is not None else settings.hunter_max_hypotheses
        self._top_functions = top_functions if top_functions is not None else settings.hunter_top_functions
        self._max_concurrency = max(
            1, max_concurrency if max_concurrency is not None else settings.hunter_max_concurrency
        )
        self._navigator_enabled = settings.hunter_navigator_enabled
        self._experts_enabled = settings.hunter_experts_enabled
        self._standalone_enabled = settings.hunter_standalone_passes_enabled
        self._navigator_leads = settings.hunter_navigator_leads
        self._navigator_max_steps = settings.hunter_navigator_max_steps
        # Breadth triage (roadmap #7). Injectable for tests; lazily real otherwise.
        self._triage_enabled = getattr(settings, "triage_enabled", True)
        self._triage_weight = getattr(settings, "triage_weight", 20.0)
        self._triager = triager
        if self._triager is None and self._triage_enabled:
            from audit_agent.agents.triage import FunctionTriager
            self._triager = FunctionTriager(
                provider=self._provider,
                model=getattr(settings, "triage_model", "") or settings.scout_model,
                batch_size=getattr(settings, "triage_batch_size", 25),
                max_concurrency=self._max_concurrency,
            )

    # ── LangGraph node entry point ─────────────────────────────────────────

    async def run(self, state: AuditState) -> AuditState:
        audit_id = state.get("audit_id", "unknown")
        self.log.info("Hunter: starting", audit_id=audit_id)

        al = get_audit_logger()
        ar = al.log_agent_start("hunter", 3, {
            "audit_id": audit_id,
            "static_findings": len(state.get("static_findings") or []),
            "invariants": len(state.get("invariants") or []),
            "scope_files": len(state.get("scope_in_scope") or []),
        }) if al else None

        analysis = self._analysis_from_state(state)
        if analysis is None:
            self.log.warning("Hunter: no slither analysis in state — returning empty hypotheses")
            if al and ar:
                al.log_agent_end(ar, {"hypotheses_count": 0},
                                 warnings=["no slither analysis in state"])
            return {**state, "hypotheses": []}

        graph = InMemoryGraph(analysis)
        context_builder = SubgraphContextBuilder(
            graph=graph,
            qdrant_search=self._qdrant_search,
            target_dir=state.get("target_dir", ""),
        )

        static_findings: list[StaticFinding] = list(state.get("static_findings") or [])
        invariants = list(state.get("invariants") or [])
        fork_protocol: Optional[str] = None
        fa = state.get("fork_analysis")
        if fa and getattr(fa, "is_fork", False):
            fork_protocol = fa.base_protocol

        documentation = (state.get("documentation") or "")[:10_000]
        in_scope_files: list[str] = list(state.get("scope_in_scope") or [])
        context_files: list[str] = list(state.get("scope_context_files") or [])

        # Threat model (roadmap #6): bias analysis toward value-concentrated code.
        threat_model = state.get("threat_model")
        priority_targets, threat_summary = _threat_targets_and_summary(threat_model)

        hypotheses = await self._generate(
            analysis=analysis,
            audit_id=audit_id,
            context_builder=context_builder,
            static_findings=static_findings,
            invariants=invariants,
            fork_protocol=fork_protocol,
            documentation=documentation,
            in_scope_files=in_scope_files,
            context_files=context_files,
            target_dir=state.get("target_dir", ""),
            priority_targets=priority_targets,
            threat_summary=threat_summary,
        )

        self.log.info("Hunter: complete", hypotheses=len(hypotheses))

        if al and ar:
            severity_dist: dict[str, int] = {}
            for h in hypotheses:
                k = h.estimated_severity
                severity_dist[k] = severity_dist.get(k, 0) + 1
            al.log_agent_end(ar, {
                "hypotheses_count": len(hypotheses),
                "hypotheses_titles": [h.title for h in hypotheses[:20]],
                "severity_distribution": severity_dist,
            })

        return {**state, "hypotheses": hypotheses}

    # ── Standalone entry point (for tests / ingest) ───────────────────────

    async def generate_hypotheses(
        self,
        analysis: SlitherAnalysis,
        audit_id: str,
        static_findings: Optional[list[StaticFinding]] = None,
        invariants: Optional[list] = None,
        fork_protocol: Optional[str] = None,
        documentation: str = "",
    ) -> list[Hypothesis]:
        """Public API used by tests and CLI."""
        context_builder = SubgraphContextBuilder(
            graph=self._graph, qdrant_search=self._qdrant_search
        )
        return await self._generate(
            analysis=analysis,
            audit_id=audit_id,
            context_builder=context_builder,
            static_findings=static_findings or [],
            invariants=invariants or [],
            fork_protocol=fork_protocol,
            documentation=documentation,
        )

    # ── Core pipeline ─────────────────────────────────────────────────────

    async def _generate(
        self,
        analysis: SlitherAnalysis,
        audit_id: str,
        context_builder: SubgraphContextBuilder,
        static_findings: list[StaticFinding],
        invariants: Optional[list] = None,
        fork_protocol: Optional[str] = None,
        documentation: str = "",
        in_scope_files: Optional[list[str]] = None,
        context_files: Optional[list[str]] = None,
        target_dir: str = "",
        priority_targets: Optional[set[str]] = None,
        threat_summary: str = "",
    ) -> list[Hypothesis]:
        # Build scope-aware scope info string for the prompt
        scope_info = _build_scope_info(in_scope_files or [], context_files or [], analysis)

        # Read full source code from .sol files for the LLM prompt.
        # context_files (dependency contracts resolved by Scout) are included with
        # a char budget so Hunter can see bodies of cross-contract calls.
        # Budgets are settings-driven (large-context defaults) so big protocols are
        # not crushed into ~800 chars/file the way fixed 40k-total budgets did.
        _settings = get_settings()
        source_code = _read_scope_sources(
            target_dir, in_scope_files or [], context_files=context_files or [],
            in_scope_budget_chars=_settings.hunter_source_budget_chars,
            context_budget_chars=_settings.hunter_context_budget_chars,
        )

        # Sort functions by attack_surface_score descending, take top-N
        # When scope is available, restrict to functions from in-scope contracts only.
        _OZ_INTERNALS = frozenset([
            "_beforeTokenTransfer", "_afterTokenTransfer", "_update", "_mint", "_burn", "_transfer",
        ])
        all_in_scope_fns: list[tuple] = []
        for c in analysis.contracts:
            if in_scope_files and not _contract_in_scope(c.source_file, in_scope_files):
                continue
            for fn in c.functions:
                if fn.name in _OZ_INTERNALS:
                    continue
                all_in_scope_fns.append((fn, c.name))

        # Content-based score boost: functions whose source body contains financial
        # computation keywords get a boost even when Slither detectors miss them
        # (e.g. computeNAV only appears in cyclomatic-complexity with score=3.0).
        # Native Slither parser never populates fn.calls, so we use source content.
        _FINANCIAL_BODY_KWS = frozenset({
            "longpremium", "shortpremium", "longamount", "shortamount",
            "poolexposure", "exposure", "haircut", "computenav", "totalnav",
            "positionidlist", "totalassets", "performancefee",
            # Crypto/auth patterns — ensure signature validation functions rank in top-N
            "ecdsa", "ecrecover", "tryrecover", ".recover(", "validateuserop",
            "validatesignature", "userophash", "encodepacked", "packvalidationdata",
        })
        _FINANCIAL_NAME_KWS = frozenset({
            "nav", "premium", "exposure", "haircut", "position",
            "compute", "price", "fee", "liquidity",
            # Crypto/auth patterns
            "validate", "signature", "recover", "permit", "auth",
        })

        def _content_boost(fn: Any) -> float:
            """Return extra score for financial computation functions."""
            name_lower = fn.name.lower()
            # Name-based boost (lightweight, always applicable)
            name_boost = 6.0 if any(kw in name_lower for kw in _FINANCIAL_NAME_KWS) else 0.0
            # Body-based boost: read source from disk via source_location
            body_boost = 0.0
            loc = getattr(fn, "source_location", None)
            if loc and getattr(loc, "file", "") and getattr(loc, "start_line", 0):
                from pathlib import Path
                src_file = loc.file
                candidates = [Path(src_file)]
                if target_dir:
                    candidates.append(Path(target_dir) / src_file)
                    if src_file.startswith("/target/"):
                        candidates.append(Path(target_dir) / src_file[len("/target/"):])
                for p in candidates:
                    try:
                        lines = p.read_text(errors="ignore").splitlines()
                        body = "\n".join(lines[loc.start_line - 1: loc.end_line]).lower()
                        if any(kw in body for kw in _FINANCIAL_BODY_KWS):
                            body_boost = 10.0
                        elif "/" in body or "*" in body:
                            body_boost = 2.0  # has arithmetic
                        break
                    except OSError:
                        continue
            return name_boost + body_boost

        # Threat-model boost (roadmap #6): functions the threat model flagged as
        # value-at-risk targets are pulled toward the front of the analysis order
        # (which also makes them navigator leads). "Aim where the money is."
        _priority_targets = priority_targets or set()
        _threat_boost_amt = _settings.hunter_threat_boost

        def _threat_boost(fn: Any) -> float:
            return _threat_boost_amt if fn.name.lower() in _priority_targets else 0.0

        # Breadth triage (roadmap #7, model cascade): a CHEAP model rates EVERY
        # in-scope function for suspicion; the result boosts selection so a
        # suspicious-but-low-surface function still reaches the EXPENSIVE depth
        # pass. This is the full-codebase-coverage safety net.
        triage_scores: dict[str, float] = {}
        if self._triage_enabled and self._triager is not None and all_in_scope_fns:
            try:
                triage_scores = await self._triager.triage(all_in_scope_fns, audit_id)
            except Exception as exc:
                log.warning("hunter.triage_failed", error=str(exc))

        def _triage_boost(fn: Any) -> float:
            return self._triage_weight * (triage_scores.get(fn.id, 0.0) / 100.0)

        scored = []
        for fn, cname in all_in_scope_fns:
            boosted = (
                fn.attack_surface_score + _content_boost(fn)
                + _threat_boost(fn) + _triage_boost(fn)
            )
            scored.append((fn, cname, boosted))

        scored = sorted(scored, key=lambda t: t[2], reverse=True)[: self._top_functions]
        scored = [(fn, cname) for fn, cname, _ in scored]

        # Per-pass hypothesis budgets — prevents any single pass from crowding out others.
        # Each pass has its own cap; the global _max_hypotheses cap applies only at the end.
        #
        # Per-function cap is applied PER FUNCTION (not as a global pool cap) so that a
        # "chatty" early function cannot starve later functions. Each function contributes
        # at most _BUDGET_PER_FUNCTION hypotheses, sorted by priority_score descending.
        # This guarantees claimRewards or _validateSignature always get their best
        # hypothesis into the pool even if _userEligibleRewards generated many before them.
        _BUDGET_PER_FUNCTION = 2   # per individual function call
        _BUDGET_TEMPORAL    = 3    # _temporal_check (second-pass, per fn)
        _BUDGET_CRYPTO2     = 3    # _crypto_check (second-pass, per fn)
        _BUDGET_CRYPTO_SCAN = 5    # _crypto_file_scan (standalone)
        _BUDGET_CROSS       = 5    # _cross_contract_check
        _BUDGET_AGGREGATION = 5    # _aggregation_check
        _BUDGET_BOUNDARY    = 5    # _boundary_check
        _BUDGET_EXPERTS_TOTAL = 20  # all expert passes combined
        _BUDGET_NAVIGATOR   = 8    # agentic navigator investigations (roadmap #3)

        all_hypotheses: list[Hypothesis] = []
        fn_contexts: list = []  # [(fn, contract_name, context)] for second-pass checks
        invariants = invariants or []

        al = get_audit_logger()
        before_dedup = 0

        # Derive protocol-level invariants once, before per-function analysis.
        # These are natural-language statements (e.g. "NAV must equal sum of all vault assets")
        # that every function analysis and the aggregation check will use.
        derived_invariants: list[str] = await self._derive_invariants(
            documentation=documentation,
            source_code=source_code[:4000],
            audit_id=audit_id,
        )

        # ── Phase 1: per-function deep analysis (parallel, concurrency-limited) ──
        # Each analyzed function is independent → fan out under a semaphore instead
        # of awaiting one LLM call at a time. Results preserve `scored` order so the
        # downstream pool composition is deterministic.
        sem = asyncio.Semaphore(self._max_concurrency)
        active_fns = [(fn, cn) for fn, cn in scored if fn.attack_surface_score > 0]
        fn_results = await _gather_limited(
            [
                self._analyze_one_function(
                    fn, contract_name, context_builder, invariants, audit_id,
                    fork_protocol, documentation, scope_info, source_code,
                    derived_invariants, _BUDGET_PER_FUNCTION, al,
                    threat_summary,
                )
                for fn, contract_name in active_fns
            ],
            sem,
        )
        for res in fn_results:
            if isinstance(res, Exception) or res is None:
                continue
            fn_hyps_capped, ctx_tuple = res
            all_hypotheses.extend(fn_hyps_capped)
            fn_contexts.append(ctx_tuple)

        # ── Phase 2: all remaining passes run concurrently ──────────────────────
        # Expert / temporal / crypto (depend on fn_contexts) + the four standalone
        # passes are mutually independent. Fan them all out under the same limiter.
        # Ablation flags (HUNTER_EXPERTS_ENABLED / HUNTER_STANDALONE_PASSES_ENABLED)
        # replace a disabled pass with an empty result; defaults run everything.
        async def _skipped() -> list[Hypothesis]:
            return []

        standalone = self._standalone_enabled
        temporal_coros = [
            self._temporal_check(_fn.name, _ctx.code_slice or "", audit_id)
            for _fn, _cn, _ctx in fn_contexts
        ] if standalone else []
        crypto2_coros = [
            self._crypto_check(_fn.name, _ctx.code_slice or "", audit_id)
            for _fn, _cn, _ctx in fn_contexts
        ] if standalone else []

        (
            expert_hyps,
            temporal_lists,
            crypto2_lists,
            crypto_scan_hyps,
            cross_hyps,
            agg_hyps,
            boundary_hyps,
        ) = await asyncio.gather(
            self._expert_passes(fn_contexts, audit_id, _BUDGET_PER_FUNCTION, _BUDGET_EXPERTS_TOTAL, sem)
            if self._experts_enabled else _skipped(),
            _gather_limited(temporal_coros, sem),
            _gather_limited(crypto2_coros, sem),
            self._crypto_file_scan(audit_id, in_scope_files or [], target_dir)
            if standalone else _skipped(),
            self._cross_contract_check(audit_id, in_scope_files or [], target_dir)
            if standalone else _skipped(),
            self._aggregation_check(
                audit_id, in_scope_files or [], target_dir,
                protocol_invariants=derived_invariants,
            ) if standalone else _skipped(),
            self._boundary_check(audit_id, in_scope_files or [], target_dir)
            if standalone else _skipped(),
            return_exceptions=True,
        )

        def _flatten(lists) -> list[Hypothesis]:
            out: list[Hypothesis] = []
            if isinstance(lists, Exception) or not lists:
                return out
            for item in lists:
                if isinstance(item, list):
                    out.extend(item)
            return out

        def _safe_list(x) -> list[Hypothesis]:
            return x if isinstance(x, list) else []

        all_hypotheses.extend(_safe_list(expert_hyps))
        all_hypotheses.extend(_flatten(temporal_lists)[:_BUDGET_TEMPORAL])
        all_hypotheses.extend(_flatten(crypto2_lists)[:_BUDGET_CRYPTO2])
        all_hypotheses.extend(_safe_list(crypto_scan_hyps)[:_BUDGET_CRYPTO_SCAN])
        all_hypotheses.extend(_safe_list(cross_hyps)[:_BUDGET_CROSS])
        all_hypotheses.extend(_safe_list(agg_hyps)[:_BUDGET_AGGREGATION])
        all_hypotheses.extend(_safe_list(boundary_hyps)[:_BUDGET_BOUNDARY])

        # ── Phase 3: agentic navigator investigations (roadmap #3) ──────────────
        # The LLM investigates the top leads with read-only tools, following data
        # and call edges across the WHOLE codebase to surface multi-step / chained
        # bugs that a single fixed code slice cannot reveal. Additive & guarded.
        if self._navigator_enabled and analysis is not None and (in_scope_files or []):
            nav_hyps = await self._navigator_passes(
                analysis, in_scope_files or [], target_dir, active_fns,
                derived_invariants, documentation, audit_id, sem, threat_summary,
            )
            all_hypotheses.extend(nav_hyps[:_BUDGET_NAVIGATOR])

        before_dedup = len(all_hypotheses)
        all_hypotheses = _deduplicate(all_hypotheses, static_findings)
        all_hypotheses = _dedup_same_class(all_hypotheses)
        all_hypotheses = _dedup_by_title_similarity(all_hypotheses)
        all_hypotheses = _consolidate_by_function(all_hypotheses)
        all_hypotheses = await self._llm_semantic_dedup(all_hypotheses)

        if al and before_dedup != len(all_hypotheses):
            al.log_decision("hunter", "deduplication",
                f"Before: {before_dedup}, After: {len(all_hypotheses)}, "
                f"Removed: {before_dedup - len(all_hypotheses)}",
                {"before": before_dedup, "after": len(all_hypotheses)})

        all_hypotheses.sort(key=lambda h: h.priority_score, reverse=True)
        return all_hypotheses[: self._max_hypotheses]

    async def _navigator_passes(
        self,
        analysis: SlitherAnalysis,
        in_scope_files: list[str],
        target_dir: str,
        active_fns: list[tuple],
        derived_invariants: list[str],
        documentation: str,
        audit_id: str,
        sem: "asyncio.Semaphore",
        threat_summary: str = "",
    ) -> list[Hypothesis]:
        """Run agentic navigator investigations on the top leads (roadmap #3).

        Each lead gets a bounded ReAct session that pulls code on demand and
        follows cross-function/cross-contract edges. Sessions run in parallel
        under the shared concurrency limiter. Never raises — best-effort.
        """
        if not active_fns:
            return []
        try:
            from audit_agent.agents.navigator import CodebaseNavigator, NavigatorSession
        except Exception as exc:
            log.warning("hunter.navigator_import_failed", error=str(exc))
            return []

        try:
            navigator = CodebaseNavigator(analysis, in_scope_files, target_dir)
        except Exception as exc:
            log.warning("hunter.navigator_build_failed", error=str(exc))
            return []

        leads = active_fns[: self._navigator_leads]
        al = get_audit_logger()
        if al:
            al.log_decision("hunter", "navigator_start",
                f"{len(leads)} leads, max_steps={self._navigator_max_steps}",
                {"leads": [f"{cn}.{fn.name}" for fn, cn in leads]})

        coros = []
        lead_ids: list[str] = []
        for fn, contract_name in leads:
            session = NavigatorSession(
                self._provider, self._model, navigator,
                max_steps=self._navigator_max_steps,
            )
            prompt = _navigator_lead_prompt(
                fn, contract_name, navigator, derived_invariants, documentation,
                threat_summary,
            )
            coros.append(session.investigate(prompt, audit_id))
            lead_ids.append(fn.id)

        results = await _gather_limited(coros, sem)

        out: list[Hypothesis] = []
        for raw, lead_id in zip(results, lead_ids):
            if isinstance(raw, Exception) or not raw:
                continue
            out.extend(_build_hypotheses(raw, audit_id, lead_id))

        if al:
            al.log_decision("hunter", "navigator_complete",
                f"{len(out)} hypotheses from {len(leads)} navigator leads",
                {"count": len(out)})
        out.sort(key=lambda h: h.priority_score, reverse=True)
        return out

    async def _analyze_one_function(
        self,
        fn: Any,
        contract_name: str,
        context_builder: SubgraphContextBuilder,
        invariants: list,
        audit_id: str,
        fork_protocol: Optional[str],
        documentation: str,
        scope_info: str,
        source_code: str,
        derived_invariants: list[str],
        budget_per_function: int,
        al,
        threat_summary: str = "",
    ) -> tuple[list[Hypothesis], tuple]:
        """Deep-analyze a single function. Independent → safe to run concurrently.

        Returns (capped hypotheses, (fn, contract_name, context)). The context tuple
        feeds the second-pass expert/temporal/crypto checks.
        """
        self.log.debug("Hunter: analyzing", fn_id=fn.id, score=fn.attack_surface_score)
        if al:
            al.log_decision("hunter", "analyzing_function",
                f"{contract_name}.{fn.name} (score={fn.attack_surface_score:.1f})",
                {"function": fn.name, "contract": contract_name,
                 "score": fn.attack_surface_score, "id": fn.id})

        fn_invariants = [
            inv for inv in invariants
            if fn.id in (getattr(inv, "affected_functions", None) or [])
            or fn.name in (getattr(inv, "affected_functions", None) or [])
        ]

        # Cross-contract context for functions with external calls so callers and
        # shared state writers from other contracts are included (surfaces bugs
        # spanning multiple contracts).
        if fn.has_external_calls:
            context = await context_builder.build_cross_contract_context(fn.id)
        else:
            context = await context_builder.build_context(fn.id)

        if al:
            al.log_decision("hunter", "context_built",
                f"Context for {fn.name}: {len(context.code_slice or '')} chars code, "
                f"{len(context.similar_historical_cases or [])} similar cases, "
                f"cross_contract={fn.has_external_calls}",
                {"code_slice_length": len(context.code_slice or ""),
                 "similar_cases_count": len(context.similar_historical_cases or []),
                 "fn_invariants_count": len(fn_invariants),
                 "cross_contract": fn.has_external_calls})

        purpose = f"hypotheses_{contract_name}_{fn.name}"
        fn_hyps = await self._hypothesize(
            context, audit_id, fork_protocol, documentation, fn_invariants, scope_info,
            source_code=source_code, purpose=purpose, derived_invariants=derived_invariants,
            threat_summary=threat_summary,
        )

        if al:
            for hyp in fn_hyps:
                al.log_decision("hunter", "hypothesis_generated",
                    f"[{hyp.estimated_severity}] {hyp.title}",
                    {"id": hyp.id, "title": hyp.title,
                     "severity": hyp.estimated_severity,
                     "priority_score": hyp.priority_score,
                     "vulnerability_class": hyp.vulnerability_class})

        # Each function contributes its top-scored hypotheses independently.
        fn_hyps_capped = sorted(fn_hyps, key=lambda h: h.priority_score, reverse=True)[
            :budget_per_function
        ]
        return fn_hyps_capped, (fn, contract_name, context)

    async def _hypothesize(
        self,
        context: SubgraphContext,
        audit_id: str,
        fork_protocol: Optional[str],
        documentation: str = "",
        fn_invariants: Optional[list] = None,
        scope_info: str = "",
        source_code: str = "",
        purpose: str = "hypotheses",
        derived_invariants: Optional[list[str]] = None,
        threat_summary: str = "",
    ) -> list[Hypothesis]:
        prompt = _build_prompt(
            context, fork_protocol, documentation, fn_invariants or [], scope_info,
            source_code, derived_invariants=derived_invariants,
            threat_summary=threat_summary,
        )
        raw = await self._llm_call(prompt, purpose=purpose, code_slice=context.code_slice or "")
        return _build_hypotheses(raw, audit_id, context.entry_function_id)

    # Hard wall-clock limit per LLM call. httpx.Timeout resets on each streaming
    # chunk, so a slowly-streaming model can run indefinitely. asyncio.wait_for
    # provides the absolute deadline regardless of streaming behavior.
    # Raised for slow thinking-models (qwen3.7-max). At full single-stream
    # throughput (~55 tok/s) an 8k-token response is ~150-250s, but the heaviest
    # liquidation/deleverage functions + variable provider throughput pushed even
    # concurrency-2 calls past 600s. 900s gives margin; pair with low concurrency
    # (HUNTER_MAX_CONCURRENCY=1) so each call gets full throughput.
    _LLM_TIMEOUT_SECS = 900.0   # initial call
    _RETRY_TIMEOUT_SECS = 300.0  # stripped retry (code[:2000] + max_tokens=4096)

    async def _llm_call(self, user_prompt: str, purpose: str = "hypotheses", code_slice: str = "") -> list[dict[str, Any]]:
        al = get_audit_logger()
        llm_record = None
        if al:
            llm_record = al.log_llm_call_start(
                agent="hunter", purpose=purpose, model=self._model,
                system_prompt=_HUNTER_SYSTEM, user_message=user_prompt,
                tools=[_HYPOTHESIS_TOOL], max_tokens=8192,
            )
        t0 = time.time()
        response = None
        timed_out = False
        try:
            response = await asyncio.wait_for(
                self._provider.complete(
                    model=self._model,
                    max_tokens=8192,
                    system=_HUNTER_SYSTEM,
                    tools=[_HYPOTHESIS_TOOL],
                    tool_choice={"name": "report_hypotheses"},
                    messages=[{"role": "user", "content": user_prompt}],
                ),
                timeout=self._LLM_TIMEOUT_SECS,
            )
        except asyncio.TimeoutError:
            timed_out = True
            log.warning("hunter.llm_timeout", purpose=purpose,
                        elapsed=round(time.time() - t0, 1))
        except Exception as exc:
            duration = time.time() - t0
            if al and llm_record:
                al.log_llm_call_end(llm_record, error=str(exc), duration_seconds=duration)
            log.warning("hunter.llm_unavailable", error=str(exc))
            return []

        hypotheses_raw: list[dict[str, Any]] = []
        tool_calls_logged: list[dict[str, Any]] = []

        if response is not None:
            for tc in response.tool_calls:
                if tc.name == "report_hypotheses":
                    hypotheses_raw = tc.arguments.get("hypotheses", [])
                    tool_calls_logged.append({
                        "name": tc.name,
                        "hypotheses_count": len(hypotheses_raw),
                    })

        # Retry cases:
        #   Timeout    → stripped code-only prompt with smaller max_tokens (fits in budget)
        #   Prose-only → model wrote analysis but forgot the tool call
        #   max_tokens → tool call JSON was truncated mid-generation
        needs_retry = timed_out or (not hypotheses_raw and (
            (response is not None and response.text and len(response.text) > 100)
            or (response is not None and response.stop_reason == "max_tokens")
        ))
        if needs_retry:
            try:
                if not timed_out and response is not None and response.text and len(response.text) > 100:
                    # Prose-only: include the model's analysis and demand the tool call
                    retry_messages = [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": response.text[:6000]},
                        {"role": "user", "content":
                            "Call `report_hypotheses` NOW with the vulnerabilities you identified above. "
                            "Output ONLY the tool call, no additional text."},
                    ]
                    retry_max_tokens = 4096
                else:
                    # Timeout or max_tokens truncation: minimal prompt.
                    # max_tokens MUST cover reasoning + the tool call. Thinking models
                    # (qwen3.7-max) spend ~3000 tokens reasoning BEFORE any output, so
                    # the old 768 budget was consumed entirely by reasoning and the
                    # tool call was never emitted — the retry produced nothing. 4096
                    # leaves ~1000 tokens for the (tiny) report_hypotheses call after
                    # reasoning. At ~55 tok/s that's ~60-75s, well under _RETRY_TIMEOUT_SECS.
                    fin_hint = ""
                    if code_slice and any(kw in code_slice.lower() for kw in _FIN_RETRY_KWS):
                        fin_hint = _FIN_RETRY_HINT
                    code_only = (
                        f"## Code Slice\n{code_slice[:500] if code_slice else '(no code)'}"
                        f"{fin_hint}\n\n"
                        "Report EXACTLY 1 security vulnerability (the most critical). "
                        "attack_scenario: 1 sentence. required_preconditions: 1 item. "
                        "Call `report_hypotheses` — ONLY the tool call, no prose."
                    )
                    retry_messages = [{"role": "user", "content": code_only}]
                    retry_max_tokens = 4096

                retry_response = await asyncio.wait_for(
                    self._provider.complete(
                        model=self._model,
                        max_tokens=retry_max_tokens,
                        system=_HUNTER_SYSTEM_TRIAGE,
                        tools=[_HYPOTHESIS_TOOL],
                        tool_choice={"name": "report_hypotheses"},
                        messages=retry_messages,
                    ),
                    timeout=self._RETRY_TIMEOUT_SECS,
                )
                for tc in retry_response.tool_calls:
                    if tc.name == "report_hypotheses":
                        hypotheses_raw = tc.arguments.get("hypotheses", [])
                        tool_calls_logged.append({
                            "name": tc.name,
                            "hypotheses_count": len(hypotheses_raw),
                            "retry": True,
                        })
                if response is None:
                    response = retry_response
            except Exception:
                pass

        duration = (response.duration_seconds if response else None) or (time.time() - t0)
        if al and llm_record:
            al.log_llm_call_end(
                llm_record,
                tool_calls=tool_calls_logged,
                input_tokens=response.input_tokens if response else 0,
                output_tokens=response.output_tokens if response else 0,
                stop_reason=response.stop_reason if response else "timeout",
                duration_seconds=duration,
            )
        return hypotheses_raw

    # ── Second-pass analysis methods ───────────────────────────────────────

    async def _temporal_check(self, fn_name: str, source_code: str, audit_id: str) -> list[Hypothesis]:
        """Second-pass: explicit check for temporal/boundary vulnerabilities."""
        time_keywords = [
            "epoch", "timestamp", "block.timestamp", "period", "deadline",
            "duration", "interval", "warp", "snapshot", "checkpoint",
            "lock", "unlock", "expire", "vest", "cliff", "cooldown",
        ]
        if not any(kw in source_code.lower() for kw in time_keywords):
            return []

        al = get_audit_logger()
        if al:
            al.log_decision("hunter", "temporal_check_triggered",
                f"Function {fn_name} contains time-related keywords")

        prompt = f"""Analyze this function specifically for TEMPORAL and BOUNDARY vulnerabilities.

## Source Code
```solidity
{source_code[:8000]}
```

## Temporal Checklist — answer YES/NO for each, with code evidence:

1. EPOCH/PERIOD BOUNDARIES:
   - Does this code calculate epoch/period boundaries?
   - Is the boundary inclusive (>=) or exclusive (>)?
   - Does epochEnd(N) equal epochStart(N+1)? If yes, which epoch owns that moment?
   - Can an action at exactly the boundary affect BOTH adjacent epochs?

2. SNAPSHOT TIMING:
   - Does this code read a balance/supply at a specific timestamp?
   - Can the underlying value change at that exact timestamp?
   - Can someone create/modify a position and have it counted in a past snapshot?

3. ORDERING ATTACKS:
   - Can transaction ordering within the same block affect outcomes?
   - Is there a first-come advantage at epoch transitions?

4. STALE STATE:
   - Does this code read historical data from an external contract?
   - Is that data finalized or can it still change?

For each YES: generate a hypothesis with exact attack timeline, violated invariant, and severity.
If ALL are NO: return empty list.
"""
        try:
            response = await self._provider.complete(
                model=self._model,
                max_tokens=8192,
                system="You are a temporal/boundary vulnerability specialist for smart contracts.",
                tools=[_HYPOTHESIS_TOOL],
                tool_choice={"name": "report_hypotheses"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw = _extract_hypotheses_raw(response)
            return _build_hypotheses(raw, audit_id, fn_name)
        except Exception as exc:
            log.warning("hunter.temporal_check_failed", error=str(exc))
            return []

    async def _crypto_check(self, fn_name: str, source_code: str, audit_id: str) -> list[Hypothesis]:
        """Second-pass: explicit check for signature/hash/encoding vulnerabilities."""
        crypto_keywords = [
            "ecrecover", "ECDSA", "signature", "keccak256", "abi.encodePacked",
            "digest", "signer", "validUntil", "validAfter", "nonce",
            "EIP712", "EIP191", "permit", "verify", "recover",
        ]
        if not any(kw in source_code for kw in crypto_keywords):
            return []

        al = get_audit_logger()
        if al:
            al.log_decision("hunter", "crypto_check_triggered",
                f"Function {fn_name} contains crypto keywords")

        prompt = f"""Analyze this code for SIGNATURE and ENCODING vulnerabilities.

```solidity
{source_code[:8000]}
```

## Checklist:

1. HASH COMPLETENESS:
   - List ALL fields included in the signed hash/digest.
   - List ALL fields parsed from the message/signature but NOT in the hash.
   - If ANY parsed field is NOT in the hash → HIGH SEVERITY
     (attacker can modify it without invalidating the signature)

2. ENCODING SAFETY:
   - Is abi.encodePacked used with variable-length types? → hash collision risk
   - Is the hash domain-separated (EIP-712 or chain-specific prefix)?
   - Can the same signature be replayed on a different chain/contract?

3. REPLAY PROTECTION:
   - Is there a nonce that increments on each use?
   - Is there a deadline/expiry that is BOTH checked AND signed?

For each finding: exact code line, attack scenario, severity.
"""
        try:
            response = await self._provider.complete(
                model=self._model,
                max_tokens=8192,
                system="You are a cryptographic vulnerability specialist for smart contracts.",
                tools=[_HYPOTHESIS_TOOL],
                tool_choice={"name": "report_hypotheses"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw = _extract_hypotheses_raw(response)
            return _build_hypotheses(raw, audit_id, fn_name)
        except Exception as exc:
            log.warning("hunter.crypto_check_failed", error=str(exc))
            return []

    async def _run_expert_pass(
        self,
        fn_name: str,
        code_slice: str,
        expert_name: str,
        audit_id: str,
    ) -> list[Hypothesis]:
        """Run one domain-expert LLM pass over a single function's code slice."""
        system_prompt = _EXPERT_REGISTRY.get(expert_name)
        if not system_prompt:
            return []

        al = get_audit_logger()
        if al:
            al.log_decision("hunter", "expert_pass_triggered",
                f"Expert '{expert_name}' → function {fn_name}")

        prompt = (
            f"## Function: {fn_name}\n\n"
            f"```solidity\n{code_slice[:8000]}\n```\n\n"
            f"Report 1-2 high-confidence {expert_name.replace('_', ' ')} vulnerabilities only. "
            "If none found in your domain, call report_hypotheses with an empty list."
        )
        try:
            response = await self._provider.complete(
                model=self._model,
                max_tokens=8192,
                system=system_prompt,
                tools=[_HYPOTHESIS_TOOL],
                tool_choice={"name": "report_hypotheses"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw = _extract_hypotheses_raw(response)
            return _build_hypotheses(raw, audit_id, fn_name)
        except Exception as exc:
            log.warning("hunter.expert_pass_failed", expert=expert_name, fn=fn_name, error=str(exc))
            return []

    async def _expert_passes(
        self,
        fn_contexts: list,
        audit_id: str,
        budget_per_expert: int,
        budget_total: int,
        sem: "Optional[asyncio.Semaphore]" = None,
    ) -> list[Hypothesis]:
        """Run domain-expert passes over fn_contexts, additive to the main generic loop.

        For each analyzed function the rule-based router selects relevant experts.
        Each expert contributes at most budget_per_expert hypotheses per function.
        Total output across all functions and experts is capped at budget_total.
        Runs under the shared concurrency semaphore (creates its own if not given).
        """
        if sem is None:
            sem = asyncio.Semaphore(self._max_concurrency)
        # Build the full (function, expert) task matrix, then run concurrently.
        # The router selects relevant experts per function via keyword matching.
        tasks: list = []
        for fn, _contract_name, context in fn_contexts:
            code_slice = context.code_slice or ""
            for expert_name in _select_experts(code_slice):
                tasks.append(self._run_expert_pass(fn.name, code_slice, expert_name, audit_id))

        results = await _gather_limited(tasks, sem)

        all_expert_hyps: list[Hypothesis] = []
        for hyps in results:
            if isinstance(hyps, Exception) or not hyps:
                continue
            capped = sorted(hyps, key=lambda h: h.priority_score, reverse=True)[:budget_per_expert]
            all_expert_hyps.extend(capped)

        # Global cap across all experts — keep the highest-priority findings rather
        # than first-come (the old sequential early-break was order-dependent).
        all_expert_hyps.sort(key=lambda h: h.priority_score, reverse=True)
        return all_expert_hyps[:budget_total]

    async def _derive_invariants(
        self,
        documentation: str,
        source_code: str,
        audit_id: str,
    ) -> list[str]:
        """One-shot LLM call: derive protocol-level invariants from documentation + code.

        Returns plain-text invariant statements. These complement formal Invariant objects
        from the Mapper — they capture higher-level business-logic invariants that static
        analysis cannot derive. Fed into every per-function analysis and _aggregation_check.
        """
        # Nothing to derive from → skip the LLM call entirely. This also keeps Hunter
        # from touching the model when scope excludes everything (no source, no docs).
        if not (documentation or "").strip() and not (source_code or "").strip():
            return []
        prompt = f"""You are auditing a DeFi protocol. Derive the KEY PROTOCOL INVARIANTS — properties that must ALWAYS hold, regardless of transaction sequence or input values.

## Protocol Documentation
{documentation[:3000] if documentation else "(none provided)"}

## Source Code Excerpt
{source_code[:4000]}

## Invariant Categories to Cover

1. **Value conservation**: e.g., "totalAssets() must equal the sum of all tokens the vault can claim, without double-counting or omission at any point in time"
2. **Accounting completeness**: e.g., "every token the vault holds must appear in NAV/totalAssets exactly once — not zero times, not twice"
3. **Aggregation safety**: e.g., "any max(sum, 0) floor applied to a sum must not silently discard components that are always-positive (like free cash held by the vault)"
4. **Skip-flag correctness**: e.g., "a deduplication flag set to prevent double-counting must not simultaneously block the fallback that would recover value if the primary path discarded it"
5. **User equity**: e.g., "a user's proportional claim on assets can only decrease via authorized fees, never silently"
6. **Cross-component consistency**: e.g., "if a token is marked as 'already counted' in a dedup structure, it must actually have been counted in NAV, not zeroed out by a floor"

List 5-8 invariants, one per line, in plain English. Be specific to the actual code shown. No headers, no markdown — just the invariant statements.
"""
        try:
            result = await self._provider.complete_simple(
                model=self._model,
                system=(
                    "You are a DeFi protocol invariant analyst. "
                    "Output only a plain list of invariant statements — one per line, no headers, no extra text."
                ),
                user_message=prompt,
                max_tokens=8192,
            )
            invariants: list[str] = []
            for line in result.split('\n'):
                inv = line.strip().lstrip('0123456789.-•*)> ').strip()
                if len(inv) > 30 and not inv.endswith(':'):
                    invariants.append(inv)
            al = get_audit_logger()
            if al and invariants:
                al.log_decision("hunter", "invariants_derived",
                    f"Derived {len(invariants)} protocol invariants",
                    {"invariants": invariants[:5]})
            return invariants[:8]
        except Exception as exc:
            log.warning("hunter.derive_invariants_failed", error=str(exc))
            return []

    async def _llm_semantic_dedup(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Collapse hypotheses that describe the SAME root-cause bug but escaped the
        token/function heuristics — different wording, a different entry function, or a
        contract-qualified vs bare function name (e.g. `LandManager._reconfigure` vs
        `_reconfigure`, or the same StorageKey misconfig surfaced via `_reconfigure`,
        `triggerPlotMetadata` and `updateTaxRate`).

        Candidates are clusters of hypotheses sharing ≥1 normalised affected function; a
        single LLM call partitions each cluster into distinct root-cause bugs and the
        highest-priority hypothesis per bug is kept. Fail-safe: any error or unparseable /
        incomplete response leaves that cluster untouched, so it can only ever help.
        """
        if len(hypotheses) < 2:
            return hypotheses

        def _norm_fn(f: str) -> str:
            return f.split("(")[0].strip().split(".")[-1].lstrip("_").lower()

        # Union-find over shared normalised affected functions.
        parent = list(range(len(hypotheses)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        first_seen: dict[str, int] = {}
        for i, h in enumerate(hypotheses):
            for fn in {_norm_fn(x) for x in (h.affected_functions or [])}:
                if fn in first_seen:
                    union(first_seen[fn], i)
                else:
                    first_seen[fn] = i

        from collections import defaultdict
        clusters: dict[int, list[int]] = defaultdict(list)
        for i in range(len(hypotheses)):
            clusters[find(i)].append(i)

        drop: set[int] = set()
        for members in clusters.values():
            if len(members) < 2:
                continue
            lines = []
            for idx in members:
                h = hypotheses[idx]
                lines.append(
                    f"[{idx}] title: {h.title}\n"
                    f"     class: {h.vulnerability_class or 'n/a'} | severity: {h.estimated_severity}\n"
                    f"     functions: {', '.join(h.affected_functions or []) or 'n/a'}\n"
                    f"     summary: {(h.attack_scenario or '')[:300]}"
                )
            prompt = (
                "Below are candidate vulnerability hypotheses that touch overlapping functions.\n"
                "Group together ONLY hypotheses that describe THE SAME underlying bug — the same "
                "defective code where a single fix resolves all of them. Different symptoms, entry "
                "points, or wording do NOT make them different; the ROOT CAUSE must be identical. "
                "When in doubt, keep them SEPARATE.\n\n"
                "Return ONLY a JSON array of arrays of ids that partitions EVERY id exactly once, "
                "e.g. [[0,3],[1],[2,4]].\n\n"
                + "\n".join(lines)
            )
            try:
                raw = await self._provider.complete_simple(
                    model=self._model,
                    system=(
                        "You de-duplicate security findings. Output only JSON — an array of arrays "
                        "of integer ids. Group only exact same-root-cause duplicates."
                    ),
                    user_message=prompt,
                    max_tokens=2048,
                )
                groups = _parse_id_groups(raw, set(members))
            except Exception as exc:
                log.warning("hunter.semantic_dedup_failed", error=str(exc))
                groups = None
            if not groups:
                continue
            for grp in groups:
                if len(grp) < 2:
                    continue
                keep = max(grp, key=lambda idx: hypotheses[idx].priority_score)
                for idx in grp:
                    if idx != keep:
                        drop.add(idx)

        if not drop:
            return hypotheses

        al = get_audit_logger()
        if al:
            al.log_decision(
                "hunter", "semantic_dedup",
                f"LLM semantic dedup collapsed {len(drop)} duplicate hypotheses",
                {"dropped": sorted(drop)},
            )
        return [h for i, h in enumerate(hypotheses) if i not in drop]

    async def _aggregation_check(
        self,
        audit_id: str,
        in_scope_files: list[str],
        target_dir: str,
        protocol_invariants: Optional[list[str]] = None,
    ) -> list[Hypothesis]:
        """Dedicated pass for accounting aggregation bugs in any DeFi protocol.

        Finds functions that accumulate values across multiple sources (pools, positions,
        tokens) and applies a floor (max(x, 0)) or ceiling to the result. This pattern
        can silently discard always-positive components when the sum goes below the floor.
        Also detects deduplication skip-flags that can block fallback recovery paths.

        Generic: works for any vault/accounting function, not protocol-specific.
        """
        from pathlib import Path
        import re

        _FLOOR_PATTERNS = frozenset({"math.max", "max(", "uint256(int256", "touint256"})
        _ACCUM_PATTERNS = frozenset({"+=", "balanceof(", "previewredeem(", "totalassets("})
        _DEDUP_PATTERNS = frozenset({"skip", ".push(", "alreadycounted", "seen["})

        def _is_aggregation_fn(body_lower: str) -> bool:
            has_floor = any(p in body_lower for p in _FLOOR_PATTERNS)
            has_accum = any(p in body_lower for p in _ACCUM_PATTERNS)
            return has_floor and has_accum

        def _extract_all_functions(source: str) -> list[tuple[str, str]]:
            """Return [(fn_name, fn_body)] for functions that match aggregation heuristics."""
            results = []
            for m in re.finditer(r'\bfunction\s+(\w+)\b', source):
                fn_name = m.group(1)
                start = m.start()
                depth = 0
                in_fn = False
                for i, ch in enumerate(source[start:], start):
                    if ch == '{':
                        depth += 1
                        in_fn = True
                    elif ch == '}':
                        depth -= 1
                        if in_fn and depth == 0:
                            body = source[start:i + 1]
                            if _is_aggregation_fn(body.lower()):
                                results.append((fn_name, body))
                            break
            return results

        agg_functions: list[tuple[str, str, str]] = []  # (file_name, fn_name, fn_body)
        for sol_path in in_scope_files:
            p = Path(sol_path)
            if not p.exists() and target_dir:
                p = Path(target_dir) / sol_path.lstrip("./")
            if not p.exists():
                continue
            try:
                content = p.read_text(errors="replace")
                for fn_name, fn_body in _extract_all_functions(content):
                    agg_functions.append((p.name, fn_name, fn_body))
            except OSError:
                pass

        if not agg_functions:
            return []

        al = get_audit_logger()
        if al:
            al.log_decision("hunter", "aggregation_check",
                f"Found {len(agg_functions)} aggregation function(s): "
                f"{[f'{f}/{n}' for f, n, _ in agg_functions[:5]]}",
                {"count": len(agg_functions)})

        invariant_section = ""
        if protocol_invariants:
            invariant_section = (
                "\n## Protocol Invariants\n"
                + "\n".join(f"- {inv}" for inv in protocol_invariants)
            )

        source_sections: list[str] = []
        for file_name, fn_name, fn_body in agg_functions[:5]:  # cap at 5 functions
            source_sections.append(
                f"### {file_name} — `{fn_name}`\n```solidity\n{fn_body[:6000]}\n```"
            )

        prompt = f"""You are auditing DeFi accounting functions for financial logic bugs.
{invariant_section}

## Functions to Audit
{chr(10).join(source_sections)}

## Bug Patterns to Find

**Pattern A — Floor Hiding Positives:**
A function accumulates: `total += component_i` where component_i mixes ALWAYS-POSITIVE parts
(free token balance) with POSSIBLY-NEGATIVE parts (unrealized P&L).
Then: `nav += max(total, 0)`.
When negatives dominate: `total < 0` → floor = 0 → always-positive part silently discarded.
`max(asset + liability, 0)` is WRONG if asset should always be in NAV.

**Pattern B — Deduplication Blocking Recovery:**
A "skip" flag/array is set to prevent double-counting. But if Pattern A discarded the value,
the flag now blocks the fallback recovery: `if (!skip) nav += balance` → never executes.
Result: balance counted zero times, not once.

**Pattern C — Asymmetric Conditional Inclusion:**
A balance is guarded by condition X. The invariant says it should ALWAYS be counted.
Find: is there a scenario where condition X = false yet the balance is positive and uncounted?

**Pattern D — Sign Inversion in Paired Computations:**
Two parallel values computed for token0 and token1 (e.g. via `.rightSlot()` / `.leftSlot()`).
Correct convention: both use the SAME subtraction order (e.g., both `short - long`).
Bug: token0 uses `short - long` (correct) but token1 uses `long - short` (inverted sign).
Even one inverted slot corrupts the entire aggregate. Check EVERY pair of rightSlot/leftSlot
computations for consistent subtraction order.
Example: `shortPremium.leftSlot() - longPremium.leftSlot()` is CORRECT;
`longPremium.leftSlot() - shortPremium.leftSlot()` inverts the sign → WRONG.

## Audit Questions

For Patterns A–C: trace every balanceOf/previewRedeem through the aggregate → floor → fallback.
For Pattern D: list every (A-B) subtraction, state what A and B represent, verify order is correct.
Cross-check: do token0 and token1 computations use the SAME order? A mismatch is HIGH severity.

Report each distinct bug as HIGH/CRITICAL with exact variable names. If no bugs: needs_review for the most suspicious interaction.
"""
        try:
            response = await self._provider.complete(
                model=self._model,
                max_tokens=8192,
                system=(
                    "You are a DeFi accounting specialist. Find financial logic bugs: "
                    "floor hiding positives, dedup blocking recovery, sign inversions. "
                    "Name exact variables and conditions. Use the report_hypotheses tool."
                ),
                tools=[_HYPOTHESIS_TOOL],
                tool_choice={"name": "report_hypotheses"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw = _extract_hypotheses_raw(response)
            return _build_hypotheses(raw, audit_id, "aggregation")
        except Exception as exc:
            log.warning("hunter.aggregation_check_failed", error=str(exc))
            return []

    async def _crypto_file_scan(
        self,
        audit_id: str,
        in_scope_files: list[str],
        target_dir: str,
    ) -> list[Hypothesis]:
        """Standalone pass: scan all in-scope files for signature/hash vulnerabilities.

        Unlike _crypto_check (second-pass on top-N fn_contexts only), this reads every
        in-scope file directly from disk and checks every function that contains crypto
        keywords — regardless of Slither attack_surface_score. Catches bugs in auth
        functions like _validateSignature that have no financial keywords and score low.
        """
        from pathlib import Path
        import re

        _CRYPTO_KWS = frozenset({
            "ecrecover", "ecdsa", "tryrecover", ".recover(", "keccak256",
            "encodepacked", "validatesignature", "validateuserop",
        })

        def _has_crypto(body_lower: str) -> bool:
            return any(kw in body_lower for kw in _CRYPTO_KWS)

        def _extract_crypto_functions(source: str) -> list[tuple[str, str]]:
            results = []
            for m in re.finditer(r'\bfunction\s+(\w+)\b', source):
                fn_name = m.group(1)
                start = m.start()
                depth = 0
                in_fn = False
                for i, ch in enumerate(source[start:], start):
                    if ch == '{':
                        depth += 1
                        in_fn = True
                    elif ch == '}':
                        depth -= 1
                        if in_fn and depth == 0:
                            body = source[start:i + 1]
                            if _has_crypto(body.lower()):
                                results.append((fn_name, body))
                            break
            return results

        crypto_functions: list[tuple[str, str, str]] = []  # (file_name, fn_name, fn_body)
        for sol_path in in_scope_files:
            p = Path(sol_path)
            if not p.exists() and target_dir:
                p = Path(target_dir) / sol_path.lstrip("./")
            if not p.exists():
                continue
            try:
                content = p.read_text(errors="replace")
                for fn_name, fn_body in _extract_crypto_functions(content):
                    crypto_functions.append((p.name, fn_name, fn_body))
            except OSError:
                pass

        if not crypto_functions:
            return []

        al = get_audit_logger()
        if al:
            al.log_decision("hunter", "crypto_file_scan",
                f"Found {len(crypto_functions)} crypto function(s): "
                f"{[f'{f}/{n}' for f, n, _ in crypto_functions[:5]]}",
                {"count": len(crypto_functions)})

        all_hyps: list[Hypothesis] = []
        for file_name, fn_name, fn_body in crypto_functions[:8]:
            prompt = f"""Analyze this Solidity function for SIGNATURE and ENCODING vulnerabilities.

File: {file_name}
Function: {fn_name}

```solidity
{fn_body[:6000]}
```

## Critical Check — HASH COMPLETENESS:
A signed hash must commit to EVERY field that is used after signature verification.

Step 1: List ALL fields extracted from the message/signature bytes
  (e.g. validUntil, validAfter, nonce, deadline — any bytes sliced from sig or calldata).

Step 2: List ALL fields included in the hash BEFORE ecrecover/ECDSA.recover() is called.

Step 3: For each field from Step 1 that is NOT in Step 2 → HIGH severity finding.
  Attacker can modify that field without invalidating the signature.
  Real example: validUntil/validAfter extracted from trailing 12 bytes of signature,
  but hash = keccak256(userOpHash) covers only the inner hash — validity window unsigned.

## Additional Checks:
- abi.encodePacked with variable-length types → hash collision
- Missing domain separator / chain ID → replay across chains or contracts
- Nonce not incremented after use and not in hash → replay attack
- Validity window (deadline, validUntil) included in hash? If not → attacker sets own window
"""
            try:
                response = await self._provider.complete(
                    model=self._model,
                    max_tokens=8192,
                    system=(
                        "You are a cryptographic vulnerability specialist for Solidity. "
                        "Focus on hash completeness: enumerate what is signed vs what is used. "
                        "Only report real vulnerabilities with exact code evidence."
                    ),
                    tools=[_HYPOTHESIS_TOOL],
                    tool_choice={"name": "report_hypotheses"},
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = _extract_hypotheses_raw(response)
                hyps = _build_hypotheses(raw, audit_id, fn_name)
                all_hyps.extend(hyps)
            except Exception as exc:
                log.warning("hunter.crypto_file_scan_failed", fn=fn_name, error=str(exc))

        return all_hyps

    async def _boundary_check(
        self,
        audit_id: str,
        in_scope_files: list[str],
        target_dir: str,
    ) -> list[Hypothesis]:
        """Standalone pass: find epoch/time boundary arithmetic bugs.

        Targets off-by-one errors at epoch/period boundaries — the class of bugs where
        `periodEnd(n)` equals the first moment of period n+1, causing checkpoints at the
        boundary to be attributed to the wrong period. Also catches inclusive/exclusive
        mismatch in snapshot lookups (ts <= t vs ts < t).
        """
        from pathlib import Path
        import re

        _BOUNDARY_KWS = frozenset({
            "epoch", "period", "epochlength", "epochend", "epochstart",
            "epochtimestamp", "currentepoch", "snapshotat", "pointintime",
            "checkpoint", "votingescrow", "vetoken", "getpastbalance",
            "getpastshares", "epochboundary",
        })
        _SNAPSHOT_KWS = frozenset({
            "binary", "binarysearch", "upperbound", "lowerbound",
            "ts <=", "ts <", "block.timestamp ==", "checkpoint",
        })

        def _has_boundary(content_lower: str) -> bool:
            return (
                any(kw in content_lower for kw in _BOUNDARY_KWS)
                and ("epoch" in content_lower or "period" in content_lower or "checkpoint" in content_lower)
            )

        # Collect relevant file content (up to 6000 chars each, max 4 files)
        relevant: list[tuple[str, str]] = []
        for sol_path in in_scope_files:
            p = Path(sol_path)
            if not p.exists() and target_dir:
                p = Path(target_dir) / sol_path.lstrip("./")
            if not p.exists():
                continue
            try:
                content = p.read_text(errors="replace")
                if _has_boundary(content.lower()):
                    relevant.append((p.name, content[:6000]))
            except OSError:
                pass

        if not relevant:
            return []

        al = get_audit_logger()
        if al:
            al.log_decision("hunter", "boundary_check",
                f"Found {len(relevant)} file(s) with epoch/boundary patterns: "
                f"{[f for f, _ in relevant]}",
                {"count": len(relevant)})

        source_sections = [
            f"### {name}\n```solidity\n{code}\n```"
            for name, code in relevant[:4]
        ]

        prompt = f"""Analyze these contracts for EPOCH/PERIOD BOUNDARY arithmetic bugs.

{chr(10).join(source_sections)}

## What to look for:

### 1. Off-by-one in period boundary definition
- Does `epochEnd(n)` return the LAST second of epoch n, or the FIRST second of epoch n+1?
- Example bug: `epochEnd(n) = START + (n+1) * LENGTH` equals `epochStart(n+1)`.
  A checkpoint at ts == epochEnd(n) would be in epoch n+1, not n.
- Check: what epoch ID does `currentEpoch(ts == boundary)` return?

### 2. Inclusive vs exclusive snapshot lookup
- VotingEscrow-style contracts typically use binary search over checkpoints.
- Bug: checkpoint with `ts == boundaryTimestamp` is included in BOTH epoch n (snapshot)
  and epoch n+1 (new participant), creating double-counting or wrong attribution.
- Look for: `while (lo < hi)` or `ts <= t` vs `ts < t` in checkpoint search.

### 3. Cross-contract boundary mismatch
- Contract A computes `snapshotTimestamp = epochEnd(n)` for rewards snapshot.
- Contract B's checkpoint lookup uses `ts <= snapshotTimestamp` (inclusive).
- If someone deposits at exactly `ts == epochEnd(n)`, their checkpoint falls in BOTH
  epoch n's snapshot AND becomes a valid participant in epoch n+1.

### 4. currentEpoch() off-by-one
- `currentEpoch(ts) = (ts - START) / LENGTH` uses integer division.
- At ts == START + n*LENGTH (exact boundary): this returns n, meaning the boundary
  moment belongs to epoch n, but the snapshot for epoch n-1 may be taken at this same ts.

Report only real bugs with exact line evidence and a concrete exploit scenario.
"""
        try:
            response = await self._provider.complete(
                model=self._model,
                max_tokens=8192,
                system=(
                    "You are an expert in epoch/period boundary arithmetic bugs in DeFi. "
                    "Focus on off-by-one errors at exact boundary timestamps. "
                    "Only report bugs with concrete code evidence."
                ),
                tools=[_HYPOTHESIS_TOOL],
                tool_choice={"name": "report_hypotheses"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw = _extract_hypotheses_raw(response)
            return _build_hypotheses(raw, audit_id, "boundary_check")
        except Exception as exc:
            log.warning("hunter.boundary_check_failed", error=str(exc))
            return []

    async def _cross_contract_check(
        self, audit_id: str, in_scope_files: list[str], target_dir: str
    ) -> list[Hypothesis]:
        """Single-pass: analyze interactions BETWEEN contracts in scope."""
        from pathlib import Path

        if len(in_scope_files) < 2:
            return []

        sources: dict[str, str] = {}
        for sol_path in in_scope_files:
            p = Path(sol_path)
            if not p.exists() and target_dir:
                p = Path(target_dir) / sol_path.lstrip("./")
            if p.exists():
                try:
                    sources[p.stem] = p.read_text(errors="replace")[:6000]
                except OSError:
                    pass

        if len(sources) < 2:
            return []

        contract_names = list(sources.keys())
        interactions = [
            f"- {caller} references {callee}"
            for caller, code in sources.items()
            for callee in contract_names
            if caller != callee and (callee in code or f"I{callee}" in code)
        ]

        if not interactions:
            return []

        al = get_audit_logger()
        if al:
            al.log_decision("hunter", "cross_contract_check",
                f"Found {len(interactions)} cross-contract references",
                {"interactions": interactions})

        source_sections = [
            f"### {name}\n```solidity\n{code}\n```"
            for name, code in sources.items()
        ]

        prompt = f"""Analyze INTERACTIONS between these contracts for vulnerabilities
that only manifest when contracts work TOGETHER.

## Contracts in Scope
{chr(10).join(source_sections)}

## Cross-Contract References
{chr(10).join(interactions)}

## Check each interaction for:
1. CONSISTENCY: Do both contracts agree on definitions?
   (epoch boundaries, rounding direction, token decimals, state meanings)
2. STALE READS: If A reads state from B, can B's state change between read and use?
3. ASSUMPTION MISMATCH: Does A assume something about B that B doesn't guarantee?
4. TIMING GAP: Do both contracts snapshot at the same moment? If not, exploitable?
5. SHARED CONCEPTS: If both define "epoch" or "balance", are definitions identical?

Only report vulnerabilities requiring BOTH contracts. Do not repeat single-contract findings.
"""
        try:
            response = await self._provider.complete(
                model=self._model,
                max_tokens=8192,
                system="You are a cross-contract interaction vulnerability specialist.",
                tools=[_HYPOTHESIS_TOOL],
                tool_choice={"name": "report_hypotheses"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw = _extract_hypotheses_raw(response)
            return _build_hypotheses(raw, audit_id, "cross_contract")
        except Exception as exc:
            log.warning("hunter.cross_contract_check_failed", error=str(exc))
            return []

    # ── State helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _analysis_from_state(state: AuditState) -> Optional[SlitherAnalysis]:
        import json
        from pathlib import Path

        raw = state.get("slither_analysis_raw")
        if not raw:
            # Fallback: load cached slither output directly from target_dir
            target_dir = state.get("target_dir", "")
            for candidate in [".slither_analysis.json", "slither_output.json"]:
                cached = Path(target_dir) / candidate if target_dir else None
                if cached and cached.exists():
                    try:
                        raw = json.loads(cached.read_text())
                        log.info("hunter.loaded_slither_from_disk", path=str(cached))
                        break
                    except Exception:
                        pass
        if not raw:
            return None
        try:
            return parse_slither_json(raw)
        except Exception as exc:
            log.warning("hunter.parse_analysis_failed", error=str(exc))
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_scope_sources(
    target_dir: str,
    in_scope_files: list[str],
    context_files: list[str] | None = None,
    context_budget_chars: int = 8_000,
    in_scope_budget_chars: int = 40_000,
) -> str:
    """Read all in-scope .sol files and return combined source code for Hunter prompt.

    context_files are dependency contracts included with a char budget cap so the
    Hunter can see the bodies of functions it calls (e.g. PanopticMath) without
    blowing the full context window.
    """
    from pathlib import Path

    parts: list[str] = []

    if in_scope_files:
        # Proportional budget: every in-scope file gets an equal slice of the total
        # budget. Files shorter than their slice use only what they need; the savings
        # are redistributed in a second pass so no budget is wasted.
        # This guarantees ALL files appear in the context — critical for protocols where
        # a large main contract (e.g. TrustBonding) and a small auth contract (AtomWallet)
        # both need to be visible for cross-contract bug detection.
        readable: list[tuple[Path, str]] = []
        for fpath in in_scope_files:
            p = Path(fpath)
            if not p.exists():
                continue
            try:
                readable.append((p, p.read_text(errors="ignore")))
            except OSError:
                pass

        if readable:
            n = len(readable)
            per_file = in_scope_budget_chars // n
            leftover = 0
            for p, content in readable:
                alloc = per_file + leftover
                snippet = content[:alloc]
                parts.append(f"// === {p.name} [IN SCOPE] ===\n{snippet}")
                leftover = alloc - len(snippet)  # unused budget rolls over to next file
    elif target_dir:
        tp = Path(target_dir)
        sd = tp / "src"
        src_dir = sd if sd.is_dir() else tp
        for sol in sorted(src_dir.rglob("*.sol")):
            try:
                content = sol.read_text(errors="ignore")
                parts.append(f"// === {sol.name} ===\n{content}")
            except OSError:
                pass

    # Append dependency context up to budget (truncated per-file to spread the budget)
    if context_files:
        remaining = context_budget_chars
        for fpath in sorted(context_files):
            if remaining <= 0:
                break
            try:
                p = Path(fpath)
                if p.exists():
                    content = p.read_text(errors="ignore")
                    snippet = content[:remaining]
                    parts.append(f"// === {p.name} [CONTEXT/DEPENDENCY] ===\n{snippet}")
                    remaining -= len(snippet)
            except OSError:
                pass

    return "\n\n".join(parts)


def _contract_in_scope(source_file: str, in_scope_files: list[str]) -> bool:
    """Return True if the contract's source_file matches any in-scope path."""
    if not source_file or not in_scope_files:
        return True  # no scope restriction — include everything
    for p in in_scope_files:
        if source_file == p or source_file.endswith(p) or p.endswith(source_file):
            return True
        # Compare by basename as fallback
        if source_file.split("/")[-1] == p.split("/")[-1]:
            return True
    return False


def _build_scope_info(
    in_scope_files: list[str],
    context_files: list[str],
    analysis: "SlitherAnalysis",  # type: ignore[name-defined]
) -> str:
    """Build a scope information section for the Hunter prompt."""
    if not in_scope_files:
        return ""

    def _names(file_list: list[str]) -> list[str]:
        names = []
        for f in file_list:
            basename = f.split("/")[-1].replace(".sol", "")
            names.append(basename)
        return names

    in_scope_names = _names(in_scope_files)
    context_names = _names(context_files)

    parts = ["\n## Scope Information"]
    if in_scope_names:
        parts.append(
            "Contracts IN SCOPE — look for bugs HERE:\n"
            + "\n".join(f"  - {n}" for n in in_scope_names)
        )
    if context_names:
        parts.append(
            "Dependency contracts (context only — DO NOT report bugs in these):\n"
            + "\n".join(f"  - {n}" for n in context_names)
        )
    parts.append(
        "Focus hypotheses on the IN SCOPE contracts. "
        "Cross-contract interactions with dependencies are relevant only as attack vectors "
        "INTO the in-scope contracts."
    )
    return "\n".join(parts)


def _build_prompt(
    context: SubgraphContext,
    fork_protocol: Optional[str],
    documentation: str = "",
    fn_invariants: Optional[list] = None,
    scope_info: str = "",
    source_code: str = "",
    derived_invariants: Optional[list[str]] = None,
    threat_summary: str = "",
) -> str:
    parts = [
        "## Code Slice",
        context.code_slice or "(no code available)",
    ]

    if threat_summary:
        parts.append(
            "\n## Threat Model (where value concentrates — prioritize impact here)\n"
            + threat_summary
        )

    if context.state_interactions:
        reads = [i["var_name"] for i in context.state_interactions if i["rel_type"] == "READS"]
        writes = [i["var_name"] for i in context.state_interactions if i["rel_type"] == "WRITES"]
        parts.append(f"\n## State Variables\nREADS: {reads}\nWRITES: {writes}")

    if context.external_calls:
        lines = [
            f"  {e['fn_name']} → {e['call_type']}({e['target_type']}) @ line {e['line']}"
            for e in context.external_calls
        ]
        parts.append("\n## External Calls\n" + "\n".join(lines))

    if context.modifiers:
        parts.append(f"\n## Modifiers\n{context.modifiers}")

    if fork_protocol:
        parts.append(
            f"\n## Fork Context\nThis code is a fork of {fork_protocol}. "
            "Pay special attention to modifications — they are the highest-risk area."
        )

    if context.similar_historical_cases:
        parts.append("\n## Similar Historical Vulnerabilities")
        for case in context.similar_historical_cases[:3]:
            title = case.get("title", "")
            summary = case.get("summary", "")[:200]
            parts.append(f"- {title}: {summary}")

    if derived_invariants:
        parts.append("\n## Protocol Invariants (derived from documentation + code)")
        for inv in derived_invariants:
            parts.append(f"- {inv}")
        parts.append(
            "\nFor EACH invariant above: determine if this function can violate it with "
            "any input or state combination. If yes → report a hypothesis."
        )

    if fn_invariants:
        parts.append("\n## Formal Invariants For This Function")
        for inv in fn_invariants:
            nl = getattr(inv, "natural_language", str(inv))
            confidence = getattr(inv, "confidence", None)
            conf_str = f" (confidence={confidence:.0%})" if confidence else ""
            parts.append(f"- {nl}{conf_str}")
        parts.append(
            "\nIf any invariant above mentions truncation, precision loss, or division — "
            "you MUST generate a hypothesis about integer precision loss for this function."
        )

    doc_text = documentation.strip() if documentation else ""
    if doc_text:
        parts.append(f"\n## Project Documentation\n{doc_text}")
    else:
        parts.append("\n## Project Documentation\nNo project documentation available")

    if source_code:
        parts.append(f"\n## Full Source Code\n```solidity\n{source_code}\n```")

    if scope_info:
        parts.append(scope_info)

    # Financial semantics check — injected when the code slice contains value-computation keywords.
    # This targets sign-reversal and accounting bugs (H-01 / H-02 class) that SWC pattern
    # matching misses entirely.
    _FIN_TRIGGER_KWS = {
        "premium", "exposure", "longamount", "shortamount", "nav", "totalnav",
        "haircut", "positionidlist", "poolexposure", "performancefee", "totalassets",
        "computenav", "longpremium", "shortpremium",
    }
    code_lower = (context.code_slice or "").lower()
    if any(kw in code_lower for kw in _FIN_TRIGGER_KWS):
        parts.append(
            "\n## Financial Semantics Check (MANDATORY for this function)\n"
            "This function performs financial value computations. Before reporting hypotheses:\n"
            "\n"
            "1. **Sign convention audit** — For EVERY subtraction (A - B) in this function:\n"
            "   - What does A represent? (asset / liability / long / short / fee / collateral)\n"
            "   - What does B represent?\n"
            "   - Should the result be positive or negative from the vault's perspective?\n"
            "   - Is the subtraction order CORRECT?\n"
            "     Example: `shortPremium - longPremium` = net owed to protocol (CORRECT);\n"
            "     `longPremium - shortPremium` flips the sign → OVERSTATES vault NAV (BUG).\n"
            "\n"
            "   **CRITICAL — Parallel slot consistency check:**\n"
            "   If the function computes two parallel values for token0 and token1 (e.g. via\n"
            "   `.rightSlot()` and `.leftSlot()`), verify BOTH use the SAME subtraction order.\n"
            "   A common bug: token0 uses `short - long` (correct) but token1 uses `long - short`\n"
            "   (inverted sign). Even one inverted slot corrupts the entire NAV / exposure sum.\n"
            "   Scan every pair of `.rightSlot()` / `.leftSlot()` computations for sign mismatch.\n"
            "\n"
            "2. **Token accounting loop audit** — If there is a loop over positions/legs:\n"
            "   - Is each token balance counted EXACTLY ONCE?\n"
            "   - Are tokens inside a Uniswap/AMM pool counted differently than free tokens?\n"
            "   - Could a token be double-counted (once as free balance, once as LP value)?\n"
            "   - Could a token be missed entirely when it transitions between in-pool and free?\n"
            "\n"
            "3. **NAV / share price integrity** — If this feeds into share price or NAV:\n"
            "   - An inflated NAV lets depositors buy cheap shares → loss for existing holders.\n"
            "   - A deflated NAV lets withdrawers exit at discount → loss for remaining holders.\n"
            "   - Does any arithmetic error here systematically bias NAV in one direction?\n"
            "\n"
            "Report any sign error, double-count, or missed token as a HIGH/CRITICAL finding."
        )

    parts.append(
        "\nCRITICAL DIRECTIVE: Do not report missing `onlyOwner` or generic ERC20 vulnerabilities "
        "unless you have explicit proof in the slice. Focus on logical flaws related to the Protocol "
        "Context described above. If the function interacts with external bridges, composability, or "
        "reward distributions, assume basic math is correct but sequence/state logic is flawed."
    )

    # Surface the REAL bodies of functions the slice calls (resolved from the loaded
    # source), so Hunter reasons about actual cross-contract behavior, not guesses.
    if source_code:
        from audit_agent.knowledge.external_behaviors import resolve_external_call_bodies_in_text
        ext_bodies = resolve_external_call_bodies_in_text(
            context.code_slice or "", source_code
        )
        if ext_bodies:
            parts.append("\n" + ext_bodies)

    user_message = "\n".join(parts)
    user_message += (
        "\n\n---\n"
        "NOW call `report_hypotheses` with 2-5 vulnerability hypotheses. "
        "Do NOT write explanatory prose — output ONLY the tool call."
    )
    return user_message


def _extract_hypotheses_raw(response: Any) -> list[dict[str, Any]]:
    """Extract hypothesis dicts from an LLM provider response's tool calls."""
    for tc in response.tool_calls:
        if tc.name == "report_hypotheses":
            return tc.arguments.get("hypotheses", [])
    return []


def _build_hypotheses(
    raw: list[dict[str, Any]], audit_id: str, default_fn_id: str
) -> list[Hypothesis]:
    results: list[Hypothesis] = []
    for item in raw:
        title = item.get("title", "").strip()
        attack_scenario = item.get("attack_scenario", "").replace("—", "-").replace("”", '"').replace("“", '"')
        if not title:
            continue
        results.append(
            Hypothesis(
                id=str(uuid.uuid4()),
                audit_id=audit_id,
                title=title,
                vulnerability_class=item.get("vulnerability_class"),
                affected_functions=item.get("affected_functions") or [default_fn_id],
                attack_scenario=item.get("attack_scenario", ""),
                required_preconditions=item.get("required_preconditions", []),
                estimated_severity=item.get("estimated_severity", "medium"),  # type: ignore[arg-type]
                similar_historical_cases=item.get("similar_historical_cases", []),
                priority_score=float(item.get("priority_score", 0.0)),
            )
        )
    return results


def _deduplicate(
    hypotheses: list[Hypothesis], static_findings: list[StaticFinding]
) -> list[Hypothesis]:
    """Remove hypotheses whose title has ≥70% token overlap with a static finding."""
    if not static_findings:
        return hypotheses

    static_token_sets = [set(sf.title.lower().split()) for sf in static_findings]

    def _overlaps(h: Hypothesis) -> bool:
        h_tokens = set(h.title.lower().split())
        if not h_tokens:
            return False
        for s_tokens in static_token_sets:
            if not s_tokens:
                continue
            overlap = len(h_tokens & s_tokens) / min(len(h_tokens), len(s_tokens))
            if overlap >= 0.7:
                return True
        return False

    return [h for h in hypotheses if not _overlaps(h)]


def _dedup_same_class(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    """Collapse near-duplicate hypotheses within the same vulnerability class.

    When per-function analysis finds the same root cause (e.g. integer division truncation)
    in many functions, it generates N near-identical hypotheses that crowd out other findings.
    This keeps at most MAX_PER_CLASS per vulnerability_class, prioritised by priority_score.

    Critical/high-severity classes get a higher cap because each instance may have a
    different impact (e.g. two different reentrancy paths are distinct findings).
    Low/medium instances of the same class in different functions are usually the same bug.
    """
    _CAP_HIGH   = 3  # critical or high
    _CAP_MEDIUM = 2  # medium
    _CAP_LOW    = 1  # low or info

    from collections import defaultdict

    # Group by vulnerability_class (case-insensitive)
    by_class: dict[str, list[Hypothesis]] = defaultdict(list)
    for h in hypotheses:
        key = (h.vulnerability_class or "unknown").lower().strip()
        by_class[key].append(h)

    result: list[Hypothesis] = []
    for cls, group in by_class.items():
        # Sort group by priority_score descending so we keep the best ones
        group_sorted = sorted(group, key=lambda h: h.priority_score, reverse=True)

        # Determine cap based on severity of the highest-scored hypothesis in the group
        top_sev = group_sorted[0].estimated_severity.lower() if group_sorted else "low"
        if top_sev in ("critical", "high"):
            cap = _CAP_HIGH
        elif top_sev == "medium":
            cap = _CAP_MEDIUM
        else:
            cap = _CAP_LOW

        result.extend(group_sorted[:cap])

    return result


def _dedup_by_title_similarity(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    """Remove near-duplicate hypotheses based on Jaccard similarity of title tokens.

    Two hypotheses are considered duplicates when their title token overlap ≥ 60%,
    regardless of vulnerability class. The higher-priority one is kept.
    This catches cases like 'Epoch Boundary Off-by-One in X' vs
    'Epoch Boundary Off-by-One Error' that _dedup_same_class misses because
    they were assigned different SWC codes.
    """
    _SIMILARITY_THRESHOLD = 0.60

    def _tokens(title: str) -> set[str]:
        # Lower-case, strip punctuation, keep only words ≥ 3 chars
        import re
        return {w for w in re.sub(r"[^a-z0-9 ]", " ", title.lower()).split() if len(w) >= 3}

    # Sort by priority descending so we always keep the best one
    sorted_hyps = sorted(hypotheses, key=lambda h: h.priority_score, reverse=True)
    kept: list[Hypothesis] = []
    kept_tokens: list[set[str]] = []

    for h in sorted_hyps:
        h_toks = _tokens(h.title)
        if not h_toks:
            kept.append(h)
            kept_tokens.append(h_toks)
            continue
        is_dup = False
        for existing_toks in kept_tokens:
            if not existing_toks:
                continue
            intersection = len(h_toks & existing_toks)
            union = len(h_toks | existing_toks)
            jaccard = intersection / union if union else 0.0
            if jaccard >= _SIMILARITY_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            kept.append(h)
            kept_tokens.append(h_toks)

    return kept


def _parse_id_groups(raw: str, valid_ids: set[int]) -> Optional[list[list[int]]]:
    """Parse an LLM 'array of arrays of ids' response into groups.

    Returns None on ANY problem — malformed JSON, a foreign or duplicate id, or an
    incomplete partition — so the caller leaves that cluster untouched rather than
    risk dropping a genuine finding on a bad response.
    """
    import json
    import re

    m = re.search(r"\[\s*\[.*\]\s*\]", raw, re.DOTALL) or re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    groups: list[list[int]] = []
    seen: set[int] = set()
    for g in parsed:
        if not isinstance(g, list):
            return None
        grp: list[int] = []
        for x in g:
            if not isinstance(x, int) or x not in valid_ids or x in seen:
                return None
            seen.add(x)
            grp.append(x)
        if grp:
            groups.append(grp)
    if seen != valid_ids:  # must be a complete partition of the cluster
        return None
    return groups


def _consolidate_by_function(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    """Merge multiple hypotheses targeting the same function into one rich report.

    When Hunter generates N hypotheses for the same function (e.g. 7 variants of
    the same signature-bypass bug), only 1 is kept — the highest-priority one —
    but its attack_scenario is enriched with unique details from merged siblings.
    This prevents Exploit from writing N nearly-identical PoC tests for the same
    root cause.
    """
    from collections import defaultdict

    def _primary_fn(h: Hypothesis) -> str:
        fns = h.affected_functions or []
        return fns[0].split("(")[0].strip() if fns else "__no_function__"

    groups: dict[str, list[Hypothesis]] = defaultdict(list)
    for h in hypotheses:
        groups[_primary_fn(h)].append(h)

    result: list[Hypothesis] = []
    for fn, group in groups.items():
        if len(group) == 1:
            result.extend(group)
            continue

        # Sort: highest priority first
        group_sorted = sorted(group, key=lambda h: h.priority_score, reverse=True)
        winner = group_sorted[0]
        siblings = group_sorted[1:]

        # Collect unique attack angles from siblings
        extra_scenarios: list[str] = []
        winner_words = set((winner.attack_scenario or "").lower().split())
        for sib in siblings:
            sib_scenario = (sib.attack_scenario or "").strip()
            if not sib_scenario:
                continue
            unique_words = set(sib_scenario.lower().split()) - winner_words
            if len(unique_words) >= 8:
                extra_scenarios.append(
                    f"[Alt vector — {sib.title[:60]}]: {sib_scenario[:300]}"
                )

        if extra_scenarios:
            merged_scenario = (
                (winner.attack_scenario or "")
                + "\n\n--- Consolidated attack vectors ---\n"
                + "\n".join(extra_scenarios[:3])
            )
            winner = winner.model_copy(update={"attack_scenario": merged_scenario})

        result.append(winner)

    return result
