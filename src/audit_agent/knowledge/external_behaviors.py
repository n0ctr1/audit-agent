"""Curated external-protocol behavior reference + dependency body resolver.

Cross-protocol false positives dominate real audits: the in-scope code looks
buggy in isolation, but the bug rests on a WRONG assumption about how an external
contract (flashloan provider, oracle, token standard, connector) actually behaves.
The model usually only sees the in-scope file + dependency interfaces ("read for
types"), never the real behavior — so it both generates and validates these FPs.

This module supplies two things to Hunter / Skeptic / Refuter:

  EXTERNAL_PROTOCOL_FACTS — a concise reference of the well-known behaviors that
  most often cause cross-protocol FPs (verified, not speculative).

  resolve_external_call_bodies() — best-effort: scan in-scope source for external
  calls (`X.method(...)`) and pull the matching `function method(...)` definition
  out of the available dependency/lib files, so the model sees the REAL body.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Curated behavior reference (kept tight — high signal only)
# ─────────────────────────────────────────────────────────────────────────────

EXTERNAL_PROTOCOL_FACTS = (
    "## EXTERNAL PROTOCOL BEHAVIOR (verified facts — use to avoid cross-protocol false positives)\n"
    "Before claiming a bug that depends on an external contract, check it against these:\n"
    "- FLASHLOAN CALLBACKS (Morpho, Aave, Balancer, Uniswap, Maker): the provider delivers the "
    "callback (onMorphoFlashLoan / executeOperation / receiveFlashLoan / uniswapV2Call) to the "
    "ADDRESS THAT CALLED flashLoan — i.e. msg.sender — NOT to an arbitrary target. A victim "
    "contract's callback only runs when that contract itself initiated the loan, and the `data` is "
    "whatever that contract passed. So 'anyone can call the permissionless flashLoan with arbitrary "
    "data and the victim's callback receives it' is FALSE. A `require(msg.sender == PROVIDER)` guard is "
    "sufficient given this routing.\n"
    "- AAVE aTOKENS: scaledBalanceOf() returns the balance divided by the liquidity index — it is "
    "CONSTANT under interest accrual (that's its purpose). Protocols deliberately track positions in "
    "scaled units. Comparing/subtracting scaledBalanceOf against an internal 'scaled' accounting "
    "variable is NOT a unit mismatch when both sides are scaled. A drop in scaledBalanceOf signals the "
    "aTokens were seized (e.g. external liquidation).\n"
    "- ERC4626 / EVK VAULTS: shares vs assets are intentionally different units; convertToAssets/"
    "convertToShares bridge them. Subtracting shares from shares (or assets from assets) is correct; a "
    "'mismatch' claim requires showing the two sides are actually different units in THIS code.\n"
    "- PERMIT2 / safeTransferFrom(... from, ...): pulling tokens from `from` requires `from` to have "
    "approved this contract / signed a permit. An attacker cannot set `from` to an arbitrary victim "
    "unless the victim approved the contract AND the attacker controls the call path that supplies "
    "`from` (check who encodes/sets it).\n"
    "- EVC / EVK (Euler Vault Connector): cross-contract calls are authenticated via the EVC; "
    "onBehalfOfAccount and _msgSender() reflect the EVC-authenticated account, not raw msg.sender.\n"
    "- ORACLES (Chainlink/Euler routers): mind decimals (often 8) and that latestAnswer/getAssetPrice "
    "return protocol-specific scales; a 'decimal mismatch' claim must show the actual scales used.\n"
    "- GOVERNANCE-GATED INPUTS: values written only via onlyAdmin/onlyOwner/onlyRole/governance are NOT "
    "attacker-controlled; a finding that treats such a value as attacker-injected is a false positive.\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# Best-effort resolver of external call bodies
# ─────────────────────────────────────────────────────────────────────────────

# `Identifier.method(`  — captures the receiver var/type and the method name.
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\s*\(")

# Solidity keywords / common receivers we don't want to chase.
_IGNORE_RECEIVERS = frozenset({
    "this", "super", "msg", "tx", "block", "abi", "address", "type", "bytes",
    "string", "require", "assert", "emit", "return", "new", "uint", "int",
    "Math", "SafeERC20", "SafeCast",
})
_IGNORE_METHODS = frozenset({
    "length", "push", "pop", "call", "delegatecall", "staticcall", "encode",
    "encodeCall", "encodePacked", "decode", "selector", "sender", "value",
    "timestamp", "number", "min", "max", "sub", "add", "mul", "div",
})


def _extract_external_calls(source: str) -> list[str]:
    """Return distinct external method names invoked in `source`."""
    methods: list[str] = []
    seen: set[str] = set()
    for recv, method in _CALL_RE.findall(source):
        if recv in _IGNORE_RECEIVERS or method in _IGNORE_METHODS:
            continue
        if method not in seen:
            seen.add(method)
            methods.append(method)
    return methods


def _find_function_body(method: str, dep_text: str, max_lines: int = 40) -> Optional[str]:
    """Extract the source of `function <method>(...)` from dep_text (brace-matched)."""
    m = re.search(rf"\bfunction\s+{re.escape(method)}\s*\(", dep_text)
    if not m:
        return None
    start = m.start()
    # Find the opening brace of the body (skip past the signature / modifiers).
    brace = dep_text.find("{", m.end())
    semic = dep_text.find(";", m.end())
    if brace == -1 or (semic != -1 and semic < brace):
        return None  # interface decl (no body) — not useful behavior
    depth = 0
    i = brace
    while i < len(dep_text):
        c = dep_text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                body = dep_text[start : i + 1]
                lines = body.splitlines()
                if len(lines) > max_lines:
                    body = "\n".join(lines[:max_lines]) + "\n    // ... [truncated]"
                return body
        i += 1
    return None


def resolve_external_call_bodies(
    in_scope_source: str,
    dependency_files: list[str],
    target_dir: str = "",
    max_bodies: int = 12,
    char_budget: int = 12_000,
) -> str:
    """Pull the real bodies of external functions called by in-scope code.

    Returns a formatted block (or "" if nothing resolved). Skips interface-only
    declarations (no body) — only concrete implementations carry behavior.
    """
    methods = _extract_external_calls(in_scope_source)
    if not methods or not dependency_files:
        return ""

    # Load dependency texts lazily, largest-first relevance is not needed; just concat search.
    resolved: list[str] = []
    used = 0
    seen_methods: set[str] = set()

    for method in methods:
        if len(resolved) >= max_bodies or used >= char_budget:
            break
        if method in seen_methods:
            continue
        for dep in dependency_files:
            p = Path(dep)
            if not p.exists() and target_dir:
                p = Path(target_dir) / dep.lstrip("./")
            if not p.exists():
                continue
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            if f"function {method}" not in text:
                continue
            body = _find_function_body(method, text)
            if body:
                block = f"// from {p.name}: {method}()\n{body}"
                resolved.append(block)
                used += len(block)
                seen_methods.add(method)
                break

    if not resolved:
        return ""
    return (
        "## Resolved external/dependency function bodies (REAL behavior of calls made by the code above)\n"
        "```solidity\n" + "\n\n".join(resolved)[:char_budget] + "\n```\n"
    )


def resolve_external_call_bodies_in_text(
    scan_source: str,
    search_source: str,
    max_bodies: int = 10,
    char_budget: int = 10_000,
) -> str:
    """Like resolve_external_call_bodies but both inputs are already-loaded text.

    Scans `scan_source` for external calls and pulls their concrete bodies out of
    `search_source` (typically the full in-scope + dependency blob). Surfaces
    cross-contract behavior the model might miss in a large source blob.
    """
    if not scan_source or not search_source:
        return ""
    resolved: list[str] = []
    used = 0
    for method in _extract_external_calls(scan_source):
        if len(resolved) >= max_bodies or used >= char_budget:
            break
        body = _find_function_body(method, search_source)
        if body:
            block = f"// {method}()\n{body}"
            resolved.append(block)
            used += len(block)
    if not resolved:
        return ""
    return (
        "## Bodies of functions called by the code above (resolved from loaded source — REAL behavior)\n"
        "```solidity\n" + "\n\n".join(resolved)[:char_budget] + "\n```\n"
    )
