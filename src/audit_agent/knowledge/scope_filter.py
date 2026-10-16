"""Scope / known-issue self-flag.

A finding can be a genuine TRUE POSITIVE yet be UNPAYABLE because it falls under a
program's standard exclusions (MEV/sandwich/frontrunning, slippage, gas, admin
centralization, external-protocol bugs, …), matches a project-declared KNOWN ISSUE,
or sits below the paid severity tier. Scope is orthogonal to validity, so this runs
over ALL validated/needs-review findings and annotates them — it never changes a
finding's verdict, only flags whether it's worth submitting.

Example: a technically valid "missing slippage check" finding is still out of
scope when the project's docs list user-side slippage protection as a known
issue. This module flags exactly that.

Deterministic (no LLM): keyword classes + project known-issues extracted from docs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ── Standard exclusion classes (common across Immunefi / Code4rena / Sherlock) ──
# (label, keyword fragments matched case-insensitively against title+class+description)
_EXCLUSIONS: list[tuple[str, tuple[str, ...]]] = [
    ("MEV / sandwich / frontrunning",
     ("sandwich", "front-run", "frontrun", "front run", "back-run", "backrun",
      " mev", "mev ", "mempool", "transaction ordering", "reorder")),
    ("slippage (user-side responsibility)",
     ("slippage", "minamountout", "min amount out", "minimum amount out",
      "min output", "minimum output", "price impact", "amountoutmin")),
    ("gas optimization / informational / style",
     ("gas optimization", "gas saving", "gas cost", "informational", "best practice",
      "code style", "naming convention", "typo", "missing event", "unused")),
    ("centralization / privileged-role trust assumption",
     ("centralization", "admin key", "admin can", "owner can", "governance can",
      "privileged role", "trusted role", "malicious owner", "malicious admin",
      "rug pull", "onlyowner can", "onlyadmin can", "compromised admin")),
    ("griefing without fund loss",
     ("griefing", "grief attack", " grief ")),
    ("external / third-party protocol behavior",
     ("aave bug", "euler bug", "morpho bug", "chainlink down", "third-party protocol",
      "external protocol vulnerability", "upstream protocol", "external lending protocol")),
    ("network / infrastructure",
     ("reorg", "re-org", "sequencer down", "network liveness", "block stuffing",
      "chain halt", "rpc ")),
    ("theoretical / negligible impact",
     ("theoretical", "negligible", "dust amount", "rounding dust", "no real impact")),
]

# Impact phrases that usually map to an IN-SCOPE high-value category.
_IN_SCOPE_IMPACT = (
    "theft", "steal", "drain", "direct loss of funds", "loss of user funds",
    "insolven", "bad debt", "permanent freez", "permanently freez", "freeze of funds",
    "funds locked", "lock user funds", "unauthorized withdraw", "unlimited mint",
    "protocol insolven", "lose funds", "steal funds",
)

_LOW_TIERS = frozenset({"medium", "low", "info"})


@dataclass
class ScopeAssessment:
    flag: str        # likely_in_scope | out_of_scope | known_issue | below_paid_tier | needs_scope_review
    note: str
    payable: bool    # rough heuristic: worth submitting?


def extract_known_issues(documentation: str) -> str:
    """Pull a project's declared known-issues / out-of-scope / by-design text from
    its collected documentation, for substring matching against findings."""
    if not documentation:
        return ""
    lines = documentation.splitlines()
    keep: list[str] = []
    grab = 0
    hdr = re.compile(r"known\s+issue|out[\s-]?of[\s-]?scope|acceptable\s+risk|by\s+design|"
                     r"won'?t\s+fix|accepted\s+risk|not\s+in\s+scope", re.I)
    for ln in lines:
        if hdr.search(ln):
            grab = 8  # capture this header + a few following lines
        if grab > 0:
            keep.append(ln)
            grab -= 1
    return "\n".join(keep)[:8000].lower()


def assess_scope(
    title: str,
    vulnerability_class: Optional[str],
    description: str,
    severity: str,
    affected_functions: list[str],
    platform: str = "generic",
    known_issues: str = "",
) -> ScopeAssessment:
    """Classify a finding's scope/payability. Priority: known_issue > out_of_scope >
    below_paid_tier > likely_in_scope > needs_scope_review."""
    hay = " ".join([title or "", vulnerability_class or "", description or ""]).lower()

    # 1) Project-declared known issue: an affected function named in the known-issues text.
    if known_issues:
        for aff in affected_functions or []:
            fn = aff.split("(")[0].split(".")[-1].split("-")[-1].strip().lower()
            if len(fn) >= 4 and fn in known_issues:
                return ScopeAssessment(
                    "known_issue", f"Project docs list a known issue referencing `{fn}` — likely accepted/out of scope.", False)
        # also a strong title-phrase overlap
        title_words = {w for w in re.findall(r"[a-z]{5,}", (title or "").lower())}
        if title_words and len(title_words & set(re.findall(r"[a-z]{5,}", known_issues))) >= 3:
            return ScopeAssessment(
                "known_issue", "Strong overlap with the project's declared known-issues section.", False)

    # 2) Standard exclusion class.
    for label, kws in _EXCLUSIONS:
        if any(k in hay for k in kws):
            return ScopeAssessment(
                "out_of_scope", f"Matches a common program exclusion: {label}. Verify against the program's rules.", False)

    # 3) Below paid severity tier (most SC programs pay High/Critical only).
    sev = (severity or "").lower()
    if sev in _LOW_TIERS:
        return ScopeAssessment(
            "below_paid_tier",
            f"Severity '{sev}' is often below the paid tier (many programs pay only High/Critical for smart contracts). Verify program tiers.",
            False)

    # 4) Maps to an in-scope high-value impact.
    if any(p in hay for p in _IN_SCOPE_IMPACT):
        return ScopeAssessment(
            "likely_in_scope", "Impact maps to a high-value category (theft/freezing/insolvency). Confirm a runnable PoC — most programs require one.", True)

    # 5) Unclear.
    return ScopeAssessment(
        "needs_scope_review", "Could not map impact to an in-scope category from the description; review against the program's impacts-in-scope list.", True)


def scope_banner(a: ScopeAssessment) -> str:
    """One-line markdown banner for a report header."""
    icon = {
        "likely_in_scope": "✅ likely in scope",
        "needs_scope_review": "❓ scope review needed",
        "below_paid_tier": "⚠️ below paid tier",
        "out_of_scope": "❌ likely OUT OF SCOPE",
        "known_issue": "❌ known issue (out of scope)",
    }.get(a.flag, a.flag)
    return f"**Scope:** {icon} — {a.note}"
