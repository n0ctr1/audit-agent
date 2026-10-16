"""Vulnerability-class taxonomy used by the benchmark matcher.

LLM agents and official contest reports name the same bug class in many ways
("SWC-107", "Cross-function reentrancy", "CEI violation"). The matcher treats
the class as a *soft* signal: two class labels are compatible when they map to
the same canonical class, or to a pair of canonical classes declared related in
``RELATED``.

A raw label may map to several canonical classes ("Flash-loan oracle
manipulation" → {flash-loan, oracle-manipulation}). Labels that map to nothing
are only compatible with an identical label (after normalisation).

Only real SWC registry IDs (SWC-100 … SWC-136) are listed. Non-existent codes
that LLMs sometimes invent (e.g. "SWC-207") deliberately map to nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class VulnClass:
    name: str
    # Phrases matched as substrings against the normalised label
    # (lowercase, "-", "_", "/" replaced by spaces, single spaces).
    aliases: tuple[str, ...] = field(default_factory=tuple)
    swc: tuple[int, ...] = field(default_factory=tuple)


TAXONOMY: tuple[VulnClass, ...] = (
    VulnClass(
        "reentrancy",
        ("reentran", "re entran", "reentry", "checks effects interactions", "cei violation"),
        (107,),
    ),
    VulnClass(
        "access-control",
        (
            "access control", "authoriz", "authoris", "unauthori", "privilege",
            "permission", "onlyowner", "only owner", "ownership", "unprotected",
            "tx.origin", "tx origin", "missing modifier", "centraliz", "centralis",
        ),
        (100, 105, 106, 115),
    ),
    VulnClass(
        "signature-replay",
        (
            "signature", "replay", "ecrecover", "ecdsa", "malleab", "nonce",
            "eip712", "eip 712", "permit", "signed message", "signed data",
            "unsigned field", "unsigned parameter", "unsigned metadata", "validity window",
            "validuntil", "validafter",
        ),
        (117, 121, 122),
    ),
    VulnClass(
        "oracle-manipulation",
        ("oracle", "price manipulation", "price feed", "spot price", "twap", "stale price", "chainlink"),
    ),
    VulnClass(
        "rounding-precision",
        ("rounding", "round down", "round up", "precision", "truncat", "division before multiplication"),
    ),
    VulnClass(
        "arithmetic",
        ("overflow", "underflow", "arithmetic", "unchecked math", "unsafe cast", "downcast", "integer"),
        (101,),
    ),
    VulnClass(
        "logic-accounting",
        (
            "logic", "accounting", "incorrect calculation", "miscalculat", "wrong calculation",
            "double count", "state inconsisten", "inconsistent state", "incorrect state",
            "share inflation", "first depositor", "inflation attack", "donation",
            "reward calculation", "invariant",
        ),
    ),
    VulnClass(
        "dos",
        ("denial of service", "dos", "griefing", "unbounded loop", "gas limit", "brick", "out of gas"),
        (113, 128),
    ),
    VulnClass(
        "locked-funds",
        ("locked fund", "stuck fund", "locked ether", "funds locked", "funds stuck", "frozen fund", "loss of funds"),
    ),
    VulnClass(
        "front-running",
        ("front run", "frontrun", "sandwich", "mev", "slippage", "transaction order", "race condition"),
        (114,),
    ),
    VulnClass(
        "timestamp-epoch",
        (
            "timestamp", "epoch", "block time", "deadline", "expir", "temporal",
            "time window", "time based", "boundary", "off by one",
        ),
        (116,),
    ),
    VulnClass(
        "upgradeability",
        ("upgrade", "proxy", "initializ", "initialis", "storage collision", "storage layout", "uups", "constructor"),
        (109, 118, 124),
    ),
    VulnClass(
        "input-validation",
        (
            "input validation", "missing validation", "insufficient validation", "lack of validation",
            "missing check", "zero address", "sanity check", "bounds check", "parameter validation",
            "improper validation",
        ),
        (123,),
    ),
    VulnClass(
        "unchecked-call",
        ("unchecked return", "return value", "unchecked call", "unchecked transfer", "unchecked external", "low level call"),
        (104,),
    ),
    VulnClass(
        "arbitrary-call",
        ("arbitrary call", "arbitrary external call", "arbitrary calldata", "delegatecall", "untrusted call"),
        (112,),
    ),
    VulnClass(
        "randomness",
        ("random", "predictable", "prng", "entropy"),
        (120,),
    ),
    VulnClass(
        "flash-loan",
        ("flash loan", "flashloan", "economic attack"),
    ),
    VulnClass(
        "token-compatibility",
        (
            "fee on transfer", "rebasing", "erc777", "non standard token", "weird token",
            "token compatib", "missing return value", "decimals",
        ),
    ),
)

# Canonical-class pairs that describe the same bug often enough to count as
# compatible labels. Symmetric. Kept small on purpose: the matcher additionally
# accepts similar titles, and location-match is always mandatory.
RELATED: frozenset[frozenset[str]] = frozenset(
    frozenset(p)
    for p in (
        ("flash-loan", "oracle-manipulation"),
        ("flash-loan", "logic-accounting"),
        ("rounding-precision", "arithmetic"),
        ("rounding-precision", "logic-accounting"),
        ("signature-replay", "access-control"),
        ("upgradeability", "access-control"),
        ("arbitrary-call", "access-control"),
        ("front-running", "oracle-manipulation"),
        ("unchecked-call", "token-compatibility"),
        ("dos", "locked-funds"),
    )
)

_BY_NAME = {c.name: c for c in TAXONOMY}
_SWC_RE = re.compile(r"swc\s*[-_ ]?\s*(\d{3})")


def normalise_label(label: str | None) -> str:
    if not label:
        return ""
    s = label.lower().strip()
    s = re.sub(r"[-_/]+", " ", s)
    return re.sub(r"\s+", " ", s)


def canonical_classes(label: str | None) -> frozenset[str]:
    """Map a raw vulnerability-class label to its canonical classes (possibly empty)."""
    s = normalise_label(label)
    if not s:
        return frozenset()
    found: set[str] = set()
    for code in _SWC_RE.findall(label.lower() if label else ""):
        for c in TAXONOMY:
            if int(code) in c.swc:
                found.add(c.name)
    # Word-start matching so "dos" does not fire inside "cardosa", etc.
    for c in TAXONOMY:
        for alias in c.aliases:
            if re.search(r"(?<![a-z0-9])" + re.escape(alias), s):
                found.add(c.name)
                break
    return frozenset(found)


ClassRelation = Literal["exact", "related", "none"]


def class_relation(label_a: str | None, label_b: str | None) -> ClassRelation:
    """'exact' — same canonical class (or identical label); 'related' — RELATED pair; else 'none'."""
    na, nb = normalise_label(label_a), normalise_label(label_b)
    if not na or not nb:
        return "none"
    if na == nb:
        return "exact"
    ca, cb = canonical_classes(label_a), canonical_classes(label_b)
    if ca & cb:
        return "exact"
    if any(frozenset((a, b)) in RELATED for a in ca for b in cb):
        return "related"
    return "none"


def class_compatible(label_a: str | None, label_b: str | None) -> bool:
    return class_relation(label_a, label_b) != "none"


def known_class(name: str) -> bool:
    return name in _BY_NAME
