"""Execution-safety guard for generated Solidity (PoCs, property and symbolic tests).

Generated code only ever runs inside Docker as ``forge test`` / ``medusa`` /
``halmos``, i.e. against a local EVM or a read-only fork. ``forge test`` never
broadcasts transactions, but a few cheatcodes can still reach a live network or
the host:

  * ``vm.rpc(...)`` — arbitrary JSON-RPC on the fork endpoint, including
    ``eth_sendRawTransaction``;
  * ``vm.ffi(...)`` — shell commands (e.g. ``cast send``) when the target
    project enables ``ffi``;
  * ``vm.env*("PRIVATE_KEY")`` / ``vm.rememberKey`` — loading real signing keys
    (forge auto-loads the target project's ``.env``);
  * ``vm.broadcast`` / ``vm.startBroadcast`` — script-only transaction recording.

Code containing any of these is rejected before execution, and Foundry runs with
``FOUNDRY_FFI=false`` regardless of the project's config.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

# Environment forced on every Foundry container.
SAFE_FOUNDRY_ENV: dict[str, str] = {"FOUNDRY_FFI": "false"}

_KEY_NAME = r"[^\"']*(?:PRIVATE|PRIV_?KEY|MNEMONIC|SEED|DEPLOYER_KEY|SIGNER_KEY)[^\"']*"

UNSAFE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bvm\s*\.\s*ffi\s*\("), "vm.ffi (shell execution)"),
    (re.compile(r"\bvm\s*\.\s*rpc\s*\("), "vm.rpc (raw JSON-RPC to a live endpoint)"),
    (re.compile(r"eth_send(?:Raw)?Transaction"), "eth_sendTransaction / eth_sendRawTransaction"),
    (re.compile(r"\bvm\s*\.\s*(?:startBroadcast|broadcast)\s*\("), "vm.broadcast / vm.startBroadcast"),
    (re.compile(r"\bvm\s*\.\s*rememberKey\s*\("), "vm.rememberKey (signing key for broadcasting)"),
    (re.compile(r"\bvm\s*\.\s*env\w*\s*\(\s*[\"']" + _KEY_NAME + r"[\"']", re.IGNORECASE),
     "reads a private key / mnemonic from the environment"),
)


def _strip_comments(source: str) -> str:
    return re.sub(r"//[^\n]*|/\*.*?\*/", " ", source, flags=re.DOTALL)


def find_unsafe_constructs(source: str) -> list[str]:
    """Reasons why ``source`` must not be executed (empty list = safe)."""
    code = _strip_comments(source or "")
    return [reason for pattern, reason in UNSAFE_PATTERNS if pattern.search(code)]


def scan_files(paths: Iterable[Path]) -> dict[str, list[str]]:
    """{path: reasons} for files that contain unsafe constructs."""
    out: dict[str, list[str]] = {}
    for p in paths:
        try:
            reasons = find_unsafe_constructs(Path(p).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if reasons:
            out[str(p)] = reasons
    return out


def rejection_message(violations: dict[str, list[str]] | list[str]) -> str:
    if isinstance(violations, dict):
        detail = "; ".join(f"{Path(k).name}: {', '.join(v)}" for k, v in violations.items())
    else:
        detail = ", ".join(violations)
    return ("PoC rejected by safety guard (generated code may only run against a local "
            f"or read-only forked EVM): {detail}")
