"""Project test-suite cross-check — a cheap, high-signal false-positive filter.

A protocol's OWN test suite is ground truth about intended behavior. If a finding
flags `foo()` and the repo has a passing test that calls `foo()` and asserts the
flagged behavior (e.g. `vm.expectRevert(...); foo();`, or a happy-path assert),
the "bug" is almost always intended/known → false positive.

Example: a finding claims "liquidate() reverts on an unhealthy position →
permanent freeze", but the repo has a passing test that builds exactly that state
and wraps the call in `vm.expectRevert(...)` — i.e. the revert is deliberate.
This module surfaces such tests into the Refuter's context.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Test directories and file patterns across the common toolchains.
_TEST_DIRS = ("test", "tests")
_TEST_GLOBS = ("*.t.sol", "*.sol", "*.test.js", "*.test.ts", "*_test.py", "test_*.py")
# Lines that signal a test ASSERTS behavior (strongest FP signal).
_ASSERT_HINTS = ("expectRevert", "assert", "assertEq", "assertTrue", "assertFalse",
                 "assertGt", "assertLt", "should", "expect(", "revert")


def _bare_fn_names(affected_functions: list[str]) -> list[str]:
    """'Contract.foo(uint256)' / 'foo()' → 'foo'. Drops very generic names."""
    names: list[str] = []
    seen: set[str] = set()
    for aff in affected_functions or []:
        n = aff.split("(")[0].split(".")[-1].strip()
        # strip a leading scope like 'C-foo' or 'C.foo' already handled; drop noise
        n = n.split("-")[-1]
        if len(n) >= 4 and n not in seen and not n.isdigit():
            seen.add(n)
            names.append(n)
    return names


def _enclosing_test_block(text: str, idx: int, max_lines: int = 45) -> Optional[str]:
    """Return the `function ...(){...}` (or JS/TS `it('...', ...)`) block containing idx."""
    # find the nearest 'function ' / 'it(' / 'def test' before idx
    starts = [m.start() for m in re.finditer(r"\bfunction\s+\w+\s*\(|^\s*def\s+test\w*\s*\(|\bit\s*\(|\btest\s*\(", text[:idx], re.M)]
    if not starts:
        return None
    start = starts[-1]
    # brace-match (Solidity/JS) from the first '{' after start
    brace = text.find("{", start)
    if brace == -1 or brace > idx + 4000:
        # python: indentation-based — just take a window
        block = text[start: idx + 600]
    else:
        depth, i = 0, brace
        end = len(text)
        while i < len(text):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
            i += 1
        block = text[start:end]
    lines = block.splitlines()
    if len(lines) > max_lines:
        block = "\n".join(lines[:max_lines]) + "\n    // ... [truncated]"
    return block


def find_relevant_tests(
    target_dir: str,
    affected_functions: list[str],
    max_blocks: int = 8,
    char_budget: int = 12_000,
) -> str:
    """Extract test blocks from the project's test suite that exercise the flagged
    functions. Asserting tests (expectRevert/assert) are prioritized. Returns a
    formatted block or "" when nothing relevant is found.
    """
    if not target_dir:
        return ""
    fn_names = _bare_fn_names(affected_functions)
    if not fn_names:
        return ""

    test_files: list[Path] = []
    seen_paths: set[str] = set()
    for d in _TEST_DIRS:
        base = Path(target_dir) / d
        if not base.is_dir():
            continue
        for pat in _TEST_GLOBS:
            for f in base.rglob(pat):
                if str(f) not in seen_paths:
                    seen_paths.add(str(f))
                    test_files.append(f)
    if not test_files:
        return ""

    # (priority, file, block) — priority 0 = asserting test (best signal)
    found: list[tuple[int, str, str]] = []
    seen_blocks: set[str] = set()
    for tf in test_files:
        try:
            text = tf.read_text(errors="replace")
        except OSError:
            continue
        for fn in fn_names:
            # word-boundary so 'borrow' doesn't match 'borrowed'. Collect EVERY distinct
            # enclosing test block (not just the first) so the most relevant — e.g. an
            # expectRevert test later in the file — is not missed.
            for m in re.finditer(rf"\b{re.escape(fn)}\b", text):
                block = _enclosing_test_block(text, m.start())
                if not block:
                    continue
                key = (tf.name, block[:80])
                if key in seen_blocks:
                    continue
                seen_blocks.add(key)
                # expectRevert on the flagged fn is the strongest "intended" signal.
                if "expectRevert" in block:
                    prio = 0
                elif any(h in block for h in _ASSERT_HINTS):
                    prio = 1
                else:
                    prio = 2
                found.append((prio, tf.name, block))

    if not found:
        return ""
    found.sort(key=lambda t: t[0])  # asserting tests first

    out: list[str] = []
    used = 0
    for _prio, fname, block in found:
        if len(out) >= max_blocks or used >= char_budget:
            break
        snippet = f"// from {fname}\n{block}"
        out.append(snippet)
        used += len(snippet)

    return (
        "## PROJECT'S OWN TEST SUITE — tests that already exercise the flagged function(s)\n"
        "If a test below CALLS the flagged function and ASSERTS the behavior the finding claims is a "
        "bug (e.g. `expectRevert(...)` on it, or a passing happy-path), the behavior is INTENDED/known "
        "→ the finding is almost certainly a FALSE POSITIVE.\n"
        "```solidity\n" + "\n\n".join(out)[:char_budget] + "\n```\n"
    )
