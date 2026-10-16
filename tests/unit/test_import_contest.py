"""Tests for benchmarks/tools/import_contest.py (synthetic report formats, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.dataset import BenchmarkDataset, select_contests
from benchmarks.tools.import_contest import (
    build_draft,
    detect_format,
    main,
    parse_functions,
    parse_report,
)

VAULT = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Vault {
    mapping(address => uint256) public balances;

    function deposit() external payable {
        balances[msg.sender] += msg.value; // "}" in a comment
    }

    function withdraw(uint256 amount) external {
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "}");
        balances[msg.sender] -= amount;
    }

    function _fee(uint256 a) internal pure returns (uint256) {
        return a / 100;
    }
}
"""

POOL = """pragma solidity ^0.8.20;
interface IPool { function borrow(uint256) external; }
contract Pool {
    function borrow(uint256 amount) external {
        amount;
    }
}
"""

C4_REPORT = """# Overview
Some intro text.

# High Risk Findings (1)

## [[H-01] Reentrancy in `withdraw()` drains the vault](https://github.com/code-423n4/2026-01-x-findings/issues/7)
*Submitted by alice*

https://github.com/code-423n4/2026-01-x/blob/abc123/src/Vault.sol#L11-L15

### Impact
The external call happens before the balance update. See also `deposit()`.

# Medium Risk Findings (1)

## [[M-01] Fee rounds down to zero](https://github.com/code-423n4/2026-01-x-findings/issues/9)

https://github.com/code-423n4/2026-01-x/blob/abc123/src/Vault.sol#L18

Precision loss for small amounts.

# Low Risk and Non-Critical Issues

## [L-01] Something minor
"""

SHERLOCK_REPORT = """# Issue H-1: Oracle price can be manipulated in borrow

Source: https://github.com/sherlock-audit/2026-02-y-judging/issues/12

## Found by
bob

## Summary
Spot price is used.

## Code Snippet
https://github.com/sherlock-audit/2026-02-y/blob/main/contracts/src/Pool.sol#L4-L6

# Issue M-1: Missing check in `Vault.withdraw()`

Source: https://github.com/sherlock-audit/2026-02-y-judging/issues/30

## Summary
No link here.
"""


@pytest.fixture
def src(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "Vault.sol").write_text(VAULT)
    (tmp_path / "src" / "Pool.sol").write_text(POOL)
    (tmp_path / "lib" / "dep").mkdir(parents=True)
    (tmp_path / "lib" / "dep" / "Vault.sol").write_text(VAULT)  # must be ignored
    (tmp_path / "scope.txt").write_text("src/Vault.sol\n")
    return tmp_path


def test_parse_functions_spans():
    spans = {s.name: s for s in parse_functions(VAULT)}
    assert (spans["deposit"].start, spans["deposit"].end) == (7, 9)
    assert (spans["withdraw"].start, spans["withdraw"].end) == (11, 15)
    assert spans["_fee"].contract == "Vault"
    # interface declarations without a body are skipped
    assert [s.contract for s in parse_functions(POOL)] == ["Pool"]


def test_detect_format():
    assert detect_format(C4_REPORT) == "c4"
    assert detect_format(SHERLOCK_REPORT) == "sherlock"
    with pytest.raises(ValueError):
        detect_format("# nothing here")


def test_parse_c4_report_keeps_only_high_medium():
    fs = parse_report(C4_REPORT)
    assert [(f.id, f.title) for f in fs] == [
        ("H-01", "Reentrancy in `withdraw()` drains the vault"),
        ("M-01", "Fee rounds down to zero"),
    ]
    assert fs[0].url.endswith("/issues/7")
    assert "Low Risk" not in fs[1].body


def test_parse_sherlock_report():
    fs = parse_report(SHERLOCK_REPORT)
    assert [f.id for f in fs] == ["H-1", "M-1"]
    assert fs[0].url.endswith("/issues/12")


def test_build_draft_c4(src: Path):
    d = build_draft(source_dir=src, report_text=C4_REPORT, contest_id="c4-x", name="X",
                    gt_source="https://code4rena.com/reports/2026-01-x", end_date="2026-01-31")
    c = d["contests"][0]
    assert c["needs_review"] is True
    assert c["scope_file"] == "scope.txt"
    h1, m1 = c["official_findings"]
    assert h1["severity"] == "high" and m1["severity"] == "medium"
    assert h1["affected_functions"] == ["Vault.withdraw"]
    assert h1["vulnerability_class"] == "reentrancy"
    assert any("deposit" in n for n in h1["review_notes"])  # body mention not added
    assert m1["affected_functions"] == ["Vault._fee"]
    assert m1["vulnerability_class"] == "rounding-precision"


def test_build_draft_sherlock(src: Path):
    d = build_draft(source_dir=src, report_text=SHERLOCK_REPORT, contest_id="s-y", name="Y",
                    gt_source="https://audits.sherlock.xyz/contests/1", end_date=None)
    h1, m1 = d["contests"][0]["official_findings"]
    assert h1["affected_functions"] == ["Pool.borrow"]
    assert h1["vulnerability_class"] == "oracle-manipulation"
    assert m1["affected_functions"] == ["Vault.withdraw"]  # from the title
    assert m1["vulnerability_class"] == "input-validation"


def test_unresolved_link_noted(src: Path):
    report = C4_REPORT.replace("src/Vault.sol#L11-L15", "src/Missing.sol#L1")
    d = build_draft(source_dir=src, report_text=report, contest_id="c", name="c",
                    gt_source="u", end_date=None)
    h1 = d["contests"][0]["official_findings"][0]
    assert h1["affected_functions"] == ["Vault.withdraw"]  # still found via title
    assert any("not resolved" in n for n in h1["review_notes"])


def test_draft_loads_but_is_skipped_by_default(src: Path, tmp_path: Path):
    out = tmp_path / "draft.json"
    rc = main(["--source-dir", str(src), "--report", str(_write(tmp_path, C4_REPORT)),
               "--id", "c4-x", "--name", "X", "--gt-source", "https://example.org",
               "--end-date", "2026-01-31", "--out", str(out)])
    assert rc == 0
    ds = BenchmarkDataset.model_validate(json.loads(out.read_text()))
    assert select_contests(ds) == []
    assert [c.id for c in select_contests(ds, include_needs_review=True)] == ["c4-x"]


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "report.md"
    p.write_text(text)
    return p


def test_merge_into_dataset(src: Path, tmp_path: Path):
    report = _write(tmp_path, C4_REPORT)
    ds_file = tmp_path / "contests.json"
    base = ["--source-dir", str(src), "--report", str(report), "--name", "X",
            "--gt-source", "https://example.org", "--merge-into", str(ds_file)]
    assert main(base + ["--id", "c4-a"]) == 0
    assert main(base + ["--id", "c4-b"]) == 0
    assert main(base + ["--id", "c4-a"]) == 0  # re-import replaces, no duplicate
    ds = BenchmarkDataset.model_validate(json.loads(ds_file.read_text()))
    assert ds.name == "contests"
    assert sorted(c.id for c in ds.contests) == ["c4-a", "c4-b"]
