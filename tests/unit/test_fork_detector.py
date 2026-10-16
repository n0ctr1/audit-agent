"""Unit tests for fork detector — fully offline."""

from __future__ import annotations

from audit_agent.knowledge.fork_detector.fingerprints import (
    normalize_source,
    compute_text_hash,
)
from audit_agent.knowledge.fork_detector.differ import (
    DifferentialAnalyzer,
    _classify_modification,
)
from audit_agent.knowledge.fork_detector.known_protocols import (
    ALL_PROTOCOLS,
    PROTOCOL_BY_NAME,
    get_protocol,
    UNISWAP_V2,
)


# ------------------------------------------------------------------ #
# Fingerprints
# ------------------------------------------------------------------ #

def test_normalize_strips_line_comments():
    source = "uint256 x; // this is a comment\nuint256 y;"
    result = normalize_source(source)
    assert "comment" not in result
    assert "x" in result


def test_normalize_strips_block_comments():
    source = "/* license header */\nuint256 x;"
    result = normalize_source(source)
    assert "license" not in result
    assert "x" in result


def test_normalize_collapses_whitespace():
    source = "uint256   x  =   1;"
    result = normalize_source(source)
    assert "  " not in result


def test_compute_text_hash_deterministic():
    h1 = compute_text_hash("uint256 x = 1;")
    h2 = compute_text_hash("uint256 x = 1;")
    assert h1 == h2


def test_compute_text_hash_comment_invariant():
    src_with_comment = "// license\nuint256 x = 1;"
    src_without = "uint256 x = 1;"
    # After normalisation, both should produce the same hash
    assert compute_text_hash(src_with_comment) == compute_text_hash(src_without)


def test_compute_text_hash_differs_for_different_source():
    h1 = compute_text_hash("uint256 x = 1;")
    h2 = compute_text_hash("uint256 y = 2;")
    assert h1 != h2


# ------------------------------------------------------------------ #
# Known protocols registry
# ------------------------------------------------------------------ #

def test_all_protocols_have_required_fields():
    for p in ALL_PROTOCOLS:
        assert p.name, f"protocol has no name: {p}"
        assert p.version, f"{p.name} has no version"
        assert p.github_repo, f"{p.name} has no github_repo"
        assert p.key_contracts, f"{p.name} has no key_contracts"


def test_get_protocol_by_name():
    p = get_protocol("uniswap-v2")
    assert p is not None
    assert p is UNISWAP_V2


def test_get_protocol_missing_returns_none():
    assert get_protocol("nonexistent-protocol") is None


def test_protocol_by_name_covers_all():
    for p in ALL_PROTOCOLS:
        assert p.name in PROTOCOL_BY_NAME


# ------------------------------------------------------------------ #
# Differential analyzer
# ------------------------------------------------------------------ #

def test_no_match_returns_not_fork():
    analyzer = DifferentialAnalyzer()
    hashes = {
        "contracts/MyContract.sol": "deadbeef" * 8,
        "contracts/Utils.sol": "cafebabe" * 8,
    }
    result = analyzer.analyze(hashes)
    assert not result.is_fork
    assert result.base_protocol is None
    assert result.similarity_score == 0.0


def test_empty_project_is_not_fork():
    analyzer = DifferentialAnalyzer()
    result = analyzer.analyze({})
    assert not result.is_fork


def test_uniswap_v2_fork_detection():
    """Simulate a project that is 100% Uniswap V2 clone."""
    analyzer = DifferentialAnalyzer()
    # Inject exact hashes from UNISWAP_V2 fingerprint stubs — won't match (empty hashes)
    # but we can still test the structure by injecting real hashes
    real_hash = "a" * 64
    UNISWAP_V2.file_hashes["contracts/UniswapV2Pair.sol"] = real_hash
    UNISWAP_V2.file_hashes["contracts/UniswapV2Factory.sol"] = real_hash
    UNISWAP_V2.file_hashes["contracts/UniswapV2ERC20.sol"] = real_hash

    project = {
        "contracts/UniswapV2Pair.sol": real_hash,
        "contracts/UniswapV2Factory.sol": real_hash,
        "contracts/UniswapV2ERC20.sol": real_hash,
    }

    try:
        result = analyzer.analyze(project)
        assert result.is_fork
        assert result.base_protocol == "uniswap-v2"
        assert result.similarity_score == 1.0
    finally:
        # Restore stub state
        UNISWAP_V2.file_hashes["contracts/UniswapV2Pair.sol"] = ""
        UNISWAP_V2.file_hashes["contracts/UniswapV2Factory.sol"] = ""
        UNISWAP_V2.file_hashes["contracts/UniswapV2ERC20.sol"] = ""


def test_classify_modification():
    assert _classify_modification(0.98) == "minor_change"
    assert _classify_modification(0.88) == "modified_access"
    assert _classify_modification(0.70) == "changed_formula"
    assert _classify_modification(0.50) == "major_rewrite"


def test_diff_result_risk_multiplier_no_fork():
    analyzer = DifferentialAnalyzer()
    result = analyzer.analyze({})
    assert result.risk_multiplier == 1.0
