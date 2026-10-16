"""Pre-computed fork fingerprint stubs for well-known DeFi protocols.

Each entry is a dict mapping relative file path → normalized_hash.
Hashes are SHA-256 of comment-stripped, whitespace-normalised Solidity source.

STATUS: stubs — populate by running `scripts/compute_fork_fingerprints.py`
against the canonical source repos once network access is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProtocolFingerprint:
    name: str
    version: str
    github_repo: str
    commit: str
    # file_path -> normalized_hash (empty until populated)
    file_hashes: dict[str, str] = field(default_factory=dict)
    key_contracts: list[str] = field(default_factory=list)
    description: str = ""


# ---------------------------------------------------------------------------
# Uniswap V2
# ---------------------------------------------------------------------------
UNISWAP_V2 = ProtocolFingerprint(
    name="uniswap-v2",
    version="1.0.1",
    github_repo="Uniswap/v2-core",
    commit="ee547b17853e71ed4e0101ccfd52e70d5acded58",
    key_contracts=[
        "UniswapV2Factory",
        "UniswapV2Pair",
        "UniswapV2ERC20",
    ],
    description="Uniswap V2 constant-product AMM",
    file_hashes={
        # STUB — replace with real hashes
        "contracts/UniswapV2Factory.sol": "",
        "contracts/UniswapV2Pair.sol": "",
        "contracts/UniswapV2ERC20.sol": "",
    },
)

# ---------------------------------------------------------------------------
# Uniswap V3
# ---------------------------------------------------------------------------
UNISWAP_V3 = ProtocolFingerprint(
    name="uniswap-v3",
    version="1.0.0",
    github_repo="Uniswap/v3-core",
    commit="6562c52e8f75f0c10f9deaf3ab01b05d7e4b1b9e",
    key_contracts=[
        "UniswapV3Factory",
        "UniswapV3Pool",
        "TickMath",
        "SwapMath",
        "FullMath",
    ],
    description="Uniswap V3 concentrated liquidity AMM",
    file_hashes={
        "contracts/UniswapV3Factory.sol": "",
        "contracts/UniswapV3Pool.sol": "",
        "contracts/libraries/TickMath.sol": "",
        "contracts/libraries/SwapMath.sol": "",
        "contracts/libraries/FullMath.sol": "",
    },
)

# ---------------------------------------------------------------------------
# Compound V2
# ---------------------------------------------------------------------------
COMPOUND_V2 = ProtocolFingerprint(
    name="compound-v2",
    version="2.9",
    github_repo="compound-finance/compound-protocol",
    commit="a3214f67b73310d547e00fc578e8355911c9d376",
    key_contracts=[
        "Comptroller",
        "CToken",
        "CEther",
        "CErc20",
        "InterestRateModel",
        "JumpRateModelV2",
    ],
    description="Compound V2 lending protocol",
    file_hashes={
        "contracts/Comptroller.sol": "",
        "contracts/CToken.sol": "",
        "contracts/CEther.sol": "",
        "contracts/CErc20.sol": "",
        "contracts/JumpRateModelV2.sol": "",
    },
)

# ---------------------------------------------------------------------------
# Aave V2
# ---------------------------------------------------------------------------
AAVE_V2 = ProtocolFingerprint(
    name="aave-v2",
    version="2.0.0",
    github_repo="aave/protocol-v2",
    commit="ce53c4a2c8badfd5ded6748967e06f100e7f5c70",
    key_contracts=[
        "LendingPool",
        "LendingPoolConfigurator",
        "AToken",
        "StableDebtToken",
        "VariableDebtToken",
        "DefaultReserveInterestRateStrategy",
    ],
    description="Aave V2 lending/borrowing protocol",
    file_hashes={
        "contracts/protocol/lendingpool/LendingPool.sol": "",
        "contracts/protocol/tokenization/AToken.sol": "",
        "contracts/protocol/tokenization/StableDebtToken.sol": "",
        "contracts/protocol/tokenization/VariableDebtToken.sol": "",
    },
)

# ---------------------------------------------------------------------------
# Aave V3
# ---------------------------------------------------------------------------
AAVE_V3 = ProtocolFingerprint(
    name="aave-v3",
    version="3.0.2",
    github_repo="aave/aave-v3-core",
    commit="6070e82d962d9b12835c88e68210d0e63f08d035",
    key_contracts=[
        "Pool",
        "PoolConfigurator",
        "AToken",
        "StableDebtToken",
        "VariableDebtToken",
        "DefaultReserveInterestRateStrategy",
    ],
    description="Aave V3 with isolation mode, efficiency mode, portals",
    file_hashes={
        "contracts/protocol/pool/Pool.sol": "",
        "contracts/protocol/pool/PoolConfigurator.sol": "",
        "contracts/protocol/tokenization/AToken.sol": "",
    },
)

# ---------------------------------------------------------------------------
# OpenZeppelin Contracts (most common version)
# ---------------------------------------------------------------------------
OPENZEPPELIN_V4 = ProtocolFingerprint(
    name="openzeppelin-v4",
    version="4.9.6",
    github_repo="OpenZeppelin/openzeppelin-contracts",
    commit="fd81a96f01cc42ef1c9a5399364968d0e07e9e90",
    key_contracts=[
        "ERC20",
        "ERC721",
        "ERC1155",
        "ERC4626",
        "Ownable",
        "AccessControl",
        "ReentrancyGuard",
        "Pausable",
    ],
    description="OpenZeppelin Contracts v4 (most widely used)",
    file_hashes={
        "contracts/token/ERC20/ERC20.sol": "",
        "contracts/token/ERC721/ERC721.sol": "",
        "contracts/access/Ownable.sol": "",
        "contracts/security/ReentrancyGuard.sol": "",
        "contracts/security/Pausable.sol": "",
    },
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ALL_PROTOCOLS: list[ProtocolFingerprint] = [
    UNISWAP_V2,
    UNISWAP_V3,
    COMPOUND_V2,
    AAVE_V2,
    AAVE_V3,
    OPENZEPPELIN_V4,
]

PROTOCOL_BY_NAME: dict[str, ProtocolFingerprint] = {p.name: p for p in ALL_PROTOCOLS}


def get_protocol(name: str) -> Optional[ProtocolFingerprint]:
    return PROTOCOL_BY_NAME.get(name)
