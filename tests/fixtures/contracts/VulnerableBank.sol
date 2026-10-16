// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @notice Intentionally vulnerable ETH bank for testing purposes.
///
/// KNOWN VULNERABILITIES (for Hunter/Skeptic validation in Phase 3+):
///   1. Reentrancy in withdraw() — state update happens AFTER external call.
///   2. No access control on emergencyDrain() — any caller can drain the vault.
///   3. Integer precision loss in calculateShares() — division before
///      multiplication causes users to receive fewer shares than expected.
///   4. Integer precision loss in collectFee() — fee = amount * feeRate / 10000
///      truncates to 0 for small amounts (amount < 10000 / feeRate).
contract VulnerableBank {
    mapping(address => uint256) public balances;
    address public owner;
    uint256 public totalShares;
    uint256 public feeRate = 100;       // basis points (1%)
    uint256 public feesCollected;

    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);

    constructor() {
        owner = msg.sender;
    }

    function deposit() external payable {
        balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    /// @notice VULNERABLE: state updated after external call (reentrancy).
    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        // BUG: external call before state update
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "Transfer failed");
        balances[msg.sender] -= amount; // too late — reentrancy possible
        emit Withdrawn(msg.sender, amount);
    }

    /// @notice VULNERABLE: no access control — anyone can drain the contract.
    function emergencyDrain() external {
        // BUG: missing onlyOwner modifier
        payable(msg.sender).transfer(address(this).balance);
    }

    /// @notice VULNERABLE: precision loss (divide before multiply).
    function calculateShares(uint256 depositAmount, uint256 totalDeposits)
        public
        pure
        returns (uint256)
    {
        // BUG: (depositAmount / totalDeposits) truncates before * 1e18
        return (depositAmount / totalDeposits) * 1e18;
    }

    /// @notice VULNERABLE: integer precision loss — fee truncates to 0 for small amounts.
    /// @dev fee = amount * feeRate / 10000. If amount < 10000 / feeRate, fee == 0.
    function collectFee(uint256 amount) external returns (uint256) {
        // BUG: integer division truncates for small amounts
        uint256 fee = amount * feeRate / 10000;
        feesCollected += fee;
        return fee;
    }
}
