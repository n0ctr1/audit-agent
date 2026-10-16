// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @notice Intentionally vulnerable ETH bank — for audit-agent smoke testing.
/// @dev Contains three deliberate vulnerabilities:
///   1. Reentrancy in withdraw()    — CEI violated, external call before state update
///   2. Missing access control      — admin functions callable by anyone
///   3. Integer precision loss       — fee calculation truncates
contract VulnerableBank {
    mapping(address => uint256) public balances;
    address public owner;
    uint256 public feeRate = 10; // basis points — but division is integer truncated

    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);
    event OwnerChanged(address indexed newOwner);
    event FeeCollected(uint256 amount);

    constructor() {
        owner = msg.sender;
    }

    // ── VULNERABILITY 1: Reentrancy ────────────────────────────────────────
    // State update happens AFTER external call — classic CEI violation.
    // An attacker contract can re-enter withdraw() in its receive() fallback.
    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "Insufficient balance");

        // WRONG: external call before state update
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "Transfer failed");

        // State update AFTER the call — re-entrancy window here
        // unchecked: balance underflow is intentionally exploitable (0.8.x safety bypass)
        unchecked {
            balances[msg.sender] -= amount;
        }

        emit Withdrawn(msg.sender, amount);
    }

    function deposit() external payable {
        require(msg.value > 0, "Must deposit non-zero");
        balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    // ── VULNERABILITY 2: Missing access control ────────────────────────────
    // Anyone can call setOwner — no onlyOwner check.
    function setOwner(address newOwner) external {
        // Missing: require(msg.sender == owner, "Not owner");
        owner = newOwner;
        emit OwnerChanged(newOwner);
    }

    // ── VULNERABILITY 3: Integer precision loss ────────────────────────────
    // For small amounts, (amount * feeRate) / 10000 truncates to 0 fee.
    function collectFee(uint256 amount) external returns (uint256 fee) {
        // Integer division: 99 * 10 / 10000 = 0 (fee is lost)
        fee = (amount * feeRate) / 10000;
        require(balances[msg.sender] >= fee, "Insufficient balance for fee");
        balances[msg.sender] -= fee;
        balances[owner] += fee;
        emit FeeCollected(fee);
    }

    // ── Admin drain (uses broken ownership) ───────────────────────────────
    function emergencyDrain() external {
        require(msg.sender == owner, "Not owner");
        (bool ok, ) = owner.call{value: address(this).balance}("");
        require(ok, "Drain failed");
    }

    receive() external payable {
        balances[msg.sender] += msg.value;
    }
}
