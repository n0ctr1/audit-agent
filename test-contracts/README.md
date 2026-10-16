# test-contracts

Small Solidity projects used by unit tests and manual smoke runs. They are
intentionally vulnerable or minimal; do not deploy them.

| Path | Used for |
|---|---|
| `VulnerableBank.sol`, `Vault.sol`, `SimpleToken.sol`, `Counter.sol` | quick manual runs (`make smoke`) |
| `test-lending/` | Foundry lending pool used by `tests/unit/test_navigator.py` |
| `clean-lending/` | identical copy of the flattened pool, for manual runs |

`FlatPool.sol` is flattened (OpenZeppelin code inlined), so no dependencies are
needed to build it. forge-std is not vendored; install it if you add Foundry tests:

```bash
cd test-contracts/test-lending
forge install foundry-rs/forge-std
forge build
```
