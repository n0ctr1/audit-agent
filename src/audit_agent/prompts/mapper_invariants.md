# Invariant Extraction Prompt (Appendix A.4)

You are a formal verification specialist. Extract testable invariants from the
following protocol documentation and contract code.

## Input
Documentation: {docs}
Contract code: {code}

## Task
Identify invariants in three categories:

### 1. GLOBAL — always true regardless of state
Example: "totalSupply == sum of all balances"
Example: "contract ETH balance >= sum of all user balances"

### 2. TRANSITION — true before/after specific function calls
Example: "after deposit(x), balance[user] increases by exactly x"
Example: "after withdraw(x), contract balance decreases by x"

### 3. ECONOMIC — constraints on value flows
Example: "user cannot withdraw more than they deposited * maxMultiplier"
Example: "protocol fee never exceeds 10% of transaction value"
Example: "fee collected is always > 0 when feeRate > 0 and amount > 0"
Example: "cumulative fees collected == sum of (amount * feeRate / denominator) for all calls — no silent truncation to zero"
Example: "slippage checks (minAmount) must account for protocol-level or cross-chain dust removal truncations"
Example: "blacklisted users MUST NOT be able to bypass restrictions by routing deposits through intermediary composer or adapter contracts"
Example: "if fees are deducted, threshold validations must be evaluated against the NET amount, not the GROSS amount"

## For each invariant provide:
- `natural_language`: clear English statement (1-2 sentences)
- `invariant_type`: "global" | "transition" | "economic"
- `affected_functions`: list of function names this invariant constrains
- `solidity_assert`: Solidity code snippet usable inside a Foundry test
  Example: `assertEq(token.totalSupply(), sumBalances(), "invariant broken");`
- `echidna_property`: Echidna property function
  Example: `function echidna_total_supply_eq_sum() public returns (bool) { ... }`
- `halmos_spec`: bounded verification spec (optional, only if mechanically checkable)
- `confidence`: 0.0–1.0 (0.9+ for explicit NatSpec, 0.5–0.7 for inferred)

## Rules
- Only extract invariants that are **checkable in code** — not aspirations or design goals.
- Do NOT extract invariants about features not present in the code.
- Mark inferred invariants with confidence ≤ 0.7.
- **Always** look for integer precision loss: any function with division (`/`) or a fee/rate/share
  multiplier should have an invariant that the result is non-zero when inputs are non-zero.
  Check: `result = amount * rate / denominator` — if `amount` is small, Solidity truncates to 0.
- Output strict JSON array.
- **Always** look for state desyncs in `try/catch` blocks. If an external callback fails, the internal state must safely revert or handle the error, not silently continue.
