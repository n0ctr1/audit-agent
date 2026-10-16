**Scope:** ⚠️ below paid tier — Severity 'low' is often below the paid tier (many programs pay only High/Critical for smart contracts). Verify program tiers.

## Findings Requiring Manual Review

### Config yieldBps mutation between Vault read and distribution execution causes incorrect yield split

**Severity (estimated):** Low  
**Confidence:** 0%  
**Affected:** MonetrixVault.distributeYield, MonetrixConfig.userYieldBps, MonetrixConfig.insuranceYieldBps, MonetrixConfig.foundationYieldBps

**Why needs review:** The PoC did not succeed, but the vulnerability hypothesis is plausible based on code analysis. A human auditor must determine whether this is a real issue.

**Attack scenario:** 1. Vault operator initiates yield distribution flow. 2. Vault reads userYieldBps, insuranceYieldBps from MonetrixConfig to compute the three-way split (user/insurance/foundation). 3. Between the Config read and the actual USDC transfers to InsuranceFund/foundation and USDM mint to sUSDM, a governor transaction (24h timelock) executes setYieldBps() changing the ratios. 4. The Vault distributes using stale ratios that no longer reflect governance intent. If the old ratios allocated more to foundation/insurance and new ratios reduce those shares, the distribution uses the old (higher) amounts, potentially over-distributing to those recipients. The inverse is also possible: under-distribution to users. While the 24h timelock makes this difficult to exploit in practice, a queued governance action landing in the same block as a distribution could create a measurable discrepancy.

**Skeptic notes:** Not validated: SKEPTIC_ENABLED=false.
