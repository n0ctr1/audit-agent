# Changelog

## Unreleased

### Added
- `benchmark retrace <results_dir>...`: re-runs the stored PoCs that Exploit marked
  `exploited` with `forge test -vvvv` (Docker, temporary copy of the contest source, same
  PoC guard; no LLM calls) and saves the traces to `run_<k>_retrace/`.
- Strict exploited check in `benchmark analysis` (analysis only; Exploit and the PoC guard
  are unchanged): a PoC counts as strict exploited only if it calls an in-scope contract,
  has no `assertTrue(true)` / `assert(true)`, no assertion with identical sides, and its
  trace shows a passing assertion after a successful in-scope call. The report shows
  exploited (raw) / exploited (strict) and the downgrade reasons. Passing assertions of
  forge-std ≥ 1.10 are not traced; (d) is then reported as undetermined.
- Manual near-miss list (`near_misses.csv`, with a selection criterion per GT finding)
  shown as a separate analysis section, never counted in recall.
- `unmatched_findings.csv` also lists validated findings that the manual labels put
  outside GT (`unmatched_by` column); `metrics.json` gets a human-label precision
  (`human_labels.precision_validated`).
- `run_forge_test(verbosity=...)` (default unchanged: `-vv`).

### Fixed
- `benchmark report`: the per-run limitation notes (`needs_review`, contaminated) were read
  from the `meta.json` snapshot taken at run time, so a run rescored with `--dataset` after
  the GT review was still reported as having unreviewed ground truth. They now come from the
  rescored `metrics.json`.
- **PoC base-harness selection** (`exploit/harness_context.py`): only a contract that
  inherits forge-std `Test` — directly or through its parents — can be chosen as the base
  test harness for generated PoCs. Previously any contract named `*Harness`, `*Base` or
  `*Setup` in a test file with a `setUp()` qualified, including plain helper contracts.
  Found through compile failures on a test contest (Code4rena 2026-04 Monetrix, full
  preset): the generator picked the helper `ToL1WeiHarness`, so PoCs had no `vm`/`assert*`
  and could not reach the project's test state; 9 of 14 PoCs that failed to compile
  stopped on this (7 inherited the helper, 2 used state of another test they did not
  inherit). Analysis script: per-PoC final compile error from the run's audit logs.
