# Pilot results

Stored runs behind the numbers in the main README (*Evaluation → Results*).
`RESULTS.md` is written by `audit-agent benchmark report`, `ANALYSIS.md` by
`audit-agent benchmark analysis`.

Per run directory:

* `meta.json`, `metrics.json` — run configuration and metrics (`rescore` output).
* `<contest>/run_0.json` — raw final pipeline state (input to `rescore`).
* `labels_<contest>_run0.csv`, `unmatched_findings.csv` — manual labels.
* `<contest>/run_0_reports/` — generated finding reports.
* `<contest>/run_0_retrace/` — `forge test -vvvv` re-runs of `exploited` PoCs.
* `<contest>/run_0_logs/.../tools/*forge*` — every PoC attempt with its code and
  compiler / test output.

Not included, to keep the repository small and avoid redistributing contest
code: the remaining raw audit logs (LLM prompts and responses, agent traces,
static-analysis output). Local contest paths are replaced by
`<BENCHMARK_CONTESTS_DIR>`. Recomputing the metrics needs the contest sources
(see `meta.json` for the dataset snapshot).
