# audit-agent — Architecture

> A multi-agent LLM pipeline for security review of Solidity smart contracts.
> This document describes the technology stack, the agent architecture, the
> analysis pipeline step by step and how the components interact, with
> references to modules in the code base. It is maintained by hand; update it
> together with the code.

**Python package:** `src/audit_agent/`
**Entry point:** `audit_agent.cli:app` (Typer CLI, see `pyproject.toml`)
**Benchmark harness:** `benchmarks/` (see the README, section *Evaluation*)

---

## 1. Technology stack

### 1.1 Core / orchestration

| Technology | Role | Where |
|---|---|---|
| **Python ≥3.12** | implementation language | whole package |
| **LangGraph** | **Main orchestration engine.** A `StateGraph` state machine drives the pipeline of ten agents; checkpointing and resume | `graph/workflow.py`, `graph/state.py` |
| **langgraph-checkpoint-postgres** | persists graph state in Postgres → `resume` an interrupted audit | `graph/checkpointer.py`, `cli.py:_create_checkpointer` |
| **Typer + Rich** | CLI (`audit`, `ingest`, `resume`, `benchmark`, `feedback`, `db-init`, `neo4j-init`, `qdrant-init`) | `cli.py`, `benchmarks/cli.py` |
| **Pydantic / pydantic-settings** | all domain models and env-based configuration | `models/schemas.py`, `config.py` |

> The project uses **LangGraph, not LangChain agents or LlamaIndex.**
> `langchain-anthropic` is listed in `pyproject.toml`, but agents call LLMs
> through the project's own provider layer (`src/audit_agent/llm/`).

### 1.2 LLM layer (provider-agnostic)

| Component | Role | File |
|---|---|---|
| **`LLMProvider` interface** | `complete()` (with tool use) and `complete_simple()` (text). All agents go through it and never import an SDK directly | `llm/base.py` |
| **Anthropic SDK** | `AnthropicProvider` (Claude models) | `llm/anthropic_provider.py` |
| **OpenAI SDK** | `OpenAIProvider` for `openai` and `openai_compatible` (vLLM, OpenRouter, local servers) | `llm/openai_provider.py` |
| **MockProvider** | deterministic provider for offline tests | `llm/mock_provider.py` |
| **UsageTrackingProvider** | wraps the provider during benchmark runs and records real token usage per model | `llm/usage.py` |

Default models (`config.py`) are Claude: Hunter and Skeptic use the strongest
model, the other roles a cheaper one. Provider choice and keys are resolved in
`llm/factory.py` (`get_provider()`, a singleton; OpenRouter headers and vLLM
`chat_template_kwargs.enable_thinking` are handled there).

### 1.3 Smart-contract analysis tools

| Tool | Role | Wrapper |
|---|---|---|
| **Slither** | static analysis → `StaticFinding`, plus raw JSON for the code graph | `tools/slither.py`, `docker/slither/Dockerfile` |
| **Aderyn** | additional static analysis (Rust/AST) → `StaticFinding`, run by Scout next to Slither | `tools/aderyn.py`, `docker/aderyn/Dockerfile` |
| **Semgrep** | pattern matching with a Solidity ruleset baked into the image (`/rules`) | `tools/semgrep.py`, `docker/semgrep/Dockerfile` |
| **Foundry (forge)** | compiles and runs PoC exploits (`forge test`) | `tools/foundry.py`, `docker/foundry/Dockerfile` |
| **Medusa** | coverage-guided property fuzzing for `fuzz` hypotheses | `tools/medusa.py`, `docker/medusa/Dockerfile` |
| **Halmos** | symbolic execution for `symbolic` hypotheses and the invariant campaign | `tools/halmos.py`, `docker/halmos/Dockerfile` |
| **Docker SDK** | sandbox: every external tool runs in an isolated container with CPU/RAM limits | `sandbox/docker_runner.py` |

> **Static analysis:** Slither + Aderyn + Semgrep all write to
> `state.static_findings`; Aderyn/Semgrep are switched by `ADERYN_ENABLED` /
> `SEMGREP_ENABLED` and degrade independently. Mythril/Manticore/Hardhat are not
> integrated. **Fuzzing:** Medusa (it replaced Echidna and consumes the same
> `echidna_*` / `property_*` property contracts through crytic-compile).
> **Symbolic:** `TestType="symbolic"` runs Halmos (`exploit/halmos_gen.py` →
> `tools/halmos.py`), falling back to Foundry when no spec can be derived.

**Execution safety.** Generated Solidity (PoCs, property contracts, symbolic
tests) is scanned by `tools/poc_guard.py` before it runs. Code that uses
`vm.rpc`, `vm.ffi`, `vm.broadcast` / `vm.startBroadcast`, `vm.rememberKey`,
`eth_send(Raw)Transaction` or reads a private key / mnemonic from the
environment is rejected, and Foundry always runs with `FOUNDRY_FFI=false`.
Nothing in the code base broadcasts transactions.

### 1.4 Storage and knowledge infrastructure

| Technology | Role | Module |
|---|---|---|
| **PostgreSQL** (SQLAlchemy async + asyncpg + Alembic) | persisted runs (`AuditRun`) and findings (`Finding`); LangGraph checkpoints | `storage/postgres.py`, `storage/migrations/` |
| **Neo4j** | code graph (contracts / functions / variables / edges) for production use | `knowledge/graph/neo4j_client.py`, `graph/schema.py`, `graph/queries.py` |
| **Qdrant** | vector search over historical vulnerabilities (RAG) — **collections are empty; currently a stub** | `knowledge/search.py`, `knowledge/vectors/` |
| **MinIO / S3** | artifact storage (PoCs, traces) | `config.py` |
| **Redis** | configured, reserved for queues/cache | `config.py` |

### 1.5 Observability

| Technology | Role | Module |
|---|---|---|
| **structlog** | structured logging | `observability/logging.py` |
| **prometheus-client** | metrics (LLM tokens, tool runs, findings by severity/status) | `observability/metrics.py` |
| **AuditLogger** | full audit trail of a run: every LLM call (prompt + response), tool run and agent decision → JSON in `{target}/audit_logs/{audit_id}/` | `observability/audit_logger.py` |
| **LangSmith** | optional tracing | `config.py` |

---

## 2. Agent architecture

### 2.1 Type: orchestrated multi-agent state graph

This is **not** a single ReAct loop. It is a **deterministic pipeline of ten
specialised agents** connected as a directed LangGraph state graph
(`graph/workflow.py:build_graph`):

```
                          ┌─ review mode ──┐
scout ──(after_scout)──┐  │                ▼
                       ├─ review ─┴──> static_analysis ──> scribe ──> END
                       │
                       └─ audit ──> mapper ──> threat_modeler ──> hunter ──> invariant_prover ──> chainer ──┐
                                                                                                             ▼
   ┌──────────────── exploit ──> skeptic ──> refuter ──(after_skeptic)──┐
   │                                                                    ├─ any TP / needs review ─> scribe ─> END
   └─────────────────────────────────────────────────────────────────── └─ all FP ─────────────────────────> END
```

Every agent derives from `BaseAgent` (`agents/base.py`) with a single async
`run(state) -> state`. Graph nodes (`graph/nodes.py`) are thin wrappers that
create the agent and record metrics. The `refuter` node also snapshots each
finding's Skeptic verdict into `ValidatedFinding.skeptic_status` before the
Refuter runs, so evaluation can attribute downgrades to the right stage.

| Agent | File | Responsibility |
|---|---|---|
| **Scout** | `agents/scout.py` | scope resolution, Slither/Aderyn/Semgrep, fork detection, documentation, protocol summary |
| **Mapper** | `agents/mapper.py` | Slither JSON → code graph, attack-surface scoring, invariant mining |
| **ThreatModeler** | `agents/threat_modeler.py` | value-centric threat model: actors / assets / trust boundaries and ranked `prioritized_targets`; Hunter boosts them |
| **Hunter** | `agents/hunter.py` | vulnerability hypothesis generation (the core of the system) |
| **InvariantProver** | `agents/invariant_prover.py` | protocol-wide Medusa + Halmos campaign over mined invariants; violations → high-priority hypotheses |
| **Chainer** | `agents/chainer.py` | composes hypotheses into multi-step `AttackChain`s over real write→read / call edges |
| **Exploit** | `agents/exploit.py` | turns hypotheses into executable Foundry / Medusa / Halmos PoCs |
| **Skeptic** | `agents/skeptic.py` | adversarial validation: true positive / false positive / needs review |
| **Refuter** | `agents/refuter.py` | devil's advocate: re-examines each finding for reachability/control and external-contract behaviour; **only downgrades** (validated → needs review → false positive) |
| **Scribe** | `agents/scribe.py` | platform-specific report generation |

Inside agents other patterns appear: **Hunter** uses a **mixture of experts**
(router + domain experts, §3 step 4), **Exploit** runs a
**generate → run → diagnose → fix** loop (ReAct-like over the compiler and test
runner). ThreatModeler / InvariantProver / Chainer are controlled by
`THREAT_MODEL_ENABLED` / `INVARIANT_PROVER_ENABLED` / `CHAINER_ENABLED` and pass
the state through unchanged when disabled or failing.

### 2.2 State (shared memory)

The pipeline's short-term memory is a single `TypedDict`, `AuditState`
(`graph/state.py`), accumulated between nodes:

- **Configuration:** `audit_id`, `mode`, `target_dir`, `docs_dir`, `platform`, budgets.
- **Scope:** `scope_in_scope`, `scope_context_files`, `scope_source`, `scope_file`.
- **Accumulated context:** `protocol_summary`, `fork_analysis`, `scope_map`,
  `invariants`, `static_findings`, `threat_model`, `hypotheses`, `chains`,
  `exploits`, `validated_findings`.
- **Transient data:** `slither_analysis_raw` (raw Slither JSON, Mapper → Hunter),
  `documentation`.
- **Output:** `reports` (`finding_id → Markdown`).
- **Accounting:** `tokens_used`, `cost_usd`, `errors`, `degraded_tools`.
  (Agents do not update `tokens_used` / `cost_usd`; benchmark runs measure usage
  with `llm/usage.py` instead.)

All domain objects are Pydantic models (`models/schemas.py`): `Hypothesis`,
`ExploitResult`, `ValidatedFinding` (linked to its hypothesis by
`hypothesis_id`), `Invariant`, `StaticFinding`, `ProtocolSummary`,
`ForkAnalysis`, `ScopeMap`, `ThreatModel`, `AttackChain`.

### 2.3 Memory: short-term / long-term

- **Short-term:** `AuditState` within one run plus LangGraph checkpoints in
  Postgres (`graph/checkpointer.py`) for resuming from the point of failure
  (`cli.py:resume`).
- **Long-term (between runs):**
  - code graph in Neo4j;
  - historical vulnerabilities in Qdrant (RAG; Hunter would add "similar
    historical cases" to its context) — **stub, collections are empty**;
  - learning loop (`learning/feedback.py`, `cli.py:feedback`) — **stub**;
  - persisted findings in Postgres.

### 2.4 How decisions are made

1. **Deterministic graph routing** (`graph/edges.py`): `after_scout` sends
   `review` mode to the light `static_analysis` branch; `after_skeptic` sends the
   run to Scribe if at least one finding is `validated` / `unverified`, else to `END`.
2. **Heuristic scoring** without an LLM (`agents/scoring.py:attack_surface_score`)
   plus content boosts in Hunter.
3. **LLM tool use** — structured decisions via forced tool calls
   (`report_hypotheses`, `validate_exploit`, `refute_finding`, `generate_report`),
   which guarantees machine-readable JSON.
4. **Adversarial verification** — Skeptic and Refuter re-check Hunter's claims
   against the source code.

### 2.5 System prompts (key ones)

- **Hunter** — `_HUNTER_SYSTEM`: two-phase method "PHASE 1: understand intent →
  PHASE 2: attack every assumption", with checklists for temporal, economic,
  cryptographic, cross-contract and aggregation bugs and an SWC classification
  table. `EXTERNAL_PROTOCOL_FACTS` (`knowledge/external_behaviors.py`), a
  reference of real external-protocol behaviour (flash-loan callbacks go to the
  caller, Aave scaled balances, ERC4626, Permit2, EVC/EVK, governance-gated
  inputs), is appended to avoid cross-protocol false positives.
- **MoE experts** — `_EXPERT_REGISTRY`: 8 narrow system prompts (`reentrancy`,
  `arithmetic`, `access_control`, `cryptography`, `oracle_price`, `economic_mev`,
  `time_epoch`, `cross_contract`).
- **Skeptic** — `_SKEPTIC_SYSTEM`, balanced towards **recall**: source code is
  the primary evidence; hard rejection only on concrete grounds (mock PoC,
  explicit guard, impossible preconditions, test misses the target); doubts about
  reachability / external behaviour → `needs_review`, leaving the final FP
  decision to the Refuter.
- **Refuter** — `_REFUTER_SYSTEM`: "assume this is a false positive" plus a
  checklist (reachability/control, external-contract behaviour, intentional
  same-unit design, guard elsewhere in the call path, **cross-check with the
  project's test suite**).
- **Scribe** — `_SCRIBE_SYSTEM`. **Scout** — `_SCOUT_SYSTEM`.

---

## 3. The pipeline step by step

Start: `audit-agent audit <target_dir> [--mode --docs --platform --scope --fuzz ...]`
(`cli.py:audit` → `_run_audit` → `build_graph().ainvoke(initial_state)`).

### Step 0. Initialisation (`cli.py:_run_audit`)

An `audit_id` (UUID) is generated, a checkpointer is created (Postgres or an
in-memory fallback), `AuditLogger` is initialised, `initial_state` is built and
an `AuditRun(status="running")` row is written to Postgres if available.

### Step 1. SCOUT — intake (`agents/scout.py`)

1. **Scope** (`scope.py:resolve_scope`). Source priority: explicit `--scope`
   file → `scope.txt` in the target → all non-library `.sol` files. Imports of
   in-scope files are resolved recursively into `context_files` (read for types,
   never audited), honouring `remappings.txt` and `foundry.toml`.
2. **Static analysis** — Slither + Aderyn + Semgrep in Docker. Slither
   detectors become `StaticFinding`s; the raw JSON is cached in
   `{target}/.slither_analysis.json` for Mapper's offline mode. **Degradation:**
   with `AUDIT_AGENT_TOOL_MODE=strict` a tool failure stops the audit; `lenient`
   adds the tool to `degraded_tools` and continues; `mock` skips Docker.
3. **Fork detection** (`knowledge/fork_detector/differ.py`) — compares file
   hashes with fingerprints of known protocols. **Stub:** the fingerprint
   database is empty, so this always reports "not a fork".
4. **Protocol summary** — one LLM call (up to 10 files, first 200 lines each) →
   `ProtocolSummary{name, category, key_mechanisms, user_roles}`; heuristic
   fallback on failure.
5. **Documentation** — all `*.md/*.txt/*.rst` outside `lib/` and `node_modules/`,
   50k-character budget → `state.documentation`.

> **Review mode:** `after_scout` routes to `static_analysis_node`
> (`graph/nodes.py`), which promotes `StaticFinding`s to `ValidatedFinding`s
> (status `unverified`, confidence 0.6) and goes straight to Scribe.

### Step 2. MAPPER — code graph and invariants (`agents/mapper.py`)

1. **Load analysis**: injected fixture → `.slither_analysis.json` → empty.
   Parsing (`knowledge/graph/builders.py:parse_slither_json`) supports the
   normalised format and native Slither detector output.
2. **Supplement from source**: Slither `--json` only reports functions hit by a
   detector, so plain public/external functions are added by regex from the
   `.sol` files.
3. **Build the graph** (`GraphBuilder.build`): Contract / Function / StateVar
   nodes and CALLS / READS / WRITES edges (`InMemoryGraphStore` by default,
   Neo4j optional). Every function gets an `attack_surface_score`
   (`agents/scoring.py`): visibility, value handling, external calls
   (`delegatecall`, user-controlled targets), oracle reads, state writes,
   privileges, arithmetic, missing `nonReentrant` around external calls.
4. **Invariant mining** (`agents/invariant_miner.py`) from NatSpec (regex),
   documentation (LLM) and code patterns (LLM).

### Step 3. THREAT_MODELER (`agents/threat_modeler.py`)

1. Deterministic fact extraction from the Slither model: value-moving,
   privileged and oracle-reading functions, trusted external contracts,
   value-bearing state variables.
2. LLM enrichment (forced tool `build_threat_model`) → `ThreatModel{actors,
   assets, trust_boundaries, external_dependencies, value_flows,
   prioritized_targets}`.
3. Deterministic fallback when the LLM is unavailable.

Hunter adds `HUNTER_THREAT_BOOST` (default +15) to the score of functions in
`prioritized_targets`. Disabled with `THREAT_MODEL_ENABLED=false`.

### Step 4. HUNTER — hypothesis generation (`agents/hunter.py`)

**A. Context.** `_read_scope_sources` reads all in-scope files with a
proportional budget (every file is represented) plus dependencies;
`_build_scope_info` separates "look for bugs here" from "context only".

**B. Function prioritisation.** In-scope functions are sorted by
`attack_surface_score` plus content boosts (financial / cryptographic keywords
in names and bodies) plus optional triage suspicion (`TRIAGE_ENABLED`, a cheap
model rates every function). The top N (`HUNTER_TOP_FUNCTIONS`) get a deep pass.

**C. Protocol invariants** (`_derive_invariants`): one LLM call over docs and
code → 5–8 business-level invariants passed to every later analysis.

**D. Per-function analysis.** For each top-N function a minimal code slice is
built (`knowledge/graph/context.py`: BFS over CALLS, cross-contract writers
for external calls), a rich prompt is assembled and the LLM is called with the
forced tool `report_hypotheses`, with a hard deadline and a shortened-prompt
retry for thinking models.

**E. Additional passes** (additive, each with its own hypothesis budget):
- **MoE experts** (`_expert_passes`, `HUNTER_EXPERTS_ENABLED`): a keyword router
  picks relevant experts out of 8; one LLM call each.
- **Standalone passes** (`HUNTER_STANDALONE_PASSES_ENABLED`): `_temporal_check`,
  `_crypto_check`, `_crypto_file_scan` (scans *all* in-scope files for
  signature/hash code regardless of scoring), `_cross_contract_check`,
  `_aggregation_check`, `_boundary_check`.
- **Navigator** (`HUNTER_NAVIGATOR_ENABLED`): the LLM follows leads with
  read-only tools (`read_function`, `get_callers`, `get_state_writers`, `grep`).

**F. Deduplication.** `_deduplicate` (overlap with static findings),
`_dedup_same_class` (cap per class and severity), `_dedup_by_title_similarity`,
`_consolidate_by_function`; sorted by `priority_score`, capped at
`HUNTER_MAX_HYPOTHESES`.

### Step 5. INVARIANT_PROVER (`agents/invariant_prover.py`)

Checks all mined invariants at once: a Medusa campaign over a generated property
contract (with `fuzz_enabled`) and Halmos `check_*` specs for arithmetic
invariants. Broken properties / counterexamples become hypotheses. No-op without
invariants or with `INVARIANT_PROVER_ENABLED=false`. **Known limitation:** the
flat Medusa workspace fails in crytic-compile on projects with heavy external
imports.

### Step 6. CHAINER (`agents/chainer.py`)

Mines structural connections (one hypothesis' functions write a state variable
another's functions read, or call them), asks the LLM (forced tool
`report_chains`) to compose chains of ≥2 hypotheses, and emits each chain as an
`AttackChain` and a composite `Hypothesis` (`vulnerability_class="Attack
Chain"`). Capped by `CHAINER_MAX_CHAINS`.

### Step 7. EXPLOIT (`agents/exploit.py`)

For each of the top-20 hypotheses by priority:
1. **Test type** (`determine_test_type`): `unit` / `fork` / `fuzz` / `symbolic`.
2. **Project metadata**: importable in-scope paths, remappings, working imports
   from existing tests, real constructor/function signatures.
3. **PoC generation** (`exploit/foundry_gen.py`): Jinja2 template by
   vulnerability class first, LLM fallback; import/case fix-ups.
4. **Run** (`tools/foundry.py:run_forge_test` in Docker): the PoC is written
   into the project's test directory (removed afterwards) and run with
   `forge test -vv`. Fork projects run under their fork profile
   (`_detect_fork_profile`); `exploit/harness_context.py` adds the project's
   base test harness, helpers and an example test to the prompt.
   `_looks_trivial_poc` downgrades passing tests without real assertions.
5. **Parsing** (`exploit/trace_parser.py`) → `exploited` / `blocked` / `partial`
   / `error` (compile errors are recognised separately).
6. **Repair loop:** compile errors → error-specific retry context; runtime
   failures → diagnostics (`console.log`) → targeted fix. Three identical errors
   in a row stop the loop.
7. **Fuzz branch** (Foundry and Medusa in parallel) and **symbolic branch**
   (Halmos, Foundry fallback).

### Step 8. SKEPTIC (`agents/skeptic.py`)

For each `ExploitResult`: anti-hallucination checks on the PoC (stub / mock
target), source selection with affected functions first, LLM validation (forced
tool `validate_exploit`: verdict, confidence, reasoning, suggested severity),
text and heuristic fallbacks, mapping to `FindingStatus`.

### Step 9. REFUTER (`agents/refuter.py`)

Re-examines every `validated` / `unverified` finding assuming it is a false
positive: (1) reachability/control of the malicious input, (2) assumptions about
external contracts, (3) intentional same-unit design, (4) guards elsewhere in
the call path, (5) the project's own tests (`knowledge/test_suite.py`): a passing
test that asserts the flagged behaviour (e.g. `expectRevert` on the flagged
function) means the behaviour is intended. Forced tool `refute_finding` →
`upheld` / `needs_review` / `false_positive`; it **only downgrades**. Disabled
with `REFUTER_ENABLED=false`; on failure the Skeptic verdict stands.

### Step 10. SCRIBE (`agents/scribe.py`)

For every reportable finding: report fields via the forced tool
`generate_report` (deterministic fallback), a deterministic scope self-flag
(`knowledge/scope_filter.py`: likely in scope / out of scope / known issue /
below paid tier / needs review — orthogonal to validity) and platform rendering
(`c4`, `sherlock`, `immunefi`, `cantina` / `spearbit` / `generic`).

**CLI post-processing** (`cli.py:_save_reports`): per-finding `*-finding.md`,
`SUMMARY.md` and debug dumps (`hypotheses.json`, `exploits.json`,
`validated_findings.json`, `invariants.json`, `chains.json`).

---

## 4. Component interaction

### 4.1 The LangGraph state as the bus

Components never call each other; they read and enrich the shared `AuditState`:

```
 SCOUT          writes: scope_*, static_findings, protocol_summary, fork_analysis, documentation
 MAPPER         reads scope_in_scope; writes: slither_analysis_raw, invariants, graph_id
 THREAT_MODELER reads slither_analysis_raw, protocol_summary, documentation; writes: threat_model
 HUNTER         reads slither_analysis_raw, invariants, threat_model, documentation, scope_*; writes: hypotheses
 INVARIANT_PROVER reads invariants, scope_*; writes: hypotheses (+ invariant violations)
 CHAINER        reads hypotheses, slither_analysis_raw; writes: chains, hypotheses (+ composites)
 EXPLOIT        reads hypotheses, scope_*; writes: exploits
 SKEPTIC        reads exploits, hypotheses, sources; writes: validated_findings
 REFUTER        reads validated_findings, sources, project tests; writes: validated_findings (downgrades)
 SCRIBE         reads validated_findings, hypotheses, exploits; writes: reports
```

### 4.2 Parser → static analyzer → graph

```
Slither (Docker, tools/slither.py)
   ├─ StaticFinding[] ─▶ Scout ─▶ state.static_findings
   └─ raw JSON ─▶ .slither_analysis.json ─▶ state.slither_analysis_raw
                                   │ parse_slither_json (+ source supplement)
                                   ▼
                     GraphBuilder.build ─▶ GraphStore (InMemory | Neo4j)
                                   │ attack_surface_score per function
                                   ▼
                     HunterAgent ─▶ SubgraphContextBuilder (BFS slice)
```

### 4.3 LLM abstraction

Agents obtain the provider through `get_provider()` or take it in the
constructor (tests), so Anthropic ↔ OpenAI ↔ openai_compatible ↔ Mock can be
switched without touching agent code. Structured output goes through
`ToolDefinition` (`llm/base.py`), mapped to each provider's native tool use.

### 4.4 Tool sandbox

`DockerRunner` (`sandbox/docker_runner.py`) is the single entry point for
running Slither / Aderyn / Semgrep / Foundry / Medusa / Halmos in containers
(memory/CPU limits, wall-clock timeout). `AUDIT_AGENT_TOOL_MODE`
(`strict` / `lenient` / `mock`) decides what happens when Docker is unavailable;
`mock` returns `MockRunResult`, which makes the whole pipeline runnable offline.

### 4.5 Checkpointing and resume

`build_graph(checkpointer=AsyncPostgresSaver)` saves the state after every node
under `thread_id=audit_id`; `audit-agent resume <audit_id>` continues from the
interrupted node. Without Postgres, `MemorySaver` is used and resume is
unavailable.

### 4.6 Observability

`AuditLogger` records agent start/end, every LLM call with full prompt and
response, every tool run (PoC code, stdout/stderr, exit code) and key decisions.
Prometheus metrics are written in the graph nodes.

### 4.7 Benchmark harness (`benchmarks/`)

`--preset baseline` swaps the whole pipeline for `agents/baseline.py`: one LLM
with `bash` (in a network-less Foundry container, `tools/shell_session.py`) and
`file_edit` on a copy of the project, under token / time / step budgets.

`dataset.py` (schema, splits, contest selection), `matching.py` + `taxonomy.py`
(prediction ↔ GT matching), `metrics.py` (GT-based metrics), `runner.py`
(sequential runs, raw-state persistence, `rescore`), `presets.py` (ablation
presets), `cli.py` (`audit-agent benchmark ...`), `tools/import_contest.py`
(draft datasets from local reports), `human_labels.py` (per-hypothesis manual GT
labels, human recall, matcher agreement). See the README for the methodology.

---

## 5. Notes and limitations

- **Many LLM calls per run.** The MoE experts and standalone Hunter passes are a
  deliberate trade of cost and time for class coverage; per-pass hypothesis
  budgets (`_BUDGET_*` in `hunter.py`) keep chatty passes from crowding out the
  others.
- **Thinking models / throughput.** Thinking models spend thousands of tokens
  on reasoning (Hunter calls use `max_tokens=8192`); the Hunter call deadline is
  900 s with a shortened retry. On providers with shared throughput, a high
  `HUNTER_MAX_CONCURRENCY` makes parallel streams share bandwidth and hit the
  deadline; use 1–2 for real runs.
- **False-positive reduction.** Skeptic is tuned for recall; the Refuter is the
  strict FP gate with three independent signals (reachability/control, external
  protocol facts, the project's test suite). It was added after a real-world run
  in which 4 of 7 `validated` findings were false positives; its effect has not
  been measured on the benchmark yet.
- **Scope self-flag** is orthogonal to validity: a true positive can be out of
  scope.
- **PoCs.** Fork-harness PoC generation produces compilable harness-inheriting
  PoCs, but end-to-end `exploited > 0` on the fork harness has **not** been
  confirmed in a live run. `_looks_trivial_poc` prevents empty PoCs from
  counting as exploits.
- **Optional infrastructure.** Neo4j/Qdrant degrade to `InMemoryGraphStore` and
  no RAG. Qdrant, fork detection and the learning loop are stubs.
- **Default models in `config.py` are Claude**; any OpenAI-compatible endpoint
  can be used per role.
