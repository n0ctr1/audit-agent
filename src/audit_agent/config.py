"""Configuration via environment variables (Pydantic Settings)."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Anthropic
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # LLM Provider
    llm_provider: str = Field(default="anthropic", alias="LLM_PROVIDER")
    llm_base_url: str | None = Field(default=None, alias="LLM_BASE_URL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")

    # Postgres
    postgres_url: str = Field(
        default="postgresql+asyncpg://audit:audit@localhost:5432/audit_agent",
        alias="POSTGRES_URL",
    )
    postgres_url_sync: str = Field(
        default="postgresql://audit:audit@localhost:5432/audit_agent",
        alias="POSTGRES_URL_SYNC",
    )

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="auditpassword", alias="NEO4J_PASSWORD")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")

    # MinIO / S3
    minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="audit-artifacts", alias="MINIO_BUCKET")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # Models
    scout_model: str = Field(default="claude-sonnet-4-6", alias="SCOUT_MODEL")
    mapper_model: str = Field(default="claude-sonnet-4-6", alias="MAPPER_MODEL")
    hunter_model: str = Field(default="claude-opus-4-7", alias="HUNTER_MODEL")
    exploit_model: str = Field(default="claude-sonnet-4-6", alias="EXPLOIT_MODEL")
    skeptic_model: str = Field(default="claude-opus-4-7", alias="SKEPTIC_MODEL")
    scribe_model: str = Field(default="claude-sonnet-4-6", alias="SCRIBE_MODEL")

    # Docker
    docker_socket: str = Field(default="unix:///var/run/docker.sock", alias="DOCKER_SOCKET")
    slither_image: str = Field(default="audit-agent/slither:latest", alias="SLITHER_IMAGE")
    foundry_image: str = Field(default="audit-agent/foundry:latest", alias="FOUNDRY_IMAGE")

    # Budget defaults
    default_budget_tokens: int = Field(default=500_000, alias="DEFAULT_BUDGET_TOKENS")
    default_budget_usd: float = Field(default=50.0, alias="DEFAULT_BUDGET_USD")

    # Hunter tuning — coverage & throughput.
    # Concurrency caps simultaneous in-flight LLM calls across all Hunter passes
    # (per-function + experts + temporal/crypto + standalone). Raise on a fast
    # provider, lower if you hit rate limits.
    hunter_max_concurrency: int = Field(default=8, alias="HUNTER_MAX_CONCURRENCY")
    # How many top-attack-surface functions get the full per-function deep dive.
    hunter_top_functions: int = Field(default=12, alias="HUNTER_TOP_FUNCTIONS")
    # Global cap on hypotheses returned to Exploit.
    hunter_max_hypotheses: int = Field(default=40, alias="HUNTER_MAX_HYPOTHESES")
    # Char budget for the full in-scope source bundle injected into prompts.
    # Default tuned for large-context models; every in-scope file gets a fair slice.
    hunter_source_budget_chars: int = Field(default=200_000, alias="HUNTER_SOURCE_BUDGET_CHARS")
    # Char budget for dependency/context source appended after in-scope sources.
    hunter_context_budget_chars: int = Field(default=40_000, alias="HUNTER_CONTEXT_BUDGET_CHARS")

    # Agentic codebase navigator (roadmap piece #3). When enabled, Hunter runs an
    # additional pass where the LLM investigates leads with read-only tools
    # (read_function / get_callers / get_state_writers / grep / ...), following
    # data and call edges across the whole codebase to surface multi-step bugs.
    hunter_navigator_enabled: bool = Field(default=True, alias="HUNTER_NAVIGATOR_ENABLED")
    # How many top-attack-surface functions to launch a navigator investigation on.
    hunter_navigator_leads: int = Field(default=4, alias="HUNTER_NAVIGATOR_LEADS")
    # Max tool/LLM steps per navigator investigation before forced report.
    hunter_navigator_max_steps: int = Field(default=6, alias="HUNTER_NAVIGATOR_MAX_STEPS")

    # Ablation switches for Hunter's additional passes (both on by default; the
    # `no_experts` benchmark preset turns them off).
    # Mixture-of-experts passes (_expert_passes).
    hunter_experts_enabled: bool = Field(default=True, alias="HUNTER_EXPERTS_ENABLED")
    # Standalone passes: temporal, crypto (second pass + file scan), cross-contract,
    # aggregation, boundary.
    hunter_standalone_passes_enabled: bool = Field(default=True, alias="HUNTER_STANDALONE_PASSES_ENABLED")

    # Exploit / Skeptic stages. With both off (`hunter_only` preset) every
    # hypothesis is reported as an `unverified` finding (graph/nodes.py).
    exploit_enabled: bool = Field(default=True, alias="EXPLOIT_ENABLED")
    skeptic_enabled: bool = Field(default=True, alias="SKEPTIC_ENABLED")

    # Baseline agent (benchmark preset `baseline`): one LLM with bash + file_edit
    # in the Foundry image, compared with the pipeline at an equal budget. Set the
    # limits to the full pipeline's measured per-contest usage.
    baseline_model: str = Field(default="", alias="BASELINE_MODEL")  # empty → HUNTER_MODEL
    baseline_max_tokens: int = Field(default=2_000_000, alias="BASELINE_MAX_TOKENS")
    baseline_max_seconds: int = Field(default=5400, alias="BASELINE_MAX_SECONDS")
    baseline_max_steps: int = Field(default=150, alias="BASELINE_MAX_STEPS")
    baseline_command_timeout: int = Field(default=120, alias="BASELINE_COMMAND_TIMEOUT")

    # Protocol-wide invariant testing stage (roadmap piece #4): runs a stateful
    # Medusa campaign + Halmos symbolic checks over ALL mined invariants and
    # turns violations into high-priority hypotheses. Medusa obeys --fuzz/--no-fuzz.
    invariant_prover_enabled: bool = Field(default=True, alias="INVARIANT_PROVER_ENABLED")

    # Refuter stage: adversarial false-positive filter after Skeptic. Re-examines
    # each validated/needs-review finding and downgrades those whose attack path is
    # unreachable or rests on a wrong external-contract assumption. Downgrade-only.
    refuter_enabled: bool = Field(default=True, alias="REFUTER_ENABLED")

    # Chainer stage (roadmap piece #5): composes individual hypotheses into
    # multi-step attack chains using structural (write→read / call) connections.
    chainer_enabled: bool = Field(default=True, alias="CHAINER_ENABLED")
    chainer_max_chains: int = Field(default=6, alias="CHAINER_MAX_CHAINS")

    # Threat-model stage (roadmap piece #6): builds a value-centric threat model
    # after Mapper; Hunter boosts its prioritized targets ("where the money is").
    threat_model_enabled: bool = Field(default=True, alias="THREAT_MODEL_ENABLED")
    # Attack-surface score bonus added to a function named in the threat model's
    # prioritized targets, steering deep analysis + navigator leads toward value.
    hunter_threat_boost: float = Field(default=15.0, alias="HUNTER_THREAT_BOOST")

    # Breadth triage (roadmap piece #7, model cascade): a cheap model rates EVERY
    # in-scope function 0-100 for suspicion; Hunter folds that into its selection
    # score so suspicious-but-low-surface functions still reach the expensive
    # depth pass. Closes full-codebase coverage within budget.
    triage_enabled: bool = Field(default=True, alias="TRIAGE_ENABLED")
    # Dedicated cheap model for triage; empty → falls back to SCOUT_MODEL.
    triage_model: str = Field(default="", alias="TRIAGE_MODEL")
    triage_batch_size: int = Field(default=25, alias="TRIAGE_BATCH_SIZE")
    # Max score contribution from triage suspicion (suspicion/100 * weight).
    triage_weight: float = Field(default=20.0, alias="TRIAGE_WEIGHT")

    # Extra static analyzers run alongside Slither in Scout (additive findings).
    # Each degrades independently: if its image/Docker is unavailable it is added
    # to degraded_tools and the audit continues (subject to tool_mode).
    aderyn_enabled: bool = Field(default=True, alias="ADERYN_ENABLED")
    semgrep_enabled: bool = Field(default=True, alias="SEMGREP_ENABLED")
    # Semgrep ruleset path *inside the image* (Decurity rules baked to /rules).
    semgrep_config: str = Field(default="/rules", alias="SEMGREP_CONFIG")

    # Tool execution mode
    # "strict"  (default) — raise on any infrastructure problem, audit fails loudly
    # "lenient" — catch tool errors, mark tool as degraded, audit continues
    # "mock"    — skip Docker entirely, return MockRunResult; use in tests without Docker
    tool_mode: str = Field(default="strict", alias="AUDIT_AGENT_TOOL_MODE")

    # LLM generation options
    # Set to true for Qwen3 and other thinking-mode models to suppress chain-of-thought
    # tokens that would otherwise exhaust the max_tokens budget before tool calls.
    llm_disable_thinking: bool = Field(default=False, alias="LLM_DISABLE_THINKING")
    # OpenRouter only: cap on reasoning tokens per call (0 = provider default).
    # Must leave room under the smallest max_tokens agents use (4096) for the answer.
    llm_reasoning_max_tokens: int = Field(default=0, alias="LLM_REASONING_MAX_TOKENS")
    # OpenAI-compatible providers: raise every call's max_tokens to at least this
    # (0 = use each agent's own limit). For thinking models whose reasoning tokens
    # count against max_tokens.
    llm_min_output_tokens: int = Field(default=0, alias="LLM_MIN_OUTPUT_TOKENS")

    # Report output
    reports_dir: Path = Field(default=Path("./audit-reports"), alias="AUDIT_REPORTS_DIR")

    # Benchmark: base directory for relative contest source_dir paths in datasets
    # (empty → repository root).
    benchmark_contests_dir: str = Field(default="", alias="BENCHMARK_CONTESTS_DIR")

    # Observability
    prometheus_port: int = Field(default=9090, alias="PROMETHEUS_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="audit-agent", alias="LANGSMITH_PROJECT")

    # Audit logging
    audit_log_enabled: bool = Field(default=True, alias="AUDIT_LOG_ENABLED")
    audit_log_llm_prompts: bool = Field(default=True, alias="AUDIT_LOG_LLM_PROMPTS")
    audit_log_llm_responses: bool = Field(default=True, alias="AUDIT_LOG_LLM_RESPONSES")
    audit_log_tool_stdout: bool = Field(default=True, alias="AUDIT_LOG_TOOL_STDOUT")
    audit_log_poc_code: bool = Field(default=True, alias="AUDIT_LOG_POC_CODE")
    audit_log_max_prompt_length: int = Field(default=50000, alias="AUDIT_LOG_MAX_PROMPT_LENGTH")
    audit_log_max_response_length: int = Field(default=50000, alias="AUDIT_LOG_MAX_RESPONSE_LENGTH")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Force re-read of environment variables. Useful in tests."""
    global _settings
    _settings = None
