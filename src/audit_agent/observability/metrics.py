"""Prometheus metrics."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

_registry = CollectorRegistry()

# Counters
audit_runs_total = Counter(
    "audit_runs_total",
    "Total audit runs started",
    ["mode"],
    registry=_registry,
)

findings_total = Counter(
    "findings_total",
    "Total findings produced",
    ["severity", "status"],
    registry=_registry,
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total LLM tokens consumed",
    ["model", "direction"],
    registry=_registry,
)

tool_runs_total = Counter(
    "tool_runs_total",
    "Total tool invocations",
    ["tool", "status"],
    registry=_registry,
)

# Gauges
active_audits = Gauge(
    "active_audits",
    "Number of currently running audits",
    registry=_registry,
)

# Histograms
audit_duration_seconds = Histogram(
    "audit_duration_seconds",
    "Time to complete an audit",
    ["mode"],
    registry=_registry,
)

tool_duration_seconds = Histogram(
    "tool_duration_seconds",
    "Tool execution duration",
    ["tool"],
    registry=_registry,
)


class Metrics:
    audit_runs_total = audit_runs_total
    findings_total = findings_total
    llm_tokens_total = llm_tokens_total
    tool_runs_total = tool_runs_total
    active_audits = active_audits
    audit_duration_seconds = audit_duration_seconds
    tool_duration_seconds = tool_duration_seconds

    def start_server(self, port: int = 9090) -> None:
        start_http_server(port, registry=_registry)


_metrics = Metrics()


def get_metrics() -> Metrics:
    return _metrics
