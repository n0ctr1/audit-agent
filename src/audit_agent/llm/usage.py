"""Token / cost accounting for a pipeline run.

Agents do not update ``AuditState.tokens_used`` / ``cost_usd``, so the benchmark
wraps the global provider with ``UsageTrackingProvider`` for the duration of a
run. Token counts come from the provider responses (API ``usage`` fields).
Cost is derived from those tokens with the provider's price table; models
missing from the table are listed in ``unpriced_models`` and make the total
cost ``None`` rather than a silently low number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from audit_agent.llm.base import LLMProvider, LLMResponse, ToolDefinition


@dataclass
class ModelUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    priced: bool = True


@dataclass
class UsageSummary:
    by_model: dict[str, ModelUsage] = field(default_factory=dict)

    @property
    def calls(self) -> int:
        return sum(m.calls for m in self.by_model.values())

    @property
    def input_tokens(self) -> int:
        return sum(m.input_tokens for m in self.by_model.values())

    @property
    def output_tokens(self) -> int:
        return sum(m.output_tokens for m in self.by_model.values())

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def unpriced_models(self) -> list[str]:
        return sorted(k for k, m in self.by_model.items() if not m.priced and m.calls)

    @property
    def cost_usd(self) -> Optional[float]:
        if self.unpriced_models:
            return None
        return round(sum(m.cost_usd for m in self.by_model.values()), 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "cost_source": "provider_price_table",
            "unpriced_models": self.unpriced_models,
            "by_model": {
                k: {"calls": m.calls, "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens, "cost_usd": round(m.cost_usd, 6),
                    "priced": m.priced}
                for k, m in sorted(self.by_model.items())
            },
        }


class UsageTrackingProvider(LLMProvider):
    """Transparent wrapper that records usage of every ``complete`` call."""

    def __init__(self, inner: LLMProvider) -> None:
        self.inner = inner
        self.usage = UsageSummary()

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        resp = await self.inner.complete(
            model=model, system=system, messages=messages, tools=tools,
            tool_choice=tool_choice, temperature=temperature, max_tokens=max_tokens,
        )
        self._record(model, resp)
        return resp

    async def complete_simple(
        self,
        *,
        model: str,
        system: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        # Same request the concrete providers build in their own complete_simple.
        resp = await self.complete(
            model=model, system=system,
            messages=[{"role": "user", "content": user_message}],
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.text

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        return self.inner.estimate_cost(model, input_tokens, output_tokens)

    def has_pricing(self, model: str) -> bool:
        return self.inner.has_pricing(model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def _record(self, model: str, resp: LLMResponse) -> None:
        mu = self.usage.by_model.setdefault(model, ModelUsage())
        mu.calls += 1
        mu.input_tokens += int(resp.input_tokens or 0)
        mu.output_tokens += int(resp.output_tokens or 0)
        if not (resp.input_tokens or resp.output_tokens):
            return
        if not self.inner.has_pricing(model):
            mu.priced = False
            return
        mu.cost_usd += self.inner.estimate_cost(model, resp.input_tokens, resp.output_tokens)
