"""Universal LLM interface for Audit Agent.

All agents call LLM through this interface. Backend is selected via config:
  LLM_PROVIDER=anthropic  (default)
  LLM_PROVIDER=openai
  LLM_PROVIDER=openai_compatible

Agents never import anthropic/openai directly — only this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolDefinition:
    """Provider-agnostic tool definition."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass
class ToolCall:
    """A single tool invocation from the LLM response."""
    id: str
    name: str
    arguments: dict[str, Any]  # Parsed JSON arguments


@dataclass
class LLMResponse:
    """Provider-agnostic LLM response."""
    text: str                          # Concatenated text content (may be empty if tool_use)
    tool_calls: list[ToolCall]         # Tool calls (may be empty if pure text)
    input_tokens: int
    output_tokens: int
    stop_reason: str                   # "end_turn" | "tool_use" | "max_tokens" | "stop"
    model: str                         # Actual model used
    raw_response: Any = None           # Original provider response for debugging
    duration_seconds: float = 0.0


class LLMProvider(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
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
        """Send a completion request.

        Args:
            model: Model identifier (provider-specific)
            system: System prompt
            messages: Conversation messages with role/content keys
            tools: Optional tool definitions
            tool_choice: "auto", "none", or {"name": "tool_name"} to force a tool
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Returns:
            LLMResponse with text and/or tool_calls
        """
        ...

    @abstractmethod
    async def complete_simple(
        self,
        *,
        model: str,
        system: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Simplified interface for text-only completions. Returns just the text."""
        ...

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD. Override per provider."""
        return 0.0

    def has_pricing(self, model: str) -> bool:
        """True when ``estimate_cost`` uses a real price for ``model`` (not a fallback)."""
        return False
