"""Anthropic API provider (Claude models)."""

from __future__ import annotations

import time
from typing import Any

import anthropic

from audit_agent.llm.base import LLMProvider, LLMResponse, ToolCall, ToolDefinition


ANTHROPIC_PRICING = {
    "claude-opus-4-7": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "claude-haiku-4-5": (0.80, 4.0),
}


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API via official SDK."""

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=api_key, **kwargs)

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
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]

        if tool_choice is not None:
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

        start = time.time()
        response = await self.client.messages.create(**kwargs)
        duration = time.time() - start

        return self._parse_response(response, model, duration)

    async def complete_simple(
        self,
        *,
        model: str,
        system: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        resp = await self.complete(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user_message}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.text

    def has_pricing(self, model: str) -> bool:
        return model in ANTHROPIC_PRICING

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        prices = ANTHROPIC_PRICING.get(model, (3.0, 15.0))
        return round(
            input_tokens * prices[0] / 1_000_000 + output_tokens * prices[1] / 1_000_000,
            6,
        )

    def _convert_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    def _convert_tool_choice(self, choice: str | dict) -> dict[str, Any]:
        if isinstance(choice, dict) and "name" in choice:
            return {"type": "tool", "name": choice["name"]}
        if choice == "auto":
            return {"type": "auto"}
        if choice == "none":
            return {"type": "none"}
        return {"type": "auto"}

    def _parse_response(self, response: Any, model: str, duration: float) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason or "end_turn",
            model=model,
            raw_response=response,
            duration_seconds=duration,
        )
