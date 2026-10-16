"""Mock LLM provider for testing — no API calls."""

from __future__ import annotations

from typing import Any, Callable

from audit_agent.llm.base import LLMProvider, LLMResponse, ToolDefinition


class MockProvider(LLMProvider):
    """Returns canned responses. For use in pytest fixtures.

    Usage:
        # Canned responses in order:
        provider = MockProvider([resp1, resp2])

        # Always raise an exception:
        provider = MockProvider(side_effect=Exception("auth error"))

        # Dynamic callback (async):
        async def cb(**kwargs): return LLMResponse(...)
        provider = MockProvider(callback=cb)
    """

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        side_effect: Exception | None = None,
        callback: Callable[..., Any] | None = None,
    ) -> None:
        self._responses = responses or []
        self._side_effect = side_effect
        self._callback = callback
        self._call_index = 0
        self.calls: list[dict[str, Any]] = []  # Records all calls for assertions

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
        self.calls.append({
            "model": model,
            "system": system,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

        if self._side_effect is not None:
            raise self._side_effect

        if self._callback is not None:
            return await self._callback(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp

        return LLMResponse(
            text="",
            tool_calls=[],
            input_tokens=0,
            output_tokens=0,
            stop_reason="end_turn",
            model=model,
        )

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
