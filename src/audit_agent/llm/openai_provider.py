"""OpenAI API provider (GPT-4o, o3, etc.) and OpenAI-compatible APIs."""

from __future__ import annotations

import json
import time
from typing import Any

from audit_agent.llm.base import LLMProvider, LLMResponse, ToolCall, ToolDefinition


OPENAI_PRICING = {
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3": (10.0, 40.0),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # OpenRouter — Qwen3.7 Max. Price per 1M tokens as listed by the OpenRouter
    # models API on 2026-10-15: input $1.475, output $4.425 (was $2.50 / $7.50).
    "qwen/qwen3.7-max": (1.475, 4.425),
    # Qwen3.8 Max snapshot 2026-09-03 — https://openrouter.ai/qwen/qwen3.8-max-0902
    # $2 / $6 per 1M (cache read $0.25 per 1M is not modeled: cached prompt
    # tokens are billed here at the input rate, so cost is an upper bound).
    "qwen/qwen3.8-max-0902": (2.00, 6.00),
}


class OpenAIProvider(LLMProvider):
    """OpenAI API and OpenAI-compatible endpoints (vLLM, Ollama, etc.)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        extra_body: dict | None = None,
        default_headers: dict | None = None,
        omit_tool_choice: bool = False,
        min_max_tokens: int = 0,
        **kwargs: Any,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: uv add openai"
            )

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        client_kwargs.update(kwargs)

        # 10-minute timeout prevents connection pool from hanging indefinitely
        # when the local LLM server stops responding mid-generation.
        import httpx
        client_kwargs.setdefault("timeout", httpx.Timeout(600.0, connect=10.0))
        self.client = AsyncOpenAI(**client_kwargs)
        self._extra_body: dict = extra_body or {}
        self._omit_tool_choice = omit_tool_choice
        # Floor for max_tokens on every call: thinking models spend part of the
        # budget on reasoning, so agents' fixed limits (4096/8192) can leave no
        # room for the answer or tool call.
        self._min_max_tokens = max(0, int(min_max_tokens or 0))

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
        oai_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for msg in messages:
            oai_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "max_tokens": max(max_tokens, self._min_max_tokens),
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]
            if tool_choice is not None and not self._omit_tool_choice:
                kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

        if self._extra_body:
            kwargs["extra_body"] = self._extra_body

        start = time.time()
        response = await self.client.chat.completions.create(**kwargs)
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
        return model in OPENAI_PRICING

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        prices = OPENAI_PRICING.get(model, (0.0, 0.0))
        return round(
            input_tokens * prices[0] / 1_000_000 + output_tokens * prices[1] / 1_000_000,
            6,
        )

    def _convert_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _convert_tool_choice(self, choice: str | dict) -> Any:
        if isinstance(choice, dict) and "name" in choice:
            return {"type": "function", "function": {"name": choice["name"]}}
        if choice in ("auto", "none"):
            return choice
        return "auto"

    def _parse_response(self, response: Any, model: str, duration: float) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message

        text = message.content or ""
        tool_calls: list[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id or "",
                    name=tc.function.name,
                    arguments=args,
                ))

        stop_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
            "content_filter": "end_turn",
        }
        stop_reason = stop_map.get(choice.finish_reason or "", choice.finish_reason or "end_turn")

        usage = response.usage
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            stop_reason=stop_reason,
            model=model,
            raw_response=response,
            duration_seconds=duration,
        )
