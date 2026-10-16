"""Unit tests for the LLM provider abstraction layer.

All tests are offline — no API keys, no network calls.
Covers: ToolDefinition, ToolCall, LLMResponse, MockProvider, factory,
        AnthropicProvider conversion helpers, OpenAIProvider conversion helpers.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from audit_agent.llm.base import LLMProvider, LLMResponse, ToolCall, ToolDefinition
from audit_agent.llm.mock_provider import MockProvider


# ─────────────────────────────────────────────────────────────────────────────
# 1. ToolDefinition, ToolCall, LLMResponse data classes
# ─────────────────────────────────────────────────────────────────────────────

class TestDataClasses:
    def test_tool_definition_fields(self):
        td = ToolDefinition(
            name="my_tool",
            description="Does something",
            parameters={"type": "object", "properties": {}},
        )
        assert td.name == "my_tool"
        assert td.description == "Does something"
        assert td.parameters["type"] == "object"

    def test_tool_call_fields(self):
        tc = ToolCall(id="call-1", name="extract", arguments={"key": "val"})
        assert tc.id == "call-1"
        assert tc.name == "extract"
        assert tc.arguments == {"key": "val"}

    def test_llm_response_defaults(self):
        resp = LLMResponse(
            text="hello",
            tool_calls=[],
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
            model="test-model",
        )
        assert resp.text == "hello"
        assert resp.tool_calls == []
        assert resp.raw_response is None
        assert resp.duration_seconds == 0.0

    def test_llm_response_with_tool_calls(self):
        tc = ToolCall(id="x", name="fn", arguments={"a": 1})
        resp = LLMResponse(
            text="",
            tool_calls=[tc],
            input_tokens=100,
            output_tokens=200,
            stop_reason="tool_use",
            model="claude-sonnet-4-6",
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "fn"


# ─────────────────────────────────────────────────────────────────────────────
# 2. MockProvider
# ─────────────────────────────────────────────────────────────────────────────

class TestMockProvider:
    def _make_response(self, text: str = "hello") -> LLMResponse:
        return LLMResponse(
            text=text,
            tool_calls=[],
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_canned_response_returned(self):
        resp = self._make_response("response text")
        provider = MockProvider([resp])
        result = await provider.complete(
            model="test", system="sys", messages=[{"role": "user", "content": "hi"}]
        )
        assert result.text == "response text"

    @pytest.mark.asyncio
    async def test_multiple_responses_in_order(self):
        r1 = self._make_response("first")
        r2 = self._make_response("second")
        provider = MockProvider([r1, r2])

        out1 = await provider.complete(model="m", system="s", messages=[{"role": "user", "content": "a"}])
        out2 = await provider.complete(model="m", system="s", messages=[{"role": "user", "content": "b"}])
        assert out1.text == "first"
        assert out2.text == "second"

    @pytest.mark.asyncio
    async def test_empty_response_after_list_exhausted(self):
        provider = MockProvider([self._make_response("only")])
        await provider.complete(model="m", system="s", messages=[])
        # Second call: no more canned responses → empty default
        result = await provider.complete(model="m", system="s", messages=[])
        assert result.text == ""
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_side_effect_raises(self):
        provider = MockProvider(side_effect=ValueError("auth failed"))
        with pytest.raises(ValueError, match="auth failed"):
            await provider.complete(model="m", system="s", messages=[])

    @pytest.mark.asyncio
    async def test_callback_called_with_kwargs(self):
        received_kwargs: list[dict] = []

        async def cb(*, model, system, messages, **kwargs):
            received_kwargs.append({"model": model, "system": system, "messages": messages})
            return LLMResponse(text="cb-resp", tool_calls=[], input_tokens=0, output_tokens=0,
                               stop_reason="end_turn", model=model)

        provider = MockProvider(callback=cb)
        await provider.complete(model="mymodel", system="mysys", messages=[{"role": "user", "content": "x"}])

        assert len(received_kwargs) == 1
        assert received_kwargs[0]["model"] == "mymodel"
        assert received_kwargs[0]["system"] == "mysys"

    @pytest.mark.asyncio
    async def test_calls_recorded(self):
        provider = MockProvider([self._make_response()])
        await provider.complete(
            model="m", system="s", messages=[{"role": "user", "content": "q"}],
            max_tokens=1024,
        )
        assert len(provider.calls) == 1
        assert provider.calls[0]["model"] == "m"
        assert provider.calls[0]["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_complete_simple_returns_text(self):
        provider = MockProvider([self._make_response("simple text")])
        result = await provider.complete_simple(model="m", system="s", user_message="q")
        assert result == "simple text"

    @pytest.mark.asyncio
    async def test_tool_calls_in_response(self):
        tc = ToolCall(id="t1", name="my_tool", arguments={"x": 42})
        resp = LLMResponse(text="", tool_calls=[tc], input_tokens=0, output_tokens=0,
                           stop_reason="tool_use", model="test")
        provider = MockProvider([resp])
        result = await provider.complete(model="m", system="s", messages=[])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"x": 42}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Factory
# ─────────────────────────────────────────────────────────────────────────────

class TestFactory:
    def test_create_anthropic_provider(self):
        from audit_agent.llm.factory import create_provider
        from audit_agent.llm.anthropic_provider import AnthropicProvider
        provider = create_provider("anthropic", api_key="test-key")
        assert isinstance(provider, AnthropicProvider)

    def test_create_unknown_provider_raises(self):
        from audit_agent.llm.factory import create_provider
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_provider("unknown_backend")

    def test_set_and_get_provider(self):
        from audit_agent.llm.factory import get_provider, set_provider
        mock = MockProvider()
        set_provider(mock)
        assert get_provider() is mock
        set_provider(None)  # reset

    def test_get_provider_resets_after_set_none(self):
        from audit_agent.llm.factory import get_provider, set_provider
        mock = MockProvider()
        set_provider(mock)
        set_provider(None)
        # After reset, get_provider() would try to create from config.
        # We just verify set_provider(None) doesn't raise.


# ─────────────────────────────────────────────────────────────────────────────
# 4. AnthropicProvider conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestAnthropicProviderConversion:
    def _make_provider(self):
        from audit_agent.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key="test-key")

    def test_convert_tool_produces_input_schema(self):
        provider = self._make_provider()
        td = ToolDefinition(
            name="find_bug",
            description="Locates bugs",
            parameters={"type": "object", "properties": {"code": {"type": "string"}}},
        )
        converted = provider._convert_tool(td)
        assert converted["name"] == "find_bug"
        assert converted["description"] == "Locates bugs"
        assert "input_schema" in converted
        assert "parameters" not in converted

    def test_convert_tool_choice_by_name(self):
        provider = self._make_provider()
        result = provider._convert_tool_choice({"name": "extract_finding"})
        assert result == {"type": "tool", "name": "extract_finding"}

    def test_convert_tool_choice_auto(self):
        provider = self._make_provider()
        result = provider._convert_tool_choice("auto")
        assert result == {"type": "auto"}

    def test_convert_tool_choice_none(self):
        provider = self._make_provider()
        result = provider._convert_tool_choice("none")
        assert result == {"type": "none"}

    def test_parse_response_text_block(self):
        provider = self._make_provider()
        block = MagicMock()
        block.type = "text"
        block.text = "hello world"
        response = MagicMock()
        response.content = [block]
        response.usage.input_tokens = 10
        response.usage.output_tokens = 5
        response.stop_reason = "end_turn"
        result = provider._parse_response(response, "claude-test", 0.5)
        assert result.text == "hello world"
        assert result.tool_calls == []
        assert result.input_tokens == 10
        assert result.stop_reason == "end_turn"

    def test_parse_response_tool_use_block(self):
        provider = self._make_provider()
        block = MagicMock()
        block.type = "tool_use"
        block.id = "tool-123"
        block.name = "find_bug"
        block.input = {"code": "uint x = 0;"}
        response = MagicMock()
        response.content = [block]
        response.usage.input_tokens = 20
        response.usage.output_tokens = 30
        response.stop_reason = "tool_use"
        result = provider._parse_response(response, "claude-test", 1.0)
        assert result.text == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "find_bug"
        assert result.tool_calls[0].arguments == {"code": "uint x = 0;"}

    def test_estimate_cost_known_model(self):
        provider = self._make_provider()
        cost = provider.estimate_cost("claude-sonnet-4-6", 1_000_000, 1_000_000)
        # Input: 1M * $3/M = $3.0, Output: 1M * $15/M = $15.0
        assert abs(cost - 18.0) < 0.01

    def test_estimate_cost_unknown_model_uses_default(self):
        provider = self._make_provider()
        cost = provider.estimate_cost("unknown-model", 0, 0)
        assert cost == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. OpenAIProvider conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenAIProviderConversion:
    def _make_provider(self):
        try:
            from audit_agent.llm.openai_provider import OpenAIProvider
            return OpenAIProvider(api_key="test-key")
        except ImportError:
            pytest.skip("openai package not installed")

    def test_convert_tool_produces_function_wrapper(self):
        provider = self._make_provider()
        td = ToolDefinition(
            name="find_bug",
            description="Locates bugs",
            parameters={"type": "object", "properties": {"code": {"type": "string"}}},
        )
        converted = provider._convert_tool(td)
        assert converted["type"] == "function"
        assert converted["function"]["name"] == "find_bug"
        assert "parameters" in converted["function"]
        assert "input_schema" not in converted["function"]

    def test_convert_tool_choice_by_name(self):
        provider = self._make_provider()
        result = provider._convert_tool_choice({"name": "extract_finding"})
        assert result == {"type": "function", "function": {"name": "extract_finding"}}

    def test_convert_tool_choice_auto(self):
        provider = self._make_provider()
        result = provider._convert_tool_choice("auto")
        assert result == "auto"

    def test_parse_response_text(self):
        provider = self._make_provider()
        choice = MagicMock()
        choice.message.content = "OpenAI says hello"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        response = MagicMock()
        response.choices = [choice]
        response.usage.prompt_tokens = 15
        response.usage.completion_tokens = 8
        result = provider._parse_response(response, "gpt-4o", 0.3)
        assert result.text == "OpenAI says hello"
        assert result.tool_calls == []
        assert result.stop_reason == "end_turn"  # mapped from "stop"

    def test_parse_response_tool_call(self):
        import json
        provider = self._make_provider()
        tc_mock = MagicMock()
        tc_mock.id = "call-abc"
        tc_mock.function.name = "extract_finding"
        tc_mock.function.arguments = json.dumps({"title": "Bug", "severity": "high"})
        choice = MagicMock()
        choice.message.content = ""
        choice.message.tool_calls = [tc_mock]
        choice.finish_reason = "tool_calls"
        response = MagicMock()
        response.choices = [choice]
        response.usage.prompt_tokens = 50
        response.usage.completion_tokens = 100
        result = provider._parse_response(response, "gpt-4o", 0.5)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "extract_finding"
        assert result.tool_calls[0].arguments["title"] == "Bug"
        assert result.stop_reason == "tool_use"  # mapped from "tool_calls"

    def test_estimate_cost_known_model(self):
        provider = self._make_provider()
        cost = provider.estimate_cost("gpt-4o", 1_000_000, 1_000_000)
        # Input: 1M * $2.50/M = $2.50, Output: 1M * $10/M = $10.0
        assert abs(cost - 12.5) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 6. LLMProvider is abstract — cannot be instantiated directly
# ─────────────────────────────────────────────────────────────────────────────

def test_llm_provider_is_abstract():
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]


def test_openrouter_qwen38_snapshot_pricing():
    from audit_agent.llm.openai_provider import OpenAIProvider

    p = OpenAIProvider(api_key="x", base_url="http://localhost:1/v1")
    assert p.has_pricing("qwen/qwen3.8-max-0902")
    assert not p.has_pricing("qwen/qwen3.8-max")  # alias is not priced, only the snapshot id
    assert p.estimate_cost("qwen/qwen3.8-max-0902", 1_000_000, 1_000_000) == 8.0


class TestOpenRouterReasoningConfig:
    """LLM_DISABLE_THINKING / LLM_REASONING_MAX_TOKENS must reach OpenRouter (they were ignored)."""

    def _provider(self, monkeypatch, **env):
        from audit_agent.config import reset_settings
        from audit_agent.llm.factory import get_provider, set_provider

        monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
        monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        reset_settings()
        set_provider(None)
        try:
            return get_provider()
        finally:
            set_provider(None)

    def test_reasoning_budget(self, monkeypatch):
        p = self._provider(monkeypatch, LLM_REASONING_MAX_TOKENS="2048")
        assert p._extra_body == {"reasoning": {"max_tokens": 2048}}
        assert p._omit_tool_choice is True

    def test_disable_thinking(self, monkeypatch):
        p = self._provider(monkeypatch, LLM_DISABLE_THINKING="true", LLM_REASONING_MAX_TOKENS="2048")
        assert p._extra_body == {"reasoning": {"enabled": False}}

    def test_default_sends_nothing(self, monkeypatch):
        p = self._provider(monkeypatch)
        assert p._extra_body == {}

    def test_vllm_path_unchanged(self, monkeypatch):
        from audit_agent.config import reset_settings
        from audit_agent.llm.factory import get_provider, set_provider

        monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8001/v1")
        monkeypatch.setenv("LLM_DISABLE_THINKING", "true")
        reset_settings()
        set_provider(None)
        try:
            assert get_provider()._extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
        finally:
            set_provider(None)


class TestMinOutputTokens:
    async def test_floor_applied_to_requests(self):
        from unittest.mock import AsyncMock, MagicMock

        from audit_agent.llm.openai_provider import OpenAIProvider

        p = OpenAIProvider(api_key="x", base_url="http://localhost:1/v1", min_max_tokens=32768)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None), finish_reason="stop")]
        resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        resp.model = "m"
        p.client.chat.completions.create = AsyncMock(return_value=resp)
        await p.complete(model="m", system="s", messages=[{"role": "user", "content": "u"}], max_tokens=4096)
        assert p.client.chat.completions.create.call_args.kwargs["max_tokens"] == 32768
        await p.complete(model="m", system="s", messages=[{"role": "user", "content": "u"}], max_tokens=65536)
        assert p.client.chat.completions.create.call_args.kwargs["max_tokens"] == 65536

    def test_factory_passes_setting(self, monkeypatch):
        from audit_agent.config import reset_settings
        from audit_agent.llm.factory import get_provider, set_provider

        monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
        monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        monkeypatch.setenv("LLM_MIN_OUTPUT_TOKENS", "32768")
        reset_settings()
        set_provider(None)
        try:
            assert get_provider()._min_max_tokens == 32768
        finally:
            set_provider(None)


def test_openrouter_qwen37_current_pricing():
    from audit_agent.llm.openai_provider import OpenAIProvider

    p = OpenAIProvider(api_key="x", base_url="http://localhost:1/v1")
    assert p.estimate_cost("qwen/qwen3.7-max", 1_000_000, 1_000_000) == 5.9
