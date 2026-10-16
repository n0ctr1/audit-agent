"""Factory for creating LLM providers based on configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from audit_agent.llm.base import LLMProvider


def create_provider(
    provider: str = "anthropic",
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> "LLMProvider":
    """Create an LLM provider instance.

    Args:
        provider: "anthropic", "openai", or "openai_compatible"
        api_key: API key (provider-specific)
        base_url: Base URL (for openai_compatible, e.g. http://localhost:8000/v1)
        **kwargs: Additional provider-specific arguments
    """
    if provider == "anthropic":
        from audit_agent.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=api_key, **kwargs)

    elif provider in ("openai", "openai_compatible"):
        from audit_agent.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=api_key, base_url=base_url, **kwargs)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Supported: anthropic, openai, openai_compatible"
        )


_global_provider: "LLMProvider | None" = None


def get_provider() -> "LLMProvider":
    """Get the global LLM provider singleton. Created on first call from config."""
    global _global_provider
    if _global_provider is None:
        from audit_agent.config import get_settings
        settings = get_settings()

        base_url = settings.llm_base_url or ""
        _is_openrouter = "openrouter.ai" in base_url

        # chat_template_kwargs is vLLM-specific — do not send to OpenRouter.
        extra_body: dict = {}
        if (
            settings.llm_disable_thinking
            and settings.llm_provider == "openai_compatible"
            and not _is_openrouter
        ):
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        # OpenRouter takes a unified `reasoning` object instead. Thinking models
        # (e.g. Qwen3.x Max) otherwise spend the whole max_tokens budget on
        # reasoning and never emit the tool call.
        if _is_openrouter:
            if settings.llm_disable_thinking:
                extra_body = {"reasoning": {"enabled": False}}
            elif settings.llm_reasoning_max_tokens > 0:
                extra_body = {"reasoning": {"max_tokens": settings.llm_reasoning_max_tokens}}

        # OpenRouter requires HTTP-Referer for proper routing and rate-limit attribution.
        default_headers: dict = {}
        if _is_openrouter:
            default_headers = {
                "HTTP-Referer": "https://github.com/audit-agent",
                "X-Title": "audit-agent",
            }

        # extra_body / omit_tool_choice are OpenAI-provider options; the Anthropic
        # SDK client rejects unknown constructor kwargs.
        provider_kwargs: dict = {}
        if settings.llm_provider in ("openai", "openai_compatible"):
            provider_kwargs = {
                "extra_body": extra_body or None,
                "default_headers": default_headers or None,
                "omit_tool_choice": _is_openrouter,
                "min_max_tokens": settings.llm_min_output_tokens,
            }

        _global_provider = create_provider(
            provider=settings.llm_provider,
            api_key=_resolve_api_key(settings),
            base_url=base_url or None,
            **provider_kwargs,
        )
    return _global_provider


def set_provider(provider: "LLMProvider | None") -> None:
    """Override the global provider (for testing)."""
    global _global_provider
    _global_provider = provider


def _resolve_api_key(settings) -> str | None:
    if settings.llm_provider == "anthropic":
        return settings.anthropic_api_key or None
    elif settings.llm_provider in ("openai", "openai_compatible"):
        # Prefer OPENROUTER_API_KEY when using OpenRouter, then OPENAI_API_KEY.
        return (
            getattr(settings, "openrouter_api_key", None)
            or getattr(settings, "openai_api_key", None)
            or "not-needed"
        )
    return None
