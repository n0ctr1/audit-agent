"""Universal LLM interface — agents import from here."""

from audit_agent.llm.base import LLMProvider, LLMResponse, ToolCall, ToolDefinition
from audit_agent.llm.factory import create_provider, get_provider, set_provider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ToolCall",
    "ToolDefinition",
    "create_provider",
    "get_provider",
    "set_provider",
]
