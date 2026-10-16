"""Knowledge base package — search interface and ingestion pipeline."""

from audit_agent.knowledge.search import KnowledgeBase
from audit_agent.knowledge.models import Finding
from audit_agent.knowledge.vectors.collections import setup_collections

__all__ = ["KnowledgeBase", "Finding", "setup_collections"]
