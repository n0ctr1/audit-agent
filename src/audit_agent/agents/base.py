"""BaseAgent with common logic."""

from __future__ import annotations

from abc import ABC, abstractmethod

from audit_agent.graph.state import AuditState
from audit_agent.observability.logging import get_logger


class BaseAgent(ABC):
    def __init__(self) -> None:
        self.log = get_logger(self.__class__.__name__)

    @abstractmethod
    async def run(self, state: AuditState) -> AuditState: ...
