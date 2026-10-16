# Intentionally minimal — importing workflow here causes a circular dependency:
# graph.__init__ → workflow → nodes → agents.scout → agents.base → graph.state → graph.__init__
# Callers should import build_graph directly from audit_agent.graph.workflow.
from audit_agent.graph.state import AuditState

__all__ = ["AuditState"]
