"""LangGraph StateGraph definition."""

from __future__ import annotations

from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from audit_agent.graph.edges import after_scout, after_skeptic
from audit_agent.graph.nodes import (
    chainer_node,
    exploit_node,
    hunter_node,
    invariant_prover_node,
    mapper_node,
    refuter_node,
    scout_node,
    skeptic_node,
    scribe_node,
    static_analysis_node,
    threat_modeler_node,
)
from audit_agent.graph.state import AuditState


def build_graph(checkpointer: Optional[BaseCheckpointSaver] = None) -> Any:
    """Build the audit LangGraph.

    Pipeline: scout → mapper → threat_modeler → hunter → invariant_prover → chainer
              → exploit → skeptic → refuter → scribe → END   ↘ (all FP) → END

    refuter is an adversarial false-positive filter: it re-examines each validated
    finding and downgrades those whose attack path is unreachable or rests on a
    wrong external-contract assumption (downgrade-only).

    threat_modeler builds a value-centric threat model that Hunter uses to prioritize
    "where the money is" (roadmap #6).
    invariant_prover runs a protocol-wide Echidna + Halmos campaign over the mined
    invariants and appends violations as high-priority hypotheses (roadmap #4).
    chainer composes individual hypotheses into multi-step attack chains (roadmap #5).

    Args:
        checkpointer: Optional checkpoint saver. Pass AsyncPostgresSaver for
            production (enables resume), MemorySaver for tests, or None to
            disable checkpointing.
    """
    builder: StateGraph[AuditState] = StateGraph(AuditState)  # type: ignore[type-var]

    builder.add_node("scout", scout_node)
    builder.add_node("mapper", mapper_node)
    builder.add_node("threat_modeler", threat_modeler_node)
    builder.add_node("hunter", hunter_node)
    builder.add_node("invariant_prover", invariant_prover_node)
    builder.add_node("chainer", chainer_node)
    builder.add_node("exploit", exploit_node)
    builder.add_node("skeptic", skeptic_node)
    builder.add_node("refuter", refuter_node)
    builder.add_node("scribe", scribe_node)
    builder.add_node("static_analysis", static_analysis_node)

    builder.set_entry_point("scout")

    builder.add_conditional_edges(
        "scout",
        after_scout,
        {
            "static_analysis": "static_analysis",
            "mapper": "mapper",
        },
    )

    builder.add_edge("static_analysis", "scribe")

    builder.add_edge("mapper", "threat_modeler")
    builder.add_edge("threat_modeler", "hunter")
    builder.add_edge("hunter", "invariant_prover")
    builder.add_edge("invariant_prover", "chainer")
    builder.add_edge("chainer", "exploit")
    builder.add_edge("exploit", "skeptic")
    builder.add_edge("skeptic", "refuter")

    builder.add_conditional_edges(
        "refuter",
        after_skeptic,
        {
            "scribe": "scribe",
            "end": END,
        },
    )

    builder.add_edge("scribe", END)

    return builder.compile(checkpointer=checkpointer)
