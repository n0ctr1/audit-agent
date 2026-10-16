"""Breadth triage — cheap pass over EVERY function (roadmap #7, model cascade).

Hunter's deep per-function analysis is expensive, so it only runs on the top-N
attack-surface functions — leaving most of a large codebase unexamined. This
triager closes that gap cheaply: a fast/cheap model rates EVERY in-scope function
0–100 on "how likely is a non-trivial bug here, worth a deep look?". Hunter folds
that suspicion into its selection score, so a function with low static surface but
high triage suspicion still gets pulled into the expensive depth pass (and the
navigator). Breadth from the cheap model, depth from the expensive one.

Batched + concurrency-limited so cost stays bounded even on big protocols.
Fully injectable/mock-aware → offline-testable.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from audit_agent.config import get_settings
from audit_agent.llm import get_provider
from audit_agent.llm.base import LLMProvider, ToolDefinition
from audit_agent.observability.audit_logger import get_audit_logger
from audit_agent.observability.logging import get_logger

log = get_logger("agents.triage")


_TRIAGE_TOOL = ToolDefinition(
    name="report_triage",
    description=(
        "Rate each listed function 0-100 on how likely it hides a non-trivial "
        "security vulnerability worth deep analysis. Call exactly once."
    ),
    parameters={
        "type": "object",
        "required": ["flags"],
        "properties": {
            "flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["index", "suspicion"],
                    "properties": {
                        "index": {"type": "integer", "description": "Index from the list."},
                        "suspicion": {"type": "integer", "minimum": 0, "maximum": 100},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
    },
)

_TRIAGE_SYSTEM = (
    "You are a fast smart-contract triage analyst. You are NOT doing a deep audit — "
    "you are deciding WHERE a deep audit should focus. For each function, give a "
    "suspicion score 0-100:\n"
    "- 80-100: handles value, accounting/pricing, signature/auth, or complex math; "
    "likely worth deep analysis.\n"
    "- 40-79: state-changing but routine; check if time permits.\n"
    "- 0-39: trivial getters/setters, pure views, standard ERC boilerplate.\n"
    "Err toward higher scores for anything touching funds, oracles, access control, "
    "or cross-contract calls. Be fast. Output ONLY the tool call."
)


class FunctionTriager:
    """Cheap, batched breadth triage of functions → {function_id: suspicion 0-100}."""

    _BATCH_TIMEOUT_SECS = 120.0

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        batch_size: int = 25,
        max_concurrency: int = 8,
    ) -> None:
        settings = get_settings()
        self._provider = provider or get_provider()
        # Default to the cheap Scout model unless a dedicated triage model is set.
        self._model = model or getattr(settings, "triage_model", "") or settings.scout_model
        self._batch_size = max(1, batch_size)
        self._max_concurrency = max(1, max_concurrency)

    async def triage(self, functions: list[tuple[str, Any]], audit_id: str = "") -> dict[str, float]:
        """functions: list of (contract_name, SlitherFunction). Returns id→suspicion."""
        if not functions:
            return {}

        batches = [
            functions[i:i + self._batch_size]
            for i in range(0, len(functions), self._batch_size)
        ]
        sem = asyncio.Semaphore(self._max_concurrency)

        async def _run(batch):
            async with sem:
                return await self._triage_batch(batch)

        results = await asyncio.gather(
            *(_run(b) for b in batches), return_exceptions=True
        )

        scores: dict[str, float] = {}
        for res in results:
            if isinstance(res, dict):
                scores.update(res)

        al = get_audit_logger()
        if al and scores:
            flagged = sum(1 for v in scores.values() if v >= 80)
            al.log_decision("hunter", "triage",
                f"triaged {len(functions)} fns in {len(batches)} batch(es); "
                f"{flagged} high-suspicion",
                {"functions": len(functions), "high_suspicion": flagged})
        return scores

    async def _triage_batch(self, batch: list[tuple[str, Any]]) -> dict[str, float]:
        lines = []
        for i, (cname, fn) in enumerate(batch):
            tainted = ", ".join(fn.tainted_state_writes or []) or "-"
            ext = ", ".join(
                f"{t.get('dest')}.{t.get('fn')}" for t in (fn.external_targets or [])[:4]
            ) or "-"
            lines.append(
                f"[{i}] {cname}.{fn.name} vis={fn.visibility} "
                f"writes={[w.split('-')[-1] for w in fn.writes][:6]} "
                f"tainted={tainted} ext_calls=[{ext}] "
                f"oracle={fn.reads_oracle} value={fn.handles_value} priv={fn.is_privileged}"
            )
        prompt = (
            "Rate each function's vulnerability suspicion 0-100:\n\n"
            + "\n".join(lines)
            + "\n\nCall report_triage with a flag for every index."
        )
        try:
            resp = await asyncio.wait_for(
                self._provider.complete(
                    model=self._model, max_tokens=2048, system=_TRIAGE_SYSTEM,
                    tools=[_TRIAGE_TOOL], tool_choice={"name": "report_triage"},
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=self._BATCH_TIMEOUT_SECS,
            )
        except Exception as exc:
            log.warning("triage.batch_failed", error=str(exc))
            return {}

        out: dict[str, float] = {}
        for tc in resp.tool_calls:
            if tc.name != "report_triage":
                continue
            for flag in (tc.arguments.get("flags") or []):
                try:
                    idx = int(flag.get("index"))
                    susp = float(flag.get("suspicion", 0))
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(batch):
                    _cname, fn = batch[idx]
                    out[fn.id] = max(0.0, min(100.0, susp))
        return out
