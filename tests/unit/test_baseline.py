"""Tests for the single-agent baseline (agents/baseline.py), offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from audit_agent.agents.baseline import BaselineAgent, _file_edit, _to_hypotheses
from audit_agent.llm.base import LLMResponse, ToolCall
from audit_agent.llm.mock_provider import MockProvider
from audit_agent.tools.shell_session import MockShellSession

HYP = {"title": "Reentrancy in withdraw", "vulnerability_class": "reentrancy",
       "affected_functions": ["VulnerableBank.withdraw"], "attack_scenario": "x",
       "required_preconditions": [], "estimated_severity": "high", "priority_score": 90}


def _resp(*calls, tokens=(10, 5)):
    return LLMResponse(text="", tool_calls=[ToolCall(id=str(i), name=n, arguments=a)
                                            for i, (n, a) in enumerate(calls)],
                       input_tokens=tokens[0], output_tokens=tokens[1], stop_reason="tool_use",
                       model="m")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    p = tmp_path / "proj"
    (p / "src").mkdir(parents=True)
    (p / "src" / "Bank.sol").write_text("contract Bank { function withdraw() external {} }")
    return p


async def test_agent_uses_tools_then_reports(project: Path):
    shells: list[MockShellSession] = []

    def factory(copy_dir):
        shells.append(MockShellSession(copy_dir, {"cat": "contract Bank {}"}))
        return shells[-1]

    provider = MockProvider([
        _resp(("bash", {"command": "cat src/Bank.sol"})),
        _resp(("file_edit", {"path": "test/PoC.t.sol", "new_string": "contract PoC {}"})),
        _resp(("report_hypotheses", {"hypotheses": [HYP, {"title": ""}]})),
    ])
    agent = BaselineAgent(provider=provider, model="m", shell_factory=factory,
                          max_tokens=10_000, max_seconds=100, max_steps=10)
    hyps = await agent.run(str(project), ["src/Bank.sol"], "audit-1")

    assert [h.title for h in hyps] == ["Reentrancy in withdraw"]
    assert hyps[0].affected_functions == ["VulnerableBank.withdraw"]
    assert hyps[0].priority_score == 90
    assert agent.stop_reason == "reported" and agent.steps == 2 and agent.tokens_used == 45
    assert shells[0].commands == ["cat src/Bank.sol"]
    assert "src/Bank.sol" in provider.calls[0]["messages"][0]["content"]
    # MockProvider keeps a reference to the growing message list: [task, a1, obs1, a2, obs2]
    assert "contract Bank {}" in provider.calls[1]["messages"][2]["content"]
    # works on a copy: the original project is untouched and the copy is removed
    assert not (project / "test").exists()
    assert not Path(shells[0].project_dir).exists()


async def test_budget_forces_final_report(project: Path):
    provider = MockProvider([_resp(("bash", {"command": "ls"}))] * 2
                            + [_resp(("report_hypotheses", {"hypotheses": [HYP]}))])
    agent = BaselineAgent(provider=provider, model="m",
                          shell_factory=lambda d: MockShellSession(d),
                          max_tokens=10_000, max_seconds=100, max_steps=2)
    hyps = await agent.run(str(project), ["src/Bank.sol"], "a")
    assert len(hyps) == 1
    assert agent.stop_reason == "budget"
    last = provider.calls[-1]
    assert last["tool_choice"] == {"name": "report_hypotheses"}
    assert [t.name for t in last["tools"]] == ["report_hypotheses"]


async def test_token_budget(project: Path):
    provider = MockProvider([_resp(("bash", {"command": "ls"}), tokens=(600, 0)),
                             _resp(("report_hypotheses", {"hypotheses": []}))])
    agent = BaselineAgent(provider=provider, model="m", shell_factory=lambda d: MockShellSession(d),
                          max_tokens=500, max_seconds=100, max_steps=50)
    assert await agent.run(str(project), [], "a") == []
    assert provider.calls[-1]["tool_choice"] == {"name": "report_hypotheses"}


def test_file_edit_stays_inside_project(tmp_path: Path):
    proj = tmp_path / "p"
    proj.mkdir()
    assert "inside /project" in _file_edit(proj, {"path": "../escape.txt", "new_string": "x"})
    assert not (tmp_path / "escape.txt").exists()
    assert _file_edit(proj, {"path": "/project/a.txt", "new_string": "hello"}).startswith("wrote")
    assert (proj / "a.txt").read_text() == "hello"
    assert _file_edit(proj, {"path": "a.txt", "old_string": "hello", "new_string": "bye"}) == "edited a.txt"
    assert "found 0 times" in _file_edit(proj, {"path": "a.txt", "old_string": "zzz", "new_string": "q"})


def test_to_hypotheses_normalises():
    hs = _to_hypotheses([{"title": "T", "estimated_severity": "CRITICAL", "priority_score": "x"},
                         {"title": "U", "estimated_severity": "urgent"}, "junk"], "a")
    assert [(h.estimated_severity, h.priority_score) for h in hs] == [("critical", 0.0), ("medium", 0.0)]
