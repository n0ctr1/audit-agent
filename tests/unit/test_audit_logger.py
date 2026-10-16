"""Unit tests for AuditLogger."""

import json
from pathlib import Path

import pytest

from audit_agent.observability.audit_logger import (
    AuditLogger,
    get_audit_logger,
    set_audit_logger,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure singleton is cleared after each test."""
    yield
    set_audit_logger(None)


class TestAuditLoggerDirectoryStructure:
    def test_creates_directory_structure(self, tmp_path):
        al = AuditLogger(tmp_path, "test_001")
        assert (tmp_path / "audit_logs" / "test_001" / "agents").is_dir()
        assert (tmp_path / "audit_logs" / "test_001" / "llm_calls").is_dir()
        assert (tmp_path / "audit_logs" / "test_001" / "tools").is_dir()
        assert (tmp_path / "audit_logs" / "test_001" / "poc").is_dir()
        al.write_summary({})

    def test_audit_log_file_created(self, tmp_path):
        al = AuditLogger(tmp_path, "test_log_file")
        al.write_summary({})
        assert (tmp_path / "audit_logs" / "test_log_file" / "audit.log").exists()
        assert (tmp_path / "audit_logs" / "test_log_file" / "audit.jsonl").exists()


class TestLLMCallLogging:
    def test_llm_call_logged(self, tmp_path):
        al = AuditLogger(tmp_path, "test_002")
        record = al.log_llm_call_start(
            agent="hunter",
            purpose="test_purpose",
            model="claude-opus-4-7",
            system_prompt="You are a security auditor.",
            user_message="Analyze this function...",
        )
        al.log_llm_call_end(
            record,
            response_text="I found a vulnerability...",
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            duration_seconds=2.5,
        )
        al.write_summary({})

        llm_files = list((tmp_path / "audit_logs" / "test_002" / "llm_calls").glob("*.json"))
        assert len(llm_files) == 1

        data = json.loads(llm_files[0].read_text())
        assert data["agent"] == "hunter"
        assert data["system_prompt"] == "You are a security auditor."
        assert data["response_text"] == "I found a vulnerability..."
        assert data["input_tokens"] == 100
        assert data["output_tokens"] == 50
        assert data["stop_reason"] == "end_turn"

    def test_llm_call_id_sequential(self, tmp_path):
        al = AuditLogger(tmp_path, "test_seq")
        r1 = al.log_llm_call_start("scout", "p1", "model", "sys", "usr")
        r2 = al.log_llm_call_start("hunter", "p2", "model", "sys", "usr")
        al.log_llm_call_end(r1)
        al.log_llm_call_end(r2)
        al.write_summary({})

        assert r1.call_id == "001"
        assert r2.call_id == "002"

    def test_llm_call_filename_includes_agent_purpose(self, tmp_path):
        al = AuditLogger(tmp_path, "test_fname")
        record = al.log_llm_call_start(
            "scout", "summarize_protocol", "claude-sonnet-4-6", "sys", "usr"
        )
        al.log_llm_call_end(record, response_text="ok")
        al.write_summary({})

        llm_files = list((tmp_path / "audit_logs" / "test_fname" / "llm_calls").glob("*.json"))
        assert len(llm_files) == 1
        assert "scout" in llm_files[0].name
        assert "summarize_protocol" in llm_files[0].name

    def test_llm_call_with_tool_calls(self, tmp_path):
        al = AuditLogger(tmp_path, "test_tool_calls")
        record = al.log_llm_call_start("hunter", "hypotheses", "model", "sys", "usr",
                                        tools=[{"name": "report_hypotheses"}])
        al.log_llm_call_end(record,
                            tool_calls=[{"name": "report_hypotheses", "count": 3}],
                            input_tokens=200, output_tokens=800,
                            stop_reason="tool_use", duration_seconds=5.0)
        al.write_summary({})

        data = json.loads(list(
            (tmp_path / "audit_logs" / "test_tool_calls" / "llm_calls").glob("*.json")
        )[0].read_text())
        assert data["stop_reason"] == "tool_use"
        assert len(data["tool_calls"]) == 1
        assert data["tools"] == [{"name": "report_hypotheses"}]


class TestToolRunLogging:
    def test_tool_run_logged_with_poc(self, tmp_path):
        al = AuditLogger(tmp_path, "test_003")
        al.log_tool_run(
            agent="exploit",
            tool_name="forge",
            purpose="test_hyp_001",
            command="forge test -vv",
            poc_code="// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\ncontract Test {}",
            hypothesis_id="hyp_001",
            attempt_number=1,
            exit_code=0,
            stdout="PASS",
            status="success",
            duration_seconds=5.0,
        )
        al.write_summary({})

        poc_file = tmp_path / "audit_logs" / "test_003" / "poc" / "hyp_001_attempt1.sol"
        assert poc_file.exists()
        assert "pragma solidity" in poc_file.read_text()

    def test_tool_run_file_created(self, tmp_path):
        al = AuditLogger(tmp_path, "test_tr_file")
        tr = al.log_tool_run(
            agent="scout", tool_name="slither", purpose="static_analysis",
            command="slither . --json out.json",
            exit_code=0, stdout="OK", status="success", duration_seconds=10.0,
        )
        al.write_summary({})

        tool_files = list((tmp_path / "audit_logs" / "test_tr_file" / "tools").glob("*.json"))
        assert len(tool_files) == 1
        data = json.loads(tool_files[0].read_text())
        assert data["tool_name"] == "slither"
        assert data["status"] == "success"
        assert data["exit_code"] == 0

    def test_tool_run_ids_are_sequential(self, tmp_path):
        al = AuditLogger(tmp_path, "test_tr_seq")
        tr1 = al.log_tool_run("scout", "slither", "p1")
        tr2 = al.log_tool_run("exploit", "forge", "p2")
        al.write_summary({})
        assert tr1.run_id == "tool_001"
        assert tr2.run_id == "tool_002"

    def test_poc_without_attempt_number(self, tmp_path):
        al = AuditLogger(tmp_path, "test_poc_no_attempt")
        al.log_tool_run(
            agent="exploit", tool_name="forge", purpose="test_hyp_x",
            poc_code="pragma solidity ^0.8.0;",
            hypothesis_id="hyp_xyz",
            attempt_number=0,
        )
        al.write_summary({})
        # attempt_number=0 → no _attempt suffix
        poc_file = tmp_path / "audit_logs" / "test_poc_no_attempt" / "poc" / "hyp_xyz.sol"
        assert poc_file.exists()

    def test_stdout_truncated_at_50kb(self, tmp_path):
        al = AuditLogger(tmp_path, "test_truncate")
        huge_stdout = "x" * 100_000
        al.log_tool_run("scout", "slither", "test", stdout=huge_stdout)
        al.write_summary({})

        data = json.loads(list(
            (tmp_path / "audit_logs" / "test_truncate" / "tools").glob("*.json")
        )[0].read_text())
        assert len(data["stdout"]) == 50_000


class TestAgentRecordLogging:
    def test_agent_record(self, tmp_path):
        al = AuditLogger(tmp_path, "test_004")
        record = al.log_agent_start("hunter", 3, {"functions": 10})
        al.log_agent_end(record, output_summary={"hypotheses": 5}, llm_call_ids=["001", "002"])
        al.write_summary({})

        agent_file = tmp_path / "audit_logs" / "test_004" / "agents" / "03_hunter.json"
        assert agent_file.exists()

    def test_agent_record_content(self, tmp_path):
        al = AuditLogger(tmp_path, "test_ar_content")
        record = al.log_agent_start("scout", 1, {"target_dir": "/foo", "mode": "audit"})
        al.log_agent_end(record,
                         output_summary={"files_in_scope": 5},
                         llm_call_ids=["001"],
                         tool_run_ids=["tool_001"],
                         warnings=["mock mode"])
        al.write_summary({})

        data = json.loads(
            (tmp_path / "audit_logs" / "test_ar_content" / "agents" / "01_scout.json").read_text()
        )
        assert data["agent"] == "scout"
        assert data["phase_number"] == 1
        assert data["input_summary"]["target_dir"] == "/foo"
        assert data["output_summary"]["files_in_scope"] == 5
        assert data["llm_call_ids"] == ["001"]
        assert data["tool_run_ids"] == ["tool_001"]
        assert "mock mode" in data["warnings"]
        assert data["duration_seconds"] >= 0.0

    def test_agent_record_multiple_phases(self, tmp_path):
        al = AuditLogger(tmp_path, "test_phases")
        for phase, name in [(1, "scout"), (2, "mapper"), (3, "hunter")]:
            record = al.log_agent_start(name, phase, {})
            al.log_agent_end(record, {})
        al.write_summary({})

        for phase, name in [(1, "scout"), (2, "mapper"), (3, "hunter")]:
            f = tmp_path / "audit_logs" / "test_phases" / "agents" / f"{phase:02d}_{name}.json"
            assert f.exists()


class TestSummary:
    def test_summary_written(self, tmp_path):
        al = AuditLogger(tmp_path, "test_005")
        al.write_summary({
            "hypotheses": [1, 2, 3],
            "validated_findings": [{"status": "validated"}, {"status": "false_positive"}],
        })

        summary = json.loads((tmp_path / "audit_logs" / "test_005" / "summary.json").read_text())
        assert summary["hypotheses_count"] == 3
        assert summary["validated_findings_count"] == 2

    def test_summary_counts_by_status(self, tmp_path):
        al = AuditLogger(tmp_path, "test_summary_counts")
        al.write_summary({
            "validated_findings": [
                {"status": "validated"},
                {"status": "validated"},
                {"status": "false_positive"},
            ]
        })
        summary = json.loads(
            (tmp_path / "audit_logs" / "test_summary_counts" / "summary.json").read_text()
        )
        assert summary["findings_by_verdict"]["validated"] == 2
        assert summary["findings_by_verdict"]["false_positive"] == 1

    def test_summary_includes_timing(self, tmp_path):
        al = AuditLogger(tmp_path, "test_timing")
        al.write_summary({})
        summary = json.loads((tmp_path / "audit_logs" / "test_timing" / "summary.json").read_text())
        assert "total_duration_seconds" in summary
        assert summary["total_duration_seconds"] >= 0.0
        assert "timestamp_start" in summary
        assert "timestamp_end" in summary

    def test_summary_llm_and_tool_counts(self, tmp_path):
        al = AuditLogger(tmp_path, "test_counts")
        r = al.log_llm_call_start("scout", "p", "m", "s", "u")
        al.log_llm_call_end(r)
        al.log_tool_run("scout", "slither", "p")
        al.write_summary({})

        summary = json.loads((tmp_path / "audit_logs" / "test_counts" / "summary.json").read_text())
        assert summary["total_llm_calls"] == 1
        assert summary["total_tool_runs"] == 1


class TestJSONLEvents:
    def test_jsonl_events(self, tmp_path):
        al = AuditLogger(tmp_path, "test_006")
        al.log_decision("hunter", "test_decision", "just testing")
        al.write_summary({})

        jsonl_lines = (
            tmp_path / "audit_logs" / "test_006" / "audit.jsonl"
        ).read_text().splitlines()
        assert len(jsonl_lines) >= 2  # audit_start + decision

    def test_jsonl_valid_json_each_line(self, tmp_path):
        al = AuditLogger(tmp_path, "test_jsonl_valid")
        al.log_decision("scout", "scope_resolved", "5 files", {"count": 5})
        r = al.log_llm_call_start("scout", "p", "m", "s", "u")
        al.log_llm_call_end(r, response_text="ok")
        al.write_summary({})

        lines = (
            tmp_path / "audit_logs" / "test_jsonl_valid" / "audit.jsonl"
        ).read_text().splitlines()
        for line in lines:
            parsed = json.loads(line)
            assert "ts" in parsed
            assert "event" in parsed

    def test_jsonl_has_audit_start_and_end(self, tmp_path):
        al = AuditLogger(tmp_path, "test_start_end")
        al.write_summary({})

        lines = [
            json.loads(l)
            for l in (tmp_path / "audit_logs" / "test_start_end" / "audit.jsonl")
            .read_text().splitlines()
        ]
        events = [l["event"] for l in lines]
        assert "audit_start" in events


class TestHumanReadableLog:
    def test_human_readable_log(self, tmp_path):
        al = AuditLogger(tmp_path, "test_007")
        record = al.log_llm_call_start("scout", "summarize", "claude-sonnet-4-6", "sys", "usr")
        al.log_llm_call_end(record, response_text="response", input_tokens=10, output_tokens=5,
                             stop_reason="end_turn")
        al.write_summary({})

        log_text = (tmp_path / "audit_logs" / "test_007" / "audit.log").read_text()
        assert "LLM CALL #001" in log_text
        assert "scout" in log_text
        assert "AUDIT COMPLETE" in log_text

    def test_human_readable_includes_agent_sections(self, tmp_path):
        al = AuditLogger(tmp_path, "test_agent_sections")
        record = al.log_agent_start("hunter", 3, {"scope": "10 files"})
        al.log_agent_end(record, {"hypotheses": 5})
        al.write_summary({})

        log_text = (tmp_path / "audit_logs" / "test_agent_sections" / "audit.log").read_text()
        assert "AGENT: hunter" in log_text
        assert "Phase 3" in log_text

    def test_human_readable_includes_tool_run(self, tmp_path):
        al = AuditLogger(tmp_path, "test_tool_in_log")
        al.log_tool_run("scout", "slither", "static_analysis",
                         command="slither .", exit_code=0, status="success")
        al.write_summary({})

        log_text = (tmp_path / "audit_logs" / "test_tool_in_log" / "audit.log").read_text()
        assert "TOOL RUN" in log_text
        assert "slither" in log_text


class TestDecisionLogging:
    def test_decision_logged_to_jsonl(self, tmp_path):
        al = AuditLogger(tmp_path, "test_decision")
        al.log_decision("hunter", "skip_function", "score too low",
                         {"function": "foo", "score": 0.1})
        al.write_summary({})

        lines = [
            json.loads(l)
            for l in (tmp_path / "audit_logs" / "test_decision" / "audit.jsonl")
            .read_text().splitlines()
        ]
        decisions = [l for l in lines if l["event"] == "decision"]
        assert len(decisions) == 1
        assert decisions[0]["decision"] == "skip_function"
        assert decisions[0]["agent"] == "hunter"

    def test_decision_logged_to_text(self, tmp_path):
        al = AuditLogger(tmp_path, "test_decision_text")
        al.log_decision("exploit", "retry", "compile error: undeclared identifier")
        al.write_summary({})

        log_text = (tmp_path / "audit_logs" / "test_decision_text" / "audit.log").read_text()
        assert "DECISION [exploit]: retry" in log_text


class TestSingleton:
    def test_set_and_get_singleton(self, tmp_path):
        al = AuditLogger(tmp_path, "test_singleton")
        set_audit_logger(al)
        assert get_audit_logger() is al
        al.write_summary({})

    def test_get_returns_none_by_default(self):
        assert get_audit_logger() is None

    def test_set_none_clears_singleton(self, tmp_path):
        al = AuditLogger(tmp_path, "test_clear")
        set_audit_logger(al)
        set_audit_logger(None)
        assert get_audit_logger() is None
        al.write_summary({})
