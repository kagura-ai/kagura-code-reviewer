import json

from kagura_code_reviewer.agent import ChatMessage, ToolCall
from kagura_code_reviewer.report import Severity
from kagura_code_reviewer.review.skill import review


class ScriptedClient:
    def __init__(self, scripted):
        self._scripted = scripted
        self.calls = 0

    def chat(self, messages, tools=None):
        msg = self._scripted[self.calls]
        self.calls += 1
        return msg


class StubRepo:
    def read_file(self, path, max_bytes=20000):
        return "file contents"

    def grep(self, pattern, max_results=50):
        return "no matches"

    def list_files(self, subdir="."):
        return ["a.py"]


def test_review_returns_report_from_submit(monkeypatch):
    payload = {"findings": [
        {"dimension": "security", "severity": "high", "file": "a.py",
         "line": 5, "title": "issue", "rationale": "why", "suggestion": "fix"}
    ]}
    scripted = [
        ChatMessage(content=None, tool_calls=[ToolCall("1", "submit_findings", json.dumps(payload))]),
    ]
    report = review(ScriptedClient(scripted), StubRepo(), diff="--- diff ---", context=None)
    assert report.findings[0].severity is Severity.HIGH
    assert report.exit_code() == 1


def test_review_exhaustion_is_blocking():
    """An incomplete review (no findings submitted within the budget) must NOT
    pass the gate — it exits non-zero so CI cannot mistake it for a clean run."""
    scripted = [ChatMessage(content=None, tool_calls=[ToolCall("1", "read_file", '{"path":"a.py"}')])] * 30
    report = review(ScriptedClient(scripted), StubRepo(), diff="d", context=None, max_iters=3)
    assert report.findings  # a meta finding is reported
    assert report.exit_code() == 1


def test_context_is_included_in_messages():
    captured = {}

    class CaptureClient(ScriptedClient):
        def chat(self, messages, tools=None):
            captured["messages"] = messages
            return ChatMessage(content=None, tool_calls=[ToolCall("1", "submit_findings", '{"findings": []}')])

    review(CaptureClient([]), StubRepo(), diff="d", context="REMEMBERED-RULE")
    joined = json.dumps(captured["messages"])
    assert "REMEMBERED-RULE" in joined
