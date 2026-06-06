import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

import kagura_code_review.cli as cli_mod
from kagura_code_review.agent import ChatMessage, ToolCall


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "t@t.t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-m", "init")
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-m", "second")
    return tmp_path


class FakeClient:
    def __init__(self, *_, **__):
        self._sent = False

    def chat(self, messages, tools=None):
        import json
        payload = {"findings": [
            {"dimension": "security", "severity": "high", "file": "a.py",
             "line": 2, "title": "bad", "rationale": "r", "suggestion": "s"}
        ]}
        return ChatMessage(content=None, tool_calls=[ToolCall("1", "submit_findings", json.dumps(payload))])


def test_cli_exits_nonzero_on_blocking(repo: Path, monkeypatch):
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout: FakeClient())
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.app,
        ["--base", "HEAD~1", "--repo", str(repo), "--format", "md"],
    )
    assert result.exit_code == 1
    assert "bad" in result.stdout


def test_cli_writes_json_out(repo: Path, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout: FakeClient())
    out = tmp_path / "r.json"
    runner = CliRunner()
    runner.invoke(
        cli_mod.app,
        ["--base", "HEAD~1", "--repo", str(repo), "--format", "json", "--out", str(out)],
    )
    assert "findings" in out.read_text()
