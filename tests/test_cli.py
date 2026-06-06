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


def test_cli_bad_ref_exits_cleanly(repo: Path, monkeypatch):
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout: FakeClient())
    # CliRunner in this Typer version does not support mix_stderr; use output instead.
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.app,
        ["--base", "nonexistent-ref-xyz", "--repo", str(repo)],
    )
    assert result.exit_code == 2
    assert "git diff failed" in result.output


def test_cli_rejects_invalid_format(repo: Path, monkeypatch):
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout: FakeClient())
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.app,
        ["--base", "HEAD~1", "--repo", str(repo), "--format", "xml"],
    )
    assert result.exit_code != 0


def test_cli_handles_ollama_failure(repo: Path, monkeypatch):
    import httpx

    class BoomClient:
        def chat(self, messages, tools=None):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout: BoomClient())
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo)])
    assert result.exit_code == 3
    assert "Ollama request failed" in result.output


def test_cli_doctor_flag(monkeypatch, tmp_path: Path):
    from kagura_code_review import doctor as doctor_mod
    from kagura_code_review.doctor import CheckResult

    monkeypatch.setattr(doctor_mod, "check_ollama", lambda base_url: CheckResult("ollama daemon", True, "ok"))
    monkeypatch.setattr(doctor_mod, "check_model", lambda base_url, model: CheckResult(f"model {model}", True, "pulled"))
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["--doctor"])
    assert result.exit_code == 0
    assert "ollama daemon" in result.output
