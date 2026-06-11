"""Regression: every text-file read/write must pin ``encoding="utf-8"`` rather
than fall back to the locale default. On a non-UTF-8 default locale (cp932 on
Japanese Windows) an unencoded ``read_text``/``write_text`` of UTF-8 content
crashes with ``UnicodeDecodeError``/``UnicodeEncodeError`` — or, with
``errors="replace"``, silently corrupts non-ASCII bytes.

Spying ``io.open`` and asserting the ``encoding`` argument is ``"utf-8"`` (not
the ``"locale"`` sentinel that ``read_text``/``write_text`` pass when no encoding
is given) is locale- and UTF-8-mode-independent, so the test fails on the bug on
every CI runner regardless of the runner's own locale (issue #20). Pattern
mirrors kagura-engineer's ``tests/setup/test_scaffold.py`` ``TestEncodingPinnedToUtf8``.
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

import kagura_code_reviewer.cli as cli_mod
from kagura_code_reviewer.agent import ChatMessage, ToolCall
from kagura_code_reviewer.tools import RepoTools


def _spy_text_opens(monkeypatch: pytest.MonkeyPatch, seen: dict[str, list]) -> None:
    """Record the encoding of every text-mode ``io.open`` keyed by file path.

    pathlib calls ``io.open(file, mode, buffering, encoding, errors, newline)``
    positionally; read the encoding from arg 3 (or the kwarg) and forward
    everything verbatim to avoid a duplicate-argument ``TypeError``.
    """
    real_open = io.open

    def spy(*args, **kwargs):
        file = args[0] if args else kwargs.get("file", "")
        mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
        enc = args[3] if len(args) > 3 else kwargs.get("encoding")
        if "b" not in mode:
            seen.setdefault(str(file), []).append(enc)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(io, "open", spy)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "t@t.t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1  # コメント\n", encoding="utf-8")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-m", "init")
    (tmp_path / "a.py").write_text("x = 1  # コメント\ny = 2  # 世界\n", encoding="utf-8")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-m", "second")
    return tmp_path


class FakeClient:
    def __init__(self, *_, **__):
        pass

    def chat(self, messages, tools=None):
        import json

        payload = {"findings": [
            {"dimension": "security", "severity": "high", "file": "a.py",
             "line": 2, "title": "bad", "rationale": "r", "suggestion": "s"}
        ]}
        return ChatMessage(
            content=None,
            tool_calls=[ToolCall("1", "submit_findings", json.dumps(payload))],
        )


class TestEncodingPinnedToUtf8:
    def test_read_file_pins_utf8(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, list] = {}
        _spy_text_opens(monkeypatch, seen)
        tools = RepoTools(repo)
        assert "世界" in tools.read_file("a.py")
        encs = seen.get(str(repo / "a.py"))
        assert encs, "expected a.py to be opened as text"
        assert all(e == "utf-8" for e in encs), f"unencoded open observed: {encs}"

    def test_grep_pins_utf8(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, list] = {}
        _spy_text_opens(monkeypatch, seen)
        tools = RepoTools(repo)
        assert "a.py" in tools.grep("世界")
        encs = seen.get(str(repo / "a.py"))
        assert encs, "expected a.py to be opened as text by grep"
        assert all(e == "utf-8" for e in encs), f"unencoded open observed: {encs}"

    def test_cli_context_read_and_report_write_pin_utf8(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # We assert the *encoding* of the context read and the report write, not
        # that the review harness runs. Stub client_factory (so
        # build_review_client makes no network call) and review_harness (so the
        # command deterministically reaches out.write_text on every platform —
        # driving the real harness with a FakeClient exits early under some
        # CPU/concurrency configurations and never reaches the write).
        monkeypatch.setattr(
            cli_mod, "client_factory", lambda spec, timeout, seed=None: FakeClient()
        )

        class _FakeReport:
            def to_markdown(self) -> str:
                return "# レビュー結果 🎉\n"

            def to_json(self) -> str:
                return '{"findings": []}'

            def exit_code(self) -> int:
                return 0

        monkeypatch.setattr(cli_mod, "review_harness", lambda *a, **k: _FakeReport())

        ctx = tmp_path / "context.md"
        ctx.write_text("# 文脈 🎉\n", encoding="utf-8")
        out = tmp_path / "report.md"
        seen: dict[str, list] = {}
        _spy_text_opens(monkeypatch, seen)
        runner = CliRunner()
        result = runner.invoke(
            cli_mod.app,
            ["--base", "HEAD~1", "--repo", str(repo),
             "--context-file", str(ctx), "--format", "md", "--out", str(out)],
        )
        assert result.exit_code == 0, result.output
        ctx_encs = seen.get(str(ctx))
        assert ctx_encs, "expected the context file to be opened as text"
        assert all(e == "utf-8" for e in ctx_encs), f"context read unencoded: {ctx_encs}"
        out_encs = seen.get(str(out))
        assert out_encs, "expected the report file to be opened as text"
        assert all(e == "utf-8" for e in out_encs), f"report write unencoded: {out_encs}"
