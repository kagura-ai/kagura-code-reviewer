"""Tests for the eval runner (eval_cli) — drives review_harness with a fake
deterministic ChatClient, so no Ollama is needed and the 138-test suite stays
network-free."""
from __future__ import annotations

import json
import textwrap

from kagura_code_reviewer.agent import ChatMessage, ToolCall
from kagura_code_reviewer.eval_harness import GoldenBug, GoldenCase
from kagura_code_reviewer.report import Severity
from kagura_code_reviewer.review.harness import resolve_tier


def _submit(findings):
    return ChatMessage(content=None,
                       tool_calls=[ToolCall("1", "submit_findings",
                                            json.dumps({"findings": findings}))])


def _verdict(v):
    return ChatMessage(content=None,
                       tool_calls=[ToolCall("1", "submit_verdict",
                                            json.dumps({"verdict": v}))])


class FakeClient:
    """Stateless: finder calls (submit_findings in tools) return one seeded index
    bug at a.py:10; verifier calls return CONFIRMED."""

    def chat(self, messages, tools=None):
        names = {t["function"]["name"] for t in (tools or [])}
        if "submit_findings" in names:
            return _submit([{
                "dimension": "correctness", "severity": "high",
                "file": "a.py", "line": 10,
                "title": "IndexError: list index out of range",
                "rationale": "index out of range", "suggestion": "guard",
            }])
        return _verdict("CONFIRMED")


def _case():
    return GoldenCase(
        name="idx", diff="--- a/a.py\n+++ b/a.py\n@@\n-x\n+y\n", source="seeded",
        bugs=[GoldenBug(file="a.py", line=10, dimension="correctness",
                        severity=Severity.HIGH, symptom="index")],
    )


def test_run_eval_scores_each_repeat():
    from kagura_code_reviewer.eval_cli import run_eval
    results = run_eval([_case()], FakeClient(), tier=resolve_tier("low"),
                       repeats=2, max_concurrency=1)
    assert len(results) == 2
    for r in results:
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert (r.tp, r.fp, r.fn) == (1, 0, 0)


def test_run_eval_counts_a_miss_as_fn():
    from kagura_code_reviewer.eval_cli import run_eval
    case = GoldenCase(
        name="miss", diff="d", source="seeded",
        bugs=[GoldenBug(file="a.py", line=10, dimension="correctness",
                        severity=Severity.HIGH, symptom="zerodivision")],
    )  # FakeClient only emits an "index" symptom -> never matches -> FN
    results = run_eval([case], FakeClient(), tier=resolve_tier("low"), repeats=1)
    r = results[0]
    assert (r.tp, r.fn) == (0, 1)
    assert r.recall == 0.0


def test_eval_cli_json_output(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from kagura_code_reviewer import cli as cli_mod
    from kagura_code_reviewer import eval_cli

    (tmp_path / "c.diff").write_text("--- a/a.py\n+++ b/a.py\n@@\n-x\n+y\n")
    (tmp_path / "manifest.toml").write_text(textwrap.dedent("""\
        [[case]]
        name = "idx"
        source = "seeded"
        diff_file = "c.diff"
        [[case.bug]]
        file = "a.py"
        line = 10
        symptom = "index"
        dimension = "correctness"
        severity = "high"
    """))

    monkeypatch.setattr(cli_mod, "build_review_client",
                        lambda *a, **k: (FakeClient(), "fake-model"), raising=False)

    res = CliRunner().invoke(
        eval_cli.app,
        ["--golden-dir", str(tmp_path), "--effort", "low", "--format", "json"],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["model"] == "fake-model"
    assert payload["summary"]["precision_mean"] == 1.0
    assert payload["summary"]["recall_mean"] == 1.0


def test_dictrepo_grep_emits_line_numbers():
    from kagura_code_reviewer.eval_cli import _DictRepo
    repo = _DictRepo({"a.py": "import os\nx = 1\nuse(os)\n"})
    out = repo.grep("os")
    assert "a.py:1: import os" in out
    assert "a.py:3: use(os)" in out
    assert "a.py:2:" not in out  # line without the pattern is not emitted


def test_dictrepo_read_file_error_not_truncated():
    from kagura_code_reviewer.eval_cli import _DictRepo
    repo = _DictRepo({})
    msg = repo.read_file("missing.py", max_bytes=5)
    assert msg == "error: file not found: missing.py"  # full error, not sliced to 5


def test_dictrepo_read_file_returns_content_capped():
    from kagura_code_reviewer.eval_cli import _DictRepo
    repo = _DictRepo({"a.py": "0123456789"})
    assert repo.read_file("a.py") == "0123456789"
    assert repo.read_file("a.py", max_bytes=4) == "0123"


def test_dictrepo_list_files_filters_subdir():
    from kagura_code_reviewer.eval_cli import _DictRepo
    repo = _DictRepo({"a.py": "x", "src/b.py": "y", "src/c.py": "z"})
    assert repo.list_files() == ["a.py", "src/b.py", "src/c.py"]
    assert repo.list_files("src") == ["src/b.py", "src/c.py"]
    assert repo.list_files("src/") == ["src/b.py", "src/c.py"]


def test_eval_cli_md_output_to_file(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from kagura_code_reviewer import cli as cli_mod
    from kagura_code_reviewer import eval_cli

    (tmp_path / "manifest.toml").write_text(textwrap.dedent("""\
        [[case]]
        name = "idx"
        source = "seeded"
        diff = "d"
        [[case.bug]]
        file = "a.py"
        line = 10
        symptom = "index"
        dimension = "correctness"
        severity = "high"
    """))
    monkeypatch.setattr(cli_mod, "build_review_client",
                        lambda *a, **k: (FakeClient(), "fake-model"), raising=False)
    out = tmp_path / "report.md"
    res = CliRunner().invoke(eval_cli.app, [
        "--golden-dir", str(tmp_path), "--effort", "low",
        "--format", "md", "--out", str(out),
    ])
    assert res.exit_code == 0, res.output
    text = out.read_text()
    assert "Eval baseline" in text
    assert "precision (seeded): mean 100.00%" in text  # _pct rendered the md path
    assert "By dimension" in text                       # EvalResult.to_markdown ran


def test_eval_cli_empty_golden_dir_errors(tmp_path):
    from typer.testing import CliRunner

    from kagura_code_reviewer import eval_cli
    (tmp_path / "manifest.toml").write_text("")  # no [[case]] entries
    res = CliRunner().invoke(eval_cli.app, ["--golden-dir", str(tmp_path)])
    assert res.exit_code == 2
