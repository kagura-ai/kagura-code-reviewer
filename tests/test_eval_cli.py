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


def test_eval_cli_empty_golden_dir_errors(tmp_path):
    from typer.testing import CliRunner

    from kagura_code_reviewer import eval_cli
    (tmp_path / "manifest.toml").write_text("")  # no [[case]] entries
    res = CliRunner().invoke(eval_cli.app, ["--golden-dir", str(tmp_path)])
    assert res.exit_code == 2
