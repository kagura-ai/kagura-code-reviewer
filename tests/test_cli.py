import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

import kagura_code_reviewer.cli as cli_mod
from kagura_code_reviewer.agent import ChatMessage, ToolCall


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
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout, seed=None: FakeClient())
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.app,
        ["--base", "HEAD~1", "--repo", str(repo), "--format", "md"],
    )
    assert result.exit_code == 1
    assert "bad" in result.stdout


def test_cli_writes_json_out(repo: Path, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout, seed=None: FakeClient())
    out = tmp_path / "r.json"
    runner = CliRunner()
    runner.invoke(
        cli_mod.app,
        ["--base", "HEAD~1", "--repo", str(repo), "--format", "json", "--out", str(out)],
    )
    assert "findings" in out.read_text()


def test_cli_bad_ref_exits_cleanly(repo: Path, monkeypatch):
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout, seed=None: FakeClient())
    # CliRunner in this Typer version does not support mix_stderr; use output instead.
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.app,
        ["--base", "nonexistent-ref-xyz", "--repo", str(repo)],
    )
    assert result.exit_code == 2
    assert "git diff failed" in result.output


def test_cli_rejects_invalid_format(repo: Path, monkeypatch):
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout, seed=None: FakeClient())
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

    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout, seed=None: BoomClient())
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo)])
    assert result.exit_code == 3
    assert "request failed" in result.output


def test_cli_doctor_flag(monkeypatch, tmp_path: Path):
    from kagura_code_reviewer import doctor as doctor_mod
    from kagura_code_reviewer.doctor import CheckResult

    monkeypatch.setattr(doctor_mod, "check_ollama", lambda base_url: CheckResult("ollama daemon", True, "ok"))
    monkeypatch.setattr(doctor_mod, "check_model", lambda base_url, model: CheckResult(f"model {model}", True, "pulled"))
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["--doctor"])
    assert result.exit_code == 0
    assert "ollama daemon" in result.output


def test_cli_effort_option_invokes_harness(repo: Path, monkeypatch):
    from kagura_code_reviewer.report import Report
    captured = {}

    def fake_harness(finder_client, verifier_client, repo_, diff, context, tier,
                     max_iters=12, max_concurrency=1, min_confidence=0.0):
        captured["tier"] = tier.name
        return Report(findings=[])

    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout, seed=None: object())
    monkeypatch.setattr(cli_mod, "review_harness", fake_harness, raising=False)
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.app,
        ["--base", "HEAD~1", "--repo", str(repo), "--effort", "high"],
    )
    assert result.exit_code == 0
    assert captured["tier"] == "high"


from kagura_code_reviewer.advisor import Recommendation
from kagura_code_reviewer.report import Report


@pytest.fixture(autouse=True)
def _stub_advisor(monkeypatch):
    """Keep the advisor default path offline/deterministic for all CLI tests."""
    monkeypatch.setattr(cli_mod, "list_models", lambda base_url: ["qwen2.5-coder:7b"], raising=False)
    monkeypatch.setattr(
        cli_mod, "recommend",
        lambda hw, installed, prefer_local=True: Recommendation(
            "qwen2.5-coder:7b", "qwen2.5-coder:7b", "stub", True),
        raising=False,
    )


def test_cli_zero_config_uses_advisor(repo: Path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli_mod, "_USER", repo / "no-user-config.toml")  # force zero-config path
    monkeypatch.setattr(cli_mod.RepoTools, "git_diff", lambda self, b, h, p=None: "DIFF")
    monkeypatch.setattr(
        cli_mod, "recommend",
        lambda hw, installed, prefer_local=True: Recommendation(
            "qwen3.5:27b", "qwen3.5:27b", "fits GPU VRAM", True),
    )

    def spy_factory(spec, timeout, seed=None):
        captured["model"] = spec.ollama_model
        return object()
    monkeypatch.setattr(cli_mod, "client_factory", spy_factory)
    monkeypatch.setattr(cli_mod, "review_harness", lambda *a, **k: Report(findings=[]), raising=False)

    result = CliRunner().invoke(cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo)])
    assert result.exit_code == 0
    assert captured["model"] == "qwen3.5:27b"
    assert "Auto-selected" in result.output


def test_cli_explicit_model_skips_advisor(repo: Path, monkeypatch):
    monkeypatch.setattr(cli_mod.RepoTools, "git_diff", lambda self, b, h, p=None: "DIFF")
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout, seed=None: object())
    monkeypatch.setattr(cli_mod, "review_harness", lambda *a, **k: Report(findings=[]), raising=False)

    def boom_recommend(*a, **k):
        raise AssertionError("advisor must not run when --model is given")
    monkeypatch.setattr(cli_mod, "recommend", boom_recommend)

    result = CliRunner().invoke(
        cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo), "--model", "review-local"])
    assert result.exit_code == 0


def test_resolve_spec_bare_model_tag_falls_back(monkeypatch, tmp_path):
    """A bare ollama tag (not a config alias) resolves to a direct spec, not KeyError."""
    import kagura_code_reviewer.config as cfgmod
    monkeypatch.setattr(cfgmod, "_USER", tmp_path / "none.toml")  # shipped aliases only
    spec = cli_mod._resolve_spec("qwen3:14b", local=False, cloud=False)
    assert spec.ollama_model == "qwen3:14b"


def test_cli_bare_model_tag_resolves(repo: Path, monkeypatch, tmp_path):
    """--model with a real ollama tag works end-to-end (no alias required)."""
    import kagura_code_reviewer.config as cfgmod
    monkeypatch.setattr(cfgmod, "_USER", tmp_path / "none.toml")
    monkeypatch.setattr(cli_mod, "_USER", tmp_path / "none.toml")
    monkeypatch.setattr(cli_mod.RepoTools, "git_diff", lambda self, b, h, p=None: "DIFF")
    captured = {}

    def spy_factory(spec, timeout, seed=None):
        captured["model"] = spec.ollama_model
        return object()
    monkeypatch.setattr(cli_mod, "client_factory", spy_factory)
    monkeypatch.setattr(cli_mod, "review_harness", lambda *a, **k: Report(findings=[]), raising=False)

    def boom_recommend(*a, **k):
        raise AssertionError("advisor must not run when --model is given")
    monkeypatch.setattr(cli_mod, "recommend", boom_recommend)

    result = CliRunner().invoke(
        cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo), "--model", "qwen3:14b"])
    assert result.exit_code == 0
    assert captured["model"] == "qwen3:14b"


def test_cli_advisor_none_exits_with_guidance(repo: Path, monkeypatch):
    monkeypatch.setattr(cli_mod, "_USER", repo / "no-user-config.toml")  # force zero-config path
    monkeypatch.setattr(cli_mod.RepoTools, "git_diff", lambda self, b, h, p=None: "DIFF")
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout, seed=None: object())
    monkeypatch.setattr(
        cli_mod, "recommend",
        lambda hw, installed, prefer_local=True: Recommendation(None, None, "no suitable local model", False))

    result = CliRunner().invoke(cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo)])
    assert result.exit_code != 0
    assert "no suitable" in result.output


def test_cli_provider_openai_uses_compat_client(repo: Path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli_mod.RepoTools, "git_diff", lambda self, b, h, p=None: "DIFF")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live")

    def spy_build(provider, model, local, cloud, timeout, seed=None, auto=False):
        captured["provider"] = provider
        return object(), "gpt-4o"
    monkeypatch.setattr(cli_mod, "build_review_client", spy_build)
    monkeypatch.setattr(cli_mod, "review_harness", lambda *a, **k: Report(findings=[]), raising=False)

    result = CliRunner().invoke(cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo),
                                              "--provider", "openai"])
    assert result.exit_code == 0
    assert captured["provider"] == "openai"


def test_cli_provider_missing_key_errors(repo: Path, monkeypatch):
    monkeypatch.setattr(cli_mod.RepoTools, "git_diff", lambda self, b, h, p=None: "DIFF")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = CliRunner().invoke(cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo),
                                              "--provider", "openai"])
    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in result.output


def test_cli_seed_threads_to_client(repo: Path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli_mod.RepoTools, "git_diff", lambda self, b, h, p=None: "DIFF")

    def spy_build(provider, model, local, cloud, timeout, seed=None, auto=False):
        captured["seed"] = seed
        return object(), "m"
    monkeypatch.setattr(cli_mod, "build_review_client", spy_build)
    monkeypatch.setattr(cli_mod, "review_harness", lambda *a, **k: Report(findings=[]), raising=False)
    result = CliRunner().invoke(cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo), "--seed", "7"])
    assert result.exit_code == 0 and captured["seed"] == 7


def test_cli_auto_persists_recommendation(repo: Path, monkeypatch, tmp_path):
    import kagura_code_reviewer.config as cfgmod
    target = tmp_path / "u.toml"
    monkeypatch.setattr(cfgmod, "_USER", target)
    monkeypatch.setattr(cli_mod, "_USER", target)  # _no_user_config() -> True
    monkeypatch.setattr(cli_mod.RepoTools, "git_diff", lambda self, b, h, p=None: "DIFF")
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout, seed=None: object())
    monkeypatch.setattr(cli_mod, "review_harness", lambda *a, **k: Report(findings=[]), raising=False)
    result = CliRunner().invoke(cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo), "--auto"])
    assert result.exit_code == 0
    assert target.is_file()
    assert "qwen2.5-coder:7b" in target.read_text()


def test_cli_out_suppresses_stdout(repo: Path, monkeypatch, tmp_path):
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout, seed=None: FakeClient())
    out = tmp_path / "r.md"
    result = CliRunner().invoke(cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo), "--out", str(out)])
    assert "bad" in out.read_text()
    assert "bad" not in result.stdout


def test_cli_concurrency_and_min_confidence_threaded(repo: Path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli_mod.RepoTools, "git_diff", lambda self, b, h, p=None: "DIFF")
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout, seed=None: object())

    def fake_harness(fc, vc, repo_, diff, context, tier, max_iters=12,
                     max_concurrency=1, min_confidence=0.0):
        captured["concurrency"] = max_concurrency
        captured["min_confidence"] = min_confidence
        return Report(findings=[])
    monkeypatch.setattr(cli_mod, "review_harness", fake_harness, raising=False)

    result = CliRunner().invoke(cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo),
                                              "--concurrency", "4", "--min-confidence", "0.7"])
    assert result.exit_code == 0
    assert captured["concurrency"] == 4 and captured["min_confidence"] == 0.7


def test_cli_pr_conflicts_with_base(repo: Path, monkeypatch):
    """--pr is mutually exclusive with --base/--head/--repo (gate1/CTO note)."""
    def boom(*a, **k):
        raise AssertionError("resolve_pr must not run when flags conflict")
    monkeypatch.setattr(cli_mod, "resolve_pr", boom)
    result = CliRunner().invoke(
        cli_mod.app,
        ["--pr", "https://github.com/o/r/pull/1", "--base", "HEAD~1"],
    )
    assert result.exit_code != 0
    assert "--pr" in result.output and "cannot be combined" in result.output


def test_cli_pr_invokes_resolver_and_cleans_up(repo: Path, monkeypatch):
    """--pr resolves a DiffSource, reviews its worktree, and always cleans up."""
    from kagura_code_reviewer.pr_source import DiffSource

    cleaned = {"v": False}

    def fake_resolve(url, *, keep=False):
        return DiffSource(
            repo_root=repo, base="HEAD~1", head="HEAD",
            cleanup=lambda: cleaned.__setitem__("v", True),
        )
    monkeypatch.setattr(cli_mod, "resolve_pr", fake_resolve)
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout, seed=None: FakeClient())

    result = CliRunner().invoke(
        cli_mod.app, ["--pr", "https://github.com/o/r/pull/1"])
    assert result.exit_code == 1          # FakeClient emits a blocking finding
    assert "bad" in result.stdout
    assert cleaned["v"] is True           # cleanup ran even though review exited nonzero


def test_cli_pr_cleans_up_on_harness_error(repo: Path, monkeypatch):
    """cleanup must run even if the review raises (finally path)."""
    from kagura_code_reviewer.pr_source import DiffSource

    cleaned = {"v": False}
    monkeypatch.setattr(
        cli_mod, "resolve_pr",
        lambda url, *, keep=False: DiffSource(
            repo_root=repo, base="HEAD~1", head="HEAD",
            cleanup=lambda: cleaned.__setitem__("v", True)),
    )
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout, seed=None: object())

    def boom_harness(*a, **k):
        raise ConnectionError("backend down")
    monkeypatch.setattr(cli_mod, "review_harness", boom_harness, raising=False)

    result = CliRunner().invoke(
        cli_mod.app, ["--pr", "https://github.com/o/r/pull/1"])
    assert result.exit_code == 3
    assert cleaned["v"] is True


def test_cli_version_flag_prints_version_and_exits_zero():
    from kagura_code_reviewer import __version__

    result = CliRunner().invoke(cli_mod.app, ["--version"])
    assert result.exit_code == 0
    # Pin the full "<prog> <version>" line: the _version_callback(prog) factory's
    # whole job is the per-CLI program-name prefix, so assert it, not a substring.
    assert result.output.strip() == f"kagura-code-reviewer {__version__}"


def test_cli_version_flag_is_eager():
    from kagura_code_reviewer import __version__

    # --version short-circuits before any review work: a bad base ref must NOT be
    # reached (eager callback runs first), so it exits 0 AND prints the version
    # rather than failing on the ref. Asserting the output proves the short-circuit.
    result = CliRunner().invoke(
        cli_mod.app, ["--base", "nonexistent-ref-xyz", "--version"]
    )
    assert result.exit_code == 0
    assert result.output.strip() == f"kagura-code-reviewer {__version__}"
