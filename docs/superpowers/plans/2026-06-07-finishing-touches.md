# Finishing Touches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans or subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Report provenance, determinism mode, advisor persistence, and residual v1 robustness fixes — the "perfectly finished" pass.

**Architecture:** Small, localized changes across `report.py`, `providers/openai_compat.py`, `config.py`, `cli.py`, `tools.py`, `doctor.py`. Each is independently TDD'd; all existing 94 tests stay green.

**Spec:** `docs/superpowers/specs/2026-06-07-finishing-touches-design.md`

---

## Task 1: Report provenance (md + json)

**Files:** Modify `src/kagura_code_reviewer/report.py`; Test `tests/test_report.py`

- [ ] **Step 1: Failing tests**

```python
# append to tests/test_report.py
def test_markdown_shows_provenance_when_present():
    f = Finding("correctness", Severity.HIGH, "a.py", 2, "bug", "r", "s",
                angles=["cross-file", "correctness-linescan"],
                votes={"CONFIRMED": 2}, merge_count=2)
    md = Report(findings=[f]).to_markdown()
    assert "cross-file" in md and "CONFIRMED" in md


def test_markdown_omits_provenance_when_absent():
    f = Finding("perf", Severity.LOW, "a.py", 1, "t", "r", "s")
    md = Report(findings=[f]).to_markdown()
    assert "Seen by" not in md


def test_json_includes_provenance():
    f = Finding("correctness", Severity.HIGH, "a.py", 2, "bug", "r", "s",
                angles=["reuse"], votes={"PLAUSIBLE": 1}, merge_count=3)
    import json
    d = json.loads(Report(findings=[f]).to_json())["findings"][0]
    assert d["angles"] == ["reuse"] and d["votes"] == {"PLAUSIBLE": 1} and d["merge_count"] == 3
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_report.py -q` → FAIL.

- [ ] **Step 3: Implement** — in `report.py`:

`to_dict()` add the three fields:
```python
            "suggestion": self.suggestion,
            "angles": self.angles,
            "votes": self.votes,
            "merge_count": self.merge_count,
```

In `to_markdown()`, after the `- **Fix:**` line for each finding, add:
```python
            if f.angles or f.votes or f.merge_count > 1:
                seen = ", ".join(f.angles) if f.angles else "—"
                count = f" ×{f.merge_count}" if f.merge_count > 1 else ""
                votes = ("; votes: " + ", ".join(f"{k} {v}" for k, v in f.votes.items())) if f.votes else ""
                lines.append(f"- **Seen by:** {seen}{count}{votes}")
```
(Insert the provenance line before the trailing newline entry so it stays within the finding block.)

- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_report.py -q` → PASS.

- [ ] **Step 5: Commit** `feat: surface finding provenance in md/json reports`

---

## Task 2: Determinism `--seed`

**Files:** Modify `providers/openai_compat.py`, `cli.py`; Test `tests/test_providers.py`, `tests/test_cli.py`

- [ ] **Step 1: Failing test**

```python
# append to tests/test_providers.py
def test_compat_client_includes_seed_when_set(httpserver: HTTPServer):
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {"choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": None}}]}
    )
    client = OpenAICompatClient(base_url=httpserver.url_for("/v1"), model="qwen",
                                api_key="ollama", timeout=5.0, seed=42)
    client.chat([{"role": "user", "content": "hi"}])
    body = json.loads(httpserver.log[0][0].get_data())
    assert body["seed"] == 42
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** — `OpenAICompatClient.__init__` add `seed: int | None = None`; store `self.seed = seed`. In `chat()` after temperature:
```python
        if self.seed is not None:
            kwargs["seed"] = self.seed
```

- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_providers.py -q` → PASS.

- [ ] **Step 5: CLI wiring** — add `seed: int = typer.Option(None, "--seed", help="Seed for reproducible local reviews.")` to `main`; thread it: change `build_review_client(provider, model, local, cloud, timeout, seed=None)` and pass `seed` into `OpenAICompatClient`/`OllamaClient` (ollama path: `client_factory(spec, timeout, seed)`). Update `client_factory` to accept and forward `seed`. Anthropic ignores it.
  Failing CLI test:
```python
# append to tests/test_cli.py
def test_cli_seed_threads_to_client(repo: Path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli_mod.RepoTools, "git_diff", lambda self, b, h, p=None: "DIFF")
    def spy_build(provider, model, local, cloud, timeout, seed=None):
        captured["seed"] = seed
        return object(), "m"
    monkeypatch.setattr(cli_mod, "build_review_client", spy_build)
    monkeypatch.setattr(cli_mod, "review_harness", lambda *a, **k: Report(findings=[]), raising=False)
    result = CliRunner().invoke(cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo), "--seed", "7"])
    assert result.exit_code == 0 and captured["seed"] == 7
```

- [ ] **Step 6: Run** `.venv/bin/pytest tests/test_cli.py tests/test_providers.py -q` → PASS.

- [ ] **Step 7: Commit** `feat: --seed for reproducible local reviews`

---

## Task 3: Advisor persistence (`write_user_model` + `--auto`)

**Files:** Modify `config.py`, `cli.py`; Test `tests/test_config.py`, `tests/test_cli.py`

- [ ] **Step 1: Failing test (config)**

```python
# append to tests/test_config.py
def test_write_user_model_writes_default_alias(tmp_path, monkeypatch):
    import kagura_code_reviewer.config as cfgmod
    target = tmp_path / "cfg.toml"
    monkeypatch.setattr(cfgmod, "_USER", target)
    cfgmod.write_user_model("qwen3.5:27b", "http://localhost:11434/v1", 32768)
    text = target.read_text(encoding="utf-8")
    assert 'default_alias = "auto"' in text
    assert "qwen3.5:27b" in text
    # round-trips through load_config override
    cfg = cfgmod.load_config()
    assert cfg["models"]["auto"]["ollama_model"] == "qwen3.5:27b"
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** — in `config.py`:
```python
def write_user_model(name: str, base_url: str, num_ctx: int, path: Path | None = None) -> Path:
    target = path or _USER
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f'default_alias = "auto"\n\n[models.auto]\n'
        f'ollama_model = "{name}"\nbase_url = "{base_url}"\nnum_ctx = {int(num_ctx)}\n',
        encoding="utf-8",
    )
    return target
```

- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_config.py -q` → PASS.

- [ ] **Step 5: CLI `--auto`** — add `auto: bool = typer.Option(False, "--auto", help="Persist the advisor's model pick to user config.")`. In the ollama branch of `build_review_client`, when `auto` is set and the advisor picked a model, call `write_user_model(...)` and echo the path. Thread `auto` into `build_review_client(..., auto=False)`. (Simplest: only persist on the zero-config/advisor path; if `--model`/user-config already set, `--auto` is a no-op with a note.)
  Failing CLI test:
```python
# append to tests/test_cli.py
def test_cli_auto_persists_recommendation(repo: Path, monkeypatch, tmp_path):
    import kagura_code_reviewer.config as cfgmod
    monkeypatch.setattr(cfgmod, "_USER", tmp_path / "u.toml")
    monkeypatch.setattr(cli_mod.RepoTools, "git_diff", lambda self, b, h, p=None: "DIFF")
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout, seed=None: object())
    monkeypatch.setattr(cli_mod, "review_harness", lambda *a, **k: Report(findings=[]), raising=False)
    # _stub_advisor autouse returns qwen2.5-coder:7b
    result = CliRunner().invoke(cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo), "--auto"])
    assert result.exit_code == 0
    assert (tmp_path / "u.toml").is_file()
    assert "qwen2.5-coder:7b" in (tmp_path / "u.toml").read_text()
```
  (Note: `_no_user_config()` reads `config._USER`; the autouse advisor stub + the `--auto` write must use the same monkeypatched `_USER`. In `cli.py`, reference `config._USER`/`config.write_user_model` via the module so the monkeypatch on `cfgmod._USER` is seen — import `from . import config as _config` and call `_config.write_user_model`, `_config._USER`.)

- [ ] **Step 6: Run** `.venv/bin/pytest tests/test_cli.py -q` → PASS.

- [ ] **Step 7: Commit** `feat: --auto persists advisor model pick to user config`

---

## Task 4: Residual robustness fixes (one test+fix per item, one commit)

**Files:** `providers/openai_compat.py`, `tools.py`, `doctor.py`, `cli.py`, `config.py`; Tests in the matching test files.

- [ ] **4a — empty choices guard.** Test (`tests/test_providers.py`): a fake-served response with `{"choices": []}` → `client.chat(...)` returns `ChatMessage(content="" or None, tool_calls=[])`, no exception. Fix: in `chat()`, `if not resp.choices: return ChatMessage(content="", tool_calls=[])` before indexing.

```python
def test_compat_client_handles_empty_choices(httpserver: HTTPServer):
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json({"choices": []})
    client = OpenAICompatClient(base_url=httpserver.url_for("/v1"), model="m", api_key="k", timeout=5.0)
    msg = client.chat([{"role": "user", "content": "hi"}])
    assert msg.tool_calls == [] and (msg.content == "" or msg.content is None)
```

- [ ] **4b — list_files flag injection.** Test (`tests/test_tools.py`): monkeypatch `RepoTools._git` to capture args; `list_files("--deleted")` → `_git` called with `("ls-files", "--", "--deleted")`. Fix: `tools.py` `list_files` → `self._git("ls-files", "--", subdir)`.

- [ ] **4c — read_file truncation marker.** Test: write a file > max_bytes; `read_file(path, max_bytes=10)` ends with `...[truncated]`. Fix: in `read_file`, `data = target.read_text(errors="replace")`; `if len(data) > max_bytes: return data[:max_bytes] + "\n...[truncated]"`; else return data.

- [ ] **4d — grep cap marker.** Test: create files with > max_results matches; `grep(pattern, max_results=2)` last line == `...[more matches hidden]`. Fix: when `len(hits) >= max_results`, append the marker before returning.

- [ ] **4e — doctor trailing slash.** Test (`tests/test_doctor.py`): `_ollama_root("http://h:1/v1/") == "http://h:1"`. Fix: `b = base_url.rstrip("/"); return b[:-3] if b.endswith("/v1") else b`.

- [ ] **4f — `--out` suppresses stdout.** Test (`tests/test_cli.py`): with `--out file`, `result.stdout` does not contain the report body (e.g. "No changes"/findings). Fix: `cli.py` final block → `if out: out.write_text(rendered)` else `typer.echo(rendered)`.

- [ ] **4g — config utf-8.** Fix only (covered by existing config tests): `config.py` both `read_text()` → `read_text(encoding="utf-8")`. Add a test that a user config containing a non-ASCII comment loads without error.

For each of 4a–4g: write the test, run it (FAIL), implement, run (PASS).

- [ ] **Final: full suite + commit.** Run `.venv/bin/pytest -q` → all green. Commit `fix: close residual v1 robustness issues (empty choices, ls-files injection, silent truncation, doctor slash, --out echo, utf-8)`.

---

## Self-Review
- Spec A→Task 1; B→Task 2; C→Task 3; D(1-7)→Task 4a-4g.
- No placeholders; each step has test + fix.
- `build_review_client`/`client_factory` signature change (add `seed`, `auto`) applied consistently in Tasks 2 & 3 and their CLI tests.
- All existing 94 tests must remain green (verified in the Final step).
