# Multi-Provider Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenAI, Anthropic, and Gemini as review backends behind the existing `ChatClient` seam, with free-local Ollama as the default and API keys read from env vars only.

**Architecture:** A `providers/` package. `OpenAICompatClient` generalizes `OllamaClient` (Ollama/OpenAI/Gemini via base_url + key). `AnthropicClient` wraps the `anthropic` SDK with pure OpenAI↔Anthropic adapter functions. `cli.py` gains `--provider`; `client_factory` dispatches by provider kind. `OllamaClient` becomes a thin subclass so the existing 83 tests pass unchanged.

**Tech Stack:** Python 3.11, `openai` SDK (existing), `anthropic` SDK (new), pytest + pytest-httpserver + Typer CliRunner.

**Spec:** `docs/superpowers/specs/2026-06-07-multiprovider-design.md`

---

## File Structure

- **Create** `src/kagura_code_reviewer/providers/__init__.py`
- **Create** `src/kagura_code_reviewer/providers/openai_compat.py` — `OpenAICompatClient`
- **Create** `src/kagura_code_reviewer/providers/anthropic_client.py` — adapter funcs + `AnthropicClient`
- **Modify** `src/kagura_code_reviewer/ollama_client.py` — `OllamaClient` becomes a subclass of `OpenAICompatClient`
- **Modify** `src/kagura_code_reviewer/config.toml` — `[providers.*]` tables
- **Modify** `src/kagura_code_reviewer/cli.py` — `--provider`, provider-aware client building
- **Modify** `pyproject.toml` — add `anthropic` dependency
- **Tests** `tests/test_providers.py` (new); additions to `tests/test_cli.py`

---

## Task 1: `OpenAICompatClient` (generalize OllamaClient)

**Files:**
- Create: `src/kagura_code_reviewer/providers/__init__.py` (empty)
- Create: `src/kagura_code_reviewer/providers/openai_compat.py`
- Modify: `src/kagura_code_reviewer/ollama_client.py`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers.py
import json
from pytest_httpserver import HTTPServer
from kagura_code_reviewer.providers.openai_compat import OpenAICompatClient


def test_compat_client_sends_auth_header_and_no_num_ctx_by_default(httpserver: HTTPServer):
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {"choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": None}}]}
    )
    client = OpenAICompatClient(base_url=httpserver.url_for("/v1"), model="gpt-4o",
                                api_key="sk-test", timeout=5.0)
    msg = client.chat([{"role": "user", "content": "hello"}])
    assert msg.content == "hi"
    req = httpserver.log[0][0]
    assert req.headers["Authorization"] == "Bearer sk-test"
    body = json.loads(req.get_data())
    assert "extra_body" not in body and "options" not in body  # num_ctx omitted for non-ollama
    assert body["temperature"] == 0


def test_compat_client_sends_num_ctx_when_set(httpserver: HTTPServer):
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {"choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": None}}]}
    )
    client = OpenAICompatClient(base_url=httpserver.url_for("/v1"), model="qwen",
                                api_key="ollama", timeout=5.0, num_ctx=16384)
    client.chat([{"role": "user", "content": "hello"}])
    body = json.loads(httpserver.log[0][0].get_data())
    assert body["options"]["num_ctx"] == 16384
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_providers.py -q`
Expected: FAIL (`ModuleNotFoundError: ...providers.openai_compat`).

- [ ] **Step 3: Write minimal implementation**

`src/kagura_code_reviewer/providers/__init__.py`: empty file.

`src/kagura_code_reviewer/providers/openai_compat.py`:

```python
from __future__ import annotations

from openai import OpenAI

from ..agent import ChatMessage, ToolCall


class OpenAICompatClient:
    """OpenAI-compatible chat client (Ollama, OpenAI, Gemini) over base_url + key."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 120.0,
        temperature: float = 0.0,
        num_ctx: int | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatMessage:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.num_ctx is not None:
            kwargs["extra_body"] = {"options": {"num_ctx": self.num_ctx}}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            )
        return ChatMessage(content=msg.content, tool_calls=tool_calls)
```

Replace `src/kagura_code_reviewer/ollama_client.py` with a thin subclass:

```python
from __future__ import annotations

from .providers.openai_compat import OpenAICompatClient


class OllamaClient(OpenAICompatClient):
    """OpenAI-compatible client pointed at an Ollama endpoint (free local default)."""

    def __init__(self, base_url, model, num_ctx=8192, timeout=120.0,
                 api_key="ollama", temperature=0.0):
        super().__init__(base_url=base_url, model=model, api_key=api_key,
                         timeout=timeout, temperature=temperature, num_ctx=num_ctx)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_providers.py tests/test_ollama_client.py -q`
Expected: PASS (new provider tests + existing Ollama tests via the subclass).

- [ ] **Step 5: Commit**

```bash
git add src/kagura_code_reviewer/providers/ src/kagura_code_reviewer/ollama_client.py tests/test_providers.py
git commit -m "feat: OpenAICompatClient generalizes OllamaClient (num_ctx opt-in)"
```

---

## Task 2: Anthropic adapter — `to_anthropic_tools`

**Files:**
- Create: `src/kagura_code_reviewer/providers/anthropic_client.py`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_providers.py
from kagura_code_reviewer.providers.anthropic_client import to_anthropic_tools


def test_to_anthropic_tools_converts_schema():
    openai_tools = [{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                           "required": ["path"]},
        },
    }]
    out = to_anthropic_tools(openai_tools)
    assert out == [{
        "name": "read_file",
        "description": "Read a file.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                         "required": ["path"]},
    }]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_providers.py::test_to_anthropic_tools_converts_schema -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

Create `src/kagura_code_reviewer/providers/anthropic_client.py`:

```python
from __future__ import annotations

import json

from ..agent import ChatMessage, ToolCall


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_providers.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kagura_code_reviewer/providers/anthropic_client.py tests/test_providers.py
git commit -m "feat: to_anthropic_tools adapter (OpenAI tool schema -> input_schema)"
```

---

## Task 3: Anthropic adapter — `to_anthropic_system_and_messages`

**Files:**
- Modify: `src/kagura_code_reviewer/providers/anthropic_client.py`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_providers.py
from kagura_code_reviewer.providers.anthropic_client import to_anthropic_system_and_messages


def test_to_anthropic_messages_extracts_system_and_converts_tool_turns():
    messages = [
        {"role": "system", "content": "You review code."},
        {"role": "user", "content": "review this"},
        {"role": "assistant", "content": "ok",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "file body"},
    ]
    system, msgs = to_anthropic_system_and_messages(messages)
    assert system == "You review code."
    assert msgs[0] == {"role": "user", "content": "review this"}
    # assistant turn carries a tool_use block with parsed input
    assert msgs[1]["role"] == "assistant"
    tool_use = [b for b in msgs[1]["content"] if b["type"] == "tool_use"][0]
    assert tool_use == {"type": "tool_use", "id": "c1", "name": "read_file",
                        "input": {"path": "a.py"}}
    # tool result becomes a user message with a tool_result block
    assert msgs[2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1", "content": "file body"}]}


def test_to_anthropic_messages_merges_consecutive_tool_results():
    messages = [
        {"role": "assistant", "content": "",
         "tool_calls": [
             {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
             {"id": "c2", "type": "function", "function": {"name": "g", "arguments": "{}"}},
         ]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
    ]
    _, msgs = to_anthropic_system_and_messages(messages)
    assert msgs[-1] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1", "content": "r1"},
        {"type": "tool_result", "tool_use_id": "c2", "content": "r2"},
    ]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_providers.py::test_to_anthropic_messages_extracts_system_and_converts_tool_turns -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

Add to `anthropic_client.py`:

```python
def to_anthropic_system_and_messages(messages: list[dict]):
    system_parts: list[str] = []
    out: list[dict] = []
    pending_tool_results: list[dict] = []

    def flush_tool_results():
        if pending_tool_results:
            out.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for m in messages:
        role = m.get("role")
        if role == "system":
            if m.get("content"):
                system_parts.append(m["content"])
            continue
        if role == "tool":
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": m.get("content", ""),
            })
            continue
        flush_tool_results()
        if role == "assistant":
            blocks: list[dict] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls", []):
                fn = tc["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                blocks.append({"type": "tool_use", "id": tc["id"],
                               "name": fn["name"], "input": args})
            out.append({"role": "assistant", "content": blocks})
        else:  # user
            out.append({"role": "user", "content": m.get("content", "")})
    flush_tool_results()
    return "\n".join(system_parts), out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_providers.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kagura_code_reviewer/providers/anthropic_client.py tests/test_providers.py
git commit -m "feat: to_anthropic_system_and_messages adapter (system + tool turns)"
```

---

## Task 4: Anthropic adapter — `from_anthropic_response`

**Files:**
- Modify: `src/kagura_code_reviewer/providers/anthropic_client.py`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_providers.py
from types import SimpleNamespace
from kagura_code_reviewer.providers.anthropic_client import from_anthropic_response


def test_from_anthropic_response_maps_text_and_tool_use():
    resp = SimpleNamespace(content=[
        SimpleNamespace(type="text", text="some text"),
        SimpleNamespace(type="tool_use", id="c1", name="read_file", input={"path": "a.py"}),
    ])
    msg = from_anthropic_response(resp)
    assert msg.content == "some text"
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "c1"
    assert msg.tool_calls[0].name == "read_file"
    assert msg.tool_calls[0].arguments == '{"path": "a.py"}'


def test_from_anthropic_response_no_blocks():
    resp = SimpleNamespace(content=[])
    msg = from_anthropic_response(resp)
    assert msg.content == "" and msg.tool_calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_providers.py::test_from_anthropic_response_maps_text_and_tool_use -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

Add to `anthropic_client.py`:

```python
def from_anthropic_response(resp) -> ChatMessage:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in getattr(resp, "content", []) or []:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name,
                                       arguments=json.dumps(block.input)))
    return ChatMessage(content="".join(text_parts), tool_calls=tool_calls)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_providers.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kagura_code_reviewer/providers/anthropic_client.py tests/test_providers.py
git commit -m "feat: from_anthropic_response adapter (blocks -> ChatMessage)"
```

---

## Task 5: `AnthropicClient.chat` (injected client)

**Files:**
- Modify: `src/kagura_code_reviewer/providers/anthropic_client.py`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_providers.py
from kagura_code_reviewer.providers.anthropic_client import AnthropicClient


class FakeAnthropic:
    def __init__(self):
        self.captured = {}
        self.messages = self  # so .messages.create works

    def create(self, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="done")])


def test_anthropic_client_chat_builds_request_without_temperature():
    fake = FakeAnthropic()
    client = AnthropicClient(model="claude-sonnet-4-6", api_key="x",
                             max_tokens=2048, client=fake)
    msg = client.chat(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f", "description": "d",
                                                 "parameters": {"type": "object"}}}],
    )
    assert msg.content == "done"
    assert fake.captured["model"] == "claude-sonnet-4-6"
    assert fake.captured["max_tokens"] == 2048
    assert fake.captured["system"] == "sys"
    assert "temperature" not in fake.captured          # Opus 4.8/4.7 reject it
    assert fake.captured["tools"][0]["name"] == "f"
    assert fake.captured["tool_choice"] == {"type": "auto"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_providers.py::test_anthropic_client_chat_builds_request_without_temperature -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

Add to `anthropic_client.py`:

```python
def AnthropicClient_import_guard():  # placeholder removed below
    pass
```

Replace with the real class (add near the top of the file the lazy SDK import inside `__init__` so tests that inject `client=` never need the SDK installed):

```python
class AnthropicClient:
    """Anthropic Messages API client implementing the ChatClient protocol."""

    def __init__(self, model: str, api_key: str = "", max_tokens: int = 4096,
                 timeout: float = 120.0, client=None):
        self.model = model
        self.max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            from anthropic import Anthropic  # lazy: only needed for the real path
            self._client = Anthropic(api_key=api_key, timeout=timeout)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatMessage:
        system, msgs = to_anthropic_system_and_messages(messages)
        kwargs: dict = {"model": self.model, "max_tokens": self.max_tokens, "messages": msgs}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
            kwargs["tool_choice"] = {"type": "auto"}
        resp = self._client.messages.create(**kwargs)
        return from_anthropic_response(resp)
```

(Delete the placeholder function — it exists only to make Step 2 fail cleanly if you ran it before writing the class.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_providers.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kagura_code_reviewer/providers/anthropic_client.py tests/test_providers.py
git commit -m "feat: AnthropicClient.chat (system/tools/tool_result round-trip, no temperature)"
```

---

## Task 6: Provider config tables

**Files:**
- Modify: `src/kagura_code_reviewer/config.toml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_config.py
def test_shipped_config_has_providers():
    from kagura_code_reviewer.config import load_config
    cfg = load_config()
    assert set(cfg["providers"]) >= {"ollama", "openai", "anthropic", "gemini"}
    assert cfg["providers"]["anthropic"]["kind"] == "anthropic"
    assert cfg["providers"]["openai"]["api_key_env"] == "OPENAI_API_KEY"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py::test_shipped_config_has_providers -q`
Expected: FAIL (`KeyError: 'providers'`).

- [ ] **Step 3: Write minimal implementation**

Append to `config.toml`:

```toml
[providers.ollama]
kind = "openai_compat"
base_url = "http://localhost:11434/v1"
api_key_env = ""

[providers.openai]
kind = "openai_compat"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
default_model = "gpt-4o"

[providers.gemini]
kind = "openai_compat"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
api_key_env = "GEMINI_API_KEY"
default_model = "gemini-2.0-flash"

[providers.anthropic]
kind = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
default_model = "claude-sonnet-4-6"
max_tokens = 4096
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kagura_code_reviewer/config.toml tests/test_config.py
git commit -m "feat: ship provider tables in config.toml"
```

---

## Task 7: CLI `--provider` + provider-aware client building

**Files:**
- Modify: `src/kagura_code_reviewer/cli.py`
- Modify: `pyproject.toml` (add `anthropic`)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_cli.py
def test_cli_provider_openai_uses_compat_client(repo: Path, monkeypatch):
    from kagura_code_reviewer.report import Report
    captured = {}
    monkeypatch.setattr(cli_mod.RepoTools, "git_diff", lambda self, b, h, p=None: "DIFF")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live")

    def spy_build(provider, model, local, cloud, timeout):
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
```

(Keep the `_stub_advisor` autouse fixture from sub-project 2 — the default
`--provider ollama` path still goes through the advisor.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py::test_cli_provider_openai_uses_compat_client -q`
Expected: FAIL (`--provider` unknown / `build_review_client` absent).

- [ ] **Step 3: Write minimal implementation**

In `pyproject.toml`, add `"anthropic>=0.40"` to `dependencies`.

In `cli.py`:

1. Imports:
```python
import os
from .providers.openai_compat import OpenAICompatClient
from .providers.anthropic_client import AnthropicClient
```

2. Add the enum near `Effort`:
```python
class Provider(str, Enum):
    ollama = "ollama"
    openai = "openai"
    anthropic = "anthropic"
    gemini = "gemini"
```

3. Add `build_review_client` at module level:
```python
def build_review_client(provider: str, model: str | None, local: bool, cloud: bool, timeout: float):
    """Return (client, model_label). Ollama keeps the advisor default path."""
    if provider == "ollama":
        spec = _resolve_spec(model, local, cloud)
        return client_factory(spec, timeout), spec.ollama_model
    cfg = load_config().get("providers", {}).get(provider, {})
    key_env = cfg.get("api_key_env", "")
    api_key = os.environ.get(key_env, "") if key_env else ""
    if key_env and not api_key:
        typer.echo(f"Provider '{provider}' needs ${key_env} set (or use --provider ollama).", err=True)
        raise typer.Exit(code=2)
    chosen = model or cfg.get("default_model")
    if not chosen:
        typer.echo(f"No model for provider '{provider}'; pass --model.", err=True)
        raise typer.Exit(code=2)
    if cfg.get("kind") == "anthropic":
        return AnthropicClient(model=chosen, api_key=api_key,
                               max_tokens=int(cfg.get("max_tokens", 4096)),
                               timeout=timeout), chosen
    return OpenAICompatClient(base_url=cfg["base_url"], model=chosen,
                              api_key=api_key, timeout=timeout), chosen
```

4. Add the option to `main(...)` (after `cloud`):
```python
    provider: Provider = typer.Option(Provider.ollama, "--provider", help="Review backend: ollama|openai|anthropic|gemini."),
```

5. Replace the spec/client block. Where `main` currently does
   `spec = _resolve_spec(model, local, cloud)` (non-doctor path) and later
   `client = client_factory(spec, timeout)`, replace both with a single call
   after the diff is obtained:
```python
    client, _model_label = build_review_client(provider.value, model, local, cloud, timeout)
```
   Remove the now-redundant `spec = _resolve_spec(...)` line and the
   `client = client_factory(spec, timeout)` line from the review path. (The
   `--doctor` branch keeps its own hardware/recommendation block from
   sub-project 2 — leave it unchanged; it does not use `build_review_client`.)
   The `review_harness(client, client, ...)` call is unchanged (finder ==
   verifier client).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cli.py -q`
Expected: PASS (new provider tests + all pre-existing CLI tests).

- [ ] **Step 5: Commit**

```bash
git add src/kagura_code_reviewer/cli.py pyproject.toml tests/test_cli.py
git commit -m "feat: --provider flag with env-only keys (ollama default, openai/anthropic/gemini opt-in)"
```

---

## Task 8: Install dep, full suite, smoke

**Files:** none (verification)

- [ ] **Step 1: Install the new dependency**

Run: `.venv/bin/pip install -e . -q`
Expected: `anthropic` installed; no errors.

- [ ] **Step 2: Full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS (83 prior + new provider/config/cli tests).

- [ ] **Step 3: CLI smoke (offline — verifies wiring + key gate)**

Run: `.venv/bin/kagura-code-reviewer --base main --provider openai`
Expected (no `OPENAI_API_KEY` set): non-zero exit with
"Provider 'openai' needs $OPENAI_API_KEY set (or use --provider ollama)."

Run: `.venv/bin/kagura-code-reviewer --doctor`
Expected: unchanged hardware + recommendation output (ollama path intact).

- [ ] **Step 4: Commit (only if a verification fix was needed)**

```bash
git add -A && git commit -m "chore: multi-provider wiring verified"
```

---

## Self-Review

**Spec coverage:**
- OpenAI/Gemini via OpenAICompatClient → Task 1 (+ Task 6 base_urls, Task 7 dispatch).
- Anthropic via SDK + adapter → Tasks 2–5.
- Free-local Ollama default + advisor preserved → Task 7 (`build_review_client` ollama branch) + `_stub_advisor` fixture.
- `--provider` opt-in → Task 7.
- Keys from env only, friendly missing-key error → Task 7 + Task 8 Step 3.
- `OllamaClient` alias keeps 83 tests green → Task 1.
- `anthropic` dependency → Task 7/8.
- Anthropic adapter pure + network-free tests → Tasks 2–5 (injected client, SimpleNamespace).

**Placeholder scan:** No TBD/TODO. Task 5 Step 3 explicitly instructs deleting
the throwaway `AnthropicClient_import_guard` placeholder once the class is in.

**Type consistency:** `OpenAICompatClient(base_url, model, api_key, timeout,
temperature, num_ctx)`, `OllamaClient(base_url, model, num_ctx, timeout,
api_key, temperature)`, `to_anthropic_tools(tools)->list[dict]`,
`to_anthropic_system_and_messages(messages)->(str, list[dict])`,
`from_anthropic_response(resp)->ChatMessage`, `AnthropicClient(model, api_key,
max_tokens, timeout, client)`, `build_review_client(provider, model, local,
cloud, timeout)->(client, model_label)` are consistent across tasks. All clients
satisfy the existing `ChatClient.chat(messages, tools)->ChatMessage` protocol.
