# kagura-code-reviewer v2 — Multi-Provider Backends Design Spec

**Date:** 2026-06-07
**Status:** Draft (awaiting user review)
**Repo:** `~/works/kagura-code-review` (package `kagura_code_reviewer`)
**Branch:** `feat/v2-multiprovider`
**Scope:** Sub-project (4) of the v2 effort.

---

## 1. Overview

Today the review brain is Ollama-only. The `ChatClient` protocol in `agent.py`
(`chat(messages, tools) -> ChatMessage`) is already the seam the harness,
finders, and verifiers depend on — none of them know about Ollama. This
sub-project adds OpenAI, Anthropic, and Gemini as alternative backends behind
that same seam.

**The core principle stays intact:** free local Ollama remains the zero-config
default (with the advisor). Paid providers are an explicit opt-in. This
generalizes the advisor's "free local default / cloud opt-in" posture to
"free local default / any provider opt-in."

### Goals
1. Add OpenAI, Anthropic, Gemini as selectable review backends.
2. Keep the harness/finders/verifiers untouched — providers implement the
   existing `ChatClient` protocol.
3. **Default stays free-local Ollama**; providers are explicit (`--provider`).
4. **API keys come from environment variables only** — never stored in config,
   memory, or anywhere persistent (OWASP/secret-hygiene).

### Non-goals (this sub-project)
- Streaming responses (harness is non-streaming).
- Cross-provider auto-selection in the advisor (advisor stays local-focused;
  providers are chosen explicitly).
- Cost estimation/telemetry (future).
- Provider-specific features (Anthropic thinking, server tools, caching).

---

## 2. Architecture

```
                    ChatClient protocol (agent.py) — unchanged seam
                                   ▲
        ┌──────────────────────────┼───────────────────────────┐
        │                          │                            │
 OpenAICompatClient          AnthropicClient            (OpenAICompatClient
 (providers/openai_compat)   (providers/anthropic)        reused for Gemini)
   Ollama / OpenAI / Gemini    anthropic SDK +
   via base_url + api_key      OpenAI<->Anthropic adapter
```

### 2.1 `providers/` package

- **`openai_compat.py` — `OpenAICompatClient`** — generalize the current
  `OllamaClient` into one client parameterized by `base_url`, `model`,
  `api_key`, `num_ctx`, `timeout`, `temperature`. Covers three providers:
  | provider | base_url | api_key source | temperature |
  |---|---|---|---|
  | ollama | `http://localhost:11434/v1` | `"ollama"` (dummy) | `0.0` |
  | openai | `https://api.openai.com/v1` | `OPENAI_API_KEY` | `0.0` |
  | gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `GEMINI_API_KEY` | `0.0` |
  `OllamaClient` becomes a thin alias/subclass of `OpenAICompatClient` so the
  existing 83 tests keep passing.

- **`anthropic_client.py` — `AnthropicClient`** — uses the `anthropic` SDK. The
  Anthropic Messages API differs from OpenAI in four ways the adapter bridges
  (verified against the Anthropic API reference, 2026-06-07):
  1. **System prompt is a top-level `system=` param**, not a message — extract
     `role: "system"` messages out of the list and concatenate into `system`.
  2. **Tools are `{name, description, input_schema}`** (not nested under
     `function`) — convert from the harness's OpenAI tool schema.
  3. **Assistant tool calls return as `tool_use` content blocks** with `id`,
     `name`, `input` (already a dict) — map to `ToolCall(id, name,
     arguments=json.dumps(input))`.
  4. **Tool results go back as a `user` message with a `tool_result` block**
     (`{type, tool_use_id, content}`) — convert the harness's
     `{role: "tool", tool_call_id, content}` messages.
  Also: `max_tokens` is required (config value, default 4096); **do not send
  `temperature`** (Opus 4.8/4.7 reject it with a 400).

  The four conversions are **pure functions** —
  `to_anthropic_system_and_messages(messages)`, `to_anthropic_tools(tools)`,
  `from_anthropic_response(resp) -> ChatMessage` — unit-tested without network.

### 2.2 Provider config (`config.toml`)

```toml
[providers.ollama]
kind = "openai_compat"
base_url = "http://localhost:11434/v1"
api_key_env = ""                  # local: dummy key

[providers.openai]
kind = "openai_compat"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
default_model = "gpt-4o"          # user-overridable via --model

[providers.gemini]
kind = "openai_compat"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
api_key_env = "GEMINI_API_KEY"
default_model = "gemini-2.0-flash"

[providers.anthropic]
kind = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
default_model = "claude-sonnet-4-6"   # balanced reviewer default; opus-4-8 available via --model
max_tokens = 4096
```

### 2.3 CLI integration

- New `--provider {ollama,openai,anthropic,gemini}` (default `ollama`).
- Precedence for the *model*: `--model` > provider's `default_model` > (ollama
  only) advisor auto-select.
- `client_factory(provider, model, ...)` dispatches by the provider's `kind`:
  `openai_compat` → `OpenAICompatClient`; `anthropic` → `AnthropicClient`.
- **Key resolution:** read `os.environ[api_key_env]`. If a non-ollama provider's
  key env var is unset → friendly error + non-zero exit ("set $OPENAI_API_KEY
  or use --provider ollama"). Keys are never written anywhere.
- `ollama` keeps the full advisor default path from sub-project (2).

---

## 3. Error handling

| Situation | Behavior |
|---|---|
| Provider key env var unset | Friendly message naming the env var + `--provider ollama` fallback; exit non-zero before any call. |
| Backend connectivity error | Existing harness re-raise-on-total-outage path (sub-project 1) → CLI exit-3 friendly message. |
| Anthropic returns no text/tool blocks | `from_anthropic_response` yields `ChatMessage(content="", tool_calls=[])` → harness treats as an empty finder turn (no crash). |
| Unknown `--provider` | Typer enum rejects it before execution. |

Secret hygiene: the API key is read from the environment at client-construction
time and handed to the SDK only. It is never logged, echoed, written to config,
or saved to Kagura Memory.

---

## 4. Testing (TDD)

- **`OpenAICompatClient`**: existing Ollama tests cover it (it's the same class);
  add one test pointing it at a fake OpenAI base_url via httpserver and asserting
  the `Authorization` header carries the key.
- **Anthropic adapter (pure, no network):**
  - `to_anthropic_system_and_messages`: a `[system, user, assistant+tool_calls,
    tool]` list → correct `(system_str, [user, assistant(tool_use), user(tool_result)])`.
  - `to_anthropic_tools`: harness OpenAI tool schema → `{name, description,
    input_schema}`.
  - `from_anthropic_response`: a fake response with text + `tool_use` blocks →
    `ChatMessage` with matching `content` and `ToolCall`s (arguments JSON-encoded).
  - temperature is **not** present in the request kwargs.
- **`AnthropicClient.chat`** with a fake `anthropic` client (injected) → end-to-end
  message/tool round-trip without network.
- **CLI**: `--provider openai` with key set → builds `OpenAICompatClient` with the
  right base_url/model; key unset → non-zero exit + guidance; `--provider anthropic`
  → builds `AnthropicClient`. `--provider ollama` (default) → unchanged advisor path.
- All existing 83 tests stay green (`OllamaClient` alias preserves them).

---

## 5. Dependencies

Add `anthropic` to `pyproject.toml`. `openai` is already a dependency (reused for
OpenAI + Gemini via the compatible endpoint). No `google-*` SDK needed.

---

## 6. Open questions (resolve during planning)

- Anthropic default model: `claude-sonnet-4-6` (balanced, cheaper) vs
  `claude-opus-4-8` (most capable). Spec picks sonnet-4-6 as the cost-balanced
  default for a frequently-run reviewer; `--model claude-opus-4-8` overrides.
- OpenAI/Gemini default model strings may drift — keep them in config so they're
  editable without code changes.
- Whether to surface provider/model in the report header (defer to sub-project 3
  report polish).
