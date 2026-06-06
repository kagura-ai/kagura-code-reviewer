# kagura-code-review — Design Spec

**Date:** 2026-06-06
**Status:** Draft (awaiting user review)
**Repo:** `~/works/kagura-code-review` (new, independent package)

---

## 1. Overview

`kagura-code-review` is a focused, **cost-free code-review agent**. The review
"brain" runs on **Ollama** (cloud or local model), so reviews consume no
Anthropic billing. It is invoked from a Claude Code session (via Bash or a
slash command), performs a multi-dimension review of a git diff in the actual
repository, and returns a concise report to the main session. It uses
**Kagura Memory Cloud** (`kagura-memory`) as a learning memory backbone so the
agent gets smarter over time.

The defining slogan: **"memory = the outer Claude; review labor = Ollama."**

### Goals
1. Code-review's main model is **Ollama-powered (cloud/local)** → zero token cost.
2. **Kagura Memory integration** → recall past conventions/findings, remember
   new durable knowledge, learn from feedback. A review agent that improves.
3. Invoked from Claude Code, returns results to the main working session.
4. Distributable on **PyPI** (`pip install kagura-code-review`), self-contained.

### Non-goals (YAGNI for v1)
- Not a general autonomous coding/ops agent (that is `kagura-engineer`).
- Not an MCP server (no remote/multi-machine review in v1).
- No nested Claude Code, no LiteLLM proxy, no middleware (rejected — see §9).
- No memory clustering/broadlistening in v1 (`analyze_context` deferred).

---

## 2. Positioning vs `kagura-engineer`

`kagura-engineer` (aka `kagura-agent`) is a **future, unstarted** autonomous
coding harness with a **Claude** brain (Claude Agent SDK) and Kagura Memory as
backbone. Its memory patterns (failure-mode learning via `prevents` edges,
pinned goals, checkpoints, sub-agent dispatch) **overlap heavily** with what
this review agent needs.

Decision (user, 2026-06-06): **build `kagura-code-review` as an independent
tool now.** Differentiators: **Ollama (free)**, **review-focused**, **shippable
immediately**. The memory-pattern duplication is accepted for v1.

**Future convergence:** common agent-memory helpers (recall+explore context
build, `prevents`-edge failure learning, pinned-goal loading) should be
extracted into a shared library (candidate: `kagura-memory-python-sdk`) so
`kagura-engineer` can reuse them when it starts. This spec marks those helpers
as extraction candidates but does not block v1 on it.

---

## 3. Architecture

Two layers with a clean split of responsibility:

```
 Main Claude Code session (Anthropic — billed)        kagura-code-review CLI (Ollama — free)
 ─────────────────────────────────────────────        ────────────────────────────────────
  (1) resolve repo context_id (list_contexts)
  (2) load_pinned + recall(HyDE) + explore   ──────►   (4) compute git diff (base...head)
      → /tmp/ctx.md  (the "/goal")                     (5) agent loop: read_file/grep/git
  (3) Bash/slash: kagura-code-review \                      → reads real code for context
        --base main --context-file /tmp/ctx.md          (6) multi-dimension review
                                                        (7) structured findings → report
  (8) finish: remember + feedback + Time     ◄──────       (JSON + Markdown) + exit code
      Memory (durable knowledge only)
```

- **Memory ops live in the outer Claude**, which already holds the
  `kagura-memory` MCP connection (a claude.ai-hosted connector). This sidesteps
  the headless-auth problem: a standalone Ollama CLI cannot reliably reach a
  claude.ai MCP connector. The CLI never speaks MCP.
- **The review runs in-repo** with full filesystem/git/grep access, which is the
  single biggest driver of review quality.
- **Only the final report** enters the billed context; the heavy reasoning
  happens on Ollama.

---

## 4. Components (`src/kagura_code_review/`)

| Module | Responsibility |
|---|---|
| `cli.py` | Typer entry. Flags: `--base main`, `--head HEAD` (or `--branch`), `--paths`, `--context-file`, `--model <alias>`, `--local`/`--cloud`, `--format json\|md`, `--out`, (later) `--background`. |
| `ollama_client.py` | Thin wrapper over Ollama's **OpenAI-compatible** `/v1/chat/completions`. Model-alias resolution, cloud/local switch, `num_ctx`, per-call timeout. |
| `agent.py` | Tool-calling loop (OpenAI Python SDK). Retries, timeouts, and a **diff-only fallback** when tool-calls are malformed. |
| `tools.py` | In-repo tools exposed to the model: `read_file`, `grep`, `git_diff`, `list_files`. **Sandboxed to repo root** (no access outside). |
| `review/skill.py` + prompts | Implements "goal → dimension-split review". Dimensions: correctness, security, performance, conventions, tests. |
| `report.py` | Findings → JSON + Markdown. Each finding: severity, `file:line`, rationale, suggested fix. Exit code derived from max severity. |
| `doctor.py` | Checks: ollama daemon up, model pulled, cloud signin (if `--cloud`), inside a git repo. |
| `config.toml` | Default review model aliases (cloud/local), thresholds. User override at `~/.config/kagura-code-review/config.toml`. |
| `.claude/commands/kagura-code-review.md` | Slash command orchestrating the memory steps + CLI call (see §6). |

---

## 5. Memory integration — `/goal finish /code-review` mapped to memory lanes

Each repository maps to **one Kagura Memory context**. Resolve `context_id` via
`list_contexts()`; cache the `repo path → context_id` mapping in a local
`.kagura-review` file. All memory operations are performed by the **outer
Claude** (which holds the MCP).

### 🎯 `/goal` = durable policy (pinned)
Review goals / guardrails / conventions are stored with
`remember(delivery_mode='always')` and loaded each run via `load_pinned()`.
`load_pinned` is **deterministic** — it returns the complete, unranked set every
call — so the review baseline never drifts.
Examples: "APIs must stay backward compatible", "no N+1 queries",
"never log secrets".

### 🔎 `/code-review` body = recall + explore context build
1. `recall` for findings relevant to the diff. Use **HyDE** (draft a
   hypothetical answer, search with it) for better hits; filter by
   `type`/`importance`/tags.
2. **Security (required):** filter `{"trust_tier": "trusted"}` to **exclude
   untrusted/connector-ingested memories** before injecting them into the
   prompt (OWASP LLM01/LLM03 — defend against prompt injection via memory).
3. `explore` from top recall hits (depth=2) to surface related past bugs /
   design decisions via the knowledge graph.
4. `reference` to fetch full content when a summary is insufficient.
5. Write the assembled context to `--context-file` for the CLI (Ollama).

### ✅ `finish` = write-back (the learning loop)
- `remember`: only **durable** new conventions / recurring findings (not
  one-off nits). Scale `importance` by severity/recurrence.
- `create_edge(type="prevents")`: link a fix memory to the pattern it prevents
  (failure-mode learning — extraction candidate shared with `kagura-engineer`).
- `feedback`: reinforce which recalled memories were actually useful → improves
  future recall via Neural Memory.
- **Time Memory** (`remember(type='time', details={'trigger': {...}})`): record
  deferred items, e.g. "remove this workaround before v2 release"; later
  surfaced by `recall_upcoming`.

### 🧠 run-state lane (`set_state`/`get_state`) — used mainly in v2
Holds ephemeral review-job state (job_id, branch, step, progress, partial
findings) with a TTL. Structurally excluded from `recall()`, so it never
pollutes knowledge search. Light use in v1 (sync); central to v2 (`--background`).

### (Future, optional) `analyze_context` / `get_cluster`
Cluster accumulated findings to surface recurring review themes
(e.g. "N+1 keeps recurring"). Deferred past v1.

---

## 6. Slash command flow (`.claude/commands/kagura-code-review.md`)

The slash command instructs the outer Claude to:

```
1. list_contexts → resolve this repo's context_id (cache in .kagura-review)
2. load_pinned(context_id)                       # the /goal: durable policy
   + recall(query=HyDE(diff summary), filters={trust_tier: trusted})
   + explore(top hits, depth=2)
   → write assembled context to /tmp/kcr-ctx.md
3. Bash: kagura-code-review --base main --context-file /tmp/kcr-ctx.md --format md
4. Present the report to the user
5. finish (only durable value):
   - remember(new conventions / recurring findings)
   - create_edge(prevents) for fix↔pattern links
   - feedback(useful recalled memories)
   - Time Memory for deferred items
```

This keeps headless auth out of the picture entirely: Claude is the memory
keeper, Ollama is the labor.

---

## 7. Data flow & interface

- **Input:** `git diff base...head` + changed-file list + optional `--context-file`.
- **Output:** Markdown report to stdout (concise, for the billed context);
  JSON via `--out report.json`.
- **Exit code:** `0` = no blocking issues; non-zero = blocking issues found
  (usable by pre-push hooks and by Claude for branching).

---

## 8. Error handling

| Condition | Behavior |
|---|---|
| Ollama daemon down | doctor-style message, non-zero exit. |
| Model not pulled | Instruct `ollama pull <model>`. |
| Malformed tool-call from model | Retry N times, then degrade to diff-only review. |
| Per-call timeout | Configurable; fail the call, continue with partial findings. |
| Not a git repo | Clear error, non-zero exit. |

---

## 9. Rejected alternative: nested Claude Code on Ollama (kagura-code style)

Reusing `kagura-code` (launch a full Claude Code instance pointed at Ollama via
LiteLLM proxy + middleware) was considered and **rejected** for v1:
- Heavy: LiteLLM proxy subprocess, FastAPI middleware, port allocation, nested
  Claude Code env/session-id collisions.
- Depends on an **archived** package's internal API.
- The general-purpose machinery (tool filtering, compression) is unnecessary for
  a single-purpose review agent.

The chosen design (Ollama-direct lightweight agent) is smaller, more reliable,
and PyPI-clean.

---

## 10. Packaging / PyPI

```toml
[project]
name = "kagura-code-review"
requires-python = ">=3.11"
dependencies = ["typer", "rich", "httpx", "openai"]   # openai SDK → Ollama OpenAI-compat

[project.scripts]
kagura-code-review = "kagura_code_review.cli:app"
```

- `ollama` and `claude` (Claude Code CLI) are **system binaries**, not pip
  deps → verified by `doctor.py`, documented in README.
- Self-contained: no dependency on `kagura-code` or `kagura-engineer`.

Repo layout:
```
~/works/kagura-code-review/
├── pyproject.toml  README.md  LICENSE  CHANGELOG.md
├── src/kagura_code_review/
│   ├── cli.py  ollama_client.py  agent.py  tools.py
│   ├── review/{skill.py, prompts/}
│   ├── report.py  doctor.py  config.toml
│   └── .claude/commands/kagura-code-review.md   (or top-level .claude/)
└── tests/
```

---

## 11. Testing

- **Unit:** tools sandbox (no escape from repo root), report formatting,
  severity→exit-code mapping, model-alias resolution.
- **Agent loop:** mock Ollama's OpenAI-compatible endpoint with
  `pytest-httpserver` (tool-call request/response cycles).
- **e2e:** gated behind `KAGURA_REVIEW_RUN_E2E=1` + a real Ollama backend.

---

## 12. Phasing

- **v1 (now):** synchronous, inline. Claude runs the CLI via Bash/slash, waits,
  report returns to the conversation. Memory steps done by outer Claude.
- **v2:** `--background` mode — CLI returns a job id immediately, runs detached,
  writes report to file + state to the `set_state` lane; Claude collects the
  result when ready. Enables non-blocking, parallel reviews.
- **Future:** memory clustering (`analyze_context`), shared pattern extraction
  with `kagura-engineer`, optional MCP-server form for remote review.
