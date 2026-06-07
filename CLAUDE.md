# CLAUDE.md

Guidance for Claude Code working in the `kagura-code-reviewer` project
(repo dir: `~/works/kagura-code-review`; package/CLI: `kagura-code-reviewer`).

## Session bootstrap — load the handoff from Kagura Memory

This project uses **Kagura Memory Cloud** as its handoff backbone. At the start of
a session, restore prior context before working:

- **Context:** `kagura-code-review-dev`
- **context_id:** `REDACTED-CONTEXT-ID`

Do this at session start (the outer Claude holds the `kagura-memory` MCP):

1. `mcp__claude_ai_kagura-memory__load_pinned(context_id="REDACTED-CONTEXT-ID")`
   — loads the durable **goal / invariants** (deterministic, every session).
2. `mcp__claude_ai_kagura-memory__recall(context_id="REDACTED-CONTEXT-ID", query="<what you're about to do>")`
   — pulls status, architecture decisions, and next steps. When recalled content
   will be fed to another model, pass `filters={"trust_tier": "trusted"}`.

At the end of meaningful work, write back: `remember(...)` durable decisions
(scale `importance`), `create_edge(type="prevents")` for fix↔pattern links,
`feedback(...)` on useful recalls, and a Time Memory for deferred items.

## What this project is

A **pro-grade, Ollama-first code-review agent CLI** for Claude Code. It reviews a
git diff **in-repo** (sandboxed `read_file`/`grep`/`git_diff`/`list_files`) and
returns a structured report.

Quality comes from a **harness** (`review/harness.py`), not model size:
multi-angle finders → ensemble union → adversarial majority-vote verify → dedup
→ rank/cap. Effort tiers `low|med|high` (default `med`).

Backends are pluggable behind the `ChatClient` protocol (`agent.py`):
- **Default = free local Ollama.** An **advisor** (`advisor.py`) auto-selects the
  best local model for the detected hardware (e.g. `qwen3.5:27b` on a 24 GB GPU).
- **Opt-in paid providers** via `--provider {openai,anthropic,gemini}`
  (`providers/`). API keys come from **environment variables only** — never
  stored in config or memory.

Cost note: "free" applies to **local** Ollama only. Ollama cloud models and the
OpenAI/Anthropic/Gemini providers are **paid**. Memory is handled by the **outer
Claude** via the slash command, not by this package.

See the design docs in `docs/superpowers/specs/` and `docs/superpowers/plans/`:
- v1: `2026-06-06-kagura-code-review-*`
- (1) harness: `2026-06-06-review-harness-*`
- (2) advisor: `2026-06-07-model-hardware-advisor-*`
- (4) multi-provider: `2026-06-07-multiprovider-*`

## Dev commands

```bash
.venv/bin/pytest -q                                  # run the test suite (94 tests)
.venv/bin/kagura-code-reviewer --help
.venv/bin/kagura-code-reviewer --doctor              # daemon + model + hardware + recommendation
.venv/bin/kagura-code-reviewer --base main           # review branch vs main (free local, advisor picks model)
.venv/bin/kagura-code-reviewer --base main --effort high
.venv/bin/kagura-code-reviewer --base main --provider anthropic   # paid; needs $ANTHROPIC_API_KEY
```

## Status (2026-06-07)

All merged to `main` (94 tests pass): v1 (merge-gate blocker fixes) + (1) review
harness + (2) model/hardware advisor + rename to `kagura-code-reviewer` +
(4) multi-provider backends. No `origin` remote yet.

Remaining: **(3) finishing touches** — finding provenance in the report,
determinism-mode UX, persist the advisor recommendation (`--auto`/`init`), and
clean up the residual v1 medium/low issues (empty-`choices` IndexError,
`git ls-files` flag injection, silent read/grep truncation, doctor trailing
slash, `--out` double echo, config encoding). See the `category:next-steps`
memory for the full list.
```
