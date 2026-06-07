# CLAUDE.md

Guidance for Claude Code working in the `kagura-code-review` repository.

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

A cost-free **Ollama-powered code-review agent CLI** for Claude Code. The review
brain runs on Ollama (cloud/local) → no Anthropic billing. It reviews a git diff
**in-repo** (sandboxed `read_file`/`grep`/`git_diff`/`list_files`) and returns a
structured report. Memory is handled by the **outer Claude** via the slash
command, not by this package.

See the design docs:
- Spec: `docs/superpowers/specs/2026-06-06-kagura-code-review-design.md`
- Plan: `docs/superpowers/plans/2026-06-06-kagura-code-review.md`

## Dev commands

```bash
.venv/bin/pytest -q          # run the test suite (36 tests)
.venv/bin/kagura-code-review --help
.venv/bin/kagura-code-review --doctor          # check ollama daemon + model
.venv/bin/kagura-code-review --base main       # review current branch vs main
```

## Status (2026-06-06)

v1 complete on branch `feat/v1-implementation` (not yet merged; no `origin`
remote). 36 tests pass. Next: finish the branch (merge or `gh repo create` + PR),
then v2 `--background`. See the `category:next-steps` memory for the full list.
