# kagura-code-review

Cost-free, Ollama-powered code review for Claude Code. The review "brain" runs on an Ollama model (cloud or local), so reviews consume **zero Anthropic billing**. A companion slash command (`/kagura-code-review`) integrates with Kagura Memory: the outer Claude session retrieves past conventions and findings, passes them as context to the CLI, then writes durable knowledge back after the review. The CLI itself is a self-contained tool — it does not call Kagura Memory directly.

---

## How it works

```
Claude Code (outer session)
   │  /kagura-code-review slash command
   │  1. Recalls past findings from Kagura Memory (trust_tier: trusted filter)
   │  2. Writes assembled context to /tmp/kcr-ctx.md
   │  3. Invokes CLI ──────────────────────────────────────────────────────┐
   │                                                                        │
   └── Presents report to user ←── 5. Writes durable knowledge back        │
                                                                            ▼
                                                         kagura-code-review CLI
                                                            │  git diff (base...HEAD)
                                                            │  sandboxed repo tools
                                                            │    read_file / grep / git
                                                            ▼
                                                         Ollama (local or cloud)
                                                            │  agentic review loop
                                                            ▼
                                                         Markdown / JSON report
                                                         exit 0 = clean
                                                         exit 1 = blocking issues
```

---

## Install

```bash
pip install kagura-code-review
```

### System prerequisites

These are **not** installed by pip — you must set them up separately:

- **Ollama daemon** running with at least one model pulled.  
  Default cloud alias uses `qwen3-coder:480b-cloud`; default local alias uses `qwen2.5-coder:7b`.  
  Pull with: `ollama pull qwen2.5-coder:7b`
- **`claude` CLI** — required only for the `/kagura-code-review` slash-command workflow.  
  Install via: `npm install -g @anthropic-ai/claude-code`

---

## Quickstart

```bash
# Review current branch vs main (Markdown to stdout)
kagura-code-review --base main

# Write report as JSON to a file
kagura-code-review --base main --format json --out report.json

# Use the local model alias (faster, smaller)
kagura-code-review --local

# Limit review to specific paths
kagura-code-review --base main --paths src/foo.py --paths tests/test_foo.py
```

**Exit codes:**
- `0` — no blocking issues (severities INFO / LOW / MEDIUM only)
- `1` — one or more HIGH or CRITICAL findings
- `2` — git error (bad refs, not a git repo, etc.)

---

## Slash-command / Kagura Memory workflow

Install the shipped slash command into your project's `.claude/commands/` directory:

```bash
cp "$(python -c 'import kagura_code_review, pathlib; print(pathlib.Path(kagura_code_review.__file__).parent)')/kagura-code-review.md" \
   .claude/commands/kagura-code-review.md
```

Then inside Claude Code, run:

```
/kagura-code-review
```

The command will:
1. Retrieve this repository's pinned review policy and past findings from Kagura Memory (filtered to `trust_tier: trusted` to guard against prompt injection).
2. Pass assembled context to the CLI via `--context-file`.
3. Present the report.
4. Write durable new conventions and recurring patterns back to memory.

> The `trust_tier: trusted` filter is intentional and required — the context is fed directly to another model, so untrusted memories must never act as instructions.

---

## Configuration

Model aliases and defaults are defined in the shipped `config.toml`:

```toml
default_alias = "review-cloud"

[models.review-cloud]
ollama_model = "qwen3-coder:480b-cloud"
base_url     = "http://localhost:11434/v1"
num_ctx      = 32768

[models.review-local]
ollama_model = "qwen2.5-coder:7b"
base_url     = "http://localhost:11434/v1"
num_ctx      = 16384
```

**User override:** create `~/.config/kagura-code-review/config.toml` (or set `KAGURA_CODE_REVIEW_CONFIG` to an alternate path). Only keys you set override the defaults; everything else inherits.

To add a custom alias:

```toml
default_alias = "my-model"

[models.my-model]
ollama_model = "deepseek-r1:70b"
base_url     = "http://localhost:11434/v1"
num_ctx      = 65536
```

---

## Status / scope

v0.1 is **synchronous**: the CLI blocks until the Ollama review loop completes (up to `--timeout` seconds per call, `--max-iters` agent iterations).

The following are **not yet implemented** and are not claimed above:
- `--background` mode (fire-and-forget async review with a status poll command)
- Memory-pattern sharing with `kagura-engineer` or other kagura-* tools
- MCP server mode

Design spec: [`docs/superpowers/specs/2026-06-06-kagura-code-review-design.md`](docs/superpowers/specs/2026-06-06-kagura-code-review-design.md)

---

## License

MIT — see [LICENSE](LICENSE).
