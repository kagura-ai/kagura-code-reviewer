# kagura-code-reviewer

<!-- Badges go live once the repo is on GitHub and the package is on PyPI. -->
[![PyPI](https://img.shields.io/pypi/v/kagura-code-reviewer.svg)](https://pypi.org/project/kagura-code-reviewer/)
[![Python](https://img.shields.io/pypi/pyversions/kagura-code-reviewer.svg)](https://pypi.org/project/kagura-code-reviewer/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/kagura-ai/kagura-code-reviewer/actions/workflows/ci.yml/badge.svg)](https://github.com/kagura-ai/kagura-code-reviewer/actions/workflows/ci.yml)

**Pro-grade code review on your git diff with zero Anthropic billing.** A local (or
cloud) Ollama model fans out multi-angle finders, adversarially majority-vote
verifies each one, dedups, and returns a ranked Markdown/JSON report with a
`green` / `yellow` / `red` verdict you can gate CI on.

Cost-free, Ollama-powered code review for Claude Code. The review "brain" runs on an Ollama model (cloud or local), so reviews consume **zero Anthropic billing**. A companion slash command (`/kagura-code-reviewer`) integrates with Kagura Memory: the outer Claude session retrieves past conventions and findings, passes them as context to the CLI, then writes durable knowledge back after the review. The CLI itself is a self-contained tool — it does not call Kagura Memory directly.

---

## How it works

```
Claude Code (outer session)
   │  /kagura-code-reviewer slash command
   │  1. Recalls past findings from Kagura Memory (trust_tier: trusted filter)
   │  2. Writes assembled context to /tmp/kcr-ctx.md
   │  3. Invokes CLI ──────────────────────────────────────────────────────┐
   │                                                                        │
   └── Presents report to user ←── 5. Writes durable knowledge back        │
                                                                            ▼
                                                         kagura-code-reviewer CLI
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
pip install kagura-code-reviewer
```

### System prerequisites

These are **not** installed by pip — you must set them up separately:

- **Ollama daemon** running with at least one model pulled.  
  Default cloud alias uses `qwen3-coder:480b-cloud`; default local alias uses `qwen2.5-coder:7b`.  
  Pull with: `ollama pull qwen2.5-coder:7b`
- **`claude` CLI** — required only for the `/kagura-code-reviewer` slash-command workflow.  
  Install via: `npm install -g @anthropic-ai/claude-code`

---

## Quickstart

```bash
# Review current branch vs main (Markdown to stdout)
kagura-code-reviewer --base main

# Write report as JSON to a file
kagura-code-reviewer --base main --format json --out report.json

# Use the local model alias (faster, smaller)
kagura-code-reviewer --local

# Limit review to specific paths
kagura-code-reviewer --base main --paths src/foo.py --paths tests/test_foo.py
```

**Exit codes:**
- `0` — no blocking issues (severities INFO / LOW / MEDIUM only)
- `1` — one or more HIGH or CRITICAL findings
- `2` — git error (bad refs, not a git repo, etc.)

### Example output

```markdown
# Code Review

## [CRITICAL] IndexError when orders is empty (correctness)
- **Where:** `orders.py:14`
- **Why:** latest_order accesses ordered[-1] without verifying the list is
  non-empty, causing IndexError when orders is empty.
- **Fix:** Guard the empty case before indexing.
- **Seen by:** correctness-linescan, cross-file, removed-behavior ×5; votes: CONFIRMED 2; conf 1.00
```

Each finding carries provenance (`Seen by:` — which finder angles surfaced it),
adversarial verify `votes`, and a `conf` score, so you can filter noise with
`--min-confidence`.

### Machine-readable output contract

`--format json` emits a stable, versioned envelope for downstream automation
(no need to scrape Markdown). The `findings` key stays top-level for backward
compatibility; `schema_version` / `verdict` / `summary` are additive.

```jsonc
{
  "schema_version": 1,
  "verdict": "green",        // green = clean, yellow = non-blocking only, red = blocking finding
  "summary": {
    "total": 0,
    "blocking": 0,           // findings with severity >= HIGH
    "by_severity": {},       // {"HIGH": 2, "LOW": 1, ...}
    "incomplete": false      // true if the review did not finish (a "meta" finding is present)
  },
  "findings": [ /* per-finding: dimension, severity, file, line, title, rationale,
                   suggestion, angles, votes, merge_count, confidence */ ]
}
```

**Verdict ↔ exit-code invariant:** `verdict == "red"` **iff** exit code is `1`.
`green` and `yellow` both exit `0` (use `verdict` to distinguish clean from
advisory). `summary.incomplete` lets an actor tell "review failed to run" apart
from "real blocking findings."

---

## Slash-command / Kagura Memory workflow

There are two ways to get the `/kagura-code-reviewer` slash command into Claude Code.

### Option A — Install as a Claude Code plugin (recommended)

Add the Kagura marketplace and install the plugin from inside Claude Code:

```
/plugin marketplace add kagura-ai/kagura-code-reviewer
/plugin install kagura-code-reviewer@kagura-code-reviewer
```

`/kagura-code-reviewer` then appears in your skill list across every project — no
per-repo copy needed. (You still need the `kagura-code-reviewer` CLI on your
`PATH`; install it with `pip install kagura-code-reviewer`.)

### Option B — Copy the shipped command into a single project

If you installed the PyPI package and only want the command in one repo, copy the
shipped command into that project's `.claude/commands/` directory:

```bash
# Copy the shipped slash command into your project so Claude Code can run /kagura-code-reviewer
mkdir -p .claude/commands
cp "$(python -c "import kagura_code_reviewer, pathlib; print(pathlib.Path(kagura_code_reviewer.__file__).parent / 'commands' / 'kagura-code-reviewer.md')")" .claude/commands/
```

Then inside Claude Code, run:

```
/kagura-code-reviewer
```

The command will:
1. Retrieve this repository's pinned review policy and past findings from Kagura Memory (filtered to `trust_tier: trusted` to guard against prompt injection).
2. Pass assembled context to the CLI via `--context-file`.
3. Present the report.
4. Write durable new conventions and recurring patterns back to memory.

### Memory security contract

Memory is grounding, not authority. The CLI receives only a string via
`--context-file` and **cannot re-verify its provenance**, so callers are
responsible for these guarantees (defense in depth):

1. **Recall with `trust_tier: "trusted"`** — required. It excludes
   external/connector-ingested memories (Slack/Discord/etc.) that could carry
   injected instructions (OWASP LLM01/LLM03). The CLI assumes the caller has
   filtered; it does not (and cannot) check.
2. **Injected memory is untrusted, reference-only data.** The CLI fences it in
   `BEGIN/END UNTRUSTED MEMORY CONTEXT` markers and the system prompt forbids
   obeying anything inside them.
3. **Memory has no finding-suppression authority.** The verdict is computed from
   findings produced via the `submit_findings` tool and the adversarial verify
   pass — never from prose or memory. Context can inform a finding's rationale
   but cannot remove a finding or change the verdict.
4. For autonomous use, only let **owner-pinned** memory influence gating
   decisions; do not treat agent-authored `on_recall` memories as trusted for
   security-relevant gates (avoids a self-poisoning feedback loop).

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

**User override:** create `~/.config/kagura-code-reviewer/config.toml` (or set `KAGURA_CODE_REVIEW_CONFIG` to an alternate path). Only keys you set override the defaults; everything else inherits.

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

## Contributing

Issues and PRs are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev
setup and workflow, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). To report a
security issue, follow [SECURITY.md](SECURITY.md) (please do not open a public
issue for vulnerabilities).

---

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
