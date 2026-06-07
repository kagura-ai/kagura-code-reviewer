# kagura-code-reviewer v2 — Finishing Touches Design Spec

**Date:** 2026-06-07
**Status:** Draft (approved scope: "(3)全部を進める")
**Repo:** `~/works/kagura-code-review` (package `kagura_code_reviewer`)
**Branch:** `feat/v2-polish`
**Scope:** Sub-project (3) of the v2 effort — report polish, determinism UX,
advisor persistence, and residual v1 robustness fixes.

---

## 1. Overview

The harness, advisor, and multi-provider backends are merged. This sub-project
is the "perfectly finished" pass: surface review provenance, make gating
reproducible, let users persist the advisor's pick, and close the residual
medium/low robustness issues found during the original dogfood review.

## 2. Components

### A. Report provenance (`report.py`)
`Finding` already carries `angles`, `votes`, `merge_count` (added in sub-project
1) but the renderers ignore them. Surface them:
- **Markdown**: append a provenance line per finding when present, e.g.
  `- **Seen by:** cross-file, correctness-linescan ×2 · votes: CONFIRMED 2`.
- **JSON**: include `angles`, `votes`, `merge_count` in `to_dict()` (additive;
  existing keys unchanged).
Findings without provenance (e.g. hand-constructed) render exactly as before.

### B. Determinism mode (`--seed`, providers, CLI)
- `OpenAICompatClient` gains an optional `seed: int | None`; when set, include it
  in the request (`seed` top-level + Ollama `options.seed`). Temperature is
  already pinned to 0.
- CLI `--seed INT` flows to the client via `build_review_client`. With
  `--seed` + `--effort low` (single repeat), a local run is reproducible.
- Anthropic has no seed; `--seed` is ignored there (documented).

### C. Advisor persistence (`--auto` / config write)
- New `config.py` `write_user_model(name, base_url, num_ctx)` that writes a
  `[models.auto]` entry + `default_alias = "auto"` to the **user** config file
  (`_USER`), creating parent dirs. Never writes secrets.
- CLI `--auto` flag: when set on an ollama run, after the advisor picks a model,
  persist it and print the path. Subsequent zero-config runs use it (the user
  config now exists → respected by precedence).

### D. Residual v1 robustness fixes
1. **Empty `choices` → IndexError** (`openai_compat.py`): guard
   `resp.choices[0]`; if empty, return `ChatMessage(content="", tool_calls=[])`
   (harness treats as an empty turn; finder contributes nothing — no crash).
2. **`git ls-files` flag injection** (`tools.py`): `list_files` passes `subdir`
   after `--`: `self._git("ls-files", "--", subdir)`.
3. **Silent `read_file` truncation** (`tools.py`): when output is truncated at
   `max_bytes`, append `\n...[truncated]`.
4. **Silent `grep` cap** (`tools.py`): when results hit `max_results`, append
   a final `...[more matches hidden]` line.
5. **`doctor._ollama_root` trailing slash** (`doctor.py`): rstrip before slicing
   so `.../v1/` yields a clean root (no double slash).
6. **`--out` double echo** (`cli.py`): only `typer.echo(rendered)` when `out` is
   not set (write-to-file suppresses stdout).
7. **Config encoding** (`config.py`): `read_text(encoding="utf-8")` on both the
   shipped and user config reads.

## 3. Non-goals
- No new providers, no streaming, no advisor-across-providers.
- No change to harness orchestration logic.

## 4. Testing (TDD)
- Provenance: md contains the provenance line when set / omits it when absent;
  json `to_dict` includes the three fields; existing report tests stay green.
- Determinism: `OpenAICompatClient(seed=...)` includes `seed` in the request
  body (httpserver); omitted when `seed=None`.
- Persistence: `write_user_model` writes a tmp user config with `default_alias`
  + `[models.auto]`; CLI `--auto` calls it (monkeypatched) with the advisor pick.
- Robustness: empty-choices → no crash, empty ChatMessage; `list_files` builds
  `["ls-files","--",subdir]`; truncated `read_file`/`grep` carry the marker;
  `_ollama_root("http://h:1/v1/")` → `http://h:1`; `--out` writes file and prints
  nothing; config reads pass `encoding="utf-8"`.
- All existing 94 tests stay green.

## 5. Open questions
- Provenance line verbosity (start terse, one line, only when non-default).
- Whether `--auto` should also pin `num_ctx` from the capability table (yes —
  use `lookup_cap(model).ctx`).
