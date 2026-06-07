# Contributing to kagura-code-reviewer

Thanks for your interest in contributing! This project is a small, focused CLI,
and we aim to keep it that way. Bug reports, docs fixes, and well-scoped features
are all welcome.

## Dev setup

```bash
git clone https://github.com/kagura-ai/kagura-code-reviewer
cd kagura-code-reviewer
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q          # 143 tests, no Ollama needed (unit tests use stubs)
```

The test suite does **not** require a running Ollama daemon — backend calls are
stubbed. You only need Ollama to actually run a review end-to-end:

```bash
.venv/bin/kagura-code-reviewer --doctor          # checks daemon + model + hardware
.venv/bin/kagura-code-reviewer --base main        # review current branch vs main
```

## Workflow

1. **Branch** off `main` with a typed name: `fix/...`, `feat/...`, `chore/...`,
   `docs/...`.
2. **Test-first.** New behavior and bug fixes start with a failing test under
   `tests/`. We follow TDD — see existing `tests/test_harness.py` for the style.
3. Keep the change focused. Match the surrounding code's conventions (naming,
   comment density, idioms).
4. Run `.venv/bin/pytest -q` and make sure it's green before opening a PR.
5. Open a PR against `main`. Fill in the PR template checklist.

## Commit messages

Use a short, imperative subject line, optionally prefixed with a type
(`fix:`, `feat:`, `docs:`, `chore:`, `refactor:`, `test:`). Explain the *why* in
the body when it isn't obvious.

## Scope

Architecture and design notes live in `docs/superpowers/specs/` and
`docs/superpowers/plans/`. For larger changes, open an issue first so we can
agree on the approach before you invest implementation time.

## Reporting security issues

Please do **not** open a public issue for vulnerabilities. See
[SECURITY.md](SECURITY.md).
