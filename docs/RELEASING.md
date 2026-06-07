# Releasing & GitHub/PyPI setup

How to take this repo public and ship releases. Publishing uses **Trusted
Publishing** (OIDC) via `.github/workflows/publish.yml` — no PyPI API tokens are
stored anywhere. This mirrors the `kagura-memory-python-sdk` setup.

GitHub org/user is assumed to be **`kagura-ai`** (matching the other kagura-*
repos). If that's wrong, update `pyproject.toml` `[project.urls]`, the README
badges, and `.github/ISSUE_TEMPLATE/config.yml`.

## 0. Pre-flight

```bash
.venv/bin/pytest -q                      # 143 tests green
```

## 1. Create the GitHub repo + settings

```bash
gh repo create kagura-ai/kagura-code-reviewer --public --source=. --remote=origin --push

gh repo edit kagura-ai/kagura-code-reviewer \
  --description "Cost-free Ollama-powered code-review agent for Claude Code — multi-angle finders, adversarial verify, green/yellow/red CI verdict." \
  --add-topic code-review --add-topic ollama --add-topic llm \
  --add-topic claude-code --add-topic static-analysis \
  --add-topic developer-tools --add-topic ai --add-topic ci

gh repo edit kagura-ai/kagura-code-reviewer --enable-discussions
```

Web UI only (no stable CLI):

- **Social preview image** (Settings → General → Social preview, 1280×640 PNG) —
  the repo's "OG image" for Slack / X / Discord shares.
- **Branch protection** for `main` (require the CI check + a PR before merge).

## 2. One-time PyPI side — set up Trusted Publishing

This is the part that lives on pypi.org / test.pypi.org, **not** in this repo. Do
it once, before the first publish.

1. **Accounts:** create accounts on <https://pypi.org> and
   <https://test.pypi.org>, and enable **2FA** on both (required to publish).
2. **GitHub Environments:** in the repo, Settings → Environments, create two
   environments named exactly **`pypi`** and **`testpypi`** (these match the
   `environment:` keys in `publish.yml`). Optionally add protection rules
   (required reviewers) to the `pypi` environment.
3. **Add a "pending publisher" on PyPI** (Your account → Publishing → Add a new
   pending publisher) with **exactly** these values:
   - PyPI Project Name: `kagura-code-reviewer`
   - Owner: `kagura-ai`
   - Repository name: `kagura-code-reviewer`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
4. **Repeat step 3 on TestPyPI**, with Environment name `testpypi`.

Because it's a *pending* publisher, you do **not** need to pre-register or
manually upload the project first — the first successful workflow run claims the
name. No tokens to create or rotate.

## 3. Dry run on TestPyPI

Actions → **Publish to PyPI** → *Run workflow* (`target=testpypi`). Then check the
TestPyPI project page: README renders, license shows **Apache-2.0**, classifiers
and project URLs look right.

## 4. Real release

```bash
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
```

The tag push triggers `publish.yml`: build → publish to PyPI (Trusted
Publishing) → create the GitHub Release with generated notes. Confirm
`pip install kagura-code-reviewer` works afterward.

## Notes

- `version` is set in `pyproject.toml` (`0.1.0`). Bump it before tagging; the tag
  and the `pyproject` version should match.
- The badges in the README go live automatically once the repo + PyPI project
  exist.
