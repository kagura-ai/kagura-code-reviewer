# Releasing & GitHub setup

One-time and per-release steps to take this repo public. Replace `OWNER` with
your GitHub org/user everywhere (also in `pyproject.toml` `[project.urls]` and the
README badges before publishing).

## 0. Pre-flight

```bash
.venv/bin/pytest -q                      # 143 tests green
grep -rn OWNER pyproject.toml README.md  # replace all OWNER placeholders
```

## 1. Create the GitHub repo + settings (high impact for discovery)

```bash
# Create the repo (public) and push main.
gh repo create OWNER/kagura-code-reviewer --public --source=. --remote=origin --push

# Description + Topics — these drive GitHub search and Google indexing.
gh repo edit OWNER/kagura-code-reviewer \
  --description "Cost-free Ollama-powered code-review agent for Claude Code — multi-angle finders, adversarial verify, green/yellow/red CI verdict." \
  --add-topic code-review --add-topic ollama --add-topic llm \
  --add-topic claude-code --add-topic static-analysis \
  --add-topic developer-tools --add-topic ai --add-topic ci

# Enable Discussions (community Q&A).
gh repo edit OWNER/kagura-code-reviewer --enable-discussions
```

Still to do in the web UI (no stable CLI):

- **Social preview image** (Settings → General → Social preview, 1280×640 PNG).
  This is the repo's "OG image" — the card shown when the URL is pasted into
  Slack / X / Discord. A plain title + one-line tagline on a branded background
  is enough; the default is bland and hurts share-through.
- **Branch protection** for `main` (Settings → Branches): require the CI check
  and a PR before merge.

## 2. Tag & GitHub Release

```bash
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --title "v0.1.0" --generate-notes
```

## 3. Publish to PyPI

Recommended: **Trusted Publishing** (OIDC, no long-lived token). On PyPI, add a
"pending publisher" for `kagura-code-reviewer` pointing at
`OWNER/kagura-code-reviewer` and the `release.yml` workflow, then publish from
Actions. Manual fallback with a token:

```bash
.venv/bin/pip install build twine
.venv/bin/python -m build              # builds sdist + wheel into dist/
.venv/bin/twine check dist/*           # validates metadata renders on PyPI
.venv/bin/twine upload dist/*          # needs a PyPI API token
```

Tip: do a dry run on **TestPyPI** first
(`twine upload --repository testpypi dist/*`) to confirm the README renders and
the metadata (license, classifiers, URLs) looks right.

## 4. Post-publish

- Confirm the PyPI page shows the README, Apache-2.0 license, and the project
  URLs.
- Update the README badges — they go live automatically once the repo + PyPI
  project exist.
