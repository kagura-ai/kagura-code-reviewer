# Releasing — version lives in FOUR files (CI `version-sync` enforces it)

The package version has **four** sources that must stay identical; the `version-sync`
job in `.github/workflows/ci.yml` fails the build on any drift:

1. `pyproject.toml` → `[project].version`  ← **single source of truth**
2. `src/kagura_code_reviewer/__init__.py` → `__version__`
3. `.claude-plugin/plugin.json` → `version`
4. `.claude-plugin/marketplace.json` → `plugins[0].version`

**`/gh-issue-driven:tag` only bumps #3 and #4** (it is written for a pure-plugin repo)
— it does **not** touch the Python version (#1, #2). A release driven by `/tag` alone
therefore leaves `version-sync` RED and makes `publish.yml` build the *old* version
(PyPI build reads `pyproject.toml`). **Before tagging, bump all four.** `publish.yml`
triggers on `push: tags: v*` and builds whatever `pyproject.toml` says at the tagged
commit, so the tag MUST point at a commit where all four are already bumped:

```bash
v=0.2.0
sed -i "s/^version = .*/version = \"$v\"/" pyproject.toml
sed -i "s/^__version__ = .*/__version__ = \"$v\"/" src/kagura_code_reviewer/__init__.py
# then /gh-issue-driven:tag bumps the two plugin manifests + CHANGELOG + creates the tag
```

(Learned the hard way on v0.2.0: `/tag` bumped only the plugin manifests → version-sync
red + PyPI built 0.1.2 → had to align the package version and move the tag.)
