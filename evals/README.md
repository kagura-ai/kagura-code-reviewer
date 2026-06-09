# Eval harness — golden set, baselines, and the regression guard

This directory measures the **quality of the review harness** (precision/recall,
FP/FN) against a small golden set of diffs with labeled known bugs.

- `golden/manifest.toml` — the golden cases (seeded synthetic diffs + one
  historical real diff). Defined in issue #5.
- `baselines/<model>.json` — committed quality baselines, one per model tier.
- The runner is `kagura-eval` (`src/kagura_code_reviewer/eval_cli.py`); the pure
  scoring core is `eval_harness.py`; the regression guard is `eval_baseline.py`.

## What a baseline is (and is not)

A baseline is a **distribution, not a point value**. The finders are stochastic,
so each baseline records the **mean ± stdev** of precision and recall across
`--repeats N` runs, plus full provenance (model, effort, seed, repeats, capture
time). `--seed` pins the backend seed for *provenance*, not bit-reproducibility —
LLM sampling, model version, and GPU kernels all vary, so re-running will land
near the recorded distribution, not exactly on it.

Two properties to keep in mind when reading the numbers:

- **Precision is seeded-only.** False positives are uncountable on a real diff
  (it may carry unknown latent bugs), so precision is computed over the seeded
  cases; recall is computed over all cases. With only one historical case today,
  **recall is seeded-dominated**.
- **A baseline is model-specific.** `qwen3:14b` and `qwen3-coder:480b-cloud` are
  different distributions and live in separate files. The guard refuses to grade
  a run against a baseline captured on a different model (see below).

## Baselines

| File | Model | Tier | Cost | Runner for the guard |
|------|-------|------|------|----------------------|
| `baselines/qwen3-coder-480b-cloud.json` | `qwen3-coder:480b-cloud` | **Ollama Cloud (primary)** | paid (cloud) | hosted `ubuntu-latest` — no GPU |
| `baselines/qwen3-14b.json` | `qwen3:14b` | zero-config local default (advisor pick on a 24 GB GPU) | free (localhost) | self-hosted GPU runner |

**Ollama Cloud is the primary eval tier** because cloud models run server-side:
the CI guard then needs **no GPU and runs on a stock hosted runner**, which is what
makes an automated regression gate practical (see CI below). The free-local
`qwen3:14b` baseline is the zero-config product default and can be gated on a
self-hosted GPU runner. The product's *default review backend* remains free local
Ollama — only the eval/CI tier leans on cloud.

> **Baselines are captured as a follow-up**, not committed in the same change as
> this harness. A baseline is a real GPU/cloud eval run (minutes), so it is produced
> by running the `eval.yml` workflow in `capture` mode (or the local command below)
> and committing the resulting JSON. Until a baseline file exists, the guard's
> `check` mode fails fast with a clear "no committed baseline" message.

## Capturing / refreshing a baseline

Authentication: Ollama Cloud is reached through the local daemon. Locally you sign
in once, interactively, with `ollama signin`; in CI the non-interactive equivalent
is an `OLLAMA_API_KEY` secret (see CI below). Local free models need no auth.

```bash
# Ollama Cloud (primary) — requires `ollama signin` locally
kagura-eval --golden-dir evals/golden \
  --model qwen3-coder:480b-cloud --effort med --repeats 5 --seed 7 \
  --baseline-out evals/baselines/qwen3-coder-480b-cloud.json

# free local default
kagura-eval --golden-dir evals/golden \
  --model qwen3:14b --effort med --repeats 5 --seed 7 \
  --baseline-out evals/baselines/qwen3-14b.json
```

Use `--repeats >= 5` where time allows (`>= 3` is acceptable): the golden set is
tiny, so a single run's precision is coarse and noisy. The committed file's
filename slug maps `:` and `/` to `-`.

## The regression guard

The guard derives the fail threshold from the **spread**, not the mean — a
mean-based gate would false-alarm on normal run-to-run noise at this small N:

```
floor = max(abs_floor, baseline_mean - k * baseline_stdev)      # k = 2.0, abs_floor = 0.0
fail if precision_mean < precision_floor  OR  recall_mean < recall_floor
```

Only **overall** precision and recall are gated. Per-dimension / per-severity
breakdowns are recorded in the baseline for context but **not** gated — they are
far too noisy with ~7 cases. The guard also fails fast on a **model mismatch**
(fresh run's model ≠ the baseline's pinned model).

Run the guard against a committed baseline:

```bash
kagura-eval --golden-dir evals/golden \
  --model qwen3:14b --effort med --repeats 5 --seed 7 \
  --check-baseline evals/baselines/qwen3-14b.json    # exit 1 on regression
```

## CI

The default CI (`.github/workflows/ci.yml`) stubs the backend, so the unit suite
stays network-free: backend-requiring tests are marked `ollama` and excluded via
`addopts = -m 'not ollama'`. Run them locally with `pytest -m ollama` (needs a
running Ollama daemon; self-skips if none is reachable).

The end-to-end eval + guard lives in `.github/workflows/eval.yml` as a **manual
`workflow_dispatch` job**. Because the primary tier is **Ollama Cloud** (models run
server-side), the job runs on a **stock hosted `ubuntu-latest` runner with no GPU** —
it starts a local `ollama` daemon only to proxy the cloud model, authenticated with
an `OLLAMA_API_KEY` secret (the non-interactive equivalent of `ollama signin`).
Create the key at <https://ollama.com/settings/keys> and add it under
Settings → Secrets → Actions. The job has two modes:

- `mode=capture` — runs the eval and uploads `evals/baselines/<model>.json` as an
  artifact to download and commit (bootstrap or refresh a baseline).
- `mode=check` — runs the eval and **fails on regression** below the committed
  baseline's σ-margin floor (requires the baseline file to exist).

For a free *local* baseline instead, dispatch the same workflow with
`runner: self-hosted` + `model: qwen3:14b` on a GPU box with a signed-in daemon.
Either way the job never blocks a PR (manual dispatch only).
