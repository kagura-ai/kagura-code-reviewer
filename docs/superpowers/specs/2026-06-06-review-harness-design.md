# kagura-code-review v2 — Review Harness Design Spec

**Date:** 2026-06-06
**Status:** Draft (awaiting user review)
**Repo:** `~/works/kagura-code-review`
**Branch:** `feat/v2-review-harness`
**Scope:** Sub-project (1) of 3 in the v2 "pro-grade reviewer" effort.

---

## 1. Overview

v1 ships a **single-pass** review: one agentic loop on one Ollama model emits a
findings list. Dogfooding on 2026-06-06 showed this is not enough to be a
trustworthy merge gate — on the same diff it produced disjoint result sets
across runs (0 / 3 / 4 / 5 findings) and **never** surfaced the real blockers
that a multi-angle Claude review found.

The lesson: **review quality comes from the harness structure, not the size of
the model.** `/code-review`'s structure — many independent finder angles, then
adversarial per-finding verification, then dedup and ranking — is what catches
real bugs, and that structure is **deterministic Python orchestration** wrapping
otherwise-ordinary LLM calls.

This sub-project ports that structure into the CLI as a **review harness** whose
every LLM call goes to a free/paid Ollama backend. It replaces the v1
single-pass `review()` while preserving the existing `Report` / exit-code
contract.

### Goals
1. **Multi-angle finders** → **ensemble union** (recall) → **adversarial
   majority-vote verify** (precision) → **dedup + rank + cap**.
2. Quality from structure, runnable on the **free local** backend; the strong
   paid cloud backend is an opt-in turbo (model *selection* is sub-project 2).
3. Turn Ollama's run-to-run **non-determinism into diversity** (union of repeated
   finder runs improves recall instead of destabilizing the verdict).
4. **effort tiers** `low|med|high` (default `med`) scale the harness shape.
5. **Robust to weak/unreliable models**: a finder that emits garbage contributes
   nothing — it can never produce a false "clean pass."

### Non-goals (this sub-project)
- **Model + hardware advisor / auto-config** — that is sub-project (2). The
  harness takes already-constructed chat client(s); it does not choose models.
- No new backends beyond the existing Ollama `ChatClient` protocol.
- No web UI, no persistence of intermediate findings between runs.
- Report provenance fields are added here (minimal); richer reporting polish is
  sub-project (3).

---

## 2. Architecture

The harness is a deterministic pipeline of LLM-backed stages. Each stage is a
pure-ish function over data; the only non-determinism is inside the model calls,
which is deliberately absorbed by the ensemble.

```
diff (+ optional memory context)
  │
  ▼
[FIND]  N angle finders  ×  R repeats          ── union ──▶ candidates[]
  │        (each: bounded agentic loop with read_file/grep/list_files)
  ▼
[DEDUP] near-duplicate collapse (file, line-bucket, normalized title)
  │
  ▼
[VERIFY] V adversarial verifiers per candidate  ── majority ──▶ survivors[]
  │        (CONFIRMED / PLAUSIBLE kept; REFUTED dropped; recall-biased)
  ▼
[RANK]  severity desc, then confidence; cap at max_findings
  │
  ▼
Report(findings[], each annotated with provenance)  →  exit code (unchanged)
```

### 2.1 Components (new module `review/harness.py`)

- **Angle catalog** — a static list of finder angles, each a `(key, prompt)`
  mirroring `/code-review`: `correctness-linescan`, `removed-behavior`,
  `cross-file`, `reuse`, `simplification`, `efficiency`, `altitude`. Correctness
  angles always rank ahead of cleanup angles when the cap forces a cut.
- **Finder** — runs one angle: a bounded agentic loop (reuses `run_agent`) given
  the diff + tools (`read_file`/`grep`/`list_files`) and an angle prompt. Returns
  candidate findings via the existing `submit_findings` terminal tool. A finder
  that exhausts iterations or never submits valid output contributes `[]` (not a
  failure of the whole review).
- **Ensemble runner** — runs the selected angles `R` times and unions the
  candidates. Repeats vary the call (e.g. distinct seeds / labels) so a
  non-deterministic backend yields diverse candidates.
- **Deduper** — collapses near-duplicates by `(file, line // BUCKET,
  normalized-title)`; keeps the highest-severity representative and records the
  merge count.
- **Verifier** — for each deduped candidate, runs `V` independent adversarial
  checks ("try to refute this finding; default to PLAUSIBLE when the failing
  state is realistic"). Each returns `CONFIRMED | PLAUSIBLE | REFUTED`. The
  candidate survives if `(#CONFIRMED + #PLAUSIBLE) > #REFUTED` (recall-biased;
  ties survive).
- **Aggregator** — sorts survivors by `(severity desc, confidence desc)` and
  caps at `max_findings`; builds the `Report`.

### 2.2 effort tiers (config-driven)

| tier | angles | repeats R | verify votes V | max_findings | wall-clock (local) |
|------|--------|-----------|----------------|--------------|--------------------|
| low  | 3 (correctness-only) | 1 | 1 | 8 | tens of seconds |
| **med** (default) | **5** | **1** | **1–2** | **10** | **~1 min** |
| high | 7 | 2 (union) | 3 (majority) | 12 | minutes |

`med` uses V=1 for cleanup-dimension candidates and V=2 for correctness
candidates (verify effort follows severity). Tiers live in `config.toml` under
`[effort.<tier>]` so they are tunable without code changes.

### 2.3 Concurrency

Finders (and per-candidate verifiers) are independent and may run concurrently.
The harness exposes a `max_concurrency` knob (default: small for serial-bound
local single-GPU; higher when a cloud/parallel backend is selected). v2(1) ships
a simple bounded thread-pool executor around the blocking client calls; ordering
of the final report is deterministic regardless of completion order.

---

## 3. Interfaces

```python
# review/harness.py
@dataclass
class EffortTier:
    angles: list[str]
    repeats: int
    verify_votes: int
    verify_votes_correctness: int
    max_findings: int

def review_harness(
    finder_client: ChatClient,
    verifier_client: ChatClient,   # may be the same object as finder_client
    repo: RepoTools,
    diff: str,
    context: str | None,
    tier: EffortTier,
    max_iters: int = 12,
    max_concurrency: int = 1,
) -> Report: ...
```

- Backend-agnostic: clients satisfy the existing `ChatClient` protocol. Using a
  cloud client for `verifier_client` only (cheap "verify with the smart model")
  is possible but is wired by the *caller* (CLI / sub-project 2), not here.
- Returns the existing `Report`; `Finding` gains optional provenance fields
  (`angles: list[str]`, `votes: dict[str,int]`, `merge_count: int`) that are
  ignored by the md/json renderers unless present (backward compatible).

The CLI's `review(...)` call site is swapped to build an `EffortTier` from
`--effort` (default `med`) and call `review_harness(...)`. v1's single-pass
`review()` is kept only as the `low`-tier degenerate path internally, or removed
if the `low` tier subsumes it (decided during implementation; tests pin the
observable behavior either way).

---

## 4. Error handling & robustness

| Failure | Behavior |
|---------|----------|
| Finder emits malformed `submit_findings` JSON | Already fed back for retry (v1 fix); if never valid within `max_iters`, finder yields `[]`. |
| Finder backend error / timeout | That finder yields `[]`; review continues with the rest. Logged to stderr. |
| **All** finders yield `[]` (no candidates) AND ≥1 finder errored | Report a **blocking** meta finding ("review could not complete"), exit non-zero — never a silent clean pass. |
| All finders succeed and genuinely found nothing | Clean pass (empty findings, exit 0) — distinguishable from the error case above by the "≥1 finder errored" flag. |
| Verifier error | That vote is dropped; if all votes for a candidate error, the candidate is **kept** (recall-biased) and flagged low-confidence. |
| Determinism mode (`--seed` + temperature 0 + tier `low`/single repeat) | Reproducible gating path for users who need a stable verdict. |

The v1 invariants stay: exhausted/incomplete review is **blocking** (HIGH), and
a parse failure can never masquerade as "no issues."

---

## 5. Testing (TDD)

All orchestration logic is tested **deterministically with scripted fake
clients** — no real Ollama in unit tests.

- **Ensemble union**: two scripted finder runs returning overlapping candidates →
  assert union with correct dedup and `merge_count`.
- **Dedup**: same defect at `file:line±BUCKET` with reworded titles → collapses
  to one, highest severity kept.
- **Majority verify**: candidate with votes `[REFUTED, PLAUSIBLE, CONFIRMED]` →
  survives; `[REFUTED, REFUTED, PLAUSIBLE]` → dropped; tie `[REFUTED, CONFIRMED]`
  → survives (recall-biased).
- **effort mapping**: `low/med/high` strings → expected `EffortTier` params;
  config override respected.
- **Robustness**: finder raising → contributes `[]`, review still completes;
  all-error → blocking meta finding, exit non-zero; verifier all-error →
  candidate kept low-confidence.
- **Ranking/cap**: correctness outranks cleanup; cap honored.
- **Backward compat**: existing 38 tests stay green; `Report` md/json unchanged
  when provenance absent.

---

## 6. Sequencing within v2

1. **(1) Review harness** — this spec. Lands the quality core on top of merged
   v1 `main`.
2. (2) Model + hardware advisor — auto-pick the best-fit free local model (e.g.
   `qwen2.5-coder:14b` / `qwen3.5:27b` on a 24 GB GPU) and a verifier model;
   `--deep`/`--cloud` for the paid turbo. Replaces the weak 7b default.
3. (3) effort/robustness/report polish — provenance rendering, calibration,
   determinism-mode UX.

Each is its own spec → plan → implementation cycle.

---

## 7. Open questions (resolve during planning)

- Exact `BUCKET` size for line-proximity dedup (start: ±5 lines).
- Whether `low` tier replaces or wraps v1 `review()` (decide by what keeps tests
  cleanest).
- Verifier prompt: single shared adversarial prompt vs per-dimension lenses
  (start shared; revisit in sub-project 3).
