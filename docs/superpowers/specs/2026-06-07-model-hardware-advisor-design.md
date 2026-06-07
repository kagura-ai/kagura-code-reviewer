# kagura-code-reviewer v2 — Model + Hardware Advisor Design Spec

**Date:** 2026-06-07
**Status:** Draft (awaiting user review)
**Repo:** `~/works/kagura-code-review` (package `kagura_code_reviewer`)
**Branch:** `feat/v2-advisor`
**Scope:** Sub-project (2) of 3 in the v2 "pro-grade reviewer" effort.

---

## 1. Overview

The v2(1) harness lifts review quality through structure, but its quality is
still bounded by the model behind each call. The shipped default model alias is
`review-cloud` (`qwen3-coder:480b-cloud`) — which is **paid** — and the free
`review-local` alias is hardcoded to `qwen2.5-coder:7b`, far weaker than the
hardware can run. The v2(1) local smoke test confirmed the consequence: the 7b
model returned "No issues found" on a real diff.

This sub-project adds an **advisor** that detects the host hardware and the
installed Ollama models, then auto-selects the best **free local** model for the
review (and verifier) role — e.g. a 14b/27b coder model on a 24 GB GPU instead
of the 7b default. Paid cloud becomes an explicit opt-in turbo.

Net effect: the default, zero-config experience becomes *free and genuinely
capable* on the user's actual machine.

### Goals
1. **Auto-select the best free local model** for the review brain (and verifier)
   from detected hardware + installed models.
2. **Free-local becomes the default backend** (replacing the paid-cloud default),
   honoring the approved "free local default / cloud opt-in" posture.
3. **Respect explicit user choices** — never silently override a user config or
   `--model`/`--local`/`--cloud`.
4. **`--doctor` upgraded** to report hardware + the recommendation (this also
   resolves the long-standing "doctor only does 2/4 checks" gap).

### Non-goals (this sub-project)
- Persisting the recommendation to a user config file (`--auto`/`init`) — deferred
  to sub-project (3). Runtime auto-selection already satisfies the requirement.
- No model downloading/pulling. The advisor only chooses among **already
  installed** models; if nothing suitable is installed it says so.
- No benchmarking. The capability table is a small curated static table, not a
  measured suite.
- No non-Ollama backends.

---

## 2. Architecture

A new `advisor.py` with three pure-ish layers plus thin probes. The pure layers
(`recommend`) are fully unit-tested; the impure probes (`detect_hardware`,
listing installed models) are thin and lightly tested via monkeypatch.

```
detect_hardware() ─┐
                   ├─▶ recommend(hardware, installed, prefer_local) ─▶ Recommendation
list_models()    ──┘        (uses MODEL_CAPABILITIES table)            (finder, verifier,
                                                                        reason, fits)
                                   │
                                   ▼
            cli.resolve_spec(...) builds a ModelSpec from the recommendation
            when the user did not pin a model  →  feeds review_harness()
            doctor prints hardware + recommendation
```

### 2.1 Components (`src/kagura_code_reviewer/advisor.py`)

- `@dataclass Hardware(vram_mb: int, ram_mb: int, cpu_threads: int, has_gpu: bool)`
- `detect_hardware() -> Hardware` — probes:
  - VRAM: `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`
    (first GPU; megabytes). On any failure → `vram_mb=0, has_gpu=False`.
  - RAM: parse `/proc/meminfo` `MemTotal` (kB → MB). On failure → `ram_mb=0`.
  - CPU: `os.cpu_count() or 1`.
  Each probe is wrapped so a missing tool never raises.
- `@dataclass ModelCap(review: float, tool_calling: str, vram_mb: int, ctx: int)`
  - `tool_calling ∈ {"good","fair","poor"}`; `review` is a 0–1 aptitude score;
    `vram_mb` is the approximate footprint for the local variant (0 for cloud);
    `ctx` is a sensible context window to request.
- `MODEL_CAPABILITIES: dict[str, ModelCap]` — curated, keyed by exact tag with a
  family-prefix fallback. Seeded from the machine's installed models (see §3).
- `is_cloud(name: str) -> bool` — `"cloud" in name`.
- `list_models(base_url) -> list[str]` — installed model tags via `/api/tags`
  (reuse the doctor's HTTP path). Returns `[]` on failure.
- `lookup_cap(name) -> ModelCap` — exact match, then longest matching family
  prefix (e.g. `qwen2.5-coder`), then a conservative default
  (`ModelCap(0.3, "fair", 6000, 8192)`).
- `@dataclass Recommendation(finder: str|None, verifier: str|None, reason: str, fits: bool)`
- `recommend(hardware, installed, prefer_local=True) -> Recommendation` — see §2.2.

### 2.2 Recommendation algorithm (pure, fully tested)

```
candidates = installed models filtered by:
  - prefer_local: local-only (not is_cloud); else cloud-only
  - tool_calling != "poor"            # review needs reliable submit_* tool calls
fitting = [c for c in candidates if cap(c).vram_mb <= hardware.vram_mb * 0.9]   # local
          (cloud: no VRAM constraint)
pool = fitting or [c for c in candidates if cap(c).vram_mb <= hardware.ram_mb]  # RAM fallback
       or candidates                                                            # last resort
pick finder   = max(pool, key=cap.review)            # strongest that fits
pick verifier = finder                               # same model in v2(2)
fits = bool(fitting)        # False => we fell back to RAM/last-resort (slower)
reason = human string: chosen model, why (VRAM/RAM fit, aptitude), and fits flag
if no candidates at all -> Recommendation(None, None, "no suitable local model installed; pull one or use --cloud", False)
```

Notes:
- `verifier == finder` keeps v2(2) simple; a smaller/faster verifier is a future
  optimization, out of scope here.
- The `0.9` VRAM safety factor leaves headroom for the KV cache.

### 2.3 Integration

- **`config.py`** gains `spec_from_model_name(name, base_url, ctx) -> ModelSpec`
  so the advisor's chosen tag becomes a `ModelSpec` without a config alias.
- **`cli.py` precedence** for choosing the model (highest first):
  1. `--model <alias>` → existing `resolve_model` path (unchanged).
  2. `--cloud` → `recommend(prefer_local=False)`.
  3. `--local` → `recommend(prefer_local=True)`.
  4. user config file defines `default_alias`/models → respect it (existing path).
  5. **otherwise (zero-config default) → `recommend(prefer_local=True)`**, and
     print the `reason` to stderr ("Auto-selected <model>: <reason>").
  A `--cloud` flag is added (mirrors the existing `--local`). When the advisor
  returns `finder=None`, the CLI prints the reason and exits non-zero with the
  friendly "pull a model or use --cloud" guidance.
- **`doctor.py`** upgraded: in addition to daemon + model checks, print detected
  hardware and the local recommendation. `--doctor` becomes the 4-ish-check
  health+advice view (closing the "2/4" gap).

The base_url comes from the shipped config's model alias entries (all point at
`http://localhost:11434/v1`); the advisor reuses that base_url.

---

## 3. Capability table seed (curated)

Seeded from this machine's installed models; extensible. Scores are relative
review aptitude, not benchmarks.

| tag / family | review | tool_calling | vram_mb | ctx | note |
|---|---|---|---|---|---|
| `qwen3.5:27b` | 0.85 | good | 17000 | 32768 | strong general+code |
| `qwen3:30b` | 0.82 | good | 18000 | 32768 | strong |
| `qwen2.5-coder:14b` | 0.80 | good | 9000 | 32768 | code-specialized, reliable tools |
| `qwen3:14b` | 0.74 | good | 9000 | 32768 | |
| `qwen2.5-coder:7b` | 0.55 | good | 5000 | 16384 | weak but reliable |
| `qwen3.5:9b` | 0.55 | fair | 6600 | 16384 | |
| `gemma4:31b` | 0.6 | fair | 19000 | 8192 | weaker tool-calling |
| `deepseek-r1` (family) | 0.4 | poor | 9000 | 8192 | reasoning; unreliable tool calls → excluded |
| `qwen3-coder:480b-cloud` | 0.95 | good | 0 | 32768 | cloud turbo |
| `qwen3.5:397b-cloud` | 0.93 | good | 0 | 32768 | cloud |
| family default | 0.3 | fair | 6000 | 8192 | unknown tags |

On the 4090 (≈22 GB usable), the local pick is **`qwen3.5:27b`** (highest
aptitude that fits) — a large, free upgrade from the 7b default.

---

## 4. Error handling

| Situation | Behavior |
|---|---|
| `nvidia-smi` absent / errors | `vram_mb=0, has_gpu=False`; recommend by RAM fallback. |
| `/proc/meminfo` unreadable | `ram_mb=0`; still pick smallest local that fits VRAM (or last resort). |
| `/api/tags` fails | `list_models` returns `[]` → `recommend` returns `finder=None`; CLI prints reason + exits non-zero with guidance. |
| No suitable local model installed | `Recommendation(None, None, reason, False)`; CLI guidance: pull one or `--cloud`. |
| `--cloud` but no cloud model installed | same `None` path, reason names the gap. |

No probe failure ever raises out of `detect_hardware()` / `list_models()`.

---

## 5. Testing (TDD)

`recommend()` is pure → exhaustive deterministic unit tests:
- 24 GB GPU + full installed set → picks `qwen3.5:27b`, `fits=True`.
- 8 GB GPU (≈7.2 GB usable) → 27b (17 GB) and 14b (9 GB) excluded; picks
  `qwen2.5-coder:7b` (5 GB ≤ 7.2 GB), `fits=True`.
- 0 GB GPU, ample RAM → RAM fallback path, `fits=False`, picks strongest ≤ RAM.
- `prefer_local=False` → picks best cloud (`qwen3-coder:480b-cloud`).
- only `deepseek-r1` installed → excluded (poor tools) → `finder=None`.
- empty installed → `finder=None` with guidance reason.
- `lookup_cap` exact vs family-prefix vs default.

`detect_hardware()` tested with monkeypatched probes (fake `nvidia-smi` output,
fake `/proc/meminfo`, `os.cpu_count`), incl. the all-missing → zeros case.

`cli` precedence tested with CliRunner + monkeypatched `recommend`:
- zero-config → advisor path invoked, reason printed, harness called with the
  recommended model spec.
- `--model`/user-config → advisor NOT consulted.
- `--cloud` → `prefer_local=False`.
- advisor returns `None` → non-zero exit + guidance text.

`doctor` test: hardware + recommendation lines present in output.

All existing 65 tests stay green.

---

## 6. Open questions (resolve during planning)

- VRAM safety factor (start `0.9`).
- Whether `qwen3:30b` vs `qwen3.5:27b` should win on ties (start: higher `review`
  score wins; `27b`=0.85 > `30b`=0.82).
- Exact `ModelSpec` num_ctx source (start: capability `ctx`, capped by the alias
  default if a user config narrows it).
