"""Regression guard for the eval harness — issue #7.

Turns a committed baseline (a precision/recall distribution captured on a pinned
model) into a pass/fail gate for a fresh eval run. This module is **pure**: no
Ollama, no network. The eval *run* lives in ``eval_cli``; here we only compare
summaries.

Design (gate1, CAIO lens):

- The golden set is tiny, so precision/recall are coarse and noisy across
  repeats. The regression floor is therefore derived from the **spread**, not the
  mean: ``floor = max(abs_floor, mean - k*stdev)`` with default ``k = 2.0``.
- Only the **overall** precision and recall are gated. Per-dimension/severity
  breakdowns are recorded in the baseline for context but not gated — far too
  noisy at N≈7.
- A baseline is **model-specific** (its provenance pins the model).
  ``check_regression`` refuses to grade across models, so an Ollama-cloud run
  cannot be graded against a free-local baseline (and vice versa). Each tier
  keeps its own ``evals/baselines/<model>.json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Bump when the committed baseline file shape changes, so loaders can reject a
# baseline written by a newer tool (mirrors report.SCHEMA_VERSION / EVAL_SCHEMA_VERSION).
BASELINE_SCHEMA_VERSION = 1

DEFAULT_K_STDEV = 2.0
DEFAULT_ABS_FLOOR = 0.0

# The metrics we gate. Per-dimension/severity are intentionally excluded.
_GATED_METRICS = ("precision", "recall")


# ----------------------------------------------------------------- floor rule

def regression_floor(mean: float | None, stdev: float | None,
                     k: float = DEFAULT_K_STDEV,
                     abs_floor: float = DEFAULT_ABS_FLOOR) -> float | None:
    """The lowest fresh value that is NOT a regression.

    ``max(abs_floor, mean - k*stdev)``. A ``None`` mean means the baseline never
    produced a sample for this metric, so it is **not gateable** -> ``None``. A
    ``None`` stdev (single-repeat baseline) is treated as ``0`` so the floor
    collapses to the mean.
    """
    if mean is None:
        return None
    spread = (stdev or 0.0) * k
    return max(abs_floor, mean - spread)


# ----------------------------------------------------------------- result types

@dataclass
class MetricCheck:
    name: str
    observed: float | None
    floor: float | None
    ok: bool
    reason: str


@dataclass
class GuardResult:
    passed: bool
    model_ok: bool
    checks: list[MetricCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        def pct(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.2%}"

        head = "PASS" if self.passed else "FAIL"
        lines = [f"# Eval regression guard — {head}", ""]
        for n in self.notes:
            lines.append(f"> {n}")
        if self.notes:
            lines.append("")
        for c in self.checks:
            mark = "ok" if c.ok else "REGRESSION"
            lines.append(f"- [{mark}] {c.name}: observed {pct(c.observed)} "
                         f"vs floor {pct(c.floor)} — {c.reason}")
        return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- the gate

def check_regression(baseline: dict, fresh_summary: dict, *,
                     fresh_model: str | None = None,
                     k: float | None = None,
                     abs_floor: float | None = None) -> GuardResult:
    """Grade ``fresh_summary`` against a committed ``baseline``.

    ``fresh_summary`` is a ``summarize_repeats`` dict (``precision_mean`` /
    ``precision_stdev`` / ``recall_mean`` / ``recall_stdev``). Guard params come
    from ``baseline["guard"]`` unless overridden by ``k`` / ``abs_floor``.

    Model safety: when ``fresh_model`` is given and differs from the baseline's
    pinned model, the result fails fast (``model_ok = False``) and the metrics
    are NOT graded — comparing across models is meaningless.
    """
    guard = baseline.get("guard", {}) or {}
    k = guard.get("k_stdev", DEFAULT_K_STDEV) if k is None else k
    abs_floor = guard.get("abs_floor", DEFAULT_ABS_FLOOR) if abs_floor is None else abs_floor

    base_summary = baseline.get("summary", {}) or {}
    base_model = (baseline.get("provenance", {}) or {}).get("model")

    notes: list[str] = []

    # --- model-match safety check
    model_ok = True
    if fresh_model is not None and base_model is not None and fresh_model != base_model:
        model_ok = False
        notes.append(
            f"model mismatch: baseline was captured on '{base_model}' but the fresh "
            f"run used '{fresh_model}'. A baseline is model-specific; grade against "
            f"the matching evals/baselines/<model>.json. Not graded."
        )
        return GuardResult(passed=False, model_ok=False, checks=[], notes=notes)

    if fresh_model is None and base_model is not None:
        notes.append(f"fresh model unknown; assuming it matches baseline '{base_model}'.")
    if fresh_model is not None and base_model is None:
        notes.append("baseline has no pinned model in provenance; model-match safety "
                     "check skipped (cannot verify the run used the right model).")

    # --- per-metric floor gate
    checks: list[MetricCheck] = []
    for m in _GATED_METRICS:
        base_mean = base_summary.get(f"{m}_mean")
        base_stdev = base_summary.get(f"{m}_stdev")
        floor = regression_floor(base_mean, base_stdev, k=k, abs_floor=abs_floor)
        observed = fresh_summary.get(f"{m}_mean")

        if floor is None:
            checks.append(MetricCheck(m, observed, None, True,
                                      "baseline n/a — not gated"))
        elif observed is None:
            # baseline expected a number; fresh produced none -> the harness
            # stopped finding anything. That is a regression, not a free pass.
            checks.append(MetricCheck(m, None, floor, False,
                                      "fresh run produced no sample where baseline had one"))
        else:
            ok = observed >= floor
            checks.append(MetricCheck(
                m, observed, floor, ok,
                "within margin" if ok else f"below floor by {floor - observed:.2%}"))

    passed = model_ok and all(c.ok for c in checks)
    return GuardResult(passed=passed, model_ok=model_ok, checks=checks, notes=notes)


# ----------------------------------------------------------------- build / load

def build_baseline(eval_payload: dict, *, provenance: dict,
                   guard: dict | None = None) -> dict:
    """Assemble a committable baseline from a ``kagura-eval --format json`` payload.

    ``eval_payload`` is the CLI envelope ``{model, effort, repeats, summary,
    runs}``. ``provenance`` carries the reproducibility metadata (model, digest,
    ollama version, effort, seed, repeats, captured_at, host). Per-dimension /
    severity breakdowns are copied from the first run for context only.
    """
    summary = eval_payload.get("summary", {}) or {}
    runs = eval_payload.get("runs", []) or []
    first = runs[0] if runs else {}
    return {
        "baseline_schema_version": BASELINE_SCHEMA_VERSION,
        "provenance": dict(provenance),
        "guard": dict(guard) if guard is not None
        else {"k_stdev": DEFAULT_K_STDEV, "abs_floor": DEFAULT_ABS_FLOOR},
        "summary": dict(summary),
        # recorded for context, NOT gated (too noisy at this N):
        "by_dimension": dict(first.get("by_dimension", {})),
        "by_severity": dict(first.get("by_severity", {})),
    }


def load_baseline(path: str | Path) -> dict:
    """Load and validate a committed baseline file.

    Raises ``ValueError`` on a malformed file or a schema version newer than this
    tool understands.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version = data.get("baseline_schema_version")
    if version != BASELINE_SCHEMA_VERSION:
        raise ValueError(
            f"baseline {str(path)!r}: schema_version {version!r} != "
            f"{BASELINE_SCHEMA_VERSION} (this tool); regenerate the baseline")
    if "summary" not in data or not isinstance(data["summary"], dict):
        raise ValueError(f"baseline {str(path)!r}: missing 'summary' block")
    if "provenance" not in data:
        raise ValueError(f"baseline {str(path)!r}: missing 'provenance' block")
    return data
