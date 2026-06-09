"""Tests for the eval regression guard (eval_baseline) — issue #7.

Pure-Python, network-free: the guard only compares summaries, it never runs the
harness. These tests pin the σ-margin floor rule, the model-match safety check,
and the n/a-metric handling agreed in the gate1 design review.
"""
from __future__ import annotations

import json

import pytest

from kagura_code_reviewer.eval_baseline import (
    BASELINE_SCHEMA_VERSION,
    GuardResult,
    build_baseline,
    check_regression,
    load_baseline,
    regression_floor,
)


# --------------------------------------------------------------- regression_floor

def test_regression_floor_uses_spread_not_mean():
    # floor = mean - k*stdev with default k=2.0
    assert regression_floor(0.80, 0.05, k=2.0) == pytest.approx(0.70)


def test_regression_floor_abs_floor_clamps_negative():
    # mean - 2*stdev would be negative; abs_floor wins
    assert regression_floor(0.10, 0.10, k=2.0, abs_floor=0.0) == 0.0


def test_regression_floor_none_mean_is_not_gateable():
    assert regression_floor(None, 0.1) is None


def test_regression_floor_none_stdev_treated_as_zero():
    # a single-repeat baseline has stdev None/0 -> floor collapses to the mean
    assert regression_floor(0.75, None, k=2.0) == pytest.approx(0.75)


# --------------------------------------------------------------- check_regression

def _baseline(precision_mean: float | None = 0.80, precision_stdev: float | None = 0.05,
              recall_mean: float | None = 0.70, recall_stdev: float | None = 0.10,
              model: str | None = "qwen3:14b"):
    return {
        "baseline_schema_version": BASELINE_SCHEMA_VERSION,
        "provenance": {"model": model, "effort": "med", "seed": 7, "repeats": 5},
        "guard": {"k_stdev": 2.0, "abs_floor": 0.0},
        "summary": {
            "repeats": 5,
            "precision_mean": precision_mean, "precision_stdev": precision_stdev,
            "recall_mean": recall_mean, "recall_stdev": recall_stdev,
        },
    }


def test_check_passes_when_fresh_at_or_above_floor():
    base = _baseline()  # precision floor 0.70, recall floor 0.50
    fresh = {"precision_mean": 0.78, "recall_mean": 0.55}
    res = check_regression(base, fresh, fresh_model="qwen3:14b")
    assert isinstance(res, GuardResult)
    assert res.passed is True
    assert res.model_ok is True
    assert all(c.ok for c in res.checks)


def test_check_fails_when_recall_below_floor():
    base = _baseline()  # recall floor = 0.70 - 2*0.10 = 0.50
    fresh = {"precision_mean": 0.78, "recall_mean": 0.40}
    res = check_regression(base, fresh, fresh_model="qwen3:14b")
    assert res.passed is False
    recall = next(c for c in res.checks if c.name == "recall")
    assert recall.ok is False
    assert recall.floor == pytest.approx(0.50)


def test_check_fails_on_model_mismatch_without_grading():
    base = _baseline(model="qwen3:14b")
    fresh = {"precision_mean": 0.99, "recall_mean": 0.99}  # would pass on metrics
    res = check_regression(base, fresh, fresh_model="qwen3-coder:480b-cloud")
    assert res.model_ok is False
    assert res.passed is False
    assert any("model" in n.lower() for n in res.notes)


def test_check_model_match_skipped_when_fresh_model_unknown():
    base = _baseline()
    fresh = {"precision_mean": 0.78, "recall_mean": 0.60}
    res = check_regression(base, fresh, fresh_model=None)  # caller didn't pass a model
    assert res.model_ok is True  # cannot mismatch what we don't know
    assert res.passed is True


def test_check_notes_when_baseline_has_no_pinned_model():
    base = _baseline()
    base["provenance"].pop("model")  # orphaned/hand-authored baseline, no model
    fresh = {"precision_mean": 0.78, "recall_mean": 0.60}
    res = check_regression(base, fresh, fresh_model="qwen3:14b")
    # cannot verify the model -> still graded, but flagged so it isn't silent
    assert res.model_ok is True
    assert any("no pinned model" in n.lower() for n in res.notes)
    assert res.passed is True


def test_check_ungated_metric_when_baseline_na():
    # baseline precision is n/a (no seeded findings captured) -> precision not gated
    base = _baseline(precision_mean=None, precision_stdev=None)
    fresh = {"precision_mean": None, "recall_mean": 0.60}
    res = check_regression(base, fresh, fresh_model="qwen3:14b")
    precision = next(c for c in res.checks if c.name == "precision")
    assert precision.ok is True
    assert precision.floor is None
    assert res.passed is True


def test_check_fails_when_fresh_na_but_baseline_expected_a_number():
    # baseline had precision 0.80; a fresh run that produces no precision sample is
    # a regression (the harness stopped finding anything), not a free pass.
    base = _baseline(precision_mean=0.80, precision_stdev=0.0)
    fresh = {"precision_mean": None, "recall_mean": 0.60}
    res = check_regression(base, fresh, fresh_model="qwen3:14b")
    precision = next(c for c in res.checks if c.name == "precision")
    assert precision.ok is False
    assert res.passed is False


def test_check_honors_override_k_and_abs_floor():
    base = _baseline(recall_mean=0.70, recall_stdev=0.10)
    fresh = {"precision_mean": 0.78, "recall_mean": 0.55}
    # k=1 -> recall floor 0.60; fresh 0.55 now fails where k=2 (floor 0.50) passed
    res = check_regression(base, fresh, fresh_model="qwen3:14b", k=1.0)
    recall = next(c for c in res.checks if c.name == "recall")
    assert recall.floor == pytest.approx(0.60)
    assert recall.ok is False


# --------------------------------------------------------------- build / load

def _eval_payload(model="qwen3:14b"):
    """Mirror the kagura-eval --format json envelope shape."""
    return {
        "model": model,
        "effort": "med",
        "repeats": 5,
        "summary": {
            "repeats": 5,
            "precision_mean": 0.82, "precision_stdev": 0.06,
            "recall_mean": 0.71, "recall_stdev": 0.09,
        },
        "runs": [{
            "schema_version": 1,
            "precision": 0.83, "recall": 0.72,
            "counts": {"tp": 5, "fp": 1, "fn": 2},
            "by_dimension": {"correctness": {"precision": 0.83, "recall": 0.72}},
            "by_severity": {"HIGH": {"precision": 0.83, "recall": 0.72}},
            "cases": [],
        }],
    }


def test_build_baseline_carries_summary_provenance_and_guard():
    payload = _eval_payload()
    prov = {"model": "qwen3:14b", "model_digest": "sha256:abc", "effort": "med",
            "seed": 7, "repeats": 5, "captured_at": "2026-06-10T00:00:00Z"}
    base = build_baseline(payload, provenance=prov)
    assert base["baseline_schema_version"] == BASELINE_SCHEMA_VERSION
    assert base["provenance"]["model"] == "qwen3:14b"
    assert base["provenance"]["model_digest"] == "sha256:abc"
    assert base["guard"] == {"k_stdev": 2.0, "abs_floor": 0.0}
    assert base["summary"]["precision_mean"] == pytest.approx(0.82)
    # per-dimension/severity recorded for context (not gated)
    assert "correctness" in base["by_dimension"]
    assert "HIGH" in base["by_severity"]


def test_build_baseline_accepts_guard_override():
    base = build_baseline(_eval_payload(), provenance={"model": "m"},
                          guard={"k_stdev": 1.5, "abs_floor": 0.1})
    assert base["guard"] == {"k_stdev": 1.5, "abs_floor": 0.1}


def test_build_then_check_roundtrip_passes_on_same_numbers():
    base = build_baseline(_eval_payload(), provenance={"model": "qwen3:14b"})
    # a fresh run identical to the baseline mean must pass
    res = check_regression(base, base["summary"], fresh_model="qwen3:14b")
    assert res.passed is True


def test_load_baseline_roundtrips(tmp_path):
    base = build_baseline(_eval_payload(), provenance={"model": "qwen3:14b"})
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps(base), encoding="utf-8")
    loaded = load_baseline(p)
    assert loaded["provenance"]["model"] == "qwen3:14b"
    assert loaded["summary"]["recall_mean"] == pytest.approx(0.71)


def test_load_baseline_rejects_missing_summary(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"baseline_schema_version": 1, "provenance": {}}),
                 encoding="utf-8")
    with pytest.raises(ValueError):
        load_baseline(p)


def test_load_baseline_rejects_unknown_schema_version(tmp_path):
    p = tmp_path / "future.json"
    p.write_text(json.dumps({"baseline_schema_version": 999, "summary": {},
                             "provenance": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_baseline(p)
