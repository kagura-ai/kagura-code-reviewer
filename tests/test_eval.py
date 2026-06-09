"""Tests for the eval (quality-measurement) harness — issue #5.

The scoring core is pure Python and fully unit-tested here with no Ollama.
The match predicate deliberately reuses harness._symptom / _norm_title so the
eval scores findings the same way the reviewer dedups them.
"""
from __future__ import annotations

import textwrap

from kagura_code_reviewer.report import Finding, Severity


def _f(file, line, title, dimension="correctness", severity=Severity.HIGH, rationale=""):
    return Finding(dimension, severity, file, line, title, rationale or title, "fix")


def _bug(file, line, symptom=None, dimension="correctness", severity=Severity.HIGH, title=""):
    from kagura_code_reviewer.eval_harness import GoldenBug
    return GoldenBug(file=file, line=line, symptom=symptom, dimension=dimension,
                     severity=severity, title=title)


# ---------------------------------------------------------------- match predicate

def test_match_same_file_symptom_and_close_line():
    from kagura_code_reviewer.eval_harness import match
    finding = _f("stats.py", 12, "ZeroDivisionError when list empty")
    bug = _bug("stats.py", 14, symptom="zerodivision")
    assert match(finding, bug) is True  # |12-14| <= 5, symptom both "zerodivision"


def test_match_line_outside_bucket_fails():
    from kagura_code_reviewer.eval_harness import match
    finding = _f("stats.py", 12, "ZeroDivisionError when list empty")
    bug = _bug("stats.py", 30, symptom="zerodivision")
    assert match(finding, bug) is False


def test_match_different_file_fails():
    from kagura_code_reviewer.eval_harness import match
    assert match(_f("a.py", 1, "index out of range"),
                 _bug("b.py", 1, symptom="index")) is False


def test_match_different_symptom_fails():
    from kagura_code_reviewer.eval_harness import match
    finding = _f("a.py", 5, "KeyError on missing key")
    bug = _bug("a.py", 5, symptom="zerodivision")
    assert match(finding, bug) is False


def test_match_lineless_bug_matches_on_file_and_symptom():
    from kagura_code_reviewer.eval_harness import match
    finding = _f("a.py", 99, "IndexError: list index out of range")
    bug = _bug("a.py", None, symptom="index")
    assert match(finding, bug) is True


def test_match_falls_back_to_norm_title_when_no_symptom():
    from kagura_code_reviewer.eval_harness import match
    # neither title hits a _SYMPTOM_PATTERN -> norm_title equality
    finding = _f("a.py", 3, "Off-by-one in slice!")
    bug = _bug("a.py", 4, symptom=None, title="off by one in slice")
    assert match(finding, bug) is True


# ---------------------------------------------------------------- score_case

def test_score_case_tp_fp_fn():
    from kagura_code_reviewer.eval_harness import score_case
    bugs = [_bug("a.py", 10, symptom="zerodivision"),
            _bug("a.py", 50, symptom="index")]
    findings = [_f("a.py", 11, "ZeroDivisionError"),          # TP for bug 0
                _f("a.py", 80, "TypeError somewhere")]        # FP (no bug there)
    s = score_case("c1", "seeded", findings, bugs)
    assert (s.tp, s.fp, s.fn) == (1, 1, 1)


def test_score_case_greedy_one_to_one():
    from kagura_code_reviewer.eval_harness import score_case
    bugs = [_bug("a.py", 10, symptom="zerodivision")]
    # two findings hit the same single bug -> 1 TP, the extra is a FP
    findings = [_f("a.py", 10, "ZeroDivisionError"),
                _f("a.py", 12, "division by zero")]
    s = score_case("c1", "seeded", findings, bugs)
    assert (s.tp, s.fp, s.fn) == (1, 1, 0)


def test_score_case_perfect():
    from kagura_code_reviewer.eval_harness import score_case
    bugs = [_bug("a.py", 10, symptom="index")]
    findings = [_f("a.py", 10, "index out of range")]
    s = score_case("c1", "seeded", findings, bugs)
    assert (s.tp, s.fp, s.fn) == (1, 0, 0)


# ---------------------------------------------------------------- aggregate + source split

def test_aggregate_precision_excludes_historical_fp():
    from kagura_code_reviewer.eval_harness import score_case, aggregate_scores
    # seeded: 1 TP, 0 FP ; historical: 1 TP plus an extra finding (would be FP)
    seeded = score_case("s", "seeded",
                        [_f("a.py", 10, "index out of range")],
                        [_bug("a.py", 10, symptom="index")])
    historical = score_case("h", "historical",
                            [_f("b.py", 10, "index out of range"),
                             _f("b.py", 99, "some other thing")],
                            [_bug("b.py", 10, symptom="index")])
    agg = aggregate_scores([seeded, historical])
    # precision counts seeded only -> 1/(1+0) = 1.0 ; the historical extra finding
    # must NOT drag precision down (FP uncountable on real diffs).
    assert agg.precision == 1.0
    # recall counts all cases -> 2 TP / 2 bugs = 1.0
    assert agg.recall == 1.0


def test_aggregate_recall_counts_all_sources():
    from kagura_code_reviewer.eval_harness import score_case, aggregate_scores
    seeded = score_case("s", "seeded", [], [_bug("a.py", 1, symptom="index")])  # missed
    historical = score_case("h", "historical",
                            [_f("b.py", 1, "index out of range")],
                            [_bug("b.py", 1, symptom="index")])               # caught
    agg = aggregate_scores([seeded, historical])
    assert agg.recall == 0.5  # 1 caught of 2 bugs


def test_aggregate_precision_none_when_no_seeded_findings():
    from kagura_code_reviewer.eval_harness import score_case, aggregate_scores
    historical = score_case("h", "historical",
                            [_f("b.py", 1, "index out of range")],
                            [_bug("b.py", 1, symptom="index")])
    agg = aggregate_scores([historical])
    assert agg.precision is None  # no seeded TP+FP to divide


def test_aggregate_breakdown_by_dimension():
    from kagura_code_reviewer.eval_harness import score_case, aggregate_scores
    s = score_case("s", "seeded",
                   [_f("a.py", 1, "index out of range", dimension="correctness"),
                    _f("a.py", 50, "perf thing", dimension="performance")],
                   [_bug("a.py", 1, symptom="index", dimension="correctness"),
                    _bug("a.py", 80, symptom="zerodivision", dimension="correctness")])
    agg = aggregate_scores([s])
    # correctness: 1 TP, 1 FN -> recall 0.5
    assert agg.by_dimension["correctness"]["recall"] == 0.5


# ---------------------------------------------------------------- repeat variance

def test_summarize_repeats_mean_and_stdev():
    from kagura_code_reviewer.eval_harness import EvalResult, summarize_repeats
    r1 = EvalResult(precision=1.0, recall=0.5, tp=1, fp=0, fn=1,
                    by_dimension={}, by_severity={}, cases=[])
    r2 = EvalResult(precision=0.5, recall=1.0, tp=2, fp=2, fn=0,
                    by_dimension={}, by_severity={}, cases=[])
    stats = summarize_repeats([r1, r2])
    assert stats["repeats"] == 2
    assert stats["precision_mean"] == 0.75
    assert stats["recall_mean"] == 0.75
    assert stats["precision_stdev"] > 0


# ---------------------------------------------------------------- manifest loading

def test_load_golden_reads_manifest_and_diffs(tmp_path):
    from kagura_code_reviewer.eval_harness import load_golden
    (tmp_path / "case1.diff").write_text(textwrap.dedent("""\
        --- a/stats.py
        +++ b/stats.py
        @@ -1,2 +1,2 @@
        -def avg(xs): return sum(xs) / max(len(xs), 1)
        +def avg(xs): return sum(xs) / len(xs)
    """))
    (tmp_path / "manifest.toml").write_text(textwrap.dedent("""\
        [[case]]
        name = "zerodiv_avg"
        source = "seeded"
        diff_file = "case1.diff"
        [[case.bug]]
        file = "stats.py"
        line = 1
        symptom = "zerodivision"
        dimension = "correctness"
        severity = "high"
    """))
    cases = load_golden(tmp_path)
    assert len(cases) == 1
    c = cases[0]
    assert c.name == "zerodiv_avg"
    assert c.source == "seeded"
    assert "len(xs)" in c.diff
    assert len(c.bugs) == 1
    assert c.bugs[0].symptom == "zerodivision"
    assert c.bugs[0].severity == Severity.HIGH


# ---------------------------------------------------------------- result serialization

def test_aggregate_dimension_precision_excludes_historical_tp():
    # seeded correctness: 1 TP + 1 FP -> precision 0.5 ; historical correctness:
    # 1 TP must NOT inflate the seeded precision numerator (it stays 0.5), but it
    # DOES count toward recall.
    from kagura_code_reviewer.eval_harness import aggregate_scores, score_case
    seeded = score_case("s", "seeded",
                        [_f("a.py", 1, "index out of range"),
                         _f("a.py", 50, "spurious thing")],
                        [_bug("a.py", 1, symptom="index")])
    historical = score_case("h", "historical",
                            [_f("b.py", 1, "index out of range")],
                            [_bug("b.py", 1, symptom="index")])
    agg = aggregate_scores([seeded, historical])
    assert agg.by_dimension["correctness"]["precision"] == 0.5  # 1/(1+1), seeded only
    assert agg.by_dimension["correctness"]["recall"] == 1.0     # 2 caught of 2 bugs


def test_aggregate_hides_historical_fp_per_case():
    from kagura_code_reviewer.eval_harness import aggregate_scores, score_case
    historical = score_case("h", "historical",
                            [_f("b.py", 1, "index out of range"),
                             _f("b.py", 99, "unknown latent thing")],
                            [_bug("b.py", 1, symptom="index")])
    agg = aggregate_scores([historical])
    case = next(c for c in agg.cases if c["name"] == "h")
    assert case["fp"] is None  # FP uncountable on historical diffs


def test_summarize_repeats_all_none_precision_yields_none_stdev():
    from kagura_code_reviewer.eval_harness import EvalResult, summarize_repeats
    # precision None on every run -> mean AND stdev are None, not 0.0
    r = EvalResult(precision=None, recall=0.5, tp=0, fp=0, fn=1,
                   by_dimension={}, by_severity={}, cases=[])
    stats = summarize_repeats([r, r])
    assert stats["precision_mean"] is None
    assert stats["precision_stdev"] is None


def test_load_golden_missing_diff_raises(tmp_path):
    from kagura_code_reviewer.eval_harness import load_golden
    (tmp_path / "manifest.toml").write_text(textwrap.dedent("""\
        [[case]]
        name = "nodiff"
        source = "seeded"
        [[case.bug]]
        file = "x.py"
        symptom = "index"
    """))
    import pytest
    with pytest.raises(ValueError):
        load_golden(tmp_path)


def test_load_golden_missing_name_raises(tmp_path):
    from kagura_code_reviewer.eval_harness import load_golden
    (tmp_path / "manifest.toml").write_text(textwrap.dedent("""\
        [[case]]
        source = "seeded"
        diff = "d"
    """))
    import pytest
    with pytest.raises(ValueError):
        load_golden(tmp_path)


def test_load_golden_missing_bug_file_raises(tmp_path):
    from kagura_code_reviewer.eval_harness import load_golden
    (tmp_path / "manifest.toml").write_text(textwrap.dedent("""\
        [[case]]
        name = "c"
        source = "seeded"
        diff = "d"
        [[case.bug]]
        symptom = "index"
    """))
    import pytest
    with pytest.raises(ValueError):
        load_golden(tmp_path)


def test_committed_golden_set_loads_and_is_wellformed():
    from pathlib import Path

    from kagura_code_reviewer.eval_harness import load_golden
    from kagura_code_reviewer.review.harness import _SYMPTOM_PATTERNS

    golden_dir = Path(__file__).resolve().parents[1] / "evals" / "golden"
    cases = load_golden(golden_dir)
    assert len(cases) >= 6  # keep the set small but non-trivial
    valid_symptoms = {name for name, _ in _SYMPTOM_PATTERNS}
    seeded = [c for c in cases if c.source == "seeded"]
    assert seeded, "golden set must contain seeded (precision+recall) cases"
    for c in cases:
        assert c.source in {"seeded", "historical"}
        assert c.diff.strip(), f"case {c.name} has an empty diff"
        assert c.bugs, f"case {c.name} has no labeled bugs"
        for b in c.bugs:
            assert b.file
            # symptoms used in the manifest must be ones the harness recognizes
            if b.symptom is not None:
                assert b.symptom in valid_symptoms, f"{c.name}: unknown symptom {b.symptom}"


def test_eval_result_to_json_roundtrips():
    import json
    from kagura_code_reviewer.eval_harness import EvalResult
    r = EvalResult(precision=1.0, recall=0.5, tp=1, fp=0, fn=1,
                   by_dimension={"correctness": {"recall": 0.5}},
                   by_severity={}, cases=[])
    payload = json.loads(r.to_json())
    assert payload["schema_version"] == 1
    assert payload["precision"] == 1.0
    assert payload["recall"] == 0.5
    assert payload["counts"] == {"tp": 1, "fp": 0, "fn": 1}
