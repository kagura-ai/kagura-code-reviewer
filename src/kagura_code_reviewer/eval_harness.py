"""Quality-measurement (eval) harness — issue #5.

Scores the review harness against a golden set of diffs with labeled known bugs
and reports precision/recall (FP/FN), broken down by dimension and severity.

Design (gate1, QA Lead):
- The match predicate REUSES the reviewer's own matching (`harness._symptom` /
  `_norm_title` + line proximity) so the eval scores findings the same way the
  reviewer dedups them.
- Metrics are split by source: `seeded` synthetic diffs give precision AND recall
  (the bug set is fully controlled); `historical` real diffs give recall only
  (false positives are uncountable — a real diff may carry unknown latent bugs).
- Finders are stochastic, so callers run N repeats and report mean ± stdev.

This module is pure (no Ollama, no network). The CLI runner that drives the real
harness lives in ``eval_cli.py``.
"""
from __future__ import annotations

import json
import statistics
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .report import Finding, Severity, parse_severity
from .review.harness import _norm_title, _symptom

DEFAULT_BUCKET = 5
_SEEDED = "seeded"

# Bump when the eval JSON envelope changes shape, so consumers can gate on it
# (mirrors report.SCHEMA_VERSION).
EVAL_SCHEMA_VERSION = 1


@dataclass
class GoldenBug:
    """A labeled known bug in a golden case.

    `symptom` is a canonical symptom name (matching harness._SYMPTOM_PATTERNS,
    e.g. "zerodivision", "index", "key", "none"). When None, matching falls back
    to normalized-title equality, so `title` should then carry the bug's title.
    """
    file: str
    line: int | None
    dimension: str
    severity: Severity
    symptom: str | None = None
    title: str = ""


@dataclass
class GoldenCase:
    name: str
    diff: str
    source: str  # "seeded" | "historical"
    bugs: list[GoldenBug]
    repo_files: dict[str, str] = field(default_factory=dict)


@dataclass
class CaseScore:
    name: str
    source: str
    tp: int
    fp: int
    fn: int
    # nested {dimension: {"tp":n,"fp":n,"fn":n}} and same for severity name
    dim_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    sev_counts: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class EvalResult:
    precision: float | None
    recall: float | None
    tp: int
    fp: int
    fn: int
    by_dimension: dict
    by_severity: dict
    cases: list

    def to_json(self) -> str:
        return json.dumps({
            "schema_version": EVAL_SCHEMA_VERSION,
            "precision": self.precision,
            "recall": self.recall,
            "counts": {"tp": self.tp, "fp": self.fp, "fn": self.fn},
            "by_dimension": self.by_dimension,
            "by_severity": self.by_severity,
            "cases": self.cases,
        }, indent=2)

    def to_markdown(self) -> str:
        def pct(x):
            return "n/a" if x is None else f"{x:.2%}"
        lines = [
            "# Eval — review harness quality",
            "",
            f"- **precision** (seeded only): {pct(self.precision)}",
            f"- **recall** (all sources): {pct(self.recall)}",
            f"- counts: TP={self.tp} FP={self.fp} FN={self.fn}",
            "",
            "## By dimension",
        ]
        for dim, m in sorted(self.by_dimension.items()):
            lines.append(f"- {dim}: precision {pct(m.get('precision'))}, "
                         f"recall {pct(m.get('recall'))}")
        lines.append("")
        lines.append("## By severity")
        for sev, m in sorted(self.by_severity.items()):
            lines.append(f"- {sev}: precision {pct(m.get('precision'))}, "
                         f"recall {pct(m.get('recall'))}")
        return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- match predicate

def _finding_key(f: Finding) -> str:
    return _symptom(f) or _norm_title(f.title)


def _bug_key(b: GoldenBug) -> str:
    return b.symptom or _norm_title(b.title)


def match(finding: Finding, bug: GoldenBug, bucket: int = DEFAULT_BUCKET) -> bool:
    """True when `finding` counts as catching `bug`.

    Same rule the harness dedup uses: same file, same symptom-class (or normalized
    title when neither has a symptom), and lines within `bucket`. A None line on
    either side means location-agnostic within the same file+symptom.
    """
    if finding.file != bug.file:
        return False
    if _finding_key(finding) != _bug_key(bug):
        return False
    if finding.line is None or bug.line is None:
        return True
    return abs(finding.line - bug.line) <= bucket


def _precision(tp: int, fp: int) -> float | None:
    return None if (tp + fp) == 0 else tp / (tp + fp)


def _recall(tp: int, fn: int) -> float | None:
    return None if (tp + fn) == 0 else tp / (tp + fn)


def _bump(table: dict[str, dict[str, int]], key: str, field_: str) -> None:
    table.setdefault(key, {"tp": 0, "fp": 0, "fn": 0})[field_] += 1


# ----------------------------------------------------------------- scoring

def score_case(name: str, source: str, findings: list[Finding],
               bugs: list[GoldenBug], bucket: int = DEFAULT_BUCKET) -> CaseScore:
    """Greedy 1:1 match of findings against labeled bugs.

    Each finding may claim at most one not-yet-claimed bug (TP); unclaimed
    findings are FP; unclaimed bugs are FN. Per-dimension/severity tallies key
    TP/FN by the bug's labels and FP by the finding's labels.
    """
    matched_bugs: set[int] = set()
    dim: dict[str, dict[str, int]] = {}
    sev: dict[str, dict[str, int]] = {}
    tp = fp = 0

    for f in findings:
        hit = None
        for bi, b in enumerate(bugs):
            if bi in matched_bugs:
                continue
            if match(f, b, bucket):
                hit = bi
                break
        if hit is not None:
            matched_bugs.add(hit)
            tp += 1
            b = bugs[hit]
            _bump(dim, b.dimension, "tp")
            _bump(sev, b.severity.name, "tp")
        else:
            fp += 1
            _bump(dim, f.dimension, "fp")
            _bump(sev, f.severity.name, "fp")

    fn = 0
    for bi, b in enumerate(bugs):
        if bi not in matched_bugs:
            fn += 1
            _bump(dim, b.dimension, "fn")
            _bump(sev, b.severity.name, "fn")

    return CaseScore(name=name, source=source, tp=tp, fp=fp, fn=fn,
                     dim_counts=dim, sev_counts=sev)


def _sum_counts(scores: list[CaseScore], attr: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for s in scores:
        for key, c in getattr(s, attr).items():
            d = out.setdefault(key, {"tp": 0, "fp": 0, "fn": 0})
            d["tp"] += c.get("tp", 0)
            d["fp"] += c.get("fp", 0)
            d["fn"] += c.get("fn", 0)
    return out


def _breakdown(all_scores: list[CaseScore], seeded_scores: list[CaseScore],
               attr: str) -> dict:
    # precision is seeded-only (TP and FP both controlled); recall is all-cases.
    seeded = _sum_counts(seeded_scores, attr)
    allc = _sum_counts(all_scores, attr)
    out = {}
    for key in set(seeded) | set(allc):
        sp = seeded.get(key, {"tp": 0, "fp": 0})
        ac = allc.get(key, {"tp": 0, "fn": 0})
        out[key] = {"precision": _precision(sp["tp"], sp["fp"]),
                    "recall": _recall(ac["tp"], ac["fn"])}
    return out


def aggregate_scores(scores: list[CaseScore]) -> EvalResult:
    """Aggregate per-case scores into precision/recall with the source split:
    precision is computed over SEEDED cases only (both TP and FP are controlled
    there; FP is uncountable on real diffs); recall is computed over ALL cases.
    The same split applies to the per-dimension/severity breakdown.
    """
    seeded = [s for s in scores if s.source == _SEEDED]
    seeded_tp = sum(s.tp for s in seeded)
    seeded_fp = sum(s.fp for s in seeded)
    all_tp = sum(s.tp for s in scores)
    all_fn = sum(s.fn for s in scores)

    # historical FP is meaningless (unknown latent bugs) -> hide it per-case too.
    cases = [{"name": s.name, "source": s.source, "tp": s.tp,
              "fp": s.fp if s.source == _SEEDED else None, "fn": s.fn}
             for s in scores]

    return EvalResult(
        precision=_precision(seeded_tp, seeded_fp),
        recall=_recall(all_tp, all_fn),
        tp=all_tp,
        fp=seeded_fp,
        fn=all_fn,
        by_dimension=_breakdown(scores, seeded, "dim_counts"),
        by_severity=_breakdown(scores, seeded, "sev_counts"),
        cases=cases,
    )


def summarize_repeats(results: list[EvalResult]) -> dict:
    """Mean ± stdev of precision/recall across repeated eval runs (finders are
    stochastic). None precision/recall values are dropped before averaging."""
    precisions = [r.precision for r in results if r.precision is not None]
    recalls = [r.recall for r in results if r.recall is not None]

    def _stdev(xs):
        if not xs:
            return None  # no valid samples -> stdev is as meaningless as the mean
        return statistics.stdev(xs) if len(xs) > 1 else 0.0

    return {
        "repeats": len(results),
        "precision_mean": statistics.fmean(precisions) if precisions else None,
        "precision_stdev": _stdev(precisions),
        "recall_mean": statistics.fmean(recalls) if recalls else None,
        "recall_stdev": _stdev(recalls),
    }


# ----------------------------------------------------------------- golden loading

def load_golden(path: str | Path) -> list[GoldenCase]:
    """Load golden cases from a directory containing manifest.toml.

    manifest.toml shape:

        [[case]]
        name = "zerodiv_avg"
        source = "seeded"          # "seeded" | "historical"
        diff_file = "case1.diff"   # or inline: diff = '''...'''
          [[case.bug]]
          file = "stats.py"
          line = 1                 # optional
          symptom = "zerodivision" # optional (falls back to title)
          dimension = "correctness"
          severity = "high"
          title = ""               # optional, used when symptom is absent
    """
    root = Path(path)
    manifest = root / "manifest.toml"
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    cases: list[GoldenCase] = []
    for entry in data.get("case", []):
        diff = entry.get("diff")
        if diff is None and entry.get("diff_file"):
            diff = (root / entry["diff_file"]).read_text(encoding="utf-8")
        if diff is None:
            raise ValueError(
                f"golden case {entry.get('name')!r}: needs 'diff' or 'diff_file'")
        bugs = [
            GoldenBug(
                file=str(b["file"]),
                line=b.get("line"),
                dimension=str(b.get("dimension", "correctness")),
                severity=parse_severity(str(b.get("severity", "info"))),
                symptom=b.get("symptom"),
                title=str(b.get("title", "")),
            )
            for b in entry.get("bug", [])
        ]
        cases.append(GoldenCase(
            name=str(entry["name"]),
            diff=diff,
            source=str(entry.get("source", _SEEDED)),
            bugs=bugs,
            repo_files=dict(entry.get("repo_files", {})),
        ))
    return cases
