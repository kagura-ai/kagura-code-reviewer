"""`kagura-eval` — run the review harness over a golden set and report FP/FN.

This is the I/O / orchestration layer for the eval harness; the scoring core is
pure and lives in ``eval_harness.py``. The default run drives real Ollama
(zero-config advisor), but ``run_eval`` takes an injected client so the scoring
path is fully unit-testable without a backend.

Scope (issue #5): this prints precision/recall for a one-off run. Capturing a
committed baseline number and a CI regression guard are a deliberate follow-up.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import typer

from .cli import Effort, OutputFormat, Provider
from .eval_baseline import build_baseline, check_regression, load_baseline
from .eval_harness import (
    DEFAULT_BUCKET,
    EvalResult,
    GoldenCase,
    aggregate_scores,
    load_golden,
    score_case,
    summarize_repeats,
)
from .report import Report
from .review.harness import resolve_tier, review_harness


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

app = typer.Typer(add_completion=False, help="Measure review-harness quality (FP/FN) on a golden set.")


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.2%}"


class _DictRepo:
    """In-memory repo for golden cases: serves labeled source files to the
    sandboxed read_file/grep/list_files tools. Real cases can leave repo_files
    empty — the diff alone is enough for the finders."""

    def __init__(self, files: dict[str, str]):
        self._files = dict(files)

    def read_file(self, path: str, max_bytes: int = 20000) -> str:
        if path not in self._files:
            return f"error: file not found: {path}"  # don't truncate the error
        return self._files[path][:max_bytes]

    def grep(self, pattern: str, max_results: int = 50) -> str:
        # Mirror RepoTools.grep's "path:lineno: line" shape so a finder gets
        # navigable matches (substring, not regex — safe for an in-memory stub).
        hits = []
        for p, content in self._files.items():
            for i, line in enumerate(content.splitlines(), start=1):
                if pattern in line:
                    hits.append(f"{p}:{i}: {line.strip()}")
        return "\n".join(hits[:max_results]) or "no matches"

    def list_files(self, subdir: str = ".") -> list[str]:
        # Mirror RepoTools.list_files: honor the subdir filter the model passes.
        if subdir in (".", "", None):
            return sorted(self._files)
        prefix = subdir.rstrip("/") + "/"
        return sorted(p for p in self._files if p.startswith(prefix))


def _default_repo(case: GoldenCase) -> _DictRepo:
    return _DictRepo(case.repo_files)


def run_eval(cases, client, *, tier=None, repeats: int = 1, bucket: int = DEFAULT_BUCKET,
             repo_factory=None, max_concurrency: int = 1,
             min_confidence: float = 0.0) -> list[EvalResult]:
    """Run the review harness over each case `repeats` times, scoring each pass.

    `client` implements the ChatClient protocol (chat(messages, tools)) and is
    used as both finder and verifier — pass a fake for tests, a real Ollama
    client for the baseline. Returns one EvalResult per repeat.
    """
    tier = tier or resolve_tier("med")
    repo_factory = repo_factory or _default_repo
    results: list[EvalResult] = []
    for _ in range(repeats):
        scores = []
        for case in cases:
            report: Report = review_harness(
                client, client, repo_factory(case), case.diff, None, tier,
                max_concurrency=max_concurrency, min_confidence=min_confidence,
            )
            scores.append(score_case(case.name, case.source, report.findings,
                                     case.bugs, bucket))
        results.append(aggregate_scores(scores))
    return results


@app.command()
def main(
    golden_dir: Path = typer.Option(Path("evals/golden"), "--golden-dir",
                                    help="Directory containing manifest.toml + diffs."),
    effort: Effort = typer.Option(Effort.med, "--effort", help="Review effort tier."),
    provider: Provider = typer.Option(Provider.ollama, "--provider", help="Backend provider."),
    model: str = typer.Option(None, "--model", help="Override the model."),
    seed: int = typer.Option(None, "--seed", help="Pin the backend seed for reproducibility."),
    repeats: int = typer.Option(1, "--repeats", min=1, help="Repeat runs (finders are stochastic)."),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    min_confidence: float = typer.Option(0.0, "--min-confidence"),
    bucket: int = typer.Option(DEFAULT_BUCKET, "--bucket", help="Line-proximity match window."),
    fmt: OutputFormat = typer.Option(OutputFormat.md, "--format", help="Output format."),
    out: Path = typer.Option(None, "--out", help="Write the report here instead of stdout."),
    baseline_out: Path = typer.Option(
        None, "--baseline-out",
        help="Write a committable baseline (summary + provenance + guard) to this path."),
    check_baseline: Path = typer.Option(
        None, "--check-baseline",
        help="Compare this run against a committed baseline; exit 1 on regression."),
) -> None:
    cases = load_golden(golden_dir)
    if not cases:
        typer.echo(f"no golden cases found in {golden_dir}", err=True)
        raise typer.Exit(2)

    # Lazy: build_review_client builds the real client (and is monkeypatched in tests
    # via the cli module, so it must be looked up at call time, not import time).
    from .cli import build_review_client
    client, label = build_review_client(provider.value, model, local=False, cloud=False,
                                        timeout=120.0, seed=seed)
    tier = resolve_tier(effort.value)

    results = run_eval(cases, client, tier=tier, repeats=repeats, bucket=bucket,
                       max_concurrency=concurrency, min_confidence=min_confidence)
    stats = summarize_repeats(results)

    payload = {
        "model": label,
        "effort": effort.value,
        "repeats": repeats,
        "summary": stats,
        "runs": [json.loads(r.to_json()) for r in results],
    }

    if fmt is OutputFormat.json:
        rendered = json.dumps(payload, indent=2)
    else:
        rendered = (f"# Eval baseline — {label} (effort={effort.value}, repeats={repeats})\n\n"
                    f"- precision (seeded): mean {_pct(stats['precision_mean'])} "
                    f"± {_pct(stats['precision_stdev'])}\n"
                    f"- recall (all): mean {_pct(stats['recall_mean'])} "
                    f"± {_pct(stats['recall_stdev'])}\n\n"
                    + results[0].to_markdown())

    if out:
        out.write_text(rendered, encoding="utf-8")
    else:
        typer.echo(rendered)

    # Optional: persist a committable baseline (summary + provenance + guard).
    if baseline_out:
        baseline = build_baseline(payload, provenance={
            "model": label,
            "effort": effort.value,
            "seed": seed,
            "repeats": repeats,
            "captured_at": _utc_now(),
            "host": platform.node(),
        })
        baseline_out.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        typer.echo(f"wrote baseline → {baseline_out}", err=True)

    # Optional: gate this run against a committed baseline (exit 1 on regression).
    # The guard report is the primary output of a --check-baseline run, so it goes
    # to stdout (capturable by pipelines); only the exit code signals pass/fail.
    if check_baseline:
        guard = check_regression(load_baseline(check_baseline), stats, fresh_model=label)
        typer.echo(guard.to_markdown())
        if not guard.passed:
            raise typer.Exit(1)


if __name__ == "__main__":  # pragma: no cover
    app()
