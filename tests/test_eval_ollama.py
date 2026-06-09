"""End-to-end eval smoke against a REAL Ollama backend — issue #7.

Marked `ollama`, so it is excluded from the default network-free suite
(`addopts = -m 'not ollama'`). It runs only when selected explicitly
(`pytest -m ollama`) — the gated eval workflow does this on a GPU/self-hosted
runner. It also self-skips when no local daemon is reachable, so
`pytest -m ollama` on a machine without Ollama is a skip, not an error.

This is the one test that proves the marker actually gates a backend-requiring
test (the rest of the eval suite uses a fake client and stays network-free).
"""
from __future__ import annotations

import urllib.request

import pytest

pytestmark = pytest.mark.ollama

_OLLAMA_URL = "http://localhost:11434"


def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen(_OLLAMA_URL + "/api/tags", timeout=3):
            return True
    except Exception:
        return False


def test_eval_end_to_end_real_ollama():
    """Drive the real harness over one tiny seeded case on the zero-config local
    default. Asserts the end-to-end path produces a well-formed EvalResult — NOT
    that the model finds the bug (that is model-dependent and lives in the
    committed baseline, not in a pass/fail unit test).

    The daemon probe lives in the body (not a module-level skipif decorator) so
    the default `-m 'not ollama'` collection never makes a network call."""
    if not _ollama_up():
        pytest.skip("no local Ollama daemon reachable")
    from kagura_code_reviewer.cli import build_review_client
    from kagura_code_reviewer.eval_cli import run_eval
    from kagura_code_reviewer.eval_harness import GoldenBug, GoldenCase
    from kagura_code_reviewer.report import Severity
    from kagura_code_reviewer.review.harness import resolve_tier

    case = GoldenCase(
        name="zerodiv",
        source="seeded",
        diff=(
            "--- a/s.py\n+++ b/s.py\n@@ -1,2 +1,2 @@\n"
            " def mean(v):\n"
            "-    return sum(v) / len(v) if v else 0.0\n"
            "+    return sum(v) / len(v)\n"
        ),
        bugs=[GoldenBug(file="s.py", line=2, dimension="correctness",
                        severity=Severity.HIGH, symptom="zerodivision")],
    )
    # local=True forces the advisor's local pick (the zero-config free default).
    client, _ = build_review_client("ollama", None, local=True, cloud=False, timeout=120.0)
    results = run_eval([case], client, tier=resolve_tier("low"), repeats=1)

    assert len(results) == 1
    r = results[0]
    assert (r.tp, r.fp, r.fn) == tuple(int(x) for x in (r.tp, r.fp, r.fn))  # all ints
    assert r.recall is None or 0.0 <= r.recall <= 1.0
    assert r.precision is None or 0.0 <= r.precision <= 1.0
