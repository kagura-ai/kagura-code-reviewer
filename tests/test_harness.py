import json

from kagura_code_review.agent import ChatMessage, ToolCall
from kagura_code_review.report import Finding, Report, Severity
from kagura_code_review.review.angles import ANGLE_PROMPTS, CORRECTNESS_ANGLES
from kagura_code_review.review.harness import EffortTier, resolve_tier


# ---- shared test doubles -------------------------------------------------

class StubRepo:
    def read_file(self, path, max_bytes=20000):
        return "file contents"

    def grep(self, pattern, max_results=50):
        return "no matches"

    def list_files(self, subdir="."):
        return ["a.py"]


def _submit(findings):
    return ChatMessage(
        content=None,
        tool_calls=[ToolCall("1", "submit_findings", json.dumps({"findings": findings}))],
    )


def _verdict(v):
    return ChatMessage(
        content=None,
        tool_calls=[ToolCall("1", "submit_verdict", json.dumps({"verdict": v}))],
    )


def _finding(file, line, title, sev="medium", dim="correctness"):
    return {"dimension": dim, "severity": sev, "file": file, "line": line,
            "title": title, "rationale": "r", "suggestion": "s"}


def _F(file, line, title, sev=Severity.MEDIUM, angle="x"):
    return Finding("correctness", sev, file, line, title, "r", "s", angles=[angle])


class ScriptedClient:
    def __init__(self, scripted):
        self._scripted = scripted
        self.calls = 0

    def chat(self, messages, tools=None):
        msg = self._scripted[self.calls]
        self.calls += 1
        return msg


class SeqClient:
    """Returns the next scripted message on each chat() call."""
    def __init__(self, msgs):
        self._msgs = list(msgs)
        self.calls = 0

    def chat(self, messages, tools=None):
        m = self._msgs[self.calls]
        self.calls += 1
        return m


# ---- tier resolution -----------------------------------------------------

def test_resolve_default_tiers():
    low, med, high = resolve_tier("low"), resolve_tier("med"), resolve_tier("high")
    assert med.repeats == 1 and med.max_findings == 10
    assert len(low.angles) == 3 and len(med.angles) == 5 and len(high.angles) == 7
    assert high.repeats == 2 and high.verify_votes == 3
    assert med.verify_votes == 1 and med.verify_votes_correctness == 2


def test_resolve_unknown_tier_defaults_to_med():
    assert resolve_tier("bogus").name == "med"


def test_resolve_tier_config_override():
    cfg = {"effort": {"med": {"max_findings": 99}}}
    assert resolve_tier("med", config=cfg).max_findings == 99
    assert resolve_tier("med", config=cfg).repeats == 1



def test_angle_catalog_has_seven_angles():
    assert set(ANGLE_PROMPTS) == {
        "correctness-linescan", "removed-behavior", "cross-file",
        "reuse", "simplification", "efficiency", "altitude",
    }
    assert all(isinstance(v, str) and len(v) > 20 for v in ANGLE_PROMPTS.values())


def test_correctness_angles_subset():
    assert CORRECTNESS_ANGLES == {"correctness-linescan", "removed-behavior", "cross-file"}
    assert CORRECTNESS_ANGLES <= set(ANGLE_PROMPTS)


# ---- single-angle finder -------------------------------------------------

def test_run_finder_returns_findings_with_angle_provenance():
    from kagura_code_review.review.harness import FinderOutcome, run_finder
    finding = _finding("a.py", 5, "bug", "high")
    out = run_finder(ScriptedClient([_submit([finding])]), StubRepo(),
                     diff="d", context=None, angle="cross-file", max_iters=4)
    assert isinstance(out, FinderOutcome)
    assert out.errored is False
    assert len(out.findings) == 1
    assert out.findings[0].angles == ["cross-file"]


def test_run_finder_without_submit_contributes_nothing():
    from kagura_code_review.review.harness import run_finder
    loop = ChatMessage(content=None, tool_calls=[ToolCall("1", "read_file", '{"path":"a.py"}')])
    out = run_finder(ScriptedClient([loop] * 30), StubRepo(),
                     diff="d", context=None, angle="reuse", max_iters=3)
    assert out.findings == []
    assert out.errored is False


def test_run_finder_marks_errored_on_exception():
    from kagura_code_review.review.harness import run_finder

    class Boom:
        def chat(self, messages, tools=None):
            raise RuntimeError("backend down")
    out = run_finder(Boom(), StubRepo(), diff="d", context=None,
                     angle="reuse", max_iters=3)
    assert out.findings == []
    assert out.errored is True


# ---- ensemble runner -----------------------------------------------------

class PerAngleClient:
    """Returns a submit message keyed by the angle named in the system prompt."""
    def __init__(self, by_angle):
        self.by_angle = by_angle

    def chat(self, messages, tools=None):
        sys = messages[0]["content"]
        angle = next(a for a in self.by_angle if a in sys)
        return _submit(self.by_angle[angle])


def test_run_finders_unions_across_angles_and_repeats():
    from kagura_code_review.review.harness import run_finders
    tier = EffortTier("t", ["correctness-linescan", "cross-file"], repeats=2,
                      verify_votes=1, verify_votes_correctness=1, max_findings=10)
    client = PerAngleClient({
        "correctness-linescan": [_finding("a.py", 1, "A")],
        "cross-file": [_finding("b.py", 2, "B")],
    })
    candidates, any_errored = run_finders(client, StubRepo(), "d", None, tier,
                                          max_iters=4, max_concurrency=1)
    assert len(candidates) == 4
    assert any_errored is False
    assert {c.file for c in candidates} == {"a.py", "b.py"}


def test_run_finders_reports_any_errored():
    from kagura_code_review.review.harness import run_finders
    tier = EffortTier("t", ["reuse"], repeats=1, verify_votes=1,
                      verify_votes_correctness=1, max_findings=10)

    class Boom:
        def chat(self, messages, tools=None):
            raise RuntimeError("down")
    candidates, any_errored = run_finders(Boom(), StubRepo(), "d", None, tier,
                                          max_iters=3, max_concurrency=1)
    assert candidates == []
    assert any_errored is True
