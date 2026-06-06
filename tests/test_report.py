from kagura_code_review.report import Finding, Report, Severity, parse_severity


def test_parse_severity_case_insensitive():
    assert parse_severity("HIGH") is Severity.HIGH
    assert parse_severity("low") is Severity.LOW
    assert parse_severity("unknown") is Severity.INFO


def test_exit_code_zero_when_no_blocking():
    r = Report(findings=[Finding("style", Severity.LOW, "a.py", 1, "t", "r", "s")])
    assert r.exit_code() == 0


def test_exit_code_nonzero_when_high_or_above():
    r = Report(findings=[Finding("security", Severity.HIGH, "a.py", 2, "t", "r", "s")])
    assert r.exit_code() == 1


def test_from_payload_builds_findings():
    payload = {"findings": [
        {"dimension": "security", "severity": "critical", "file": "a.py",
         "line": 10, "title": "SQLi", "rationale": "concat", "suggestion": "param"}
    ]}
    r = Report.from_payload(payload)
    assert r.findings[0].severity is Severity.CRITICAL
    assert r.findings[0].file == "a.py"


def test_markdown_contains_title_and_file():
    r = Report(findings=[Finding("perf", Severity.MEDIUM, "x.py", 3, "N+1", "loop", "batch")])
    md = r.to_markdown()
    assert "N+1" in md and "x.py" in md


def test_json_roundtrips_severity_as_name():
    import json
    r = Report(findings=[Finding("perf", Severity.MEDIUM, "x.py", None, "t", "r", "s")])
    data = json.loads(r.to_json())
    assert data["findings"][0]["severity"] == "MEDIUM"
    assert data["findings"][0]["line"] is None


def test_finding_has_optional_provenance_defaults():
    f = Finding("correctness", Severity.HIGH, "a.py", 5, "t", "r", "s")
    assert f.angles == []
    assert f.votes == {}
    assert f.merge_count == 1


def test_finding_accepts_provenance():
    f = Finding("correctness", Severity.HIGH, "a.py", 5, "t", "r", "s",
                angles=["cross-file"], votes={"CONFIRMED": 2}, merge_count=3)
    assert f.angles == ["cross-file"]
    assert f.votes == {"CONFIRMED": 2}
    assert f.merge_count == 3
