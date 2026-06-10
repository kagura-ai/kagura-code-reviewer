from kagura_code_reviewer.report import Finding, Report, Severity, parse_severity


def test_parse_severity_case_insensitive():
    assert parse_severity("HIGH") is Severity.HIGH
    assert parse_severity("low") is Severity.LOW


def test_parse_severity_unknown_falls_back_to_medium():
    # An unrecognised value must NOT demote to the least-severe level: a model
    # typo like "HIGHT" would otherwise flip a blocking finding to INFO and
    # silently turn a red verdict green. Fall back to a visible, non-blocking
    # level instead.
    assert parse_severity("unknown") is Severity.MEDIUM
    assert parse_severity("HIGHT") is Severity.MEDIUM
    assert parse_severity(None) is Severity.MEDIUM  # type: ignore[arg-type]


def test_parse_severity_unknown_logs_warning(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        parse_severity("HIGHT")
    assert any("HIGHT" in r.getMessage() for r in caplog.records)


def test_exit_code_zero_when_no_blocking():
    r = Report(findings=[Finding("style", Severity.LOW, "a.py", 1, "t", "r", "s")])
    assert r.exit_code() == 0


def test_exit_code_nonzero_when_high_or_above():
    r = Report(findings=[Finding("security", Severity.HIGH, "a.py", 2, "t", "r", "s")])
    assert r.exit_code() == 1


def test_verdict_green_yellow_red():
    assert Report(findings=[]).verdict() == "green"
    assert Report(findings=[Finding("style", Severity.LOW, "a.py", 1, "t", "r", "s")]).verdict() == "yellow"
    assert Report(findings=[Finding("sec", Severity.HIGH, "a.py", 1, "t", "r", "s")]).verdict() == "red"


def test_verdict_red_iff_exit_code_one():
    # invariant: an actor can rely on red <-> exit 1
    for findings in ([], [Finding("s", Severity.MEDIUM, "a", 1, "t", "r", "s")],
                     [Finding("s", Severity.CRITICAL, "a", 1, "t", "r", "s")]):
        r = Report(findings=findings)
        assert (r.verdict() == "red") == (r.exit_code() == 1)


def test_json_envelope_has_schema_verdict_summary():
    import json
    r = Report(findings=[
        Finding("sec", Severity.HIGH, "a.py", 1, "t", "r", "s"),
        Finding("style", Severity.LOW, "b.py", 2, "t", "r", "s"),
    ])
    d = json.loads(r.to_json())
    assert d["schema_version"] >= 1
    assert d["verdict"] == "red"
    assert d["summary"]["total"] == 2
    assert d["summary"]["blocking"] == 1
    assert d["summary"]["by_severity"]["HIGH"] == 1
    assert len(d["findings"]) == 2  # findings stays top-level (backward compatible)


def test_json_summary_flags_incomplete():
    import json
    r = Report(findings=[Finding("meta", Severity.HIGH, "", None, "Review incomplete", "r", "s")])
    d = json.loads(r.to_json())
    assert d["summary"]["incomplete"] is True
    assert d["verdict"] == "red"


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


def test_markdown_shows_provenance_when_present():
    f = Finding("correctness", Severity.HIGH, "a.py", 2, "bug", "r", "s",
                angles=["cross-file", "correctness-linescan"],
                votes={"CONFIRMED": 2}, merge_count=2)
    md = Report(findings=[f]).to_markdown()
    assert "cross-file" in md and "CONFIRMED" in md


def test_markdown_omits_provenance_when_absent():
    f = Finding("perf", Severity.LOW, "a.py", 1, "t", "r", "s")
    md = Report(findings=[f]).to_markdown()
    assert "Seen by" not in md


def test_json_includes_provenance():
    import json
    f = Finding("correctness", Severity.HIGH, "a.py", 2, "bug", "r", "s",
                angles=["reuse"], votes={"PLAUSIBLE": 1}, merge_count=3)
    d = json.loads(Report(findings=[f]).to_json())["findings"][0]
    assert d["angles"] == ["reuse"] and d["votes"] == {"PLAUSIBLE": 1} and d["merge_count"] == 3


def test_from_payload_skips_non_dict_findings():
    # A model may return findings as bare strings; must not crash, just skip them.
    payload = {"findings": [
        "some free-text finding",
        {"dimension": "correctness", "severity": "high", "file": "a.py",
         "line": 1, "title": "real", "rationale": "r", "suggestion": "s"},
    ]}
    r = Report.from_payload(payload)
    assert len(r.findings) == 1 and r.findings[0].title == "real"


def test_from_payload_non_list_findings_is_empty():
    assert Report.from_payload({"findings": "oops"}).findings == []


def test_confidence_from_votes():
    from kagura_code_reviewer.report import confidence_from_votes
    assert confidence_from_votes({"CONFIRMED": 2, "PLAUSIBLE": 1}) == (2 + 0.5) / 3
    assert confidence_from_votes({"REFUTED": 1}) == 0.0
    assert confidence_from_votes({}) is None
    assert confidence_from_votes({"ERROR": 2}) is None  # only error votes -> unknown


def test_finding_confidence_in_md_and_json():
    import json
    f = Finding("correctness", Severity.HIGH, "a.py", 1, "t", "r", "s",
                votes={"CONFIRMED": 2, "PLAUSIBLE": 1}, confidence=0.83)
    md = Report(findings=[f]).to_markdown()
    assert "conf 0.83" in md
    d = json.loads(Report(findings=[f]).to_json())["findings"][0]
    assert d["confidence"] == 0.83
