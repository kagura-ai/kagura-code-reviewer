from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum


class Severity(IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


BLOCKING = Severity.HIGH

# Bump when the JSON report envelope changes shape, so actors can gate on it.
SCHEMA_VERSION = 1


def parse_severity(value: str) -> Severity:
    try:
        return Severity[value.strip().upper()]
    except (KeyError, AttributeError):
        return Severity.INFO


def confidence_from_votes(votes: dict) -> float | None:
    """Derive a 0-1 confidence from verifier votes. None when there are no
    real verdicts (empty, or only ERROR votes)."""
    c = votes.get("CONFIRMED", 0)
    p = votes.get("PLAUSIBLE", 0)
    r = votes.get("REFUTED", 0)
    total = c + p + r
    if total == 0:
        return None
    return (c + 0.5 * p) / total


@dataclass
class Finding:
    dimension: str
    severity: Severity
    file: str
    line: int | None
    title: str
    rationale: str
    suggestion: str
    angles: list[str] = field(default_factory=list)
    votes: dict = field(default_factory=dict)
    merge_count: int = 1
    confidence: float | None = None

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "severity": self.severity.name,
            "file": self.file,
            "line": self.line,
            "title": self.title,
            "rationale": self.rationale,
            "suggestion": self.suggestion,
            "angles": self.angles,
            "votes": self.votes,
            "merge_count": self.merge_count,
            "confidence": self.confidence,
        }


@dataclass
class Report:
    findings: list[Finding]

    @classmethod
    def from_payload(cls, payload: dict) -> "Report":
        out: list[Finding] = []
        raw = payload.get("findings", [])
        if not isinstance(raw, list):
            raw = []
        for f in raw:
            if not isinstance(f, dict):
                # Models sometimes emit findings as bare strings; skip non-dicts.
                continue
            out.append(
                Finding(
                    dimension=str(f.get("dimension", "general")),
                    severity=parse_severity(str(f.get("severity", "info"))),
                    file=str(f.get("file", "")),
                    line=f.get("line"),
                    title=str(f.get("title", "")),
                    rationale=str(f.get("rationale", "")),
                    suggestion=str(f.get("suggestion", "")),
                )
            )
        return cls(findings=out)

    def verdict(self) -> str:
        """Machine-readable gate verdict for downstream actors (couples to
        exit_code): red = blocking finding present, yellow = non-blocking
        findings only, green = clean."""
        if any(f.severity >= BLOCKING for f in self.findings):
            return "red"
        return "yellow" if self.findings else "green"

    def exit_code(self) -> int:
        return 1 if self.verdict() == "red" else 0

    def summary(self) -> dict:
        by_severity: dict[str, int] = {}
        for f in self.findings:
            by_severity[f.severity.name] = by_severity.get(f.severity.name, 0) + 1
        return {
            "total": len(self.findings),
            "blocking": sum(1 for f in self.findings if f.severity >= BLOCKING),
            "by_severity": by_severity,
            # a meta finding marks a review that did not complete cleanly
            "incomplete": any(f.dimension == "meta" for f in self.findings),
        }

    def to_json(self) -> str:
        return json.dumps({
            "schema_version": SCHEMA_VERSION,
            "verdict": self.verdict(),
            "summary": self.summary(),
            "findings": [f.to_dict() for f in self.findings],
        }, indent=2)

    def to_markdown(self) -> str:
        if not self.findings:
            return "# Code Review\n\nNo issues found. ✅\n"
        lines = ["# Code Review\n"]
        for f in sorted(self.findings, key=lambda x: x.severity, reverse=True):
            loc = f"{f.file}:{f.line}" if f.line is not None else f.file
            lines.append(f"## [{f.severity.name}] {f.title} ({f.dimension})")
            lines.append(f"- **Where:** `{loc}`")
            lines.append(f"- **Why:** {f.rationale}")
            lines.append(f"- **Fix:** {f.suggestion}")
            if f.angles or f.votes or f.merge_count > 1 or f.confidence is not None:
                seen = ", ".join(f.angles) if f.angles else "—"
                count = f" ×{f.merge_count}" if f.merge_count > 1 else ""
                votes = ("; votes: " + ", ".join(f"{k} {v}" for k, v in f.votes.items())) if f.votes else ""
                conf = f"; conf {f.confidence:.2f}" if f.confidence is not None else ""
                lines.append(f"- **Seen by:** {seen}{count}{votes}{conf}")
            lines.append("")
        return "\n".join(lines)
