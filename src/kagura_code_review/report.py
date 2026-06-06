from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum


class Severity(IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


BLOCKING = Severity.HIGH


def parse_severity(value: str) -> Severity:
    try:
        return Severity[value.strip().upper()]
    except (KeyError, AttributeError):
        return Severity.INFO


@dataclass
class Finding:
    dimension: str
    severity: Severity
    file: str
    line: int | None
    title: str
    rationale: str
    suggestion: str

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "severity": self.severity.name,
            "file": self.file,
            "line": self.line,
            "title": self.title,
            "rationale": self.rationale,
            "suggestion": self.suggestion,
        }


@dataclass
class Report:
    findings: list[Finding]

    @classmethod
    def from_payload(cls, payload: dict) -> "Report":
        out: list[Finding] = []
        for f in payload.get("findings", []):
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

    def exit_code(self) -> int:
        return 1 if any(f.severity >= BLOCKING for f in self.findings) else 0

    def to_json(self) -> str:
        return json.dumps({"findings": [f.to_dict() for f in self.findings]}, indent=2)

    def to_markdown(self) -> str:
        if not self.findings:
            return "# Code Review\n\nNo issues found. ✅\n"
        lines = ["# Code Review\n"]
        for f in sorted(self.findings, key=lambda x: x.severity, reverse=True):
            loc = f"{f.file}:{f.line}" if f.line is not None else f.file
            lines.append(f"## [{f.severity.name}] {f.title} ({f.dimension})")
            lines.append(f"- **Where:** `{loc}`")
            lines.append(f"- **Why:** {f.rationale}")
            lines.append(f"- **Fix:** {f.suggestion}\n")
        return "\n".join(lines)
