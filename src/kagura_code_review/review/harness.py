from __future__ import annotations

from dataclasses import dataclass, replace

_ALL_ANGLES = [
    "correctness-linescan", "removed-behavior", "cross-file",
    "reuse", "simplification", "efficiency", "altitude",
]


@dataclass
class EffortTier:
    name: str
    angles: list[str]
    repeats: int
    verify_votes: int
    verify_votes_correctness: int
    max_findings: int


_DEFAULT_TIERS = {
    "low": EffortTier("low", _ALL_ANGLES[:3], 1, 1, 1, 8),
    "med": EffortTier("med", _ALL_ANGLES[:5], 1, 1, 2, 10),
    "high": EffortTier("high", _ALL_ANGLES[:7], 2, 3, 3, 12),
}


def resolve_tier(name: str, config: dict | None = None) -> EffortTier:
    base = _DEFAULT_TIERS.get(name, _DEFAULT_TIERS["med"])
    overrides = ((config or {}).get("effort", {}) or {}).get(base.name, {})
    if not overrides:
        return base
    fields = {k: overrides[k] for k in (
        "repeats", "verify_votes", "verify_votes_correctness", "max_findings",
    ) if k in overrides}
    if "angles" in overrides:
        fields["angles"] = list(overrides["angles"])
    return replace(base, **fields)
