from __future__ import annotations

CORRECTNESS_ANGLES = {"correctness-linescan", "removed-behavior", "cross-file"}

ANGLE_PROMPTS: dict[str, str] = {
    "correctness-linescan": (
        "Scan every changed hunk line by line. For each line ask what input, "
        "state, timing, or platform makes it wrong: inverted conditions, "
        "off-by-one, null/None deref, missing await, falsy-zero checks, "
        "wrong-variable copy-paste, swallowed errors, unescaped regex metachars."
    ),
    "removed-behavior": (
        "For every line the diff deletes or replaces, name the invariant or "
        "behavior it enforced, then look for where the new code re-establishes "
        "it. If it is not re-established, report it: a removed guard, dropped "
        "error path, narrowed validation, or deleted test for a real case."
    ),
    "cross-file": (
        "For each function the diff changes, find its callers and callees and "
        "check whether the change breaks a call site: a new precondition, a "
        "changed return shape, a new exception, or a timing/ordering dependency."
    ),
    "reuse": (
        "Flag new code that re-implements something the codebase already has. "
        "Name the existing helper or module that should be called instead."
    ),
    "simplification": (
        "Flag unnecessary complexity the diff adds: redundant or derivable "
        "state, copy-paste with slight variation, deep nesting, dead code. "
        "Name the simpler form that does the same job."
    ),
    "efficiency": (
        "Flag wasted work the diff introduces: redundant computation or repeated "
        "I/O, sequential work that could be independent, blocking work on hot "
        "paths. Name the cheaper alternative."
    ),
    "altitude": (
        "Check that each change is at the right depth, not a fragile bandaid. "
        "Special cases bolted onto shared infrastructure signal the fix is not "
        "deep enough; prefer generalizing the underlying mechanism."
    ),
}
