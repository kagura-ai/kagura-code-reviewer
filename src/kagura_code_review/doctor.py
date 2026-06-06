from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _ollama_root(base_url: str) -> str:
    # Accept either ".../v1" or bare host root; /api lives at the host root.
    return base_url[: -len("/v1")] if base_url.rstrip("/").endswith("/v1") else base_url.rstrip("/")


def check_ollama(base_url: str) -> CheckResult:
    root = _ollama_root(base_url)
    try:
        resp = httpx.get(f"{root}/api/tags", timeout=3.0)
        resp.raise_for_status()
        return CheckResult("ollama daemon", True, f"reachable at {root}")
    except Exception as exc:
        return CheckResult("ollama daemon", False, f"not reachable: {exc}")


def check_model(base_url: str, model: str) -> CheckResult:
    root = _ollama_root(base_url)
    try:
        resp = httpx.get(f"{root}/api/tags", timeout=3.0)
        resp.raise_for_status()
        names = {m.get("name") for m in resp.json().get("models", [])}
        if model in names:
            return CheckResult(f"model {model}", True, "pulled")
        return CheckResult(f"model {model}", False, f"not pulled — run: ollama pull {model}")
    except Exception as exc:
        return CheckResult(f"model {model}", False, f"could not list models: {exc}")
