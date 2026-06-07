from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

_SHIPPED = Path(__file__).with_name("config.toml")
_USER = Path(
    os.environ.get(
        "KAGURA_CODE_REVIEWER_CONFIG",
        Path.home() / ".config" / "kagura-code-reviewer" / "config.toml",
    )
)


@dataclass
class ModelSpec:
    alias: str
    ollama_model: str
    base_url: str
    num_ctx: int


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | None = None) -> dict:
    data = tomllib.loads(_SHIPPED.read_text())
    user_path = path or _USER
    if user_path.is_file():
        data = _merge(data, tomllib.loads(user_path.read_text()))
    return data


def resolve_model(alias: str | None, local: bool, config: dict | None = None) -> ModelSpec:
    cfg = config or load_config()
    if alias is None:
        alias = "review-local" if local else cfg.get("default_alias", "review-cloud")
    models = cfg.get("models", {})
    if alias not in models:
        raise KeyError(f"unknown model alias: {alias}")
    entry = models[alias]
    return ModelSpec(
        alias=alias,
        ollama_model=entry["ollama_model"],
        base_url=entry["base_url"],
        num_ctx=int(entry.get("num_ctx", 8192)),
    )


def spec_from_model_name(name: str, base_url: str, num_ctx: int = 8192) -> ModelSpec:
    return ModelSpec(alias="auto", ollama_model=name, base_url=base_url, num_ctx=num_ctx)
