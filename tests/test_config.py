import pytest

from kagura_code_reviewer.config import resolve_model


def test_default_alias_resolves():
    spec = resolve_model(None, local=False)
    assert spec.ollama_model
    assert spec.base_url.endswith("/v1")


def test_explicit_alias_resolves():
    spec = resolve_model("review-cloud", local=False)
    assert spec.alias == "review-cloud"


def test_unknown_alias_raises():
    with pytest.raises(KeyError):
        resolve_model("does-not-exist", local=False)


def test_shipped_config_has_effort_tiers():
    from kagura_code_reviewer.config import load_config
    cfg = load_config()
    assert set(cfg["effort"]) >= {"low", "med", "high"}
    assert cfg["effort"]["med"]["max_findings"] == 10
