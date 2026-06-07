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


def test_spec_from_model_name_builds_spec():
    from kagura_code_reviewer.config import ModelSpec, spec_from_model_name
    spec = spec_from_model_name("qwen3.5:27b", "http://localhost:11434/v1", num_ctx=32768)
    assert isinstance(spec, ModelSpec)
    assert spec.ollama_model == "qwen3.5:27b"
    assert spec.base_url == "http://localhost:11434/v1"
    assert spec.num_ctx == 32768
    assert spec.alias == "auto"


def test_shipped_config_has_providers():
    from kagura_code_reviewer.config import load_config
    cfg = load_config()
    assert set(cfg["providers"]) >= {"ollama", "openai", "anthropic", "gemini"}
    assert cfg["providers"]["anthropic"]["kind"] == "anthropic"
    assert cfg["providers"]["openai"]["api_key_env"] == "OPENAI_API_KEY"
