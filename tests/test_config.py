import pytest

from kagura_code_reviewer.config import resolve_model


def test_default_alias_resolves():
    spec = resolve_model(None, local=False)
    assert spec.ollama_model
    assert spec.base_url.endswith("/v1")


def test_explicit_alias_resolves():
    spec = resolve_model("review-cloud", local=False)
    assert spec.alias == "review-cloud"


def test_review_local_is_a_reliable_tool_caller():
    # review-local must drive the agentic review loop. qwen2.5-coder narrates
    # instead of calling submit_findings (dogfood) -> default to a qwen3 model.
    spec = resolve_model("review-local", local=True)
    assert spec.ollama_model.startswith("qwen3")


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
    # gemini-2.0-flash was retired by Google (404 "no longer available"); the
    # shipped default must be a current model so --provider gemini works OOTB.
    assert cfg["providers"]["gemini"]["default_model"] == "gemini-2.5-flash"


def test_write_user_model_writes_default_alias(tmp_path, monkeypatch):
    import kagura_code_reviewer.config as cfgmod
    target = tmp_path / "cfg.toml"
    monkeypatch.setattr(cfgmod, "_USER", target)
    cfgmod.write_user_model("qwen3.5:27b", "http://localhost:11434/v1", 32768)
    text = target.read_text(encoding="utf-8")
    assert 'default_alias = "auto"' in text and "qwen3.5:27b" in text
    cfg = cfgmod.load_config()
    assert cfg["models"]["auto"]["ollama_model"] == "qwen3.5:27b"


def test_load_config_handles_non_ascii_user_file(tmp_path, monkeypatch):
    import kagura_code_reviewer.config as cfgmod
    target = tmp_path / "u.toml"
    target.write_text('# コメント\ndefault_alias = "review-local"\n', encoding="utf-8")
    monkeypatch.setattr(cfgmod, "_USER", target)
    cfg = cfgmod.load_config()
    assert cfg["default_alias"] == "review-local"
