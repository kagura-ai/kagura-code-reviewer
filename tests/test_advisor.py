from kagura_code_reviewer.advisor import Hardware, detect_hardware


def test_detect_hardware_uses_injected_probes():
    hw = detect_hardware(
        vram_reader=lambda: 24564,
        ram_reader=lambda: 96000,
        cpu_reader=lambda: 32,
    )
    assert hw == Hardware(vram_mb=24564, ram_mb=96000, cpu_threads=32, has_gpu=True)


def test_detect_hardware_no_gpu_when_vram_zero():
    hw = detect_hardware(vram_reader=lambda: 0, ram_reader=lambda: 8000, cpu_reader=lambda: 4)
    assert hw.has_gpu is False
    assert hw.vram_mb == 0


def test_detect_hardware_swallows_probe_errors():
    def boom():
        raise RuntimeError("no tool")
    hw = detect_hardware(vram_reader=boom, ram_reader=boom, cpu_reader=boom)
    assert hw == Hardware(vram_mb=0, ram_mb=0, cpu_threads=1, has_gpu=False)


from kagura_code_reviewer.advisor import ModelCap, is_cloud, lookup_cap


def test_is_cloud():
    assert is_cloud("qwen3-coder:480b-cloud") is True
    assert is_cloud("kimi-k2.6:cloud") is True
    assert is_cloud("qwen2.5-coder:14b") is False


def test_lookup_cap_exact_then_family_then_default():
    exact = lookup_cap("qwen2.5-coder:14b")
    # tool_calling re-rated to "fair": reliable single-shot, but narrates instead
    # of calling submit_findings in the multi-turn review-agent loop (dogfood).
    assert exact.tool_calling == "fair" and exact.vram_mb == 9000
    family = lookup_cap("qwen2.5-coder:3b")
    assert family.tool_calling in {"good", "fair"}
    default = lookup_cap("totally-unknown:1b")
    assert isinstance(default, ModelCap) and default.review <= 0.4


from pytest_httpserver import HTTPServer
from kagura_code_reviewer.advisor import list_models


def test_list_models_parses_tags(httpserver: HTTPServer):
    httpserver.expect_request("/api/tags").respond_with_json(
        {"models": [{"name": "qwen2.5-coder:14b"}, {"name": "qwen3-coder:480b-cloud"}]}
    )
    names = list_models(httpserver.url_for("/v1"))
    assert names == ["qwen2.5-coder:14b", "qwen3-coder:480b-cloud"]


def test_list_models_returns_empty_on_error():
    assert list_models("http://127.0.0.1:9/v1") == []


from kagura_code_reviewer.advisor import (
    Recommendation, effective_vram_mb, recommend,
)


def test_effective_vram_includes_kv_cache():
    """Effective footprint exceeds raw weights once the KV-cache is counted,
    and grows with context length."""
    small_ctx = ModelCap(0.8, "good", 9000, 8192)
    large_ctx = ModelCap(0.8, "good", 9000, 32768)
    assert effective_vram_mb(small_ctx) > small_ctx.vram_mb
    assert effective_vram_mb(large_ctx) > effective_vram_mb(small_ctx)

_INSTALLED = [
    "qwen3.5:27b", "qwen2.5-coder:14b", "qwen2.5-coder:7b",
    "deepseek-r1:32b", "qwen3-coder:480b-cloud",
]


def test_recommend_picks_strongest_local_that_fits_24gb():
    # qwen3.5:27b (17GB weights) overflows a 24GB GPU once the 32k KV-cache is
    # counted, so the strongest model that *actually* fits is qwen2.5-coder:14b.
    hw = Hardware(vram_mb=24564, ram_mb=96000, cpu_threads=32, has_gpu=True)
    rec = recommend(hw, _INSTALLED, prefer_local=True)
    assert rec.finder == "qwen2.5-coder:14b"
    assert rec.verifier == "qwen2.5-coder:14b"
    assert rec.fits is True


def test_recommend_prefers_reliable_tool_caller_over_higher_aptitude():
    # Both fit a 24GB GPU. qwen2.5-coder:14b has higher review aptitude but is
    # unreliable at agentic tool-calling (won't call submit_findings in the
    # review loop), so the advisor must prefer qwen3:14b, which drives the agent.
    hw = Hardware(vram_mb=24564, ram_mb=96000, cpu_threads=32, has_gpu=True)
    rec = recommend(hw, ["qwen2.5-coder:14b", "qwen3:14b"], prefer_local=True)
    assert rec.finder == "qwen3:14b"
    assert rec.fits is True


def test_recommend_excludes_models_over_vram_on_8gb():
    hw = Hardware(vram_mb=8000, ram_mb=32000, cpu_threads=8, has_gpu=True)
    rec = recommend(hw, _INSTALLED, prefer_local=True)
    assert rec.finder == "qwen2.5-coder:7b"
    assert rec.fits is True


def test_recommend_ram_fallback_when_no_gpu():
    hw = Hardware(vram_mb=0, ram_mb=40000, cpu_threads=16, has_gpu=False)
    rec = recommend(hw, _INSTALLED, prefer_local=True)
    assert rec.finder == "qwen3.5:27b"
    assert rec.fits is False


def test_recommend_excludes_poor_tool_calling():
    hw = Hardware(vram_mb=24000, ram_mb=96000, cpu_threads=32, has_gpu=True)
    rec = recommend(hw, ["deepseek-r1:32b"], prefer_local=True)
    assert rec.finder is None
    assert "no suitable" in rec.reason.lower()


def test_recommend_empty_installed():
    hw = Hardware(vram_mb=24000, ram_mb=96000, cpu_threads=32, has_gpu=True)
    rec = recommend(hw, [], prefer_local=True)
    assert rec.finder is None


def test_recommend_cloud_mode_picks_best_cloud():
    hw = Hardware(vram_mb=0, ram_mb=8000, cpu_threads=8, has_gpu=False)
    rec = recommend(hw, _INSTALLED, prefer_local=False)
    assert rec.finder == "qwen3-coder:480b-cloud"
    assert rec.fits is True
