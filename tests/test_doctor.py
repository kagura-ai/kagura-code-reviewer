from pytest_httpserver import HTTPServer

from kagura_code_reviewer.doctor import check_ollama, check_model


def test_check_ollama_ok(httpserver: HTTPServer):
    httpserver.expect_request("/api/tags").respond_with_json({"models": []})
    result = check_ollama(httpserver.url_for(""))
    assert result.ok is True


def test_check_ollama_down():
    result = check_ollama("http://127.0.0.1:1")  # nothing listening
    assert result.ok is False


def test_check_model_present(httpserver: HTTPServer):
    httpserver.expect_request("/api/tags").respond_with_json(
        {"models": [{"name": "qwen2.5-coder:7b"}]}
    )
    result = check_model(httpserver.url_for(""), "qwen2.5-coder:7b")
    assert result.ok is True


def test_check_model_missing(httpserver: HTTPServer):
    httpserver.expect_request("/api/tags").respond_with_json({"models": []})
    result = check_model(httpserver.url_for(""), "qwen2.5-coder:7b")
    assert result.ok is False


def test_format_hardware_report_includes_hw_and_recommendation():
    from kagura_code_reviewer.advisor import Hardware, Recommendation
    from kagura_code_reviewer.doctor import format_hardware_report
    hw = Hardware(vram_mb=24564, ram_mb=96000, cpu_threads=32, has_gpu=True)
    rec = Recommendation("qwen3.5:27b", "qwen3.5:27b", "fits GPU VRAM", True)
    text = format_hardware_report(hw, rec)
    assert "24564" in text and "32" in text
    assert "qwen3.5:27b" in text
    assert "fits GPU VRAM" in text
