from pytest_httpserver import HTTPServer

from kagura_code_review.doctor import check_ollama, check_model


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
