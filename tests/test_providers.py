import json

from pytest_httpserver import HTTPServer

from kagura_code_reviewer.providers.openai_compat import OpenAICompatClient


def test_compat_client_sends_auth_header_and_no_num_ctx_by_default(httpserver: HTTPServer):
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {"choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": None}}]}
    )
    client = OpenAICompatClient(base_url=httpserver.url_for("/v1"), model="gpt-4o",
                                api_key="sk-test", timeout=5.0)
    msg = client.chat([{"role": "user", "content": "hello"}])
    assert msg.content == "hi"
    req = httpserver.log[0][0]
    assert req.headers["Authorization"] == "Bearer sk-test"
    body = json.loads(req.get_data())
    assert "extra_body" not in body and "options" not in body
    assert body["temperature"] == 0


def test_compat_client_sends_num_ctx_when_set(httpserver: HTTPServer):
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {"choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": None}}]}
    )
    client = OpenAICompatClient(base_url=httpserver.url_for("/v1"), model="qwen",
                                api_key="ollama", timeout=5.0, num_ctx=16384)
    client.chat([{"role": "user", "content": "hello"}])
    body = json.loads(httpserver.log[0][0].get_data())
    assert body["options"]["num_ctx"] == 16384
