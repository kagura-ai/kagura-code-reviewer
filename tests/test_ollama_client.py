import json

from pytest_httpserver import HTTPServer

from kagura_code_reviewer.ollama_client import OllamaClient


def _native(content="hi", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"message": msg, "done": True, "done_reason": "stop"}


def test_chat_posts_to_native_api_chat_and_honors_num_ctx(httpserver: HTTPServer):
    """The whole point of the native endpoint: options.num_ctx is actually sent
    (the OpenAI-compat /v1 endpoint ignores it)."""
    httpserver.expect_request("/api/chat", method="POST").respond_with_json(_native("hi"))
    client = OllamaClient(base_url=httpserver.url_for("/v1"), model="qwen", num_ctx=8192, timeout=5.0)
    msg = client.chat([{"role": "user", "content": "hello"}])
    assert msg.content == "hi"
    assert msg.tool_calls == []
    body = json.loads(httpserver.log[0][0].get_data())
    assert body["options"]["num_ctx"] == 8192
    assert body["stream"] is False


def test_chat_sends_temperature_zero_in_options(httpserver: HTTPServer):
    """Reviews must be reproducible: temperature is pinned to 0 (in native options)."""
    httpserver.expect_request("/api/chat", method="POST").respond_with_json(_native("hi"))
    client = OllamaClient(base_url=httpserver.url_for("/v1"), model="qwen", timeout=5.0)
    client.chat([{"role": "user", "content": "hello"}])
    body = json.loads(httpserver.log[0][0].get_data())
    assert body["options"]["temperature"] == 0


def test_chat_includes_seed_in_options_when_set(httpserver: HTTPServer):
    httpserver.expect_request("/api/chat", method="POST").respond_with_json(_native("hi"))
    client = OllamaClient(base_url=httpserver.url_for("/v1"), model="qwen", timeout=5.0, seed=42)
    client.chat([{"role": "user", "content": "hi"}])
    body = json.loads(httpserver.log[0][0].get_data())
    assert body["options"]["seed"] == 42


def test_chat_parses_native_tool_calls(httpserver: HTTPServer):
    """Native /api/chat returns tool_call arguments as a JSON object; we expose
    them as the raw JSON string ToolCall expects."""
    httpserver.expect_request("/api/chat", method="POST").respond_with_json(
        _native(content="", tool_calls=[
            {"id": "call_1", "function": {"name": "read_file", "arguments": {"path": "a.py"}}}
        ])
    )
    client = OllamaClient(base_url=httpserver.url_for("/v1"), model="qwen", timeout=5.0)
    msg = client.chat([{"role": "user", "content": "review"}])
    assert msg.tool_calls[0].id == "call_1"
    assert msg.tool_calls[0].name == "read_file"
    assert json.loads(msg.tool_calls[0].arguments)["path"] == "a.py"


def test_chat_translates_openai_history_to_native(httpserver: HTTPServer):
    """run_agent builds OpenAI-format history; translate assistant tool_calls
    (arguments str -> object) and tool results (by id -> tool_name) to native."""
    httpserver.expect_request("/api/chat", method="POST").respond_with_json(_native("done"))
    client = OllamaClient(base_url=httpserver.url_for("/v1"), model="qwen", timeout=5.0)
    history = [
        {"role": "system", "content": "review"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "file body"},
    ]
    client.chat(history)
    sent = json.loads(httpserver.log[0][0].get_data())["messages"]
    assistant = sent[2]
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"path": "a.py"}
    assert "id" not in assistant["tool_calls"][0]
    tool_msg = sent[3]
    assert tool_msg == {"role": "tool", "content": "file body", "tool_name": "read_file"}


def test_chat_handles_missing_message(httpserver: HTTPServer):
    """A response without a message (model unloaded / error) yields no crash."""
    httpserver.expect_request("/api/chat", method="POST").respond_with_json({"done": True})
    client = OllamaClient(base_url=httpserver.url_for("/v1"), model="qwen", timeout=5.0)
    msg = client.chat([{"role": "user", "content": "hi"}])
    assert msg.tool_calls == [] and (msg.content == "" or msg.content is None)


def test_chat_retries_transient_5xx_then_succeeds(httpserver: HTTPServer, monkeypatch):
    """Ollama can briefly return 5xx (e.g. during a model swap). The client must
    retry transient errors — matching the resilience the OpenAI SDK gave us."""
    monkeypatch.setattr("kagura_code_reviewer.ollama_client.time.sleep", lambda s: None)
    httpserver.expect_ordered_request("/api/chat", method="POST").respond_with_data("loading", status=503)
    httpserver.expect_ordered_request("/api/chat", method="POST").respond_with_json(_native("hi"))
    client = OllamaClient(base_url=httpserver.url_for("/v1"), model="qwen", timeout=5.0)
    msg = client.chat([{"role": "user", "content": "hi"}])
    assert msg.content == "hi"
    assert len(httpserver.log) == 2  # one retry


def test_chat_raises_after_retries_exhausted(httpserver: HTTPServer, monkeypatch):
    import httpx
    import pytest
    monkeypatch.setattr("kagura_code_reviewer.ollama_client.time.sleep", lambda s: None)
    httpserver.expect_request("/api/chat", method="POST").respond_with_data("down", status=500)
    client = OllamaClient(base_url=httpserver.url_for("/v1"), model="qwen", timeout=5.0, max_retries=2)
    with pytest.raises(httpx.HTTPError):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(httpserver.log) == 3  # initial + 2 retries


def test_chat_does_not_retry_client_error(httpserver: HTTPServer, monkeypatch):
    """A 400 is a deterministic client error — don't waste retries on it."""
    import httpx
    import pytest
    monkeypatch.setattr("kagura_code_reviewer.ollama_client.time.sleep", lambda s: None)
    httpserver.expect_request("/api/chat", method="POST").respond_with_data("bad", status=400)
    client = OllamaClient(base_url=httpserver.url_for("/v1"), model="qwen", timeout=5.0, max_retries=2)
    with pytest.raises(httpx.HTTPError):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(httpserver.log) == 1  # no retry
