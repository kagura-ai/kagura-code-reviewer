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


# ---- Anthropic adapter ---------------------------------------------------

from types import SimpleNamespace

from kagura_code_reviewer.providers.anthropic_client import (
    AnthropicClient,
    from_anthropic_response,
    to_anthropic_system_and_messages,
    to_anthropic_tools,
)


def test_to_anthropic_tools_converts_schema():
    openai_tools = [{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                           "required": ["path"]},
        },
    }]
    out = to_anthropic_tools(openai_tools)
    assert out == [{
        "name": "read_file",
        "description": "Read a file.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                         "required": ["path"]},
    }]


def test_to_anthropic_messages_extracts_system_and_converts_tool_turns():
    messages = [
        {"role": "system", "content": "You review code."},
        {"role": "user", "content": "review this"},
        {"role": "assistant", "content": "ok",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "file body"},
    ]
    system, msgs = to_anthropic_system_and_messages(messages)
    assert system == "You review code."
    assert msgs[0] == {"role": "user", "content": "review this"}
    assert msgs[1]["role"] == "assistant"
    tool_use = [b for b in msgs[1]["content"] if b["type"] == "tool_use"][0]
    assert tool_use == {"type": "tool_use", "id": "c1", "name": "read_file",
                        "input": {"path": "a.py"}}
    assert msgs[2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1", "content": "file body"}]}


def test_to_anthropic_messages_merges_consecutive_tool_results():
    messages = [
        {"role": "assistant", "content": "",
         "tool_calls": [
             {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
             {"id": "c2", "type": "function", "function": {"name": "g", "arguments": "{}"}},
         ]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
    ]
    _, msgs = to_anthropic_system_and_messages(messages)
    assert msgs[-1] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1", "content": "r1"},
        {"type": "tool_result", "tool_use_id": "c2", "content": "r2"},
    ]}


def test_from_anthropic_response_maps_text_and_tool_use():
    resp = SimpleNamespace(content=[
        SimpleNamespace(type="text", text="some text"),
        SimpleNamespace(type="tool_use", id="c1", name="read_file", input={"path": "a.py"}),
    ])
    msg = from_anthropic_response(resp)
    assert msg.content == "some text"
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "c1"
    assert msg.tool_calls[0].name == "read_file"
    assert msg.tool_calls[0].arguments == '{"path": "a.py"}'


def test_from_anthropic_response_no_blocks():
    resp = SimpleNamespace(content=[])
    msg = from_anthropic_response(resp)
    assert msg.content == "" and msg.tool_calls == []


class FakeAnthropic:
    def __init__(self):
        self.captured = {}
        self.messages = self

    def create(self, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="done")])


def test_anthropic_client_chat_builds_request_without_temperature():
    fake = FakeAnthropic()
    client = AnthropicClient(model="claude-sonnet-4-6", api_key="x",
                             max_tokens=2048, client=fake)
    msg = client.chat(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f", "description": "d",
                                                 "parameters": {"type": "object"}}}],
    )
    assert msg.content == "done"
    assert fake.captured["model"] == "claude-sonnet-4-6"
    assert fake.captured["max_tokens"] == 2048
    assert fake.captured["system"] == "sys"
    assert "temperature" not in fake.captured
    assert fake.captured["tools"][0]["name"] == "f"
    assert fake.captured["tool_choice"] == {"type": "auto"}


def test_compat_client_includes_seed_when_set(httpserver: HTTPServer):
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {"choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": None}}]}
    )
    client = OpenAICompatClient(base_url=httpserver.url_for("/v1"), model="qwen",
                                api_key="ollama", timeout=5.0, seed=42)
    client.chat([{"role": "user", "content": "hi"}])
    body = json.loads(httpserver.log[0][0].get_data())
    assert body["seed"] == 42


def test_compat_client_handles_empty_choices(httpserver: HTTPServer):
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json({"choices": []})
    client = OpenAICompatClient(base_url=httpserver.url_for("/v1"), model="m", api_key="k", timeout=5.0)
    msg = client.chat([{"role": "user", "content": "hi"}])
    assert msg.tool_calls == [] and (msg.content == "" or msg.content is None)
