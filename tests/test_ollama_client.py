import json

from pytest_httpserver import HTTPServer

from kagura_code_reviewer.ollama_client import OllamaClient


def test_chat_parses_plain_content(httpserver: HTTPServer):
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {"choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": None}}]}
    )
    client = OllamaClient(base_url=httpserver.url_for("/v1"), model="qwen", timeout=5.0)
    msg = client.chat([{"role": "user", "content": "hello"}])
    assert msg.content == "hi"
    assert msg.tool_calls == []


def test_chat_sends_temperature_zero_for_determinism(httpserver: HTTPServer):
    """Reviews must be reproducible: the chat call pins temperature=0 so the
    same diff yields the same verdict (merge-gate determinism)."""
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {"choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": None}}]}
    )
    client = OllamaClient(base_url=httpserver.url_for("/v1"), model="qwen", timeout=5.0)
    client.chat([{"role": "user", "content": "hello"}])
    sent = json.loads(httpserver.log[0][0].get_data())
    assert sent["temperature"] == 0


def test_chat_parses_tool_calls(httpserver: HTTPServer):
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": json.dumps({"path": "a.py"})},
                            }
                        ],
                    }
                }
            ]
        }
    )
    client = OllamaClient(base_url=httpserver.url_for("/v1"), model="qwen", timeout=5.0)
    msg = client.chat([{"role": "user", "content": "review"}])
    assert msg.tool_calls[0].name == "read_file"
    assert json.loads(msg.tool_calls[0].arguments)["path"] == "a.py"
