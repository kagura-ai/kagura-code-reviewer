from kagura_code_review.agent import (
    ChatMessage,
    Tool,
    ToolCall,
    run_agent,
)


class FakeClient:
    """Returns scripted ChatMessages in sequence, ignoring inputs."""

    def __init__(self, scripted: list[ChatMessage]):
        self._scripted = scripted
        self.calls = 0

    def chat(self, messages, tools=None) -> ChatMessage:
        msg = self._scripted[self.calls]
        self.calls += 1
        return msg


def _echo_tool(captured: list) -> Tool:
    def handler(args: dict) -> str:
        captured.append(args)
        return "ok"

    return Tool(name="echo", description="echo", parameters={"type": "object"}, handler=handler)


def test_loop_executes_tool_then_finishes():
    captured: list = []
    scripted = [
        ChatMessage(content=None, tool_calls=[ToolCall("1", "echo", '{"x": 1}')]),
        ChatMessage(content="done", tool_calls=[]),
    ]
    result = run_agent(FakeClient(scripted), [], [_echo_tool(captured)])
    assert captured == [{"x": 1}]
    assert result.final_text == "done"
    assert result.terminal_payload is None


def test_terminal_tool_ends_loop_immediately():
    def submit(args: dict) -> str:
        return "recorded"

    terminal = Tool("submit", "submit", {"type": "object"}, submit, terminal=True)
    scripted = [
        ChatMessage(content=None, tool_calls=[ToolCall("1", "submit", '{"findings": []}')]),
    ]
    result = run_agent(FakeClient(scripted), [], [terminal])
    assert result.terminal_payload == {"findings": []}


def test_unknown_tool_does_not_crash():
    scripted = [
        ChatMessage(content=None, tool_calls=[ToolCall("1", "ghost", "{}")]),
        ChatMessage(content="recovered", tool_calls=[]),
    ]
    result = run_agent(FakeClient(scripted), [], [])
    assert result.final_text == "recovered"


def test_max_iters_returns_exhausted():
    looping = ChatMessage(content=None, tool_calls=[ToolCall("1", "echo", "{}")])
    client = FakeClient([looping] * 5)
    result = run_agent(client, [], [_echo_tool([])], max_iters=3)
    assert result.exhausted is True
