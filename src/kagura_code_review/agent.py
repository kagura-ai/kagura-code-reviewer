from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as returned by the model


@dataclass
class ChatMessage:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema for the tool's arguments
    handler: Callable[[dict], str]
    terminal: bool = False


@dataclass
class AgentResult:
    final_text: str | None = None
    terminal_payload: dict | None = None
    exhausted: bool = False


class ChatClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatMessage: ...


def _schema(tool: Tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def run_agent(
    client: ChatClient,
    messages: list[dict],
    tools: list[Tool],
    max_iters: int = 12,
) -> AgentResult:
    tool_map = {t.name: t for t in tools}
    schemas = [_schema(t) for t in tools] or None

    for _ in range(max_iters):
        msg = client.chat(messages, tools=schemas)
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        if not msg.tool_calls:
            return AgentResult(final_text=msg.content)

        for tc in msg.tool_calls:
            tool = tool_map.get(tc.name)
            try:
                args = json.loads(tc.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if tool is None:
                result = f"error: unknown tool {tc.name}"
            else:
                try:
                    result = tool.handler(args)
                except Exception as exc:  # tool failures are fed back, not fatal
                    result = f"error: {exc}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            if tool is not None and tool.terminal:
                return AgentResult(final_text=msg.content, terminal_payload=args)

    return AgentResult(exhausted=True)
