from __future__ import annotations

import json

from ..agent import ChatMessage, ToolCall


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-style tool schemas to Anthropic {name, description, input_schema}."""
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


def to_anthropic_system_and_messages(messages: list[dict]):
    """Split OpenAI-style messages into (system_str, anthropic_messages).

    - system messages -> concatenated top-level system string
    - assistant tool_calls -> tool_use content blocks (input parsed from JSON)
    - tool messages -> tool_result blocks merged into a single following user message
    """
    system_parts: list[str] = []
    out: list[dict] = []
    pending_tool_results: list[dict] = []

    def flush_tool_results():
        if pending_tool_results:
            out.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for m in messages:
        role = m.get("role")
        if role == "system":
            if m.get("content"):
                system_parts.append(m["content"])
            continue
        if role == "tool":
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": m.get("content", ""),
            })
            continue
        flush_tool_results()
        if role == "assistant":
            blocks: list[dict] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls", []):
                fn = tc["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                blocks.append({"type": "tool_use", "id": tc["id"],
                               "name": fn["name"], "input": args})
            out.append({"role": "assistant", "content": blocks})
        else:  # user
            out.append({"role": "user", "content": m.get("content", "")})
    flush_tool_results()
    return "\n".join(system_parts), out


def from_anthropic_response(resp) -> ChatMessage:
    """Map an Anthropic Messages response (text + tool_use blocks) to a ChatMessage."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in getattr(resp, "content", []) or []:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name,
                                       arguments=json.dumps(block.input)))
    return ChatMessage(content="".join(text_parts), tool_calls=tool_calls)


class AnthropicClient:
    """Anthropic Messages API client implementing the ChatClient protocol."""

    def __init__(self, model: str, api_key: str = "", max_tokens: int = 4096,
                 timeout: float = 120.0, client=None):
        self.model = model
        self.max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            from anthropic import Anthropic  # lazy: only the real path needs the SDK
            self._client = Anthropic(api_key=api_key, timeout=timeout)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatMessage:
        system, msgs = to_anthropic_system_and_messages(messages)
        kwargs: dict = {"model": self.model, "max_tokens": self.max_tokens, "messages": msgs}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
            kwargs["tool_choice"] = {"type": "auto"}
        resp = self._client.messages.create(**kwargs)
        return from_anthropic_response(resp)
