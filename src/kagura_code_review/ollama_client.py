from __future__ import annotations

from openai import OpenAI

from .agent import ChatMessage, ToolCall


class OllamaClient:
    """OpenAI-compatible chat client pointed at an Ollama endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        num_ctx: int = 8192,
        timeout: float = 120.0,
        api_key: str = "ollama",
        temperature: float = 0.0,
    ):
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatMessage:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            # Pin temperature for reproducible reviews (merge-gate determinism).
            "temperature": self.temperature,
            "extra_body": {"options": {"num_ctx": self.num_ctx}},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            )
        return ChatMessage(content=msg.content, tool_calls=tool_calls)
