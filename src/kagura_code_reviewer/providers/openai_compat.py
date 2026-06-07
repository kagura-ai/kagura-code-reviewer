from __future__ import annotations

from openai import OpenAI

from ..agent import ChatMessage, ToolCall


class OpenAICompatClient:
    """OpenAI-compatible chat client (Ollama, OpenAI, Gemini) over base_url + key."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 120.0,
        temperature: float = 0.0,
        num_ctx: int | None = None,
        seed: int | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.seed = seed
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatMessage:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            # Pin temperature for reproducible reviews (merge-gate determinism).
            "temperature": self.temperature,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        # num_ctx is an Ollama-specific option; omit for real OpenAI/Gemini (they 400 on it).
        # NOTE: Ollama's OpenAI-compat endpoint (/v1/chat/completions) currently
        # *ignores* options.num_ctx (verified on Ollama 0.20.2) — only the native
        # /api/chat honors it. The loaded context is therefore set by
        # OLLAMA_CONTEXT_LENGTH or the model's Modelfile, NOT this field. We still
        # emit it (correct per the option spec, harmless, forward-compatible); the
        # advisor uses ModelCap.ctx for its VRAM fit estimate rather than relying
        # on this taking effect.
        if self.num_ctx is not None:
            kwargs["extra_body"] = {"options": {"num_ctx": self.num_ctx}}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self._client.chat.completions.create(**kwargs)
        if not resp.choices:  # empty response (model unloaded / context overflow) — no crash
            return ChatMessage(content="", tool_calls=[])
        msg = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            )
        return ChatMessage(content=msg.content, tool_calls=tool_calls)
