from __future__ import annotations

import json
import time

import httpx

from .agent import ChatMessage, ToolCall
from .doctor import _ollama_root

# Transient HTTP statuses worth retrying (e.g. Ollama returns 5xx briefly while
# loading/swapping a model). 4xx are deterministic client errors — never retried.
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _to_native_messages(messages: list[dict]) -> list[dict]:
    """Translate OpenAI-format history (as built by run_agent) to Ollama native.

    Differences from the /v1 format:
    - assistant tool_calls carry `arguments` as an object (not a JSON string)
      and drop the `id`/`type` fields;
    - tool results are matched by `tool_name` (resolved from the call id) rather
      than `tool_call_id`.
    """
    id_to_name: dict[str, str] = {}
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                if tc.get("id"):
                    id_to_name[tc["id"]] = name
                raw = fn.get("arguments", "")
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    parsed = {}
                calls.append({"function": {"name": name, "arguments": parsed or {}}})
            out.append({"role": "assistant", "content": m.get("content", ""), "tool_calls": calls})
        elif role == "tool":
            entry = {"role": "tool", "content": m.get("content", "")}
            name = id_to_name.get(m.get("tool_call_id"))
            if name:
                entry["tool_name"] = name
            out.append(entry)
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


class OllamaClient:
    """Native Ollama /api/chat client (free local default).

    Uses the native endpoint instead of the OpenAI-compat /v1 because the compat
    endpoint silently ignores options.num_ctx — only /api/chat honors it, so the
    configured/advised context window actually takes effect.
    """

    def __init__(self, base_url, model, num_ctx=8192, timeout=120.0,
                 api_key="ollama", temperature=0.0, seed=None,
                 max_retries=2, backoff=0.5):
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.seed = seed
        self._url = f"{_ollama_root(base_url)}/api/chat"
        self._timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._max_retries = max_retries
        self._backoff = backoff

    def _post_with_retry(self, body: dict) -> httpx.Response:
        # Restore the transient-error resilience the OpenAI SDK provided: retry
        # connection errors / timeouts / transient 5xx with exponential backoff.
        for attempt in range(self._max_retries + 1):
            try:
                resp = httpx.post(self._url, json=body, timeout=self._timeout,
                                  headers=self._headers)
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as exc:
                if (exc.response.status_code not in _RETRY_STATUS
                        or attempt == self._max_retries):
                    raise
            except httpx.TransportError:  # ConnectError, ReadTimeout, etc.
                if attempt == self._max_retries:
                    raise
            time.sleep(self._backoff * (2 ** attempt))
        raise RuntimeError("unreachable")  # pragma: no cover

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatMessage:
        # Pin temperature for reproducible reviews (merge-gate determinism).
        options: dict = {"temperature": self.temperature}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.seed is not None:
            options["seed"] = self.seed
        body: dict = {
            "model": self.model,
            "messages": _to_native_messages(messages),
            "stream": False,
            "options": options,
        }
        if tools:
            body["tools"] = tools
        resp = self._post_with_retry(body)
        msg = resp.json().get("message")
        if not msg:  # no message (model unloaded / context overflow) — no crash
            return ChatMessage(content="", tool_calls=[])
        tool_calls: list[ToolCall] = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            tool_calls.append(
                ToolCall(
                    id=tc.get("id") or f"call_{i}",
                    name=fn.get("name", ""),
                    arguments=args if isinstance(args, str) else json.dumps(args),
                )
            )
        return ChatMessage(content=msg.get("content") or "", tool_calls=tool_calls)
