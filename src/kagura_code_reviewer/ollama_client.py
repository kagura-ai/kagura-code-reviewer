from __future__ import annotations

from .providers.openai_compat import OpenAICompatClient


class OllamaClient(OpenAICompatClient):
    """OpenAI-compatible client pointed at an Ollama endpoint (free local default)."""

    def __init__(self, base_url, model, num_ctx=8192, timeout=120.0,
                 api_key="ollama", temperature=0.0):
        super().__init__(base_url=base_url, model=model, api_key=api_key,
                         timeout=timeout, temperature=temperature, num_ctx=num_ctx)
