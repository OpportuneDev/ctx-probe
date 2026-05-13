from __future__ import annotations

import time

import anthropic

from ctx_probe.adapters.base import ModelAdapter, ModelResponse


class AnthropicAdapter(ModelAdapter):
    """Anthropic adapter. Caches the haystack via cache_control so repeated
    samples against the same context don't pay the full re-encode each call.
    """

    def __init__(self, model: str, *, max_tokens: int = 256, api_key: str | None = None):
        self.name = f"anthropic:{model}"
        self.model = model
        self.default_max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def complete(
        self,
        cacheable_prefix: str,
        query: str,
        *,
        max_tokens: int = 256,
        system: str | None = None,
    ) -> ModelResponse:
        content = [
            {
                "type": "text",
                "text": cacheable_prefix,
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": query},
        ]

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens or self.default_max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            kwargs["system"] = system

        t0 = time.perf_counter()
        resp = self.client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        text = "".join(block.text for block in resp.content if block.type == "text")

        return ModelResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
            latency_ms=latency_ms,
            raw={"model": resp.model, "stop_reason": resp.stop_reason},
        )
