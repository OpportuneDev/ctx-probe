from __future__ import annotations

import pytest

from ctx_probe.adapters.base import ModelAdapter, ModelResponse


class MockAdapter(ModelAdapter):
    """Records calls and returns a canned response. Use answer_fn to make the
    response depend on the query — e.g. echo whether the expected fact is
    present in the haystack.
    """

    def __init__(self, name: str = "mock:test", answer_fn=None):
        self.name = name
        self.calls: list[dict] = []
        self.answer_fn = answer_fn or (lambda prefix, query: "I don't know.")

    def complete(self, cacheable_prefix, query, *, max_tokens=256, system=None):
        self.calls.append(
            {
                "prefix_len": len(cacheable_prefix),
                "query": query,
                "system": system,
                "max_tokens": max_tokens,
            }
        )
        text = self.answer_fn(cacheable_prefix, query)
        return ModelResponse(
            text=text,
            input_tokens=100,
            output_tokens=20,
            cache_read_tokens=80,
            cache_write_tokens=20,
            latency_ms=42.0,
            raw={"mock": True},
        )


@pytest.fixture
def mock_adapter():
    return MockAdapter()


@pytest.fixture
def sample_corpus_path():
    from pathlib import Path

    return str(Path(__file__).parent.parent / "examples" / "sample_corpus")
