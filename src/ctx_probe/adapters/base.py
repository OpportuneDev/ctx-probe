from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ModelResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)


class ModelAdapter(ABC):
    """One-shot completion interface.

    Adapters are stateless. The runner builds the prompt; adapters just call the API.
    The `cacheable_prefix` argument lets providers that support prompt caching
    (Anthropic) mark the haystack portion as cached. Providers without caching
    should concatenate it normally.
    """

    name: str

    @abstractmethod
    def complete(
        self,
        cacheable_prefix: str,
        query: str,
        *,
        max_tokens: int = 256,
        system: str | None = None,
    ) -> ModelResponse: ...
