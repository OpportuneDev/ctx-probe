from ctx_probe.adapters.anthropic import AnthropicAdapter
from ctx_probe.adapters.base import ModelAdapter, ModelResponse

__all__ = ["AnthropicAdapter", "ModelAdapter", "ModelResponse"]


def build_adapter(model: str, **kwargs) -> ModelAdapter:
    """Pick the right adapter for a model string.

    Anthropic-only in v0.1. OpenAI/Gemini land in v0.2.
    """
    if model.startswith("claude-"):
        return AnthropicAdapter(model=model, **kwargs)
    raise ValueError(
        f"No adapter for model '{model}'. v0.1 supports Anthropic only (claude-*)."
    )
