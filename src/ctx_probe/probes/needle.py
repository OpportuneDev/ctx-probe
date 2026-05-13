from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ctx_probe.adapters.base import ModelAdapter
from ctx_probe.corpus import build_haystack

DEFAULT_NEEDLE = {
    "needle_text": (
        "IMPORTANT FACT: The secret access code for the project codenamed "
        '"Blue Octopus" is 47-XJ-2891.'
    ),
    "question": 'What is the secret access code for the project codenamed "Blue Octopus"?',
    "expected": "47-XJ-2891",
}

SYSTEM_PROMPT = (
    "You will be shown a long document, followed by a question about a specific "
    "fact stated somewhere inside it. Answer the question concisely using only "
    "information from the document. If the fact is not present, say so explicitly."
)


@dataclass
class NeedleResult:
    probe: str
    depth: float
    sample_idx: int
    correct: bool
    response_text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    latency_ms: float
    haystack_tokens: int
    needle_token_position: int
    expected: str = ""
    error: str = ""


@dataclass
class NeedleConfig:
    needle_text: str = DEFAULT_NEEDLE["needle_text"]
    question: str = DEFAULT_NEEDLE["question"]
    expected: str = DEFAULT_NEEDLE["expected"]
    depths: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    samples_per_depth: int = 1
    target_tokens: int = 32_000
    max_output_tokens: int = 256


def grade(response_text: str, expected: str) -> bool:
    """Case-insensitive substring match. Strict enough for unambiguous answers
    like '47-XJ-2891' — for fuzzier expected outputs, swap in an LLM judge later.
    """
    if not response_text or not expected:
        return False
    return expected.strip().lower() in response_text.strip().lower()


def run_niah(
    adapter: ModelAdapter,
    chunks: list[str],
    config: NeedleConfig,
    *,
    seed: int | None = None,
    on_result: Callable[[NeedleResult], None] | None = None,
) -> list[NeedleResult]:
    """Run the NIAH depth probe across all (depth × sample) combinations.

    If `on_result` is provided, it's called once per completed result — use it
    to persist incrementally so a crash mid-run doesn't lose work.
    """
    results: list[NeedleResult] = []
    sample_seed_base = seed if seed is not None else 0

    def _emit(r: NeedleResult) -> None:
        results.append(r)
        if on_result is not None:
            on_result(r)

    for depth in config.depths:
        for sample_idx in range(config.samples_per_depth):
            sample_seed = sample_seed_base + sample_idx * 1000 + int(depth * 100)
            try:
                haystack = build_haystack(
                    chunks,
                    target_tokens=config.target_tokens,
                    needle=config.needle_text,
                    depth=depth,
                    seed=sample_seed,
                )
                resp = adapter.complete(
                    cacheable_prefix=haystack.text,
                    query=config.question,
                    max_tokens=config.max_output_tokens,
                    system=SYSTEM_PROMPT,
                )
                _emit(
                    NeedleResult(
                        probe="niah",
                        depth=depth,
                        sample_idx=sample_idx,
                        correct=grade(resp.text, config.expected),
                        response_text=resp.text,
                        input_tokens=resp.input_tokens,
                        output_tokens=resp.output_tokens,
                        cache_read_tokens=resp.cache_read_tokens,
                        cache_write_tokens=resp.cache_write_tokens,
                        latency_ms=resp.latency_ms,
                        haystack_tokens=haystack.token_count,
                        needle_token_position=haystack.needle_token_position,
                        expected=config.expected,
                    )
                )
            except Exception as e:
                _emit(
                    NeedleResult(
                        probe="niah",
                        depth=depth,
                        sample_idx=sample_idx,
                        correct=False,
                        response_text="",
                        input_tokens=0,
                        output_tokens=0,
                        cache_read_tokens=0,
                        cache_write_tokens=0,
                        latency_ms=0.0,
                        haystack_tokens=0,
                        needle_token_position=0,
                        expected=config.expected,
                        error=f"{type(e).__name__}: {e}",
                    )
                )

    return results
