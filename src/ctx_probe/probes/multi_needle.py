from __future__ import annotations

import random
import string
from collections.abc import Callable
from dataclasses import dataclass, field

import tiktoken

from ctx_probe.adapters.base import ModelAdapter
from ctx_probe.probes.needle import NeedleResult, grade

_ENC = tiktoken.get_encoding("cl100k_base")

SYSTEM_PROMPT = (
    "You will be shown a long document that contains several similar facts, "
    "followed by a question about ONE specific fact. Answer using only "
    "information from the document. Do not confuse similar entries."
)

CODENAMES = [
    "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot",
    "Golf", "Hotel", "India", "Juliet", "Kilo", "Lima",
]


def _make_code(rng: random.Random) -> str:
    """Generates a distinctive alphanumeric code unlikely to appear in any corpus."""
    return "-".join(
        "".join(rng.choices(string.ascii_uppercase + string.digits, k=4))
        for _ in range(3)
    )


@dataclass
class MultiNeedleConfig:
    needle_counts: list[int] = field(default_factory=lambda: [4, 8])
    depths: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    samples_per_combo: int = 1
    target_tokens: int = 32_000
    max_output_tokens: int = 256


def _build_multi_needle_haystack(
    chunks: list[str],
    target_tokens: int,
    needles: list[tuple[str, str]],
    seed: int,
) -> tuple[str, int, list[int]]:
    """Insert N needles at evenly spaced depths in a haystack.

    Returns (assembled_text, total_tokens, needle_token_positions).
    """
    rng = random.Random(seed)
    pool = chunks.copy()
    rng.shuffle(pool)

    selected: list[str] = []
    selected_tokens = 0
    for chunk in pool:
        chunk_tokens = len(_ENC.encode(chunk))
        if selected_tokens + chunk_tokens > target_tokens:
            break
        selected.append(chunk)
        selected_tokens += chunk_tokens

    if not selected:
        raise ValueError("Corpus too small to assemble target haystack size.")

    n = len(needles)
    depths = [(i + 1) / (n + 1) for i in range(n)]
    target_positions = [int(selected_tokens * d) for d in depths]

    insertion_points: list[int] = []
    running = 0
    target_iter = iter(target_positions)
    next_target = next(target_iter, None)
    for i, chunk in enumerate(selected):
        chunk_tokens = len(_ENC.encode(chunk))
        while next_target is not None and running + chunk_tokens >= next_target:
            insertion_points.append(i + 1)
            next_target = next(target_iter, None)
        running += chunk_tokens
    while next_target is not None:
        insertion_points.append(len(selected))
        next_target = next(target_iter, None)

    pieces = selected.copy()
    for needle_idx, insert_at in enumerate(sorted(insertion_points, reverse=True)):
        codename, code = needles[len(insertion_points) - 1 - needle_idx]
        block = f"\n\nProject codename {codename} has security code {code}.\n\n"
        pieces.insert(insert_at, block)

    text = "\n\n".join(pieces)
    return text, len(_ENC.encode(text)), target_positions


def run_multi_needle(
    adapter: ModelAdapter,
    chunks: list[str],
    config: MultiNeedleConfig,
    *,
    seed: int | None = None,
    on_result: Callable[[NeedleResult], None] | None = None,
) -> list[NeedleResult]:
    results: list[NeedleResult] = []
    base_seed = seed if seed is not None else 0

    def _emit(r: NeedleResult) -> None:
        results.append(r)
        if on_result is not None:
            on_result(r)

    for n_needles in config.needle_counts:
        if n_needles > len(CODENAMES):
            raise ValueError(
                f"needle_count={n_needles} exceeds available codenames ({len(CODENAMES)})"
            )

        for sample_idx in range(config.samples_per_combo):
            sample_seed = base_seed + sample_idx * 1000 + n_needles
            rng = random.Random(sample_seed)
            codenames = rng.sample(CODENAMES, n_needles)
            needles = [(name, _make_code(rng)) for name in codenames]

            for target_depth in config.depths:
                target_idx = min(
                    range(n_needles),
                    key=lambda i: abs((i + 1) / (n_needles + 1) - target_depth),
                )
                target_name, target_code = needles[target_idx]
                question = (
                    f"What is the security code for project codename {target_name}? "
                    "Reply with only the code."
                )

                try:
                    text, total_tokens, positions = _build_multi_needle_haystack(
                        chunks,
                        target_tokens=config.target_tokens,
                        needles=needles,
                        seed=sample_seed,
                    )
                    resp = adapter.complete(
                        cacheable_prefix=text,
                        query=question,
                        max_tokens=config.max_output_tokens,
                        system=SYSTEM_PROMPT,
                    )
                    _emit(
                        NeedleResult(
                            probe=f"multi_needle_{n_needles}",
                            depth=(target_idx + 1) / (n_needles + 1),
                            sample_idx=sample_idx,
                            correct=grade(resp.text, target_code),
                            response_text=resp.text,
                            input_tokens=resp.input_tokens,
                            output_tokens=resp.output_tokens,
                            cache_read_tokens=resp.cache_read_tokens,
                            cache_write_tokens=resp.cache_write_tokens,
                            latency_ms=resp.latency_ms,
                            haystack_tokens=total_tokens,
                            needle_token_position=positions[target_idx],
                            expected=target_code,
                        )
                    )
                except Exception as e:
                    _emit(
                        NeedleResult(
                            probe=f"multi_needle_{n_needles}",
                            depth=target_depth,
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
                            expected=target_code,
                            error=f"{type(e).__name__}: {e}",
                        )
                    )

    return results
