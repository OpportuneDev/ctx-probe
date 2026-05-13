from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ctx_probe.adapters.base import ModelAdapter
from ctx_probe.corpus import chunk_documents, load_corpus
from ctx_probe.probes.multi_needle import MultiNeedleConfig, run_multi_needle
from ctx_probe.probes.needle import NeedleConfig, NeedleResult, run_niah


@dataclass
class RunConfig:
    model: str
    corpus_path: str
    out_dir: str
    target_tokens: int = 32_000
    depths: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    needle_counts: list[int] = field(default_factory=lambda: [4, 8])
    samples_per_depth: int = 1
    seed: int = 42
    run_niah: bool = True
    run_multi_needle: bool = True
    needle_text: str | None = None
    needle_question: str | None = None
    needle_expected: str | None = None


def run(adapter: ModelAdapter, cfg: RunConfig) -> list[NeedleResult]:
    """Run all configured probes, persist each result to results.jsonl as it
    completes, and return the full result list at the end.

    Incremental persistence: if the process crashes mid-run, results.jsonl
    contains every result completed up to the crash point — one JSON object
    per line, append-only.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    jsonl_path.write_text("")

    docs = load_corpus(cfg.corpus_path)
    chunks = chunk_documents(docs)

    def persist(r: NeedleResult) -> None:
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    all_results: list[NeedleResult] = []

    if cfg.run_niah:
        niah_kwargs: dict = {
            "depths": cfg.depths,
            "samples_per_depth": cfg.samples_per_depth,
            "target_tokens": cfg.target_tokens,
        }
        if cfg.needle_text is not None:
            niah_kwargs["needle_text"] = cfg.needle_text
        if cfg.needle_question is not None:
            niah_kwargs["question"] = cfg.needle_question
        if cfg.needle_expected is not None:
            niah_kwargs["expected"] = cfg.needle_expected
        niah_cfg = NeedleConfig(**niah_kwargs)
        all_results.extend(
            run_niah(adapter, chunks, niah_cfg, seed=cfg.seed, on_result=persist)
        )

    if cfg.run_multi_needle:
        mn_cfg = MultiNeedleConfig(
            needle_counts=cfg.needle_counts,
            samples_per_combo=cfg.samples_per_depth,
            target_tokens=cfg.target_tokens,
        )
        all_results.extend(
            run_multi_needle(adapter, chunks, mn_cfg, seed=cfg.seed, on_result=persist)
        )

    summary = {
        "model": adapter.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "results": [asdict(r) for r in all_results],
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    return all_results
