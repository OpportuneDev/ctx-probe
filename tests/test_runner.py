from __future__ import annotations

import json
from pathlib import Path

from ctx_probe.runner import RunConfig, run


def test_runner_writes_jsonl_and_json(sample_corpus_path, mock_adapter, tmp_path):
    out = tmp_path / "report"
    cfg = RunConfig(
        model="mock",
        corpus_path=sample_corpus_path,
        out_dir=str(out),
        target_tokens=1500,
        depths=[0.1, 0.5],
        needle_counts=[4],
        samples_per_depth=1,
        seed=0,
    )
    results = run(mock_adapter, cfg)

    # 2 niah (depths=[0.1, 0.5]) + 3 multi_needle (default mn depths=[0.1, 0.5, 0.9])
    assert len(results) == 5
    jsonl = out / "results.jsonl"
    json_path = out / "results.json"
    assert jsonl.exists()
    assert json_path.exists()

    lines = jsonl.read_text().strip().split("\n")
    assert len(lines) == 5
    for line in lines:
        parsed = json.loads(line)
        assert "probe" in parsed
        assert "depth" in parsed
        assert "correct" in parsed


def test_runner_jsonl_appended_in_order(sample_corpus_path, mock_adapter, tmp_path):
    out = tmp_path / "report"
    cfg = RunConfig(
        model="mock",
        corpus_path=sample_corpus_path,
        out_dir=str(out),
        target_tokens=1500,
        depths=[0.1, 0.9],
        needle_counts=[],
        samples_per_depth=1,
        seed=0,
        run_multi_needle=False,
    )
    run(mock_adapter, cfg)
    lines = [json.loads(l) for l in (out / "results.jsonl").read_text().strip().split("\n")]
    assert [r["depth"] for r in lines] == [0.1, 0.9]
    assert all(r["probe"] == "niah" for r in lines)


def test_runner_skip_flags(sample_corpus_path, mock_adapter, tmp_path):
    cfg = RunConfig(
        model="mock",
        corpus_path=sample_corpus_path,
        out_dir=str(tmp_path / "out"),
        target_tokens=1500,
        depths=[0.5],
        needle_counts=[4],
        samples_per_depth=1,
        run_niah=False,
        run_multi_needle=True,
    )
    results = run(mock_adapter, cfg)
    assert all(r.probe.startswith("multi_needle") for r in results)
