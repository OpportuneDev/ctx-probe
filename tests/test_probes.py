from __future__ import annotations

from ctx_probe.corpus import chunk_documents, load_corpus
from ctx_probe.probes.multi_needle import MultiNeedleConfig, run_multi_needle
from ctx_probe.probes.needle import NeedleConfig, grade, run_niah


def test_grade_substring_match():
    assert grade("The answer is 47-XJ-2891.", "47-XJ-2891") is True
    assert grade("47-XJ-2891", "47-XJ-2891") is True


def test_grade_case_insensitive():
    assert grade("The code is BLUE-OCTOPUS", "blue-octopus") is True


def test_grade_no_match():
    assert grade("I don't know the answer.", "47-XJ-2891") is False


def test_grade_empty():
    assert grade("", "47-XJ-2891") is False
    assert grade("hello", "") is False


def test_niah_returns_one_result_per_combo(sample_corpus_path, mock_adapter):
    chunks = chunk_documents(load_corpus(sample_corpus_path), chunk_tokens=200)
    cfg = NeedleConfig(
        depths=[0.1, 0.5, 0.9],
        samples_per_depth=2,
        target_tokens=1500,
    )
    results = run_niah(mock_adapter, chunks, cfg, seed=0)
    assert len(results) == 6
    assert {r.depth for r in results} == {0.1, 0.5, 0.9}


def test_niah_grades_correctly_when_model_answers(sample_corpus_path):
    from tests.conftest import MockAdapter

    chunks = chunk_documents(load_corpus(sample_corpus_path), chunk_tokens=200)
    cfg = NeedleConfig(depths=[0.5], samples_per_depth=1, target_tokens=1500)

    adapter = MockAdapter(answer_fn=lambda prefix, q: f"The answer is {cfg.expected}.")
    results = run_niah(adapter, chunks, cfg, seed=0)
    assert results[0].correct is True


def test_niah_marks_incorrect_when_model_misses(sample_corpus_path):
    from tests.conftest import MockAdapter

    chunks = chunk_documents(load_corpus(sample_corpus_path), chunk_tokens=200)
    cfg = NeedleConfig(depths=[0.5], samples_per_depth=1, target_tokens=1500)

    adapter = MockAdapter(answer_fn=lambda prefix, q: "I don't know.")
    results = run_niah(adapter, chunks, cfg, seed=0)
    assert results[0].correct is False


def test_niah_callback_fires_per_result(sample_corpus_path, mock_adapter):
    chunks = chunk_documents(load_corpus(sample_corpus_path), chunk_tokens=200)
    cfg = NeedleConfig(depths=[0.1, 0.5], samples_per_depth=1, target_tokens=1500)
    seen = []
    run_niah(mock_adapter, chunks, cfg, seed=0, on_result=seen.append)
    assert len(seen) == 2


def test_niah_captures_adapter_errors(sample_corpus_path):
    from tests.conftest import MockAdapter

    chunks = chunk_documents(load_corpus(sample_corpus_path), chunk_tokens=200)

    def boom(prefix, q):
        raise RuntimeError("simulated API failure")

    adapter = MockAdapter(answer_fn=boom)
    cfg = NeedleConfig(depths=[0.5], samples_per_depth=1, target_tokens=1500)
    results = run_niah(adapter, chunks, cfg, seed=0)
    assert len(results) == 1
    assert results[0].error.startswith("RuntimeError")
    assert results[0].correct is False


def test_multi_needle_inserts_target_count(sample_corpus_path, mock_adapter):
    chunks = chunk_documents(load_corpus(sample_corpus_path), chunk_tokens=200)
    cfg = MultiNeedleConfig(
        needle_counts=[4],
        depths=[0.5],
        samples_per_combo=1,
        target_tokens=2000,
    )
    results = run_multi_needle(mock_adapter, chunks, cfg, seed=0)
    assert len(results) == 1
    assert results[0].probe == "multi_needle_4"
    last_call_prefix_len = mock_adapter.calls[-1]["prefix_len"]
    assert last_call_prefix_len > 1000
