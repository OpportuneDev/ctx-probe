from __future__ import annotations

import pytest

from ctx_probe.corpus import (
    build_haystack,
    chunk_documents,
    estimate_tokens,
    load_corpus,
)


def test_load_corpus_reads_text_and_md(sample_corpus_path):
    docs = load_corpus(sample_corpus_path)
    assert len(docs) == 2
    assert any("Lorem ipsum" in d for d in docs)
    assert any("Sample document B" in d for d in docs)


def test_load_corpus_single_file(sample_corpus_path, tmp_path):
    f = tmp_path / "one.txt"
    f.write_text("Hello world")
    docs = load_corpus(str(f))
    assert docs == ["Hello world"]


def test_load_corpus_missing_path():
    with pytest.raises(FileNotFoundError):
        load_corpus("/does/not/exist")


def test_load_corpus_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="No .txt or .md"):
        load_corpus(str(tmp_path))


def test_chunk_documents_respects_size():
    docs = ["word " * 2000]
    chunks = chunk_documents(docs, chunk_tokens=100)
    assert len(chunks) >= 5
    for c in chunks[:-1]:
        assert estimate_tokens(c) <= 110


def test_build_haystack_inserts_needle():
    chunks = [" ".join(["alpha"] * 100) for _ in range(20)]
    needle = "MAGIC_NEEDLE_TOKEN_XYZ"
    hs = build_haystack(chunks, target_tokens=1000, needle=needle, depth=0.5, seed=1)
    assert needle in hs.text
    assert hs.token_count > 800


def test_build_haystack_depth_position():
    chunks = [" ".join(["alpha"] * 100) for _ in range(20)]
    needle = "MARKER"
    hs_early = build_haystack(chunks, target_tokens=1000, needle=needle, depth=0.1, seed=1)
    hs_late = build_haystack(chunks, target_tokens=1000, needle=needle, depth=0.9, seed=1)

    pos_early = hs_early.text.index(needle)
    pos_late = hs_late.text.index(needle)
    assert pos_early < pos_late, "Needle at depth=0.1 should appear before depth=0.9"


def test_build_haystack_reproducible_with_seed():
    chunks = [f"chunk_{i} " * 50 for i in range(20)]
    a = build_haystack(chunks, target_tokens=500, needle="N", depth=0.5, seed=42)
    b = build_haystack(chunks, target_tokens=500, needle="N", depth=0.5, seed=42)
    assert a.text == b.text


def test_build_haystack_invalid_depth():
    chunks = ["foo"]
    with pytest.raises(ValueError, match="depth"):
        build_haystack(chunks, target_tokens=100, needle="N", depth=1.5)


def test_build_haystack_empty_chunks():
    with pytest.raises(ValueError, match="empty"):
        build_haystack([], target_tokens=100, needle="N", depth=0.5)
