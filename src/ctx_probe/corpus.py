from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import tiktoken

# tiktoken's cl100k is an approximation for Claude's tokenizer. It's not exact,
# but it's deterministic and close enough for positional sampling — and it's
# far cheaper than calling client.messages.count_tokens() for every chunk.
# Exact token accounting comes from the API response.
_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Haystack:
    text: str
    token_count: int
    needle: str
    needle_depth: float
    needle_token_position: int


def load_corpus(corpus_path: str | Path) -> list[str]:
    """Load all .txt and .md files under a directory. Returns one string per file."""
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {path}")

    if path.is_file():
        return [path.read_text(encoding="utf-8", errors="replace")]

    docs = []
    for f in sorted(path.rglob("*")):
        if f.is_file() and f.suffix.lower() in {".txt", ".md"}:
            docs.append(f.read_text(encoding="utf-8", errors="replace"))
    if not docs:
        raise ValueError(f"No .txt or .md files found under {path}")
    return docs


def chunk_documents(docs: list[str], chunk_tokens: int = 500) -> list[str]:
    """Split documents into ~chunk_tokens-sized pieces.

    We chunk on token boundaries (not paragraphs) so haystack assembly hits
    target sizes precisely. Document boundaries are preserved by chunking each
    doc independently.
    """
    chunks: list[str] = []
    for doc in docs:
        tokens = _ENC.encode(doc)
        for i in range(0, len(tokens), chunk_tokens):
            piece = tokens[i : i + chunk_tokens]
            if piece:
                chunks.append(_ENC.decode(piece))
    return chunks


def build_haystack(
    chunks: list[str],
    target_tokens: int,
    needle: str,
    depth: float,
    *,
    seed: int | None = None,
) -> Haystack:
    """Assemble a haystack of approximately `target_tokens` tokens, inject
    `needle` at fractional position `depth` (0.0–1.0), return assembled text
    plus metadata.

    The needle is inserted at the chunk boundary closest to the requested
    depth. The actual token position is recorded in the result for grading
    and reporting.
    """
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth must be in [0, 1], got {depth}")
    if not chunks:
        raise ValueError("chunks is empty — load and chunk a corpus first")

    rng = random.Random(seed)

    selected: list[str] = []
    selected_tokens = 0
    pool = chunks.copy()
    rng.shuffle(pool)

    for chunk in pool:
        chunk_token_count = len(_ENC.encode(chunk))
        if selected_tokens + chunk_token_count > target_tokens:
            break
        selected.append(chunk)
        selected_tokens += chunk_token_count

    if not selected:
        raise ValueError(
            f"Corpus too small: cannot build a haystack of {target_tokens} tokens "
            f"from available chunks (largest chunk exceeds target)."
        )

    target_position = int(selected_tokens * depth)
    running_tokens = 0
    insert_index = 0
    for i, chunk in enumerate(selected):
        chunk_token_count = len(_ENC.encode(chunk))
        if running_tokens + chunk_token_count >= target_position:
            insert_index = i
            break
        running_tokens += chunk_token_count
    else:
        insert_index = len(selected)

    needle_block = f"\n\n{needle}\n\n"
    assembled = "\n\n".join(selected[:insert_index] + [needle_block] + selected[insert_index:])

    return Haystack(
        text=assembled,
        token_count=len(_ENC.encode(assembled)),
        needle=needle,
        needle_depth=depth,
        needle_token_position=running_tokens,
    )


def estimate_tokens(text: str) -> int:
    """Quick local token estimate. Not exact for Claude — use API for accounting."""
    return len(_ENC.encode(text))
