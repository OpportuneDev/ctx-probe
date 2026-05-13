# ctx-probe

**Measure your model's effective context window — on your data, in 5 minutes.**

Frontier models advertise million-token context windows. In practice, accuracy collapses well before that, and the drop is task- and corpus-specific. `ctx-probe` measures *your* model's degradation curve on *your* documents, so you can stop guessing where retrieval starts to silently fail.

The published benchmark numbers (MRCR, RULER, HELMET) are measured on synthetic haystacks. Your stack is not synthetic. The only number that tells you where your pipeline breaks is the one measured on your data.

---

## Install

```bash
pip install ctx-probe                  # from PyPI (when published)
# or, for now:
pip install -e .                       # from a clone of this repo
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quickstart — three recipes

### 1. Smoke test (one call, costs a fraction of a cent)

Use the bundled sample corpus to verify your install and key are wired up:

```bash
ctx-probe run \
  --model claude-sonnet-4-6 \
  --corpus examples/sample_corpus \
  --context-length 5000 \
  --depths 50 \
  --samples 1 \
  --needles "" \
  --out ./smoke-test
```

Expect: `✓ 1/1 correct · report → smoke-test/report.html`.

### 2. Full curve on your own corpus

```bash
ctx-probe run \
  --model claude-opus-4-7 \
  --corpus /path/to/your-docs \
  --context-length 128000 \
  --depths 10,25,50,75,90 \
  --samples 3 \
  --out ./report
```

Open `report/report.html`. One chart: accuracy vs. needle depth. That's your curve.

### 3. Domain-specific needle (what to do for production use cases)

The default needle is fictional ("Blue Octopus" with a made-up code). For a real diagnostic on your domain, supply a needle that **the model cannot know from training** — a fictional-but-plausible identifier with a unique value:

```bash
ctx-probe run \
  --model claude-opus-4-7 \
  --corpus /path/to/clinical-standards \
  --context-length 100000 \
  --depths 10,25,50,75,90 \
  --samples 3 \
  --needles "" \
  --needle-text "Per internal protocol SI-2026-NEPHRO-742, the serum creatinine threshold for nephrology referral in Type 2 diabetic patients is 2.3 mg/dL." \
  --needle-question "What is the serum creatinine threshold for nephrology referral specified in protocol SI-2026-NEPHRO-742?" \
  --needle-expected "2.3" \
  --out ./clinical-baseline
```

**Why fictional matters:** if the needle could be answered from the model's prior knowledge, you can't tell retrieval from hallucination. The fictional ID forces the answer to come from your document, not the model's training data.

---

## What it measures

- **NIAH depth probe** — inserts a single fact at the specified depths, asks the model to retrieve it
- **MRCR-style multi-needle probe** — inserts 4 and 8 similar facts at evenly spaced depths, asks for one specific fact; tests disambiguation
- **Prompt caching** — the haystack is cached, so depth sweeps don't pay the full encode cost on every call. Verified via `cache_read_input_tokens` in the response
- Outputs `report.html` (chart + summary), `results.json` (full run summary), and `results.jsonl` (one line per call, append-only — safe against crashes)

## What it does not do (yet)

- No RAG wrapping. We measure raw model context handling first; RAG-on-top is v0.3.
- No fix recommendations. Measurement only. If you want help reading the curve, [book 20 minutes](https://opportunedev.com/#services).
- v0.1 supports **Anthropic models only** out of the box (`claude-*`). For anything else, wrap it with a custom adapter (see below). OpenAI and Gemini land in v0.2.
- No telemetry. Zero. Read the code.

---

## CLI reference

| Flag | Default | What it does |
| --- | --- | --- |
| `--model` | required | Any `claude-*` model ID (e.g. `claude-opus-4-7`, `claude-sonnet-4-6`) |
| `--corpus` | required | Path to a directory of `.txt`/`.md` files, or a single file |
| `--context-length` | `32000` | Approximate haystack size in tokens |
| `--depths` | `10,25,50,75,90` | Needle depth positions, as percentages |
| `--needles` | `4,8` | Multi-needle counts. Empty string skips multi-needle |
| `--samples` | `1` | Samples per combination — bump to 3+ for statistical signal |
| `--out` | `./report` | Output directory |
| `--seed` | `42` | RNG seed for haystack assembly (reproducible runs) |
| `--needle-text` | (built-in fictional needle) | Custom NIAH needle |
| `--needle-question` | (matches built-in) | Question to ask about your custom needle |
| `--needle-expected` | (matches built-in) | Expected substring in the answer |
| `--skip-niah` / `--skip-multi-needle` | off | Skip a probe |

`--needle-text`, `--needle-question`, and `--needle-expected` must be provided together when overriding the default needle.

---

## Using ctx-probe as a library

CLI is fine for one-off baselines. To plug it into a CI job, a production observability dashboard, or to probe a system that isn't a raw Anthropic API call, use the Python API directly.

### Programmatic NIAH against an Anthropic model

```python
from ctx_probe.adapters import build_adapter
from ctx_probe.corpus import chunk_documents, load_corpus
from ctx_probe.probes.needle import NeedleConfig, run_niah

adapter = build_adapter("claude-opus-4-7")
chunks = chunk_documents(load_corpus("/path/to/docs"))

cfg = NeedleConfig(
    needle_text="Per protocol SI-2026-NEPHRO-742, the threshold is 2.3 mg/dL.",
    question="What is the threshold in protocol SI-2026-NEPHRO-742?",
    expected="2.3",
    depths=[0.1, 0.25, 0.5, 0.75, 0.9],
    samples_per_depth=3,
    target_tokens=100_000,
)

results = run_niah(adapter, chunks, cfg, seed=42)
for r in results:
    print(f"depth={r.depth} sample={r.sample_idx} correct={r.correct} "
          f"latency={r.latency_ms:.0f}ms cache_hit={r.cache_read_tokens}t")
```

### Custom adapter — wrap a production system

`ctx-probe` doesn't care what's behind the model call. Any class that implements the `ModelAdapter` interface plugs into the probe machinery:

```python
import time
import requests
from ctx_probe.adapters import ModelAdapter, ModelResponse


class ProductionAdapter(ModelAdapter):
    """Wrap a production LLM endpoint (your FastAPI, vLLM, SageMaker, etc.)
    so ctx-probe measures the same stack your users hit."""

    def __init__(self, endpoint: str, api_key: str):
        self.name = "prod:clinical-pipeline"
        self.endpoint = endpoint
        self.api_key = api_key

    def complete(self, cacheable_prefix, query, *, max_tokens=256, system=None):
        # Assemble the prompt however your production system expects it.
        # cacheable_prefix is the haystack; query is the user's question.
        prompt = f"{system}\n\n{cacheable_prefix}\n\nQ: {query}" if system else \
                 f"{cacheable_prefix}\n\nQ: {query}"

        t0 = time.perf_counter()
        resp = requests.post(
            self.endpoint,
            json={"prompt": prompt, "max_tokens": max_tokens},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        return ModelResponse(
            text=data["text"],
            input_tokens=data.get("usage", {}).get("input_tokens", 0),
            output_tokens=data.get("usage", {}).get("output_tokens", 0),
            cache_read_tokens=data.get("usage", {}).get("cache_read_tokens", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
            raw=data,
        )
```

Then pass `ProductionAdapter` instead of the built-in adapter:

```python
adapter = ProductionAdapter(endpoint="https://my-llm.internal/v1/complete",
                            api_key="...")
results = run_niah(adapter, chunks, cfg)
```

### Wrap an in-process Python pipeline (not an HTTP endpoint)

If your production "model" is a Python class — your own RAG retriever, a wrapped Anthropic SDK with custom prompt scaffolding, a local model behind a function — the adapter is even thinner:

```python
from ctx_probe.adapters import ModelAdapter, ModelResponse
import time

class InProcessAdapter(ModelAdapter):
    def __init__(self, pipeline):
        self.name = f"inproc:{pipeline.__class__.__name__}"
        self.pipeline = pipeline

    def complete(self, cacheable_prefix, query, *, max_tokens=256, system=None):
        t0 = time.perf_counter()
        text = self.pipeline.answer(context=cacheable_prefix, question=query)
        return ModelResponse(
            text=text,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

adapter = InProcessAdapter(my_production_pipeline)
```

### Full programmatic run with HTML report

If you want the chart and summary table, drive `run()` and `render()` directly:

```python
from ctx_probe.runner import RunConfig, run
from ctx_probe.report import render
from pathlib import Path

cfg = RunConfig(
    model="claude-opus-4-7",
    corpus_path="/path/to/docs",
    out_dir="./report",
    target_tokens=100_000,
    depths=[0.1, 0.5, 0.9],
    samples_per_depth=3,
    needle_text="...",
    needle_question="...",
    needle_expected="...",
)
results = run(adapter, cfg)        # writes results.jsonl + results.json
render(results, model=adapter.name, target_tokens=cfg.target_tokens,
       out_path=Path("./report/report.html"))
```

---

## Output schema

Each call produces a `NeedleResult` (one JSON line in `results.jsonl`, one entry in the `results` array of `results.json`):

| Field | Type | Meaning |
| --- | --- | --- |
| `probe` | str | `niah` or `multi_needle_<N>` |
| `depth` | float | Needle position in context (0.0–1.0) |
| `sample_idx` | int | Which sample of this (probe, depth) combo |
| `correct` | bool | Did the response contain the expected substring (case-insensitive)? |
| `response_text` | str | The model's full answer |
| `expected` | str | The substring we graded against |
| `input_tokens` | int | Uncached input tokens (full price) |
| `output_tokens` | int | Output tokens |
| `cache_read_tokens` | int | Tokens served from cache (~0.1× price) |
| `cache_write_tokens` | int | Tokens written to cache (~1.25× price) |
| `latency_ms` | float | End-to-end call duration |
| `haystack_tokens` | int | Assembled haystack size |
| `needle_token_position` | int | Approximate token offset of the needle |
| `error` | str | Exception message if the call failed; empty otherwise |

Anything missing for your use case is a one-line dataclass extension in `src/ctx_probe/probes/needle.py`.

---

## Cost

A single run with the defaults (5 NIAH depths × 1 sample + 2 needle counts × 3 depths × 1 sample = 11 calls, at 32K context) costs a few cents on Sonnet 4.6, somewhat more on Opus. Bumping `--context-length` to 1M and `--samples` to 5 runs into the tens of dollars — start small, validate the curve, then scale.

Cache writes (~1.25× input price) happen on the first call against a given haystack; subsequent samples at the same depth read from cache (~0.1×). Bumping `--samples` is cheaper than you'd expect for this reason — most of the per-sample cost is amortized.

## Development

```bash
git clone https://github.com/opportune/ctx-probe
cd ctx-probe
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

28 tests, runs in < 1 second, no API calls in the test suite.

## License

MIT. Built by [Opportune](https://opportunedev.com) — we help AI teams diagnose and fix context engineering failures in production.
