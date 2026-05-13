# ctx-probe

**Measure your model's effective context window — on your data, in 5 minutes.**

Frontier models advertise million-token context windows. In practice, accuracy collapses well before that, and the drop is task- and corpus-specific. `ctx-probe` measures *your* model's degradation curve on *your* documents, so you can stop guessing where retrieval starts to silently fail.

The published benchmark numbers (MRCR, RULER, HELMET) are measured on synthetic haystacks. Your stack is not synthetic. The only number that tells you where your pipeline breaks is the one measured on your data.

## Quickstart

```bash
pip install ctx-probe
export ANTHROPIC_API_KEY=sk-ant-...

ctx-probe run \
  --model claude-opus-4-7 \
  --corpus ./my-docs \
  --context-length 128000 \
  --out ./report
```

Open `report/report.html`. One chart: accuracy vs. needle depth, one line per needle count. That's your curve.

## What it measures

- **NIAH depth probe** — inserts a single fact at depths 10 / 25 / 50 / 75 / 90 % of context, asks the model to retrieve it
- **MRCR-style multi-needle probe** — inserts 4 and 8 similar facts at evenly spaced depths, asks for one specific fact; tests whether the model can distinguish them
- **Prompt caching** — the haystack is cached, so depth sweeps don't pay the full encode cost on every call
- Outputs `report.html` (the chart and summary table) plus `results.json` and `results.jsonl` (per-call detail, incrementally written so a crash mid-run doesn't lose work)

## What it does not do (yet)

- No RAG wrapping. We measure raw model context handling first; RAG-on-top is v0.3.
- No fix recommendations. Measurement only. If you want help reading the curve, [book 20 minutes](https://opportunedev.com/#services).
- v0.1 supports **Anthropic models only** (`claude-*`). OpenAI and Gemini adapters land in v0.2.
- No telemetry. Zero. Read the code.

## Configuration

| Flag | Default | What it does |
| --- | --- | --- |
| `--model` | required | Any `claude-*` model ID (e.g. `claude-opus-4-7`, `claude-sonnet-4-6`) |
| `--corpus` | required | Path to a directory of `.txt`/`.md` files, or a single file |
| `--context-length` | `32000` | Approximate haystack size in tokens |
| `--depths` | `10,25,50,75,90` | Needle depth positions, as percentages |
| `--needles` | `4,8` | Multi-needle counts. Empty string skips multi-needle |
| `--samples` | `1` | Samples per combination — bump to 3+ for statistical signal |
| `--out` | `./report` | Output directory |
| `--seed` | `42` | RNG seed for haystack assembly |
| `--skip-niah` / `--skip-multi-needle` | off | Skip a probe |

## Cost

A single run with the defaults (5 depths × 1 sample for NIAH + 2 needle counts × 3 depths × 1 sample for multi-needle = 11 calls, at 32K context) costs a few cents on Sonnet 4.6, somewhat more on Opus. Bumping `--context-length` to 1M tokens and `--samples` to 5 will run into the tens of dollars — start small.

## Development

```bash
git clone https://github.com/opportune/ctx-probe
cd ctx-probe
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT. Built by [Opportune](https://opportunedev.com) — we help AI teams diagnose and fix context engineering failures in production.
