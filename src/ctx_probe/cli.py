from __future__ import annotations

from pathlib import Path

import click

from ctx_probe.adapters import build_adapter
from ctx_probe.report import render
from ctx_probe.runner import RunConfig, run


def _parse_floats(ctx, param, value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_ints(ctx, param, value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


@click.group()
@click.version_option()
def main():
    """Measure your model's effective context window on your data."""


@main.command()
@click.option("--model", required=True, help="e.g. claude-opus-4-7, claude-sonnet-4-6")
@click.option(
    "--corpus",
    "corpus_path",
    required=True,
    type=click.Path(exists=True),
    help="Directory of .txt/.md files, or a single file.",
)
@click.option(
    "--context-length",
    "target_tokens",
    default=32_000,
    show_default=True,
    type=int,
    help="Approximate haystack size in tokens.",
)
@click.option(
    "--depths",
    default="10,25,50,75,90",
    show_default=True,
    callback=lambda c, p, v: [float(x) / 100 for x in v.split(",") if x.strip()],
    help="Depth positions (percent) to insert needles at.",
)
@click.option(
    "--needles",
    "needle_counts",
    default="4,8",
    show_default=True,
    callback=_parse_ints,
    help="Multi-needle counts. Pass empty to skip multi-needle probe.",
)
@click.option(
    "--samples",
    "samples_per_depth",
    default=1,
    show_default=True,
    type=int,
    help="Samples per (depth, needle-count) combination.",
)
@click.option(
    "--out",
    "out_dir",
    default="./report",
    show_default=True,
    type=click.Path(),
    help="Output directory for report.html and results.{json,jsonl}.",
)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--skip-niah", is_flag=True, help="Skip the single-needle NIAH probe.")
@click.option(
    "--skip-multi-needle", is_flag=True, help="Skip the multi-needle MRCR-style probe."
)
@click.option(
    "--needle-text",
    default=None,
    help="Custom NIAH needle to inject into the haystack. "
    "Must be a fact the model cannot know from training (e.g. fictional ID + value).",
)
@click.option(
    "--needle-question",
    default=None,
    help="Question to ask the model about the needle.",
)
@click.option(
    "--needle-expected",
    default=None,
    help="Expected substring in the model's answer (case-insensitive).",
)
def run_cmd(
    model,
    corpus_path,
    target_tokens,
    depths,
    needle_counts,
    samples_per_depth,
    out_dir,
    seed,
    skip_niah,
    skip_multi_needle,
    needle_text,
    needle_question,
    needle_expected,
):
    """Run probes against a model and write an HTML report."""
    custom_needle_flags = [needle_text, needle_question, needle_expected]
    if any(f is not None for f in custom_needle_flags) and not all(
        f is not None for f in custom_needle_flags
    ):
        raise click.UsageError(
            "When overriding the needle, all three of --needle-text, "
            "--needle-question, --needle-expected must be provided together."
        )

    cfg = RunConfig(
        model=model,
        corpus_path=str(corpus_path),
        out_dir=str(out_dir),
        target_tokens=target_tokens,
        depths=depths,
        needle_counts=needle_counts,
        samples_per_depth=samples_per_depth,
        seed=seed,
        run_niah=not skip_niah,
        run_multi_needle=not skip_multi_needle and bool(needle_counts),
        needle_text=needle_text,
        needle_question=needle_question,
        needle_expected=needle_expected,
    )

    adapter = build_adapter(model)
    click.echo(f"→ Running {adapter.name} against corpus={corpus_path} target={target_tokens} tokens")
    click.echo(f"  depths={depths} needles={needle_counts} samples={samples_per_depth}")

    results = run(adapter, cfg)

    report_path = render(
        results,
        model=adapter.name,
        target_tokens=target_tokens,
        out_path=Path(out_dir) / "report.html",
    )

    n_ok = sum(1 for r in results if r.correct and not r.error)
    click.echo(f"\n✓ {n_ok}/{len(results)} correct · report → {report_path}")


main.add_command(run_cmd, name="run")


if __name__ == "__main__":
    main()
