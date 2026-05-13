from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ctx_probe.probes.needle import NeedleResult


def _summary_rows(results: list[NeedleResult]) -> list[dict]:
    by_probe: dict[str, list[NeedleResult]] = defaultdict(list)
    for r in results:
        by_probe[r.probe].append(r)

    rows = []
    for probe, items in sorted(by_probe.items()):
        valid = [r for r in items if not r.error]
        if not valid:
            rows.append(
                {
                    "probe": probe,
                    "calls": len(items),
                    "accuracy": 0.0,
                    "avg_input_tokens": 0,
                    "cache_hit_rate": 0.0,
                    "avg_latency_ms": 0.0,
                }
            )
            continue
        total_input = sum(r.input_tokens + r.cache_read_tokens for r in valid)
        cache_read = sum(r.cache_read_tokens for r in valid)
        rows.append(
            {
                "probe": probe,
                "calls": len(items),
                "accuracy": sum(r.correct for r in valid) / len(valid),
                "avg_input_tokens": int(total_input / len(valid)) if valid else 0,
                "cache_hit_rate": (cache_read / total_input) if total_input else 0.0,
                "avg_latency_ms": sum(r.latency_ms for r in valid) / len(valid),
            }
        )
    return rows


def _chart_data(results: list[NeedleResult]) -> dict:
    by_probe_depth: dict[str, dict[float, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.error:
            continue
        by_probe_depth[r.probe][round(r.depth, 4)].append(r.correct)

    all_depths = sorted({d for series in by_probe_depth.values() for d in series.keys()})
    labels = [str(d) for d in all_depths]

    datasets = []
    for probe in sorted(by_probe_depth.keys()):
        series = by_probe_depth[probe]
        data = []
        for d in all_depths:
            samples = series.get(d, [])
            data.append(sum(samples) / len(samples) if samples else None)
        datasets.append({"label": probe, "data": data})

    return {"labels": labels, "datasets": datasets}


def render(
    results: list[NeedleResult],
    model: str,
    target_tokens: int,
    out_path: str | Path,
    *,
    started_at: str | None = None,
) -> Path:
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.j2")

    failures = [
        asdict(r)
        for r in results
        if not r.correct and not r.error
    ][:10]

    html = template.render(
        model=model,
        started_at=started_at or datetime.now(timezone.utc).isoformat(),
        target_tokens=target_tokens,
        total_calls=len(results),
        summary_rows=_summary_rows(results),
        chart_data_json=json.dumps(_chart_data(results)),
        failures=failures,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
