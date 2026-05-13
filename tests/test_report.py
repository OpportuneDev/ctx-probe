from __future__ import annotations

from ctx_probe.probes.needle import NeedleResult
from ctx_probe.report import _chart_data, _summary_rows, render


def _make_result(probe="niah", depth=0.5, correct=True, error=""):
    return NeedleResult(
        probe=probe,
        depth=depth,
        sample_idx=0,
        correct=correct,
        response_text="ok" if correct else "miss",
        input_tokens=100,
        output_tokens=20,
        cache_read_tokens=50,
        cache_write_tokens=10,
        latency_ms=100.0,
        haystack_tokens=1500,
        needle_token_position=750,
        expected="MARKER",
        error=error,
    )


def test_summary_rows_compute_accuracy():
    results = [
        _make_result(correct=True),
        _make_result(correct=True),
        _make_result(correct=False),
    ]
    rows = _summary_rows(results)
    assert len(rows) == 1
    assert rows[0]["calls"] == 3
    assert rows[0]["accuracy"] == 2 / 3


def test_summary_rows_excludes_errors_from_accuracy():
    results = [
        _make_result(correct=True),
        _make_result(correct=False, error="boom"),
    ]
    rows = _summary_rows(results)
    assert rows[0]["accuracy"] == 1.0  # only one non-errored, and it's correct


def test_chart_data_groups_by_probe_and_depth():
    results = [
        _make_result(probe="niah", depth=0.1, correct=True),
        _make_result(probe="niah", depth=0.5, correct=False),
        _make_result(probe="multi_needle_4", depth=0.5, correct=True),
    ]
    data = _chart_data(results)
    assert set(data["labels"]) == {"0.1", "0.5"}
    assert len(data["datasets"]) == 2
    labels = {ds["label"] for ds in data["datasets"]}
    assert labels == {"niah", "multi_needle_4"}


def test_render_writes_html(tmp_path):
    results = [_make_result(depth=0.5, correct=True), _make_result(depth=0.9, correct=False)]
    out = tmp_path / "report.html"
    rendered = render(results, model="mock", target_tokens=1500, out_path=out)
    assert rendered.exists()
    html = rendered.read_text()
    assert "Effective context probe" in html
    assert "mock" in html
    assert "Chart" in html  # Chart.js is loaded
