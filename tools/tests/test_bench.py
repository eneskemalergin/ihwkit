"""Test the statistical benchmark's pure metrics and explicit matrix."""

from __future__ import annotations

import numpy as np

from bench.__main__ import MATRIX, AttemptRow, _summarize, _wilson_interval
from bench.report import (
    METHOD_LABELS,
    PLOT_SIZES,
    TimingRow,
    _format_markdown_tables,
    _merge_current_and_peer_rows,
    _paired_validity,
    _scope_ratio,
)
from tools.benchmark_zebrac import DEFAULT_METHODS


def _null_attempt(replicate: int, rejections: int) -> AttemptRow:
    return AttemptRow(
        scenario_id="global_null",
        assumption_class="independent valid null",
        method_id="ihw_inf_cv",
        n=100,
        replicate=replicate,
        seed=2000 + replicate,
        alpha=0.1,
        status="ok",
        rejections=rejections,
        false_rejections=rejections,
        any_rejection=int(rejections > 0),
        fdp=float(rejections > 0),
        power=None,
        error_type=None,
        error_message=None,
    )


def test_global_null_summary_uses_probability_of_any_rejection() -> None:
    rows = [_null_attempt(0, 0), _null_attempt(1, 3), _null_attempt(2, 0)]

    summary = _summarize(rows)[0]

    assert summary.mean_fdp == 1.0 / 3.0
    assert summary.mean_rejections == 1.0
    assert summary.fdr_ci_low is not None
    assert summary.fdr_ci_high is not None
    assert summary.fdr_ci_low <= summary.mean_fdp <= summary.fdr_ci_high


def test_wilson_interval_is_finite_and_bounded() -> None:
    lower, upper = _wilson_interval(10, 100)

    assert np.isfinite(lower)
    assert np.isfinite(upper)
    assert 0.0 <= lower <= 0.1 <= upper <= 1.0


def test_benchmark_matrix_keeps_evidence_questions_separate() -> None:
    tracks = {row.track for row in MATRIX}

    assert {"correctness", "parity", "validity", "robustness", "performance"} <= tracks
    single_cell_levels = {
        row.data_level for row in MATRIX if row.track.startswith("single-cell")
    }
    assert len(single_cell_levels) == 3


def test_performance_defaults_to_the_changing_method_only() -> None:
    assert DEFAULT_METHODS == ("ihwkit_numpy_numba",)


def test_report_uses_short_production_label_and_explicit_scaling_sizes() -> None:
    assert METHOD_LABELS["ihwkit_numpy_numba"] == "ihwkit"
    assert PLOT_SIZES == (5_000, 15_000, 50_000)


def test_generated_markdown_tables_are_human_aligned() -> None:
    lines = [
        "| method | time |",
        "|---|---:|",
        "| ihwkit | 2.0 ms |",
        "| NumPy reference | 20.0 ms |",
    ]

    formatted = _format_markdown_tables(lines)

    assert formatted == [
        "| method          |    time |",
        "| --------------- | ------: |",
        "| ihwkit          |  2.0 ms |",
        "| NumPy reference | 20.0 ms |",
    ]


def test_scope_ratio_is_human_readable() -> None:
    warm = _timing_row("ihwkit_numpy_numba", 12.0)
    process = _timing_row("ihwkit_numpy_numba", 99.0)

    assert _scope_ratio(warm, process) == "8.2x"


def test_peer_baseline_merge_keeps_current_production_and_saved_peers() -> None:
    current = [_timing_row("ihwkit_numpy_numba", 12.0)]
    saved = [
        _timing_row("ihwkit_numpy_numba", 99.0),
        _timing_row("ihwkit_scipy", 8.0),
    ]

    merged = _merge_current_and_peer_rows(current, saved)

    assert [(row.method_id, row.wall_median_ns) for row in merged] == [
        ("ihwkit_numpy_numba", 12.0),
        ("ihwkit_scipy", 8.0),
    ]


def test_paired_validity_drops_failed_production_attempts() -> None:
    rows = [
        {
            "scenario_id": "mixture_mild",
            "method_id": "bh",
            "replicate": "0",
            "status": "ok",
            "fdp": "0.1",
            "power": "0.2",
        },
        {
            "scenario_id": "mixture_mild",
            "method_id": "ihw_inf_cv",
            "replicate": "0",
            "status": "ok",
            "fdp": "0.05",
            "power": "0.3",
        },
        {
            "scenario_id": "mixture_mild",
            "method_id": "bh",
            "replicate": "1",
            "status": "ok",
            "fdp": "0.0",
            "power": "0.1",
        },
        {
            "scenario_id": "mixture_mild",
            "method_id": "ihw_inf_cv",
            "replicate": "1",
            "status": "error",
            "fdp": "",
            "power": "",
        },
    ]

    summary = _paired_validity(rows)[0]

    assert summary.paired_replicates == 1
    assert summary.fdp_difference == -0.05
    assert np.isclose(summary.power_difference, 0.1)


def _timing_row(method_id: str, median: float) -> TimingRow:
    return TimingRow(
        measurement="warm_fit",
        dataset_id="sim_5000_seed42",
        size=5_000,
        method_id=method_id,
        status="ok",
        version="test",
        sample_count=1,
        failed_sample_count=0,
        wall_mean_ns=median,
        wall_median_ns=median,
        wall_std_ns=0.0,
        peak_rss_mean_bytes=None,
        peak_rss_median_bytes=None,
        rejection_count=1,
        error=None,
    )
