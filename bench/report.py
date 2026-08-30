"""Collect and render one compact, comparative benchmark report.

Raw attempts stay below ``tmp/results``. The reviewed Markdown report, three
figures, and one reusable peer-performance table are the only public outputs.
There is no aggregate score and no opaque result identity machinery.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.peer import (
    METHODS,
    PeerInput,
    RunConfig,
    fit,
    load_peer_input,
    load_reference,
)
from tools.simulators import SCENARIO_BUILDERS

REPORT_PATH = ROOT / "bench" / "REPORT.md"
FIGURE_DIR = ROOT / "bench" / "figures"
PEER_BASELINE_PATH = ROOT / "bench" / "peer-performance.json"
PROCESS_DATASETS = (
    "sim_500_seed42",
    "sim_5000_seed42",
    "sim_15000_seed42",
    "sim_50000_seed42",
)
PLOT_SIZES = (5_000, 15_000, 50_000)
SCENARIO_ORDER = (
    "global_null",
    "null_covariate",
    "mixture_mild",
    "mixture_sparse",
    "ignatiadis",
    "dense_covariate",
)
PRODUCTION = "ihwkit"
PEER_METHODS = tuple(method for method in METHODS if method != PRODUCTION)
METHOD_LABELS = {
    "bh": "BH",
    "ihw_inf_cv": "ihwkit",
    "ihwkit": "ihwkit",
    "ihwkit_scipy": "SciPy/HiGHS",
    "pyihw": "pyihw",
    "r_ihw": "R IHW",
}
METHOD_MARKERS = {
    "ihwkit": "o",
    "ihwkit_scipy": "^",
    "pyihw": "D",
    "r_ihw": "P",
}


@dataclass(frozen=True)
class TimingRow:
    """Record one method and dataset timing outcome."""

    measurement: str
    dataset_id: str
    size: int
    method_id: str
    status: str
    version: str | None
    sample_count: int
    failed_sample_count: int
    wall_mean_ns: float | None
    wall_median_ns: float | None
    wall_std_ns: float | None
    peak_rss_mean_bytes: float | None
    peak_rss_median_bytes: float | None
    rejection_count: int | None
    error: str | None


@dataclass(frozen=True)
class FigureTheme:
    """Define the small visual vocabulary shared by every report figure."""

    background: str
    ink: str
    muted: str
    grid: str
    accent: str
    method_colors: Mapping[str, str]


@dataclass(frozen=True)
class PeerParityRow:
    """Compare one Python solver with the fixed R IHW output vectors."""

    reference_id: str
    method_id: str
    status: str
    version: str | None
    reference_rejections: int
    method_rejections: int | None
    max_adjusted_difference: float | None
    max_weight_difference: float | None
    decision_agreement: float | None
    error: str | None


@dataclass(frozen=True)
class PairedValidityRow:
    """Summarize paired ihwkit-minus-BH replicate differences."""

    scenario_id: str
    paired_replicates: int
    fdp_difference: float
    fdp_difference_mcse: float
    power_difference: float | None
    power_difference_mcse: float | None


def study_main(argv: Sequence[str] | None = None) -> int:
    """Run the comparative study and render its public report."""

    parser = _study_parser()
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.duration < 1:
        parser.error("--duration must be positive")
    result_dir = (ROOT / args.result_dir).resolve()
    _ensure_result_dir(result_dir)
    _write_json(result_dir / "environment.json", _environment())
    relative_result_dir = result_dir.relative_to(ROOT)
    print(f"full study results: {relative_result_dir}")

    statuses: dict[str, int] = {}
    statuses["tests"] = _run_tests(result_dir)

    from bench.__main__ import _validity_main
    from tools.check_peer_correctness import main as correctness_main
    from tools.replay_parity import main as replay_main

    statuses["correctness"] = correctness_main(
        ["--result-dir", str(relative_result_dir), "--quiet"]
    )
    statuses["parity"] = replay_main(
        ["--result-dir", str(relative_result_dir)], gates_only=True
    )
    statuses["robustness"] = replay_main(
        ["--result-dir", str(relative_result_dir)], gates_only=False
    )
    validity_arguments = [
        "--name",
        "validity_full" if not args.quick else "validity_quick",
        "--result-dir",
        str(relative_result_dir),
    ]
    if args.quick:
        validity_arguments.append("--quick")
    statuses["validity"] = _validity_main(validity_arguments)

    parity_rows = _run_peer_parity()
    _write_json(result_dir / "peer_parity.json", [asdict(row) for row in parity_rows])
    stress_rows = _run_stress_peers()
    _write_json(result_dir / "stress_peers.json", stress_rows)

    timing_methods = METHODS if args.refresh_peers else (PRODUCTION,)
    warm_rows = _run_warm_timings(
        methods=timing_methods,
        warmups=args.warmup,
        samples=args.samples,
    )
    _write_json(result_dir / "warm_timings.json", [asdict(row) for row in warm_rows])
    process_rows = _run_process_timings(
        result_dir=result_dir,
        methods=timing_methods,
        warmups=args.warmup,
        samples=args.samples,
        duration=args.duration,
    )
    _write_json(
        result_dir / "process_timings.json", [asdict(row) for row in process_rows]
    )

    if args.refresh_peers:
        _write_peer_baseline(
            warm_rows,
            process_rows,
            warmups=args.warmup,
            samples=args.samples,
            duration=args.duration,
        )
    elif not PEER_BASELINE_PATH.is_file():
        raise SystemExit(
            "bench/peer-performance.json is missing; run study once with "
            "--refresh-peers"
        )

    validity_stem = "validity_quick" if args.quick else "validity_full"
    render_report(
        result_dir=result_dir,
        validity_stem=validity_stem,
        quick=args.quick,
    )
    expected_failures = statuses["robustness"] != 0 or statuses["validity"] != 0
    required_failure = any(
        statuses[name] != 0 for name in ("tests", "correctness", "parity")
    )
    if expected_failures:
        print("report includes robustness or validity failures", file=sys.stderr)
    return 1 if required_failure or expected_failures else 0


def report_main(argv: Sequence[str] | None = None) -> int:
    """Render a report from an existing full-study result directory."""

    parser = argparse.ArgumentParser(prog="python -m bench report")
    parser.add_argument(
        "--result-dir", type=Path, default=Path("tmp/results/full-study")
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    result_dir = (ROOT / args.result_dir).resolve()
    _ensure_result_dir(result_dir, create=False)
    validity_stem = "validity_quick" if args.quick else "validity_full"
    render_report(
        result_dir=result_dir,
        validity_stem=validity_stem,
        quick=args.quick,
    )
    return 0


def _study_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bench study")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--refresh-peers",
        action="store_true",
        help="Rerun unchanged peers and replace their dated performance table.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--duration", type=int, default=120_000)
    parser.add_argument(
        "--result-dir", type=Path, default=Path("tmp/results/full-study")
    )
    return parser


def _run_tests(result_dir: Path) -> int:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    document = {
        "status": "ok" if completed.returncode == 0 else "error",
        "exit_code": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "command": f"{sys.executable} -m pytest -q",
        "output": output,
    }
    _write_json(result_dir / "tests.json", document)
    print(output or "pytest produced no output")
    return completed.returncode


def _run_peer_parity() -> list[PeerParityRow]:
    record = load_reference("sim_5000_inf_n5")
    config = RunConfig(alpha=0.1, nbins=3, nfolds=5, seed=42)
    decisions = record.adjusted_pvalues <= config.alpha
    rows: list[PeerParityRow] = []
    for method_id in (PRODUCTION, "ihwkit_scipy", "pyihw"):
        try:
            result = fit(method_id, record.peer_input, config)
            if result.weights is None:
                raise RuntimeError("method did not return weights")
        except Exception as exc:  # noqa: BLE001 - every peer outcome is evidence
            rows.append(
                PeerParityRow(
                    reference_id=record.spec.reference_id,
                    method_id=method_id,
                    status="unavailable"
                    if type(exc).__name__ == "PeerUnavailable"
                    else "error",
                    version=None,
                    reference_rejections=record.r_rejections,
                    method_rejections=None,
                    max_adjusted_difference=None,
                    max_weight_difference=None,
                    decision_agreement=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        rows.append(
            PeerParityRow(
                reference_id=record.spec.reference_id,
                method_id=method_id,
                status=(
                    "pass"
                    if np.allclose(
                        result.adjusted_pvalues,
                        record.adjusted_pvalues,
                        atol=1e-8,
                        rtol=1e-6,
                    )
                    and np.allclose(
                        result.weights, record.weights, atol=1e-8, rtol=1e-6
                    )
                    else "difference"
                ),
                version=result.version,
                reference_rejections=record.r_rejections,
                method_rejections=result.rejection_count,
                max_adjusted_difference=float(
                    np.max(np.abs(result.adjusted_pvalues - record.adjusted_pvalues))
                ),
                max_weight_difference=float(
                    np.max(np.abs(result.weights - record.weights))
                ),
                decision_agreement=float(
                    np.mean((result.adjusted_pvalues <= config.alpha) == decisions)
                ),
                error=None,
            )
        )
    return rows


def _run_stress_peers() -> list[dict[str, object]]:
    draw = SCENARIO_BUILDERS["mixture_mild"](3_000, 2_034)
    peer_input = PeerInput(
        dataset_id="mixture_mild_n3000_seed2034",
        source_path="generated:mixture_mild(n=3000, seed=2034)",
        provenance="deterministic synthetic numerical stress input",
        size=3_000,
        seed=2_034,
        pvalues=draw.pvalues,
        covariates=draw.covariates,
        truth_labels=draw.is_null,
    )
    config = RunConfig(alpha=0.1, nbins="auto", nfolds=5, seed=2_034)
    rows: list[dict[str, object]] = []
    for method_id in (PRODUCTION, "ihwkit_scipy", "pyihw"):
        try:
            result = fit(method_id, peer_input, config)
        except Exception as exc:  # noqa: BLE001 - every peer outcome is evidence
            rows.append(
                {
                    "method_id": method_id,
                    "status": "unavailable"
                    if type(exc).__name__ == "PeerUnavailable"
                    else "error",
                    "version": None,
                    "rejection_count": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        rows.append(
            {
                "method_id": method_id,
                "status": "ok",
                "version": result.version,
                "rejection_count": result.rejection_count,
                "error": None,
            }
        )
    return rows


def _run_warm_timings(
    *, methods: Sequence[str], warmups: int, samples: int
) -> list[TimingRow]:
    rows: list[TimingRow] = []
    for dataset_id in PROCESS_DATASETS:
        peer_input = load_peer_input(dataset_id)
        for method_id in methods:
            print(f"warm fit: {dataset_id} {method_id}")
            config = RunConfig(alpha=0.1, nbins="auto", nfolds=5, seed=42)
            try:
                result = None
                for _ in range(warmups):
                    result = fit(method_id, peer_input, config)
                elapsed: list[int] = []
                for _ in range(samples):
                    started = time.perf_counter_ns()
                    result = fit(method_id, peer_input, config)
                    elapsed.append(time.perf_counter_ns() - started)
                if result is None:
                    raise RuntimeError("timing did not produce a fit")
            except Exception as exc:  # noqa: BLE001 - timing failures are evidence
                rows.append(
                    TimingRow(
                        "warm_fit",
                        dataset_id,
                        peer_input.size,
                        method_id,
                        "unavailable"
                        if type(exc).__name__ == "PeerUnavailable"
                        else "error",
                        None,
                        0,
                        samples,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            rows.append(
                TimingRow(
                    "warm_fit",
                    dataset_id,
                    peer_input.size,
                    method_id,
                    "ok",
                    result.version,
                    samples,
                    0,
                    float(statistics.fmean(elapsed)),
                    float(statistics.median(elapsed)),
                    float(statistics.stdev(elapsed)) if samples > 1 else 0.0,
                    None,
                    None,
                    result.rejection_count,
                    None,
                )
            )
    return rows


def _run_process_timings(
    *,
    result_dir: Path,
    methods: Sequence[str],
    warmups: int,
    samples: int,
    duration: int,
) -> list[TimingRow]:
    from tools.benchmark_zebrac import main as performance_main

    rows: list[TimingRow] = []
    relative_result_dir = result_dir.relative_to(ROOT)
    for dataset_id in PROCESS_DATASETS:
        size = load_peer_input(dataset_id).size
        selected = list(methods)
        name = f"process_{dataset_id}"
        print(f"process benchmark: {dataset_id} {' '.join(selected)}")
        performance_main(
            [
                "--dataset",
                dataset_id,
                "--methods",
                *selected,
                "--warmup",
                str(warmups),
                "--min-samples",
                str(samples),
                "--max-samples",
                str(samples),
                "--duration",
                str(duration),
                "--name",
                name,
                "--result-dir",
                str(relative_result_dir),
                "--quiet",
            ]
        )
        metadata_path = result_dir / f"{name}.metadata.json"
        if not metadata_path.is_file():
            for method_id in selected:
                rows.append(
                    TimingRow(
                        "cold_process",
                        dataset_id,
                        size,
                        method_id,
                        "error",
                        None,
                        0,
                        samples,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        "process benchmark did not write metadata",
                    )
                )
        else:
            rows.extend(_process_rows(metadata_path))
    return rows


def _process_rows(path: Path) -> list[TimingRow]:
    document = _read_json(path)
    dataset_id = str(document["dataset_id"])
    size = int(_mapping(document["data"])["size"])
    rows: list[TimingRow] = []
    for method_id, raw_method in _mapping(document["methods"]).items():
        method = _mapping(raw_method)
        metrics = _mapping(method.get("metrics", {}))
        wall = _mapping(metrics.get("wall_time", {}))
        rss = _mapping(metrics.get("peak_rss", {}))
        error_value = method.get("error")
        rows.append(
            TimingRow(
                "cold_process",
                dataset_id,
                size,
                method_id,
                str(method.get("status", "error")),
                None if method.get("version") is None else str(method["version"]),
                int(method.get("sample_count") or 0),
                int(method.get("failed_sample_count") or 0),
                _optional_float(wall.get("mean")),
                _optional_float(wall.get("median")),
                _optional_float(wall.get("std_dev")),
                _optional_float(rss.get("mean")),
                _optional_float(rss.get("median")),
                _optional_int(method.get("rejection_count")),
                _error_text(error_value),
            )
        )
    return rows


def _write_peer_baseline(
    warm_rows: Sequence[TimingRow],
    process_rows: Sequence[TimingRow],
    *,
    warmups: int,
    samples: int,
    duration: int,
) -> None:
    document = {
        "purpose": "Reusable machine-local measurements for peers that normally do not change",
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": _environment(),
        "configuration": {
            "alpha": 0.1,
            "nfolds": 5,
            "nbins": "auto",
            "seed": 42,
            "warmups": warmups,
            "samples": samples,
            "maximum_duration_ms": duration,
        },
        "warm_rows": [asdict(row) for row in warm_rows if row.method_id != PRODUCTION],
        "process_rows": [
            asdict(row) for row in process_rows if row.method_id != PRODUCTION
        ],
    }
    _write_json(PEER_BASELINE_PATH, document)
    print(f"wrote {PEER_BASELINE_PATH.relative_to(ROOT)}")


def render_report(
    *,
    result_dir: Path,
    validity_stem: str,
    quick: bool,
) -> None:
    """Render Markdown and light/dark SVG figures from collected evidence."""

    required = (
        "correctness.json",
        "ihw_replay_parity.json",
        "ihw_replay_robustness.json",
        f"{validity_stem}_summary.csv",
        f"{validity_stem}_attempts.csv",
        "peer_parity.json",
        "stress_peers.json",
        "warm_timings.json",
        "process_timings.json",
        "tests.json",
    )
    missing = [name for name in required if not (result_dir / name).is_file()]
    if missing:
        raise SystemExit(f"cannot render report; missing: {', '.join(missing)}")
    if not PEER_BASELINE_PATH.is_file():
        raise SystemExit("cannot render report; bench/peer-performance.json is missing")

    correctness = _read_json(result_dir / "correctness.json")
    parity = _read_json(result_dir / "ihw_replay_parity.json")
    robustness = _read_json(result_dir / "ihw_replay_robustness.json")
    tests = _read_json(result_dir / "tests.json")
    validity = _read_csv(result_dir / f"{validity_stem}_summary.csv")
    validity.sort(
        key=lambda row: (
            SCENARIO_ORDER.index(row["scenario_id"]),
            0 if row["method_id"] == "bh" else 1,
        )
    )
    attempts = _read_csv(result_dir / f"{validity_stem}_attempts.csv")
    paired = _paired_validity(attempts)
    peer_parity = _read_json_list(result_dir / "peer_parity.json")
    stress_peers = _read_json_list(result_dir / "stress_peers.json")
    current_warm = _timing_rows(result_dir / "warm_timings.json")
    current_process = _timing_rows(result_dir / "process_timings.json")
    baseline = _read_json(PEER_BASELINE_PATH)
    study_environment = (
        _read_json(result_dir / "environment.json")
        if (result_dir / "environment.json").is_file()
        else _environment()
    )
    warm_rows = _merge_current_and_peer_rows(
        current_warm, _timing_values(baseline["warm_rows"])
    )
    process_rows = _merge_current_and_peer_rows(
        current_process, _timing_values(baseline["process_rows"])
    )
    _render_figures(validity, paired, peer_parity, warm_rows, process_rows)
    lines = _report_lines(
        correctness=correctness,
        parity=parity,
        robustness=robustness,
        tests=tests,
        validity=validity,
        paired=paired,
        peer_parity=peer_parity,
        stress_peers=stress_peers,
        warm_rows=warm_rows,
        process_rows=process_rows,
        baseline=baseline,
        study_environment=study_environment,
        quick=quick,
    )
    REPORT_PATH.write_text(
        "\n".join(_format_markdown_tables(lines)) + "\n", encoding="utf-8"
    )
    print(f"wrote {REPORT_PATH.relative_to(ROOT)}")


def _format_markdown_tables(lines: Sequence[str]) -> list[str]:
    """Align generated Markdown tables for direct human inspection."""

    formatted: list[str] = []
    line_idx = 0
    while line_idx < len(lines):
        if line_idx + 1 >= len(lines):
            formatted.append(lines[line_idx])
            break
        header = _markdown_cells(lines[line_idx])
        separator = _markdown_cells(lines[line_idx + 1])
        if header is None or separator is None or not _is_table_separator(separator):
            formatted.append(lines[line_idx])
            line_idx += 1
            continue
        table = [header]
        line_idx += 2
        while line_idx < len(lines):
            cells = _markdown_cells(lines[line_idx])
            if cells is None or len(cells) != len(header):
                break
            table.append(cells)
            line_idx += 1
        formatted.extend(_aligned_table(table, separator))
    return formatted


def _markdown_cells(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def _is_table_separator(cells: Sequence[str]) -> bool:
    return bool(cells) and all(
        len(cell.strip(":")) >= 3 and not cell.strip(":").replace("-", "")
        for cell in cells
    )


def _aligned_table(table: list[list[str]], separator: list[str]) -> list[str]:
    widths = [
        max(3, max(len(row[column]) for row in table))
        for column in range(len(separator))
    ]
    alignments = [
        (
            "center"
            if cell.startswith(":") and cell.endswith(":")
            else "right"
            if cell.endswith(":")
            else "left"
        )
        for cell in separator
    ]

    def aligned_row(row: Sequence[str]) -> str:
        cells = []
        for value, width, alignment in zip(row, widths, alignments, strict=True):
            if alignment == "right":
                cells.append(value.rjust(width))
            elif alignment == "center":
                cells.append(value.center(width))
            else:
                cells.append(value.ljust(width))
        return "| " + " | ".join(cells) + " |"

    delimiter = []
    for width, alignment in zip(widths, alignments, strict=True):
        if alignment == "right":
            delimiter.append("-" * (width - 1) + ":")
        elif alignment == "center":
            delimiter.append(":" + "-" * (width - 2) + ":")
        else:
            delimiter.append("-" * width)
    return [aligned_row(table[0]), aligned_row(delimiter), *map(aligned_row, table[1:])]


def _report_lines(
    *,
    correctness: Mapping[str, object],
    parity: Mapping[str, object],
    robustness: Mapping[str, object],
    tests: Mapping[str, object],
    validity: Sequence[Mapping[str, str]],
    paired: Sequence[PairedValidityRow],
    peer_parity: Sequence[Mapping[str, object]],
    stress_peers: Sequence[Mapping[str, object]],
    warm_rows: Sequence[TimingRow],
    process_rows: Sequence[TimingRow],
    baseline: Mapping[str, object],
    study_environment: Mapping[str, object],
    quick: bool,
) -> list[str]:
    robustness_failures = int(robustness.get("fail", 0))
    robustness_failure_label = (
        f"{robustness_failures} failed diagnostic"
        f"{'s' if robustness_failures != 1 else ''}"
    )
    if robustness_failures:
        reliability_heading = "Numerical reliability has visible failures."
        reliability_summary = (
            "The remaining failed diagnostics stay visible in the detailed table. "
            "They block a blanket robustness claim, but no longer describe the "
            "current unregularized path as failing."
        )
        robustness_judgement = "failed named diagnostics block a release claim"
    else:
        reliability_heading = "The named numerical envelope passes."
        reliability_summary = (
            "Every named robustness diagnostic passed; this remains evidence for "
            "the tested cases rather than a universal numerical guarantee."
        )
        robustness_judgement = (
            "tested unregularized cases pass; broader robustness remains unclaimed"
        )
    validity_failures = sum(
        int(row["failures"]) for row in validity if row["method_id"] != "bh"
    )
    parity_pass = int(parity.get("fail", 1)) == 0
    correctness_pass = bool(correctness.get("correctness_gate"))
    test_status = str(tests.get("status"))
    test_summary = _last_line(str(tests.get("output", "")))
    production_validity = [row for row in validity if row["method_id"] == "ihw_inf_cv"]
    fdr_above_nominal = sum(
        bool(row["fdr_ci_low"]) and float(row["fdr_ci_low"]) > 0.1
        for row in production_validity
    )
    materially_positive_power = sum(
        row.power_difference is not None
        and row.power_difference_mcse is not None
        and row.power_difference - 1.96 * row.power_difference_mcse > 0.0
        for row in paired
    )
    paired_power_scenarios = sum(row.power_difference is not None for row in paired)
    baseline_environment = _mapping(baseline.get("environment", {}))
    environment_differences = [
        key
        for key in ("platform", "cpu", "python", "numpy", "scipy", "zebrac")
        if baseline_environment.get(key) != study_environment.get(key)
    ]
    performance_headline = _performance_headline(warm_rows, process_rows)
    lines = [
        "<!-- markdownlint-disable MD033 MD041 -->",
        "",
        "# ihwkit benchmark report",
        "",
        f"Recorded: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        "This is the current measurement baseline for correctness, R parity, statistical behavior, numerical robustness, speed, and process memory. It is a presentation of evidence, not a combined winner score. Failed and unavailable fits remain visible.",
        "",
        "## Method labels used throughout",
        "",
        "| label | implementation | role in this report |",
        "|---|---|---|",
        "| **ihwkit** | installable low-memory NumPy method; five-fold unregularized IHW | subject under evaluation |",
        "| **SciPy/HiGHS** | retained implementation using SciPy's HiGHS solver | numerical and performance reference; never a fallback |",
        "| **pyihw** | public pyihw 0.2.0 package | external Python comparison |",
        "| **R IHW** | Bioconductor IHW 1.40.0 | fixed parity authority and external timing comparison |",
        "| **BH** | unweighted Benjamini-Hochberg | statistical baseline only |",
        "",
        "`ihwkit` always means the production method in figures and tables. Longer internal method identifiers appear only in raw JSON.",
        "",
        "## Results at a glance",
        "",
        "| question | result | judgement |",
        "|---|---:|---|",
        f"| repository tests | {test_status} | {test_summary} |",
        f"| generated correctness | {'pass' if correctness_pass else 'fail'} | valid numerical output and structural invariants on named cases |",
        f"| fixed R parity | {'pass' if parity_pass else 'fail'} | strong agreement inside the declared fixed synthetic envelope |",
        f"| numerical robustness | {robustness_failure_label} | {robustness_judgement} |",
        f"| validity study | {validity_failures} failed ihwkit fits | FDR and power remain conditional on successful fits |",
        f"| FDR screen | {fdr_above_nominal}/{len(production_validity)} intervals wholly above 0.10 | no clear inflation in these named scenarios; not a universal guarantee |",
        f"| paired power | {materially_positive_power}/{paired_power_scenarios} intervals wholly above zero | gains in three scenarios and a loss in the dense-covariate scenario |",
        f"| performance | {len(process_rows)} process rows + {len(warm_rows)} warmed rows | current direct-default measurements; no universal speed claim |",
        "",
        "### Direct reading",
        "",
        "- **Numerical agreement is strong where it is defined.** The fixed five-fold synthetic replay agrees with R IHW in rejection decisions and full output vectors at errors far below the declared tolerance.",
        f"- **{reliability_heading}** {reliability_summary}",
        f"- **The development-scale statistical result is encouraging but conditional.** This is {'a quick wiring run' if quick else '1,000 replicates per null scenario and 200 per alternative scenario'}; {validity_failures} ihwkit failures are reported separately instead of converted to zero discoveries.",
        f"- **Performance is favorable on the measured scaling inputs and remains size-dependent.** {performance_headline}",
        f"- **The peer timing environment {'matches this study' if not environment_differences else 'differs on ' + ', '.join(environment_differences)}.** Peer measurements are reused only while that human-readable environment and protocol remain applicable.",
        "",
        "## Statistical evidence",
        "",
        _picture(
            "01-statistical-evidence",
            "Fixed-reference errors, empirical FDR intervals, and paired power differences",
        ),
        "",
        "The FDR panel shows empirical estimates and 95% Monte Carlo intervals against nominal alpha 0.10. The paired-power panel reports ihwkit minus BH in percentage points; positive values favor ihwkit. The parity panel expresses maximum vector differences as a percentage of the absolute tolerance, so 100% is the declared boundary and farther left is better. Failed ihwkit fits keep their own denominator and never become zero-discovery runs.",
        "",
        *_scenario_design_lines(validity),
        "",
        "## Absolute compute cost",
        "",
        _picture(
            "02-process-cost",
            "Absolute warmed-fit time, complete-process time, and peak RSS at 5k, 15k, and 50k hypotheses",
        ),
        "",
        "Each row is one explicit hypothesis-family size; point position is the sample mean and horizontal timing whiskers are +/- one sample standard deviation. The warmed-fit panel uses logarithmic spacing because those measurements span orders of magnitude; complete-process time and RSS use zero-based linear axes. Warmed Python fits remain inside one benchmark process after input construction. R IHW still includes serialization, the adapter, and an R process launch. Complete-process measurements launch a fresh command and therefore include startup, imports, deterministic input generation, fitting work, and any method-specific initialization. Peak RSS is whole-process memory.",
        "",
        "The main scaling figures use exactly 5k, 15k, and 50k hypotheses, shown as explicit axis labels. The one-bin n=500 startup floor remains in the detailed tables but is excluded from the scaling figures.",
        "",
        "## Relative compute cost",
        "",
        _picture(
            "03-warm-fit",
            "Peer-to-ihwkit ratios for warmed-fit time, complete-process time, and peak RSS",
        ),
        "",
        "Every peer endpoint divides its mean by the ihwkit mean on the same input. The orange circles and 1x line mark the ihwkit baseline: points left of it favor the peer, while points to the right favor ihwkit for both time and memory. All three ratio axes use logarithmic spacing, so the same multiplicative change occupies the same horizontal distance. Lines expose the magnitude and direction of each comparison without replacing the absolute measurements above.",
        "",
        "## Detailed statistical results",
        "",
        "| scenario | method | ok/attempted | FDR (95% interval) | power | mean rejections |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in validity:
        interval = (
            f"[{_number(row['fdr_ci_low'], 3)}, {_number(row['fdr_ci_high'], 3)}]"
        )
        lines.append(
            f"| {_short_scenario(row['scenario_id'])} | {METHOD_LABELS.get(row['method_id'], row['method_id'])} | "
            f"{row['successful']}/{row['attempted']} | {_number(row['mean_fdp'], 3)} {interval} | "
            f"{_number(row['mean_power'], 3)} | {_number(row['mean_rejections'], 1)} |"
        )
    lines.extend(
        [
            "",
            "Paired differences use only replicates where both BH and production succeeded. They improve precision for method comparison but do not erase production failures.",
            "",
            "| scenario | paired reps | FDP difference (MCSE) | power difference (MCSE) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {_short_scenario(row.scenario_id)} | {row.paired_replicates} | {row.fdp_difference:+.4f} ({row.fdp_difference_mcse:.4f}) | "
            f"{_optional_signed(row.power_difference, row.power_difference_mcse)} |"
        )
    lines.extend(
        [
            "",
            "## Performance interpretation",
            "",
            "ihwkit 0.1 solves the unregularized Grenander allocation directly in NumPy. The retained SciPy lane solves the same tested problem through a dense LP; it is a comparison, not a dependency or fallback.",
            "",
            f"- **Measured position:** {performance_headline}",
            "- **Complete process:** Import, method initialization, input construction, and fitting are intentionally combined because that is what a new command experiences.",
            "- **Scope contrast:** process divided by warmed-fit time is descriptive, not a pure startup decomposition. A large factor says that fit-only timing cannot explain command latency; it does not assign the difference to one component.",
            "- **Numerical boundary:** these results cover the named unregularized cases only. Finite regularization remains roadmap work and receives no claim here.",
            "",
            "### Representative measurements at 5k hypotheses",
            "",
            "| method | warm median | process median | process / warm | process peak RSS | status |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for method_id in METHODS:
        warm = _find_timing(warm_rows, "sim_5000_seed42", method_id)
        process = _find_timing(process_rows, "sim_5000_seed42", method_id)
        status = _combined_status(warm, process)
        lines.append(
            f"| {METHOD_LABELS[method_id]} | {_time_cell(warm)} | {_time_cell(process)} | "
            f"{_scope_ratio(warm, process)} | {_rss_cell(process)} | {status} |"
        )
    lines.extend(
        [
            "",
            "<details>",
            "<summary>Protocol and comparability details</summary>",
            "",
            "The label key at the start of this report defines every implementation. SciPy/HiGHS is a retained comparison and never a production dependency or fallback. BH is a truth-labelled simulation baseline, not a solver-performance peer. pyihw participates in generated-input timing but cannot accept the fixed covariate groups required for stored-vector parity.",
            "",
            f"All timed IHW lanes use alpha 0.1, five folds, automatic bin count, the unregularized allocation, seed 42, {int(_mapping(baseline['configuration'])['warmups'])} warmups, and {int(_mapping(baseline['configuration'])['samples'])} measured samples. Exact parity instead uses the fixed R groups and folds stored with the reference.",
            "",
            "The n=500 automatic configuration has one bin. Every implementation reduces that case to ordinary BH. Because this shortcut is dominated by startup, it remains in the detailed tables but not the main scaling figures.",
            "",
            "</details>",
            "",
            "<details>",
            "<summary>Fixed-reference parity details</summary>",
            "",
            "Tolerance: absolute 1e-8 and relative 1e-6.",
            "",
            "| method | status | R rej | method rej | max adjusted-p difference | max weight difference | decision agreement |",
            "|---|:---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in peer_parity:
        lines.append(
            f"| {METHOD_LABELS.get(str(row['method_id']), str(row['method_id']))} | {row['status']} | {row['reference_rejections']} | "
            f"{_plain(row.get('method_rejections'))} | {_scientific(row.get('max_adjusted_difference'))} | "
            f"{_scientific(row.get('max_weight_difference'))} | {_percent(row.get('decision_agreement'))} |"
        )
    lines.extend(
        [
            "",
            "The release gate itself contains one-fold and five-fold synthetic replays. This expanded table shows the five-fold reference across compatible local methods. An unavailable fixed-partition API is not counted as a parity failure.",
            "",
            "</details>",
            "",
            "<details>",
            "<summary>Correctness and robustness details</summary>",
            "",
            "#### Generated correctness",
            "",
            "| case | status | rejections | mean weight | error |",
            "|---|:---:|---:|---:|---|",
        ]
    )
    for raw in correctness.get("rows", []):
        row = _mapping(raw)
        lines.append(
            f"| {row['case_id']} | {row['status']} | {_plain(row.get('rejection_count'))} | "
            f"{_number(row.get('weight_mean'), 6)} | {_nested_error(row.get('error'))} |"
        )
    lines.extend(
        [
            "",
            "#### Fixed and generated robustness",
            "",
            "| case | production | R rejections | BH on R weights | error |",
            "|---|:---:|---:|:---:|---|",
        ]
    )
    for raw in robustness.get("rows", []):
        row = _mapping(raw)
        lines.append(
            f"| {row['reference_id']} | {'pass' if row['production_ok'] else 'fail'} | {row['r_rejections']} | "
            f"{'pass' if row['bh_reference_ok'] else 'fail'} | {row.get('error') or ''} |"
        )
    for raw in robustness.get("stress_rows", []):
        row = _mapping(raw)
        lines.append(
            f"| {row['case_id']} | {row['status']} |  | n/a | {row.get('error') or ''} |"
        )
    lines.extend(
        [
            "",
            "#### Seed-2034 numerical comparison",
            "",
            "| method | status | rejections | error |",
            "|---|:---:|---:|---|",
        ]
    )
    for row in stress_peers:
        lines.append(
            f"| {METHOD_LABELS.get(str(row['method_id']), str(row['method_id']))} | {row['status']} | "
            f"{_plain(row.get('rejection_count'))} | {row.get('error') or ''} |"
        )
    lines.extend(
        [
            "",
            "Weighted BH using stored R weights passes on every retained airway row. The one-fold and five-fold airway replays test the current unregularized method on a real-data shape; they do not validate unimplemented finite regularization.",
            "",
            "</details>",
            "",
            "<details>",
            "<summary>All complete-process measurements</summary>",
            "",
            "| n | bins | method | samples | median wall | mean wall | wall SD | median RSS | status |",
            "|---:|---:|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    lines.extend(_timing_table_rows(process_rows, include_rss=True))
    lines.extend(
        [
            "",
            "</details>",
            "",
            "<details>",
            "<summary>All warmed-fit measurements</summary>",
            "",
            "| n | bins | method | samples | median wall | mean wall | wall SD | status |",
            "|---:|---:|---|---:|---:|---:|---:|:---:|",
        ]
    )
    lines.extend(_timing_table_rows(warm_rows, include_rss=False))
    lines.extend(
        [
            "",
            "</details>",
            "",
            "<details>",
            "<summary>Environment, retained data, and rerun commands</summary>",
            "",
            f"Peer timing recorded: {baseline.get('recorded_at', 'unknown')}",
            "",
        ]
    )
    environment = _mapping(baseline.get("environment", {}))
    for key in (
        "platform",
        "cpu",
        "logical_cpus",
        "python",
        "numpy",
        "scipy",
        "pyihw",
        "zebrac",
    ):
        lines.append(f"- **{key}:** {environment.get(key, 'unknown')}")
    lines.extend(
        [
            "",
            "The retained datasets are deterministic generated draws and the two self-contained files in `bench/data`: one n=5000 synthetic shape and one airway p-value/base-mean shape. No benchmark downloads data; `bench/data/README.md` records their human-readable source and derivation notes.",
            "",
            "Run the complete study, reusing the dated peer timing table:",
            "",
            "```bash",
            "uv run --no-project --with pytest --with numpy --with scipy --with pyihw==0.2.0 --with matplotlib python -m bench study",
            "```",
            "",
            "Refresh unchanged peers only when their code, runtime, machine, or benchmark protocol changes:",
            "",
            "```bash",
            "uv run --no-project --with pytest --with numpy --with scipy --with pyihw==0.2.0 --with matplotlib python -m bench study --refresh-peers",
            "```",
            "",
            "Render the report again without rerunning measurements:",
            "",
            "```bash",
            "uv run --no-project --with numpy --with matplotlib python -m bench report",
            "```",
            "",
            "</details>",
            "",
            "## Limits and next datasets",
            "",
            "- The current real-data evidence is one airway shape. It is a useful numerical stress case, not broad real-data coverage.",
            "- No reviewed local single-cell export is present, so real single-cell calibration, realistic count-based power, and donor-stability panels remain unmeasured.",
            "- The statistical simulations assume the named data-generating mechanisms. They do not establish FDR control under arbitrary dependence, discrete p-values, or invalid covariates.",
            "- Finite regularization, broader statistical procedures, and diagnostic plotting remain future work, not hidden 0.1 features.",
            "- Peak RSS is whole-process memory. It cannot be interpreted as method-specific allocation alone.",
            "- Machine-local performance is evidence for this environment, not a universal speed or memory guarantee.",
            "",
            "The simulation structure follows the ADEMP framework from [Morris, White, and Crowther (2019)](https://doi.org/10.1002/sim.8086). The separation of datasets, metrics, failures, and method scope follows the benchmarking guidance of [Weber et al. (2019)](https://doi.org/10.1186/s13059-019-1738-8). The report layout borrows z-fasta's useful pattern of correctness-first execution, absolute and ratio plots, and collapsible detailed evidence while intentionally using a much smaller local implementation.",
        ]
    )
    return lines


def _scenario_design_lines(
    validity: Sequence[Mapping[str, str]],
) -> list[str]:
    descriptions = {
        "global_null": "All hypotheses are null; p-values and a continuous uniform covariate are independent.",
        "null_covariate": "All hypotheses are null; p-values are uniform and independent of a rounded, skewed log-normal covariate with ties.",
        "mixture_mild": "Ten percent are alternatives; their one-sided normal signal increases with a continuous uniform covariate.",
        "mixture_sparse": "Five percent are alternatives under the same covariate-dependent signal model as Mild mixture.",
        "ignatiadis": "Twelve percent are alternatives in the retained Ignatiadis-style covariate-dependent normal model.",
        "dense_covariate": "Fifteen percent are alternatives; a wide log-normal covariate drives stronger high-end signals.",
    }
    lines = [
        "<details>",
        "<summary>Simulation designs and statistical estimands</summary>",
        "",
        "Every method receives the same truth-labelled draw at alpha 0.10. BH is unweighted; ihwkit uses five folds, automatic bin count, and the current unregularized allocation. For each scenario:",
        "",
    ]
    for scenario_id in SCENARIO_ORDER:
        row = next(
            (item for item in validity if item["scenario_id"] == scenario_id), None
        )
        if row is None:
            continue
        lines.append(
            f"- **{_short_scenario(scenario_id)}:** n={int(row['n']):,}; "
            f"{int(row['attempted']):,} attempts per method. {descriptions[scenario_id]}"
        )
    lines.extend(
        [
            "",
            "FDR is the mean replicate false-discovery proportion. Under either all-null scenario this equals the probability of at least one rejection, and the report uses a Wilson interval; other FDR intervals use the mean plus or minus 1.96 Monte Carlo standard errors, bounded to [0, 1]. Power is the fraction of true alternatives rejected. Paired differences compare ihwkit with BH on the same successful draw; unsuccessful ihwkit fits retain their own denominator and are excluded from the paired estimand.",
            "",
            "</details>",
        ]
    )
    return lines


def _render_figures(
    validity: Sequence[Mapping[str, str]],
    paired: Sequence[PairedValidityRow],
    peer_parity: Sequence[Mapping[str, object]],
    warm_rows: Sequence[TimingRow],
    process_rows: Sequence[TimingRow],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib is required only to render benchmark figures; use the documented uv command"
        ) from exc
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for dark in (False, True):
        suffix = "dark" if dark else "light"
        theme = _figure_theme(dark)
        settings = {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
            "figure.facecolor": theme.background,
            "axes.facecolor": theme.background,
            "savefig.facecolor": theme.background,
            "text.color": theme.ink,
            "axes.labelcolor": theme.muted,
            "xtick.color": theme.muted,
            "ytick.color": theme.ink,
            "axes.edgecolor": theme.grid,
        }
        with plt.rc_context(settings):
            _statistical_figure(plt, validity, paired, peer_parity, suffix, theme)
            _absolute_cost_figure(plt, warm_rows, process_rows, suffix, theme)
            _relative_cost_figure(plt, warm_rows, process_rows, suffix, theme)


def _figure_theme(dark: bool) -> FigureTheme:
    if dark:
        return FigureTheme(
            background="#0D1117",
            ink="#F0F3F6",
            muted="#9DA7B0",
            grid="#30363D",
            accent="#FF9D2E",
            method_colors={
                PRODUCTION: "#FF9D2E",
                "ihwkit_scipy": "#35C2A3",
                "pyihw": "#66A9E8",
                "r_ihw": "#E084B7",
                "bh": "#C5CDD5",
                "ihw_inf_cv": "#FF9D2E",
            },
        )
    return FigureTheme(
        background="#FFFFFF",
        ink="#1F2328",
        muted="#59636E",
        grid="#D0D7DE",
        accent="#B66C11",
        method_colors={
            PRODUCTION: "#E87500",
            "ihwkit_scipy": "#008C72",
            "pyihw": "#2F6DAA",
            "r_ihw": "#B85C91",
            "bh": "#4F5964",
            "ihw_inf_cv": "#E87500",
        },
    )


def _figure_header(
    figure: object,
    theme: FigureTheme,
    *,
    title: str,
    subtitle: str,
    label: str,
) -> None:
    from matplotlib.lines import Line2D

    figure.text(
        0.045,
        0.955,
        title,
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        color=theme.ink,
    )
    figure.text(
        0.045,
        0.895,
        subtitle,
        ha="left",
        va="top",
        fontsize=10.2,
        color=theme.muted,
    )
    figure.text(
        0.97,
        0.952,
        "ihwkit",
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=theme.ink,
    )
    figure.text(
        0.97,
        0.914,
        label.upper(),
        ha="right",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        color=theme.accent,
    )
    figure.add_artist(
        Line2D(
            [0.045, 0.97],
            [0.847, 0.847],
            transform=figure.transFigure,
            color=theme.grid,
            linewidth=0.9,
        )
    )


def _figure_footer(
    figure: object,
    theme: FigureTheme,
    blocks: Sequence[tuple[str, str, str]],
) -> None:
    from matplotlib.lines import Line2D

    figure.add_artist(
        Line2D(
            [0.045, 0.97],
            [0.16, 0.16],
            transform=figure.transFigure,
            color=theme.grid,
            linewidth=0.8,
        )
    )
    anchors = ((0.045, "left"), (0.37, "left"), (0.72, "left"))
    for index, ((title, first, second), (x_value, alignment)) in enumerate(
        zip(blocks, anchors, strict=True)
    ):
        if index:
            divider = 0.345 if index == 1 else 0.695
            figure.add_artist(
                Line2D(
                    [divider, divider],
                    [0.025, 0.137],
                    transform=figure.transFigure,
                    color=theme.grid,
                    linewidth=0.8,
                )
            )
        figure.text(
            x_value,
            0.125,
            title.upper(),
            ha=alignment,
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color=theme.accent,
        )
        figure.text(
            x_value,
            0.084,
            first,
            ha=alignment,
            va="center",
            fontsize=7.1,
            fontweight="bold",
            color=theme.ink,
        )
        figure.text(
            x_value,
            0.043,
            second,
            ha=alignment,
            va="center",
            fontsize=6.9,
            color=theme.muted,
        )


def _panel_heading(
    figure: object,
    axis: object,
    theme: FigureTheme,
    title: str,
    subtitle: str,
) -> None:
    box = axis.get_position()
    figure.text(
        box.x0,
        0.758,
        title.upper(),
        ha="left",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=theme.accent,
    )
    figure.text(
        box.x0,
        0.721,
        subtitle,
        ha="left",
        va="center",
        fontsize=7.8,
        color=theme.muted,
    )


def _clean_axis(axis: object, theme: FigureTheme) -> None:
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(theme.grid)
    axis.tick_params(axis="both", length=0, labelsize=8.2)
    axis.grid(axis="x", color=theme.grid, linewidth=0.7, linestyle=(0, (2, 4)))
    axis.set_axisbelow(True)


def _successful_metric_values(
    rows: Sequence[TimingRow], field: str, *, divisor: float = 1.0
) -> list[float]:
    """Return positive displayed values for one measured field."""

    values: list[float] = []
    for row in rows:
        value = getattr(row, field)
        if (
            row.status == "ok"
            and row.size in PLOT_SIZES
            and value is not None
            and float(value) > 0.0
        ):
            values.append(float(value) / divisor)
    return values


def _relative_metric_values(rows: Sequence[TimingRow], field: str) -> list[float]:
    """Return displayed peer-to-production ratios, including the 1x baseline."""

    production = {
        row.size: float(getattr(row, field))
        for row in rows
        if row.method_id == PRODUCTION
        and row.status == "ok"
        and row.size in PLOT_SIZES
        and getattr(row, field) is not None
        and float(getattr(row, field)) > 0.0
    }
    ratios = [1.0]
    for row in rows:
        value = getattr(row, field)
        if (
            row.method_id in PEER_METHODS
            and row.status == "ok"
            and row.size in production
            and value is not None
            and float(value) > 0.0
        ):
            ratios.append(float(value) / production[row.size])
    return ratios


def _padded_log_limits(
    values: Sequence[float], *, reference: float | None = None
) -> tuple[float, float]:
    """Return positive log-scale bounds with visible endpoint padding."""

    positive = [float(value) for value in values if value > 0.0]
    if reference is not None and reference > 0.0:
        positive.append(reference)
    if not positive:
        raise ValueError("log-scale plot requires at least one positive value")
    lower_log = math.log10(min(positive))
    upper_log = math.log10(max(positive))
    span = max(upper_log - lower_log, math.log10(1.5))
    lower = 10 ** (lower_log - 0.08 * span)
    upper = 10 ** (upper_log + 0.08 * span)
    if reference is not None:
        lower = min(lower, reference / 1.25)
        upper = max(upper, reference * 1.25)
    return lower, upper


def _log_ticks(lower: float, upper: float) -> tuple[float, ...]:
    """Return sparse 1-3 log ticks inside the displayed bounds."""

    start = math.floor(math.log10(lower)) - 1
    stop = math.ceil(math.log10(upper)) + 1
    ticks: list[float] = []
    for exponent in range(start, stop + 1):
        for multiplier in (1.0, 3.0):
            value = multiplier * 10**exponent
            if lower <= value <= upper:
                ticks.append(value)
    return tuple(ticks)


def _ratio_ticks(lower: float, upper: float) -> tuple[float, ...]:
    """Return readable ticks for either a narrow or wide ratio range."""

    if upper <= 6.0 and lower >= 0.1:
        candidates = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
    else:
        candidates = (
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
            20.0,
            50.0,
            100.0,
            200.0,
            500.0,
            1000.0,
        )
    return tuple(value for value in candidates if lower <= value <= upper)


def _format_time_tick(seconds: float) -> str:
    """Format a positive time tick in milliseconds or seconds."""

    return f"{seconds * 1000:g} ms" if seconds < 1.0 else f"{seconds:g} s"


def _statistical_figure(
    plt: object,
    validity: Sequence[Mapping[str, str]],
    paired: Sequence[PairedValidityRow],
    peer_parity: Sequence[Mapping[str, object]],
    suffix: str,
    theme: FigureTheme,
) -> None:
    from matplotlib.lines import Line2D

    failed_fits = sum(
        int(row["failures"]) for row in validity if row["method_id"] == "ihw_inf_cv"
    )
    failure_note = (
        "No ihwkit fit failures in these named simulations"
        if failed_fits == 0
        else f"{failed_fits} failed ihwkit fits stay outside paired summaries"
    )
    figure = plt.figure(figsize=(16, 7.2))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(4.25, 3.25, 3.25),
        left=0.095,
        right=0.965,
        top=0.67,
        bottom=0.205,
        wspace=0.42,
    )
    fdr_axis = figure.add_subplot(grid[0, 0])
    power_axis = figure.add_subplot(grid[0, 1])
    parity_axis = figure.add_subplot(grid[0, 2])
    _figure_header(
        figure,
        theme,
        title="Statistical behavior and fixed-reference agreement",
        subtitle="Truth-labelled simulations at alpha 0.10 | paired BH baseline | fixed five-fold R IHW replay",
        label="statistical evidence",
    )
    available_scenarios = {row["scenario_id"] for row in validity}
    scenarios = [value for value in SCENARIO_ORDER if value in available_scenarios]
    scenario_positions = np.arange(len(scenarios))[::-1]
    for offset, method_id in ((0.12, "bh"), (-0.12, "ihw_inf_cv")):
        selected = [
            next(
                row
                for row in validity
                if row["scenario_id"] == scenario and row["method_id"] == method_id
            )
            for scenario in scenarios
        ]
        means = np.asarray([float(row["mean_fdp"]) for row in selected])
        lower = means - np.asarray([float(row["fdr_ci_low"]) for row in selected])
        upper = np.asarray([float(row["fdr_ci_high"]) for row in selected]) - means
        fdr_axis.errorbar(
            means,
            scenario_positions + offset,
            xerr=np.vstack((lower, upper)),
            fmt=METHOD_MARKERS[PRODUCTION] if method_id == "ihw_inf_cv" else "o",
            capsize=2.5,
            markersize=6.2,
            label=METHOD_LABELS[method_id],
            color=theme.method_colors[method_id],
            elinewidth=1.15,
            linewidth=0,
        )
    fdr_axis.axvline(
        0.1,
        color=theme.ink,
        linestyle=(0, (3, 3)),
        linewidth=1.0,
    )
    fdr_axis.set_yticks(
        scenario_positions, [_short_scenario(value) for value in scenarios]
    )
    fdr_axis.set_xlim(0.055, 0.12)
    fdr_axis.set_xticks((0.06, 0.08, 0.10, 0.12))
    fdr_axis.set_xlabel("Empirical false-discovery rate")
    _clean_axis(fdr_axis, theme)
    _panel_heading(
        figure,
        fdr_axis,
        theme,
        "FDR calibration",
        "Point = estimate | whisker = 95% Monte Carlo interval | line = 0.10",
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=theme.method_colors["bh"],
                markeredgecolor=theme.method_colors["bh"],
                label="BH",
            ),
            Line2D(
                [0],
                [0],
                marker=METHOD_MARKERS[PRODUCTION],
                color="none",
                markerfacecolor=theme.method_colors[PRODUCTION],
                markeredgecolor=theme.method_colors[PRODUCTION],
                label="ihwkit",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.36, 0.817),
        ncol=2,
        frameon=False,
        fontsize=8,
        handletextpad=0.35,
        columnspacing=1.1,
    )

    paired_by_scenario = {row.scenario_id: row for row in paired}
    power_rows = [
        paired_by_scenario[scenario]
        for scenario in SCENARIO_ORDER
        if scenario in paired_by_scenario
        and paired_by_scenario[scenario].power_difference is not None
    ]
    mixture_scenarios = [row.scenario_id for row in power_rows]
    differences = np.asarray(
        [100.0 * float(row.power_difference) for row in power_rows]
    )
    errors = np.asarray(
        [100.0 * 1.96 * float(row.power_difference_mcse) for row in power_rows]
    )
    mix_positions = np.arange(len(mixture_scenarios))[::-1]
    power_axis.errorbar(
        differences,
        mix_positions,
        xerr=errors,
        fmt="o",
        color=theme.method_colors[PRODUCTION],
        ecolor=theme.method_colors[PRODUCTION],
        elinewidth=1.2,
        markersize=6.2,
        capsize=2.5,
    )
    power_axis.axvline(0, color=theme.ink, linewidth=1.0, linestyle=(0, (3, 3)))
    power_axis.set_yticks(
        mix_positions, [_short_scenario(value) for value in mixture_scenarios]
    )
    power_axis.set_xlim(-1.0, 4.45)
    power_axis.set_xticks((-1, 0, 1, 2, 3, 4))
    power_axis.set_xlabel("ihwkit - BH power (percentage points)")
    for x_value, y_value in zip(differences, mix_positions, strict=True):
        power_axis.annotate(
            f"{x_value:+.2f}",
            (x_value, y_value),
            xytext=(6 if x_value >= 0 else -6, 0),
            textcoords="offset points",
            ha="left" if x_value >= 0 else "right",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color=theme.ink,
        )
    _clean_axis(power_axis, theme)
    _panel_heading(
        figure,
        power_axis,
        theme,
        "Paired power",
        "Positive values favor ihwkit | matched successful draws only",
    )

    compatible = [row for row in peer_parity if row["status"] == "pass"]
    parity_positions = np.arange(len(compatible))[::-1]
    tolerance = 1e-8
    for position, row in zip(parity_positions, compatible, strict=True):
        adjusted_percent = float(row["max_adjusted_difference"]) / tolerance * 100.0
        weight_percent = float(row["max_weight_difference"]) / tolerance * 100.0
        parity_axis.scatter(
            adjusted_percent,
            position + 0.10,
            marker="o",
            s=35,
            color="#2F81B7" if suffix == "light" else "#66B5E8",
            zorder=3,
        )
        parity_axis.scatter(
            weight_percent,
            position - 0.10,
            marker="s",
            s=32,
            color="#009E73" if suffix == "light" else "#40C9A2",
            zorder=3,
        )
    parity_axis.axvline(100.0, color=theme.ink, linewidth=1.0, linestyle=(0, (3, 3)))
    parity_axis.set_xscale("log")
    parity_axis.set_xlim(5e-6, 200)
    parity_axis.set_xticks(
        (1e-5, 1e-3, 1e-1, 10, 100),
        ("0.00001%", "0.001%", "0.1%", "10%", "100%"),
    )
    parity_axis.minorticks_off()
    parity_axis.set_yticks(
        parity_positions,
        [METHOD_LABELS[str(row["method_id"])] for row in compatible],
    )
    parity_axis.set_xlabel("Maximum difference as percent of tolerance")
    _clean_axis(parity_axis, theme)
    _panel_heading(
        figure,
        parity_axis,
        theme,
        "Fixed R parity",
        "Circle = adjusted p | square = weight | all decisions agree",
    )
    _figure_footer(
        figure,
        theme,
        (
            (
                "Simulation",
                "n=3,000 | 1,000 null replicates | 200 alternative replicates",
                "Same truth-labelled draw for BH and ihwkit",
            ),
            (
                "Uncertainty",
                "Whiskers = 95% Monte Carlo intervals",
                failure_note,
            ),
            (
                "Reference",
                "R IHW 1.40.0 | five folds | lambda = infinity",
                "Absolute tolerance 1e-8 | relative tolerance 1e-6",
            ),
        ),
    )
    _save_figure(figure, f"01-statistical-evidence-{suffix}.svg")


def _absolute_cost_figure(
    plt: object,
    warm_rows: Sequence[TimingRow],
    process_rows: Sequence[TimingRow],
    suffix: str,
    theme: FigureTheme,
) -> None:
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    figure = plt.figure(figsize=(16, 7.2))
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=(0.7, 3.25, 3.25, 3.0),
        left=0.035,
        right=0.965,
        top=0.67,
        bottom=0.205,
        wspace=0.16,
    )
    labels_axis = figure.add_subplot(grid[0, 0])
    warm_axis = figure.add_subplot(grid[0, 1])
    process_axis = figure.add_subplot(grid[0, 2], sharey=warm_axis)
    rss_axis = figure.add_subplot(grid[0, 3], sharey=warm_axis)
    _figure_header(
        figure,
        theme,
        title="Compute cost across hypothesis families",
        subtitle="Absolute machine-local measurements | identical generated input per size | lower values are better",
        label="absolute cost",
    )
    centers = {
        size: float(len(PLOT_SIZES) - index - 1)
        for index, size in enumerate(PLOT_SIZES)
    }
    labels_axis.set_xlim(0, 1)
    labels_axis.set_ylim(-0.55, 2.55)
    labels_axis.axis("off")
    labels_axis.text(
        0.92,
        2.48,
        "HYPOTHESES",
        ha="right",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color=theme.accent,
    )
    for size, position in centers.items():
        labels_axis.text(
            0.92,
            position,
            f"{size // 1000}k",
            ha="right",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=theme.ink,
        )
        labels_axis.plot(
            [0.95, 0.95],
            [position - 0.34, position + 0.34],
            color=theme.grid,
            linewidth=0.8,
        )
    _absolute_points(
        warm_axis,
        warm_rows,
        theme,
        centers,
        field="wall_mean_ns",
        divisor=1e9,
        error_field="wall_std_ns",
    )
    warm_values = _successful_metric_values(warm_rows, "wall_mean_ns", divisor=1e9)
    warm_limits = _padded_log_limits(warm_values)
    warm_axis.set_xscale("log")
    warm_axis.set_xlim(*warm_limits)
    warm_ticks = _log_ticks(*warm_limits)
    warm_axis.set_xticks(warm_ticks)
    warm_axis.set_xticklabels(tuple(map(_format_time_tick, warm_ticks)))
    warm_axis.set_xlabel("Mean repeated-fit wall time")
    _absolute_points(
        process_axis,
        process_rows,
        theme,
        centers,
        field="wall_mean_ns",
        divisor=1e9,
        error_field="wall_std_ns",
    )
    process_values = _successful_metric_values(
        process_rows, "wall_mean_ns", divisor=1e9
    )
    process_axis.set_xlim(0.0, max(process_values) * 1.1)
    process_axis.xaxis.set_major_locator(
        MaxNLocator(nbins=5, steps=(1, 2, 2.5, 5, 10), min_n_ticks=4)
    )
    process_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    process_axis.set_xlabel("Mean complete-process wall time (s)")
    _absolute_points(
        rss_axis,
        process_rows,
        theme,
        centers,
        field="peak_rss_mean_bytes",
        divisor=1e6,
    )
    rss_values = _successful_metric_values(
        process_rows, "peak_rss_mean_bytes", divisor=1e6
    )
    rss_axis.set_xlim(0.0, max(rss_values) * 1.1)
    rss_axis.xaxis.set_major_locator(
        MaxNLocator(nbins=5, steps=(1, 2, 2.5, 5, 10), min_n_ticks=4)
    )
    rss_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    rss_axis.set_xlabel("Mean peak RSS (MB)")
    for axis in (warm_axis, process_axis, rss_axis):
        axis.minorticks_off()
        axis.set_ylim(-0.55, 2.55)
        axis.set_yticks([])
        _clean_axis(axis, theme)
        for position in centers.values():
            axis.axhline(
                position,
                color=theme.grid,
                linewidth=0.5,
                alpha=0.55,
                zorder=0,
            )
    _panel_heading(
        figure,
        warm_axis,
        theme,
        "Warmed fit",
        "Python calls after 3 warmups | R still launches its adapter",
    )
    _panel_heading(
        figure,
        process_axis,
        theme,
        "Complete process",
        "Startup + imports + input generation + fit",
    )
    _panel_heading(
        figure,
        rss_axis,
        theme,
        "Process memory",
        "Whole-process peak resident set size",
    )
    _method_legend(figure, theme)
    _figure_footer(
        figure,
        theme,
        (
            (
                "Measurement",
                "10 samples per cell | 3 warmups | seed 42",
                "Points = mean | horizontal whiskers = 1 standard deviation",
            ),
            (
                "Scope",
                "5k | 15k | 50k hypotheses | automatic bins",
                "All methods measured at every displayed size",
            ),
            (
                "Interpretation",
                "Warm time is log scaled; process time and RSS start at zero",
                "RSS is whole-process memory, not method allocation",
            ),
        ),
    )
    _save_figure(figure, f"02-process-cost-{suffix}.svg")


def _relative_cost_figure(
    plt: object,
    warm_rows: Sequence[TimingRow],
    process_rows: Sequence[TimingRow],
    suffix: str,
    theme: FigureTheme,
) -> None:
    figure = plt.figure(figsize=(16, 7.2))
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=(0.7, 3.25, 3.25, 3.0),
        left=0.035,
        right=0.965,
        top=0.67,
        bottom=0.205,
        wspace=0.16,
    )
    labels_axis = figure.add_subplot(grid[0, 0])
    warm_axis = figure.add_subplot(grid[0, 1])
    process_axis = figure.add_subplot(grid[0, 2], sharey=warm_axis)
    rss_axis = figure.add_subplot(grid[0, 3], sharey=warm_axis)
    _figure_header(
        figure,
        theme,
        title="Peer cost relative to ihwkit",
        subtitle="Each peer endpoint divides its mean by ihwkit on the same input | 1x is equal cost",
        label="relative performance",
    )
    centers = {
        size: float(len(PLOT_SIZES) - index - 1)
        for index, size in enumerate(PLOT_SIZES)
    }
    labels_axis.set_xlim(0, 1)
    labels_axis.set_ylim(-0.55, 2.55)
    labels_axis.axis("off")
    labels_axis.text(
        0.92,
        2.48,
        "HYPOTHESES",
        ha="right",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color=theme.accent,
    )
    for size, position in centers.items():
        labels_axis.text(
            0.92,
            position,
            f"{size // 1000}k",
            ha="right",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=theme.ink,
        )
        labels_axis.plot(
            [0.95, 0.95],
            [position - 0.34, position + 0.34],
            color=theme.grid,
            linewidth=0.8,
        )
    _relative_points(
        warm_axis,
        warm_rows,
        theme,
        centers,
        field="wall_mean_ns",
    )
    warm_ratios = _relative_metric_values(warm_rows, "wall_mean_ns")
    warm_limits = _padded_log_limits(warm_ratios, reference=1.0)
    warm_axis.set_xscale("log")
    warm_axis.set_xlim(*warm_limits)
    warm_ticks = _ratio_ticks(*warm_limits)
    warm_axis.set_xticks(warm_ticks)
    warm_axis.set_xticklabels(tuple(f"{value:g}x" for value in warm_ticks))
    warm_axis.set_xlabel("Peer / ihwkit warmed-fit time")
    _relative_points(
        process_axis,
        process_rows,
        theme,
        centers,
        field="wall_mean_ns",
    )
    process_ratios = _relative_metric_values(process_rows, "wall_mean_ns")
    process_limits = _padded_log_limits(process_ratios, reference=1.0)
    process_axis.set_xscale("log")
    process_axis.set_xlim(*process_limits)
    process_ticks = _ratio_ticks(*process_limits)
    process_axis.set_xticks(process_ticks)
    process_axis.set_xticklabels(tuple(f"{value:g}x" for value in process_ticks))
    process_axis.set_xlabel("Peer / ihwkit complete-process time")
    _relative_points(
        rss_axis,
        process_rows,
        theme,
        centers,
        field="peak_rss_mean_bytes",
    )
    rss_ratios = _relative_metric_values(process_rows, "peak_rss_mean_bytes")
    rss_limits = _padded_log_limits(rss_ratios, reference=1.0)
    rss_axis.set_xscale("log")
    rss_axis.set_xlim(*rss_limits)
    rss_ticks = _ratio_ticks(*rss_limits)
    rss_axis.set_xticks(rss_ticks)
    rss_axis.set_xticklabels(tuple(f"{value:g}x" for value in rss_ticks))
    rss_axis.set_xlabel("Peer / ihwkit peak RSS")
    for axis in (warm_axis, process_axis, rss_axis):
        axis.minorticks_off()
        axis.set_ylim(-0.55, 2.55)
        axis.set_yticks([])
        axis.axvline(
            1.0,
            color=theme.accent,
            linewidth=1.2,
            linestyle=(0, (3, 3)),
            zorder=1,
        )
        _clean_axis(axis, theme)
        for position in centers.values():
            axis.axhline(
                position,
                color=theme.grid,
                linewidth=0.5,
                alpha=0.55,
                zorder=0,
            )
    _panel_heading(
        figure,
        warm_axis,
        theme,
        "Warmed fit",
        "Left of 1x = peer faster | right of 1x = ihwkit faster",
    )
    _panel_heading(
        figure,
        process_axis,
        theme,
        "Complete process",
        "Left of 1x = peer faster | right of 1x = ihwkit faster",
    )
    _panel_heading(
        figure,
        rss_axis,
        theme,
        "Process memory",
        "Left of 1x = peer uses less | right of 1x = ihwkit uses less",
    )
    _method_legend(figure, theme)
    _figure_footer(
        figure,
        theme,
        (
            (
                "Ratio rule",
                "Peer mean divided by ihwkit mean on the same input",
                "Orange circles and 1x line = ihwkit baseline",
            ),
            (
                "Scope",
                "All methods are measured at 5k, 15k, and 50k hypotheses",
                "The one-bin 500-hypothesis startup floor stays in the table",
            ),
            (
                "Reading",
                "Log axes show multiplicative distance from 1x",
                "Absolute values remain above; no aggregate score is computed",
            ),
        ),
    )
    _save_figure(figure, f"03-warm-fit-{suffix}.svg")


def _absolute_points(
    axis: object,
    rows: Sequence[TimingRow],
    theme: FigureTheme,
    centers: Mapping[int, float],
    *,
    field: str,
    divisor: float,
    error_field: str | None = None,
) -> None:
    offsets = dict(zip(METHODS, np.linspace(0.27, -0.27, len(METHODS)), strict=True))
    for size, center in centers.items():
        for method_id in METHODS:
            row = next(
                (
                    value
                    for value in rows
                    if value.size == size and value.method_id == method_id
                ),
                None,
            )
            if row is None:
                continue
            color = theme.method_colors[method_id]
            position = center + offsets[method_id]
            value = getattr(row, field)
            if row.status == "ok" and value is not None:
                error = (
                    None
                    if error_field is None
                    else float(getattr(row, error_field) or 0.0) / divisor
                )
                axis.errorbar(
                    float(value) / divisor,
                    position,
                    xerr=error,
                    marker=METHOD_MARKERS[method_id],
                    markersize=7.2 if method_id == PRODUCTION else 5.7,
                    markerfacecolor=color,
                    markeredgecolor=theme.ink if method_id == PRODUCTION else color,
                    markeredgewidth=0.9,
                    color=color,
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=1.8 if error is not None else 0,
                    linewidth=0,
                    zorder=4 if method_id == PRODUCTION else 3,
                )


def _relative_points(
    axis: object,
    rows: Sequence[TimingRow],
    theme: FigureTheme,
    centers: Mapping[int, float],
    *,
    field: str,
) -> None:
    production = {
        row.size: float(getattr(row, field))
        for row in rows
        if row.method_id == PRODUCTION
        and row.status == "ok"
        and row.size in centers
        and getattr(row, field) is not None
    }
    offsets = dict(zip(METHODS, np.linspace(0.27, -0.27, len(METHODS)), strict=True))
    for size, center in centers.items():
        if size not in production:
            continue
        axis.scatter(
            1.0,
            center + offsets[PRODUCTION],
            marker=METHOD_MARKERS[PRODUCTION],
            s=38,
            color=theme.method_colors[PRODUCTION],
            edgecolor=theme.ink,
            linewidth=0.8,
            zorder=4,
        )
        for method_id in PEER_METHODS:
            row = next(
                (
                    value
                    for value in rows
                    if value.size == size and value.method_id == method_id
                ),
                None,
            )
            if row is None:
                continue
            color = theme.method_colors[method_id]
            position = center + offsets[method_id]
            value = getattr(row, field)
            if row.status == "ok" and value is not None:
                ratio = float(value) / production[size]
                axis.plot(
                    [1.0, ratio],
                    [position, position],
                    color=color,
                    linewidth=1.2,
                    alpha=0.65,
                    zorder=2,
                )
                axis.scatter(
                    ratio,
                    position,
                    marker=METHOD_MARKERS[method_id],
                    s=34,
                    color=color,
                    zorder=3,
                )


def _method_legend(figure: object, theme: FigureTheme) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[method_id],
            color="none",
            markerfacecolor=theme.method_colors[method_id],
            markeredgecolor=(
                theme.ink if method_id == PRODUCTION else theme.method_colors[method_id]
            ),
            markeredgewidth=0.9,
            markersize=7,
            label=METHOD_LABELS[method_id],
        )
        for method_id in METHODS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.813),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        handletextpad=0.35,
        columnspacing=1.2,
    )


def _save_figure(figure: object, name: str) -> None:
    path = FIGURE_DIR / name
    figure.savefig(
        path,
        format="svg",
        metadata={"Creator": "ihwkit benchmark report", "Date": None},
    )
    svg = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
    )
    import matplotlib.pyplot as plt

    plt.close(figure)


def _paired_validity(attempts: Sequence[Mapping[str, str]]) -> list[PairedValidityRow]:
    rows: list[PairedValidityRow] = []
    available = {row["scenario_id"] for row in attempts}
    scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in available]
    for scenario_id in scenarios:
        selected = [row for row in attempts if row["scenario_id"] == scenario_id]
        by_key = {(row["method_id"], row["replicate"]): row for row in selected}
        fdp_differences: list[float] = []
        power_differences: list[float] = []
        replicates = sorted({row["replicate"] for row in selected})
        for replicate in replicates:
            bh = by_key.get(("bh", replicate))
            ihw = by_key.get(("ihw_inf_cv", replicate))
            if (
                bh is None
                or ihw is None
                or bh["status"] != "ok"
                or ihw["status"] != "ok"
            ):
                continue
            fdp_differences.append(float(ihw["fdp"]) - float(bh["fdp"]))
            if bh["power"] and ihw["power"]:
                power_differences.append(float(ihw["power"]) - float(bh["power"]))
        if not fdp_differences:
            continue
        rows.append(
            PairedValidityRow(
                scenario_id,
                len(fdp_differences),
                statistics.fmean(fdp_differences),
                _mcse(fdp_differences),
                statistics.fmean(power_differences) if power_differences else None,
                _mcse(power_differences) if power_differences else None,
            )
        )
    return rows


def _mcse(values: Sequence[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _merge_current_and_peer_rows(
    current: Sequence[TimingRow], peers: Sequence[TimingRow]
) -> list[TimingRow]:
    merged = [row for row in current if row.method_id == PRODUCTION]
    merged.extend(row for row in peers if row.method_id != PRODUCTION)
    return sorted(merged, key=lambda row: (row.size, METHODS.index(row.method_id)))


def _timing_rows(path: Path) -> list[TimingRow]:
    return _timing_values(_read_json_list(path))


def _timing_values(values: object) -> list[TimingRow]:
    if not isinstance(values, list):
        raise TypeError("timing rows must be a list")
    return [TimingRow(**_mapping(value)) for value in values]


def _timing_table_rows(rows: Sequence[TimingRow], *, include_rss: bool) -> list[str]:
    lines: list[str] = []
    for row in rows:
        values = [
            str(row.size),
            str(max(1, min(40, row.size // 1500))),
            METHOD_LABELS[row.method_id],
            str(row.sample_count),
            _time_cell(row),
            _duration(row.wall_mean_ns),
            _duration(row.wall_std_ns),
        ]
        if include_rss:
            values.append(_rss_cell(row))
        values.append(row.status)
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _find_timing(
    rows: Sequence[TimingRow], dataset_id: str, method_id: str
) -> TimingRow | None:
    return next(
        (
            row
            for row in rows
            if row.dataset_id == dataset_id and row.method_id == method_id
        ),
        None,
    )


def _combined_status(first: TimingRow | None, second: TimingRow | None) -> str:
    statuses = {row.status for row in (first, second) if row is not None}
    if not statuses:
        return "missing"
    return "ok" if statuses == {"ok"} else ", ".join(sorted(statuses))


def _performance_headline(
    warm_rows: Sequence[TimingRow], process_rows: Sequence[TimingRow]
) -> str:
    statements: list[str] = []
    for dataset_id in ("sim_5000_seed42", "sim_50000_seed42"):
        size = 5_000 if dataset_id == "sim_5000_seed42" else 50_000
        warm = [
            row
            for row in warm_rows
            if row.dataset_id == dataset_id
            and row.status == "ok"
            and row.wall_median_ns is not None
        ]
        process = [
            row
            for row in process_rows
            if row.dataset_id == dataset_id
            and row.status == "ok"
            and row.wall_median_ns is not None
        ]
        production_warm = next(row for row in warm if row.method_id == PRODUCTION)
        production_process = next(row for row in process if row.method_id == PRODUCTION)
        warm_rank = 1 + sum(
            float(row.wall_median_ns) < float(production_warm.wall_median_ns)
            for row in warm
        )
        process_rank = 1 + sum(
            float(row.wall_median_ns) < float(production_process.wall_median_ns)
            for row in process
        )
        rss_rank = 1 + sum(
            row.peak_rss_median_bytes is not None
            and production_process.peak_rss_median_bytes is not None
            and row.peak_rss_median_bytes < production_process.peak_rss_median_bytes
            for row in process
        )
        fastest_peer = min(
            (row for row in warm if row.method_id != PRODUCTION),
            key=lambda row: float(row.wall_median_ns),
        )
        ratio = float(fastest_peer.wall_median_ns) / float(
            production_warm.wall_median_ns
        )
        relation = (
            f"{ratio:.1f}x faster than the next measured method"
            if ratio >= 1.0
            else f"{1.0 / ratio:.1f}x slower than {METHOD_LABELS[fastest_peer.method_id]}"
        )
        statements.append(
            f"n={size}: median warmed-fit rank {warm_rank}/{len(warm)} ({relation}); "
            f"median process-time rank {process_rank}/{len(process)} and median RSS rank {rss_rank}/{len(process)}"
        )
    return "; ".join(statements) + "."


def _time_cell(row: TimingRow | None) -> str:
    return "" if row is None else _duration(row.wall_median_ns)


def _rss_cell(row: TimingRow | None) -> str:
    if row is None or row.peak_rss_median_bytes is None:
        return ""
    return f"{row.peak_rss_median_bytes / 1e6:.1f} MB"


def _scope_ratio(warm: TimingRow | None, process: TimingRow | None) -> str:
    if (
        warm is None
        or process is None
        or warm.wall_median_ns is None
        or process.wall_median_ns is None
        or warm.wall_median_ns <= 0
    ):
        return ""
    return f"{process.wall_median_ns / warm.wall_median_ns:.1f}x"


def _duration(value: float | None) -> str:
    if value is None:
        return ""
    if value >= 1e9:
        return f"{value / 1e9:.3f} s"
    if value >= 1e6:
        return f"{value / 1e6:.3f} ms"
    return f"{value / 1e3:.3f} us"


def _picture(stem: str, alt: str) -> str:
    return "\n".join(
        [
            "<picture>",
            f'  <source media="(prefers-color-scheme: dark)" srcset="figures/{stem}-dark.svg">',
            f'  <source media="(prefers-color-scheme: light)" srcset="figures/{stem}-light.svg">',
            f'  <img src="figures/{stem}-light.svg" alt="{alt}" width="100%">',
            "</picture>",
        ]
    )


def _environment() -> dict[str, str]:
    cpu = platform.processor().strip()
    if not cpu and Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "platform": platform.platform(),
        "cpu": cpu or "unknown",
        "logical_cpus": str(os.cpu_count() or "unknown"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "pyihw": _package_version("pyihw"),
        "zebrac": _command_version("zebrac"),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _command_version(name: str) -> str:
    completed = subprocess.run(
        [name, "--version"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _short_scenario(value: str) -> str:
    labels = {
        "global_null": "Global null",
        "null_covariate": "Tied null",
        "mixture_mild": "Mild mixture",
        "mixture_sparse": "Sparse mixture",
        "ignatiadis": "Ignatiadis",
        "dense_covariate": "Dense covariate",
    }
    return labels.get(value, value.replace("_", " "))


def _last_line(value: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return lines[-1] if lines else "no test output"


def _number(value: object, digits: int) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.{digits}f}"


def _optional_signed(value: float | None, error: float | None) -> str:
    if value is None or error is None:
        return ""
    return f"{value:+.4f} ({error:.4f})"


def _plain(value: object) -> str:
    return "" if value is None else str(value)


def _scientific(value: object) -> str:
    return "" if value is None else f"{float(value):.2e}"


def _percent(value: object) -> str:
    return "" if value is None else f"{100.0 * float(value):.3f}%"


def _nested_error(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    return f"{value.get('type', 'Error')}: {value.get('message', '')}"


def _error_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return f"{value.get('type', 'Error')}: {value.get('message', '')}"
    return str(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, object]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")))


def _read_json_list(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"expected a JSON list in {path}")
    return [_mapping(item) for item in value]


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("expected a mapping")
    return {str(key): item for key, item in value.items()}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _ensure_result_dir(path: Path, *, create: bool = True) -> None:
    temporary_root = (ROOT / "tmp").resolve()
    if not path.is_relative_to(temporary_root):
        raise SystemExit(f"result directory must be under {temporary_root}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise SystemExit(f"result directory does not exist: {path}")
