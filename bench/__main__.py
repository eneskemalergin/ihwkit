"""Run the local correctness, parity, validity, and performance studies."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import math
import platform
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ihw import _p_adjust, adjust_ihw
from tools.peer import (
    REFERENCE_SPECS,
    PeerInput,
    RunConfig,
    generate_r_reference,
    load_reference,
)
from tools.simulators import SCENARIO_BUILDERS, SimDraw

DEFAULT_ALPHA = 0.1
DEFAULT_SEED = 2026
RESULTS_DIR = ROOT / "tmp" / "results"
_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


@dataclass(frozen=True)
class MatrixRow:
    """Describe one evidence track and its appropriate data level."""

    track: str
    data_level: str
    question: str
    primary_measure: str
    command: str
    status: str


@dataclass(frozen=True)
class ValidityCase:
    """Describe one truth-labelled simulation case."""

    scenario_id: str
    n: int
    reps: int
    assumption_class: str


@dataclass(frozen=True)
class AttemptRow:
    """Record one method attempt on one simulated draw."""

    scenario_id: str
    assumption_class: str
    method_id: str
    n: int
    replicate: int
    seed: int
    alpha: float
    status: str
    rejections: int | None
    false_rejections: int | None
    any_rejection: int | None
    fdp: float | None
    power: float | None
    error_type: str | None
    error_message: str | None


@dataclass(frozen=True)
class SummaryRow:
    """Summarize successful attempts without hiding failed attempts."""

    scenario_id: str
    assumption_class: str
    method_id: str
    n: int
    attempted: int
    successful: int
    failures: int
    failure_rate: float
    mean_fdp: float | None
    fdr_mcse: float | None
    fdr_ci_low: float | None
    fdr_ci_high: float | None
    mean_power: float | None
    power_mcse: float | None
    mean_rejections: float | None
    fdp_q95: float | None


MATRIX: tuple[MatrixRow, ...] = (
    MatrixRow(
        "correctness",
        "generated edge cases and synthetic draws",
        "Does the API validate inputs and return numerically valid results?",
        "status, validation, numerical invariants",
        "python -m bench correctness",
        "implemented",
    ),
    MatrixRow(
        "parity",
        "fixed synthetic and airway R reference records",
        "Does a fixed Python configuration agree with R IHW?",
        "full vectors, rejection count, declared tolerance",
        "python -m bench parity",
        "synthetic release gates plus broader diagnostics",
    ),
    MatrixRow(
        "validity",
        "truth-labelled synthetic draws",
        "Does empirical FDR calibrate, and what power is gained over BH?",
        "FDR, power, Monte Carlo error, failure rate",
        "python -m bench validity",
        "development-scale runner and comparative report implemented",
    ),
    MatrixRow(
        "robustness",
        "synthetic stress and airway shapes",
        "Where do assumptions, conditioning, or implementation limits fail?",
        "fit status, parity residuals, and failure surface",
        "python -m bench robustness",
        "implemented diagnostics; failures remain visible",
    ),
    MatrixRow(
        "performance",
        "generated scaling draws and real-sized p-value tables",
        "What are cold process cost, warm algorithm cost, and peak memory?",
        "wall time, peak RSS, failures, problem dimensions",
        "python -m bench performance --dataset ...",
        "cold process and full-study warmed scaling implemented",
    ),
    MatrixRow(
        "single-cell calibration",
        "Kang control-only pseudobulk contrasts",
        "Do real null-like contrasts expose p-value or weighting pathologies?",
        "p-value diagnostics, discoveries, split stability",
        "not runnable until the reviewed local export exists",
        "planned",
    ),
    MatrixRow(
        "single-cell power",
        "muscat simulations based on multi-sample single-cell data",
        "How do calibration and power behave with realistic count structure?",
        "FDR, power, failure rate by cell type and signal pattern",
        "not runnable until the reviewed R export exists",
        "planned",
    ),
    MatrixRow(
        "single-cell case study",
        "paired Kang interferon pseudobulk contrasts",
        "Are weights interpretable and discoveries stable across donors?",
        "weight curves, decision boundaries, leave-one-donor-out stability",
        "not runnable until provenance and analysis are reviewed",
        "planned; no truth claim",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one explicit benchmark track."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        _print_help()
        return 0
    command, remaining = arguments[0], arguments[1:]
    if command == "matrix":
        if remaining:
            raise SystemExit("matrix does not accept arguments")
        _print_matrix()
        return 0
    if command == "validity":
        return _validity_main(remaining)
    if command == "correctness":
        from tools.check_peer_correctness import main as correctness_main

        return correctness_main(remaining)
    if command == "parity":
        from tools.replay_parity import main as parity_main

        return parity_main(remaining)
    if command == "robustness":
        from tools.replay_parity import main as parity_main

        return parity_main(remaining, gates_only=False)
    if command == "performance":
        from tools.benchmark_zebrac import main as performance_main

        return performance_main(remaining)
    if command == "study":
        from bench.report import study_main

        return study_main(remaining)
    if command == "report":
        from bench.report import report_main

        return report_main(remaining)
    if command == "references":
        return _references_main(remaining)
    raise SystemExit(f"unknown benchmark command: {command}")


def _print_help() -> None:
    print(
        "Usage: python -m bench COMMAND [options]\n\n"
        "Commands:\n"
        "  matrix       Show which data answer each benchmark question.\n"
        "  correctness  Run generated production correctness gates.\n"
        "  parity       Replay the synthetic R parity gates.\n"
        "  robustness   Replay every synthetic and airway R diagnostic.\n"
        "  validity     Run truth-labelled FDR and power simulations.\n"
        "  performance  Run cold process measurements through zebrac.\n\n"
        "  study        Run the comparative study and generate bench/REPORT.md.\n"
        "  report       Render bench/REPORT.md from an existing study directory.\n"
        "  references   List fixed R records or explicitly refresh one dataset.\n\n"
        "The study keeps every evidence track separate and computes no combined score."
    )


def _print_matrix() -> None:
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(asdict(MATRIX[0]).keys())
    for row in MATRIX:
        writer.writerow(asdict(row).values())


def _references_main(argv: Sequence[str]) -> int:
    """List fixed R records or explicitly regenerate one local dataset."""

    dataset_ids = tuple(dict.fromkeys(spec.dataset_id for spec in REFERENCE_SPECS))
    parser = argparse.ArgumentParser(prog="python -m bench references")
    parser.add_argument("--refresh", choices=dataset_ids)
    args = parser.parse_args(argv)
    if args.refresh is None:
        writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
        writer.writerow(("reference", "dataset", "gate", "nbins", "nfolds", "file"))
        for spec in REFERENCE_SPECS:
            writer.writerow(
                (
                    spec.reference_id,
                    spec.dataset_id,
                    "yes" if spec.gate else "no",
                    spec.nbins,
                    spec.nfolds,
                    spec.relative_path,
                )
            )
        return 0
    _refresh_references(args.refresh)
    return 0


def _refresh_references(dataset_id: str) -> None:
    """Replace one data file with fixed outputs freshly computed by R IHW."""

    specs = [spec for spec in REFERENCE_SPECS if spec.dataset_id == dataset_id]
    if not specs:
        raise SystemExit(f"unknown reference dataset: {dataset_id}")
    old_record = load_reference(specs[0].reference_id)
    frozen = old_record.peer_input
    peer_input = PeerInput(
        dataset_id=frozen.dataset_id,
        source_path=frozen.source_path,
        provenance=frozen.provenance,
        size=frozen.size,
        seed=frozen.seed,
        pvalues=frozen.pvalues,
        covariates=frozen.covariates,
    )
    metadata = old_record.metadata
    payload: dict[str, object] = {
        "pvalues": peer_input.pvalues,
        "covariates": peer_input.covariates,
        "dataset_id": np.asarray(dataset_id),
        "provenance": np.asarray(str(metadata["provenance"])),
        "source_url": np.asarray(str(metadata["source_url"])),
        "source_license": np.asarray(str(metadata["source_license"])),
        "r_ihw_version": np.asarray(str(metadata["r_ihw_version"])),
    }
    for spec in specs:
        print(f"R IHW: {spec.reference_id}")
        result = generate_r_reference(
            peer_input,
            RunConfig(
                alpha=spec.alpha,
                nbins=spec.nbins,
                nfolds=spec.nfolds,
                adjustment_type="bh",
                seed=spec.seed,
            ),
        )
        if result.fit.version != metadata["r_ihw_version"]:
            raise SystemExit(
                "refusing to mix R IHW versions: "
                f"stored {metadata['r_ihw_version']}, found {result.fit.version}"
            )
        prefix = f"{spec.prefix}_"
        payload.update(
            {
                prefix + "groups": result.groups,
                prefix + "folds": result.folds,
                prefix + "adjusted_pvalues": result.fit.adjusted_pvalues,
                prefix + "weights": result.fit.weights,
                prefix + "rejections": np.asarray(result.fit.rejection_count),
                prefix + "alpha": np.asarray(spec.alpha),
                prefix + "nbins": np.asarray(spec.nbins),
                prefix + "nfolds": np.asarray(spec.nfolds),
                prefix + "seed": np.asarray(spec.seed),
            }
        )
    target = ROOT / specs[0].relative_path
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=f".{target.stem}.", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {target.relative_to(ROOT)}")


def _validity_main(argv: Sequence[str]) -> int:
    args = _validity_arguments(argv)
    cases = _validity_cases(
        quick=args.quick,
        scenario_ids=args.scenario,
        n_override=args.n,
        reps_override=args.reps,
    )
    methods = ["bh", "ihw_inf_cv"]
    attempts = _run_validity(cases, methods, alpha=args.alpha)
    summaries = _summarize(attempts)
    result_dir = (ROOT / args.result_dir).resolve()
    _ensure_result_dir(result_dir)
    name = _validated_name(args.name)
    attempt_path = result_dir / f"{name}_attempts.csv"
    summary_path = result_dir / f"{name}_summary.csv"
    report_path = result_dir / f"{name}.md"
    _write_csv(attempts, attempt_path)
    _write_csv(summaries, summary_path)
    _write_report(
        summaries,
        report_path,
        alpha=args.alpha,
        quick=args.quick,
        command="python -m bench validity " + " ".join(argv),
    )
    print(f"wrote {attempt_path.relative_to(ROOT)}")
    print(f"wrote {summary_path.relative_to(ROOT)}")
    print(f"wrote {report_path.relative_to(ROOT)}")
    production_failures = sum(
        row.status != "ok" for row in attempts if row.method_id != "bh"
    )
    if production_failures:
        print(f"production fit failures: {production_failures}", file=sys.stderr)
        return 1
    return 0


def _validity_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m bench validity")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--n", type=int)
    parser.add_argument("--reps", type=int)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(SCENARIO_BUILDERS),
        help="Run only this scenario; repeat the option to select several.",
    )
    parser.add_argument("--name", default="validity")
    parser.add_argument("--result-dir", type=Path, default=Path("tmp/results"))
    args = parser.parse_args(argv)
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be strictly between zero and one")
    if args.n is not None and args.n < 1:
        parser.error("--n must be positive")
    if args.reps is not None and args.reps < 1:
        parser.error("--reps must be positive")
    return args


def _validity_cases(
    *,
    quick: bool,
    scenario_ids: Sequence[str] | None,
    n_override: int | None,
    reps_override: int | None,
) -> list[ValidityCase]:
    assumptions = {
        "global_null": "independent valid null",
        "null_covariate": "independent valid null with tied skewed covariate",
        "mixture_mild": "independent valid mixture",
        "mixture_sparse": "independent valid sparse mixture",
        "ignatiadis": "independent valid informative-covariate mixture",
        "dense_covariate": "independent valid wide-covariate mixture",
    }
    selected = list(scenario_ids) if scenario_ids else list(assumptions)
    cases: list[ValidityCase] = []
    for scenario_id in selected:
        reps = 5 if quick else (1_000 if "null" in scenario_id else 200)
        cases.append(
            ValidityCase(
                scenario_id=scenario_id,
                n=3_000 if n_override is None else n_override,
                reps=reps if reps_override is None else reps_override,
                assumption_class=assumptions[scenario_id],
            )
        )
    return cases


def _run_validity(
    cases: Sequence[ValidityCase], methods: Sequence[str], *, alpha: float
) -> list[AttemptRow]:
    rows: list[AttemptRow] = []
    for case in cases:
        print(
            f"{case.scenario_id}: n={case.n}, reps={case.reps}, "
            f"methods={','.join(methods)}"
        )
        builder = SCENARIO_BUILDERS[case.scenario_id]
        for replicate in range(case.reps):
            seed = DEFAULT_SEED + replicate
            draw = builder(case.n, seed)
            for method_id in methods:
                rows.append(
                    _run_attempt(
                        case,
                        draw,
                        method_id,
                        replicate=replicate,
                        seed=seed,
                        alpha=alpha,
                    )
                )
    return rows


def _run_attempt(
    case: ValidityCase,
    draw: SimDraw,
    method_id: str,
    *,
    replicate: int,
    seed: int,
    alpha: float,
) -> AttemptRow:
    try:
        adjusted = _run_method(method_id, draw, alpha=alpha, seed=seed)
        rejected = adjusted <= alpha
        rejections = int(np.sum(rejected))
        false_rejections = int(np.sum(np.logical_and(rejected, draw.is_null)))
        alternatives = ~draw.is_null
        alternative_count = int(np.sum(alternatives))
        power = (
            None
            if alternative_count == 0
            else float(np.sum(np.logical_and(rejected, alternatives)))
            / alternative_count
        )
        return AttemptRow(
            scenario_id=case.scenario_id,
            assumption_class=case.assumption_class,
            method_id=method_id,
            n=case.n,
            replicate=replicate,
            seed=seed,
            alpha=alpha,
            status="ok",
            rejections=rejections,
            false_rejections=false_rejections,
            any_rejection=int(rejections > 0),
            fdp=false_rejections / max(1, rejections),
            power=power,
            error_type=None,
            error_message=None,
        )
    except Exception as exc:  # noqa: BLE001 - every failed attempt is evidence
        return AttemptRow(
            scenario_id=case.scenario_id,
            assumption_class=case.assumption_class,
            method_id=method_id,
            n=case.n,
            replicate=replicate,
            seed=seed,
            alpha=alpha,
            status="error",
            rejections=None,
            false_rejections=None,
            any_rejection=None,
            fdp=None,
            power=None,
            error_type=type(exc).__name__,
            error_message=str(exc).replace("\n", " "),
        )


def _run_method(
    method_id: str, draw: SimDraw, *, alpha: float, seed: int
) -> np.ndarray:
    if method_id == "bh":
        return _p_adjust(draw.pvalues, "fdr_bh")
    if method_id == "ihw_inf_cv":
        return adjust_ihw(
            draw.pvalues,
            draw.covariates,
            alpha,
            nfolds=5,
            seed=seed,
        ).adj_pvalues
    raise ValueError(f"unknown validity method: {method_id}")


def _summarize(attempts: Sequence[AttemptRow]) -> list[SummaryRow]:
    keys = sorted({(row.scenario_id, row.method_id) for row in attempts})
    summaries: list[SummaryRow] = []
    for scenario_id, method_id in keys:
        selected = [
            row
            for row in attempts
            if row.scenario_id == scenario_id and row.method_id == method_id
        ]
        successful = [row for row in selected if row.status == "ok"]
        failed = len(selected) - len(successful)
        fdps = np.asarray([row.fdp for row in successful], dtype=np.float64)
        powers = np.asarray(
            [row.power for row in successful if row.power is not None],
            dtype=np.float64,
        )
        rejections = np.asarray(
            [row.rejections for row in successful], dtype=np.float64
        )
        is_global_null = bool(successful) and all(
            row.power is None for row in successful
        )
        if is_global_null:
            any_rejection = np.asarray(
                [row.any_rejection for row in successful], dtype=np.float64
            )
            mean_fdp = float(np.mean(any_rejection))
            fdr_mcse = _mcse(any_rejection)
            fdr_ci_low, fdr_ci_high = _wilson_interval(
                int(np.sum(any_rejection)), len(any_rejection)
            )
        elif fdps.size:
            mean_fdp = float(np.mean(fdps))
            fdr_mcse = _mcse(fdps)
            fdr_ci_low, fdr_ci_high = _mean_interval(mean_fdp, fdr_mcse)
        else:
            mean_fdp = fdr_mcse = fdr_ci_low = fdr_ci_high = None
        summaries.append(
            SummaryRow(
                scenario_id=scenario_id,
                assumption_class=selected[0].assumption_class,
                method_id=method_id,
                n=selected[0].n,
                attempted=len(selected),
                successful=len(successful),
                failures=failed,
                failure_rate=failed / len(selected),
                mean_fdp=mean_fdp,
                fdr_mcse=fdr_mcse,
                fdr_ci_low=fdr_ci_low,
                fdr_ci_high=fdr_ci_high,
                mean_power=float(np.mean(powers)) if powers.size else None,
                power_mcse=_mcse(powers) if powers.size else None,
                mean_rejections=(
                    float(np.mean(rejections)) if rejections.size else None
                ),
                fdp_q95=(float(np.quantile(fdps, 0.95)) if fdps.size else None),
            )
        )
    return summaries


def _mcse(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def _mean_interval(mean: float, standard_error: float) -> tuple[float, float]:
    return max(0.0, mean - 1.96 * standard_error), min(
        1.0, mean + 1.96 * standard_error
    )


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials < 1:
        raise ValueError("Wilson interval requires at least one trial")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _write_csv(rows: Sequence[AttemptRow] | Sequence[SummaryRow], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty benchmark table")
    mappings = [asdict(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mappings[0]))
        writer.writeheader()
        writer.writerows(mappings)


def _write_report(
    rows: Sequence[SummaryRow],
    path: Path,
    *,
    alpha: float,
    quick: bool,
    command: str,
) -> None:
    lines = [
        "# IHW validity benchmark summary",
        "",
        f"Recorded: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"Command: `{command.strip()}`",
        f"Python: {platform.python_version()}",
        f"NumPy: {np.__version__}",
        f"Imported implementation: `{Path(adjust_ihw.__code__.co_filename).resolve()}`",
        f"Nominal alpha: {alpha}",
        f"Study scale: {'quick smoke' if quick else 'development study'}",
        "",
        "Global-null FDR is estimated as Pr(R > 0) and receives a Wilson interval. Other FDR rows report mean replicate FDP with Monte Carlo standard error. Metrics summarize successful fits only; failure counts remain separate and any production failure makes the command fail.",
        "",
        "| scenario | method | ok/attempted | FDR | FDR MCSE | FDR interval | power | mean rejections |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        interval = (
            ""
            if row.fdr_ci_low is None or row.fdr_ci_high is None
            else f"[{row.fdr_ci_low:.4f}, {row.fdr_ci_high:.4f}]"
        )
        lines.append(
            f"| {row.scenario_id} | {row.method_id} | "
            f"{row.successful}/{row.attempted} | {_format(row.mean_fdp)} | "
            f"{_format(row.fdr_mcse)} | {interval} | "
            f"{_format(row.mean_power)} | {_format(row.mean_rejections, digits=2)} |"
        )
    lines.extend(
        [
            "",
            "This generated table is measurement output, not the maintainer's interpretation. A claim about calibration or power requires the planned replicate count, scenario assumptions, paired comparison, and a separately reviewed limitations paragraph.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _format(value: float | None, *, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _ensure_result_dir(path: Path) -> None:
    temporary_root = (ROOT / "tmp").resolve()
    if not path.is_relative_to(temporary_root):
        raise SystemExit(f"result directory must be under {temporary_root}")
    path.mkdir(parents=True, exist_ok=True)


def _validated_name(value: str) -> str:
    if _NAME_PATTERN.fullmatch(value) is None:
        raise SystemExit(
            "--name must contain only lowercase letters, digits, underscores, or hyphens"
        )
    return value


if __name__ == "__main__":
    raise SystemExit(main())
