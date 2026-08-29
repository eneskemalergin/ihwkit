"""Run synthetic correctness gates and qualified peer availability checks."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.peer import (
    COMPARISON_METHODS,
    FitResult,
    PeerDataError,
    PeerInput,
    RunConfig,
    fit,
    load_oracle,
    load_peer_input,
    run_peer,
)

ATOL = 1e-8
RTOL = 1e-6


def main(argv: Sequence[str] | None = None) -> int:
    """Run production gates, oracle replays, and peer preflights."""

    args = _argument_parser().parse_args(argv)
    result_dir = (ROOT / args.result_dir).resolve()
    if not result_dir.is_relative_to((ROOT / "tmp").resolve()):
        print("result directory must be under tmp", file=sys.stderr)
        return 2
    result_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    passed = True
    for dataset_id, nfolds in (
        ("sim_500_seed42", 1),
        ("dense_500_seed42", 1),
        ("sim_5000_seed42", 1),
        ("sim_50000_seed42", 1),
        ("sim_50000_seed42", 5),
    ):
        row = _production_gate(dataset_id, nfolds)
        rows.append(row)
        passed = passed and row["status"] == "ok"
    for oracle_id in ("sim_5000_inf_n1", "sim_5000_inf_n5"):
        row = _oracle_gate(oracle_id)
        rows.append(row)
        passed = passed and row["status"] == "ok"
    auto_row = _production_gate("sim_5000_seed42", 5, lambda_policy="auto")
    auto_row["case_id"] = "sim_5000_auto_native"
    rows.append(auto_row)
    passed = passed and auto_row["status"] == "ok"
    peer_rows = _peer_availability(args)
    rows.extend(peer_rows)
    airway_row = _airway_diagnostic()
    rows.append(airway_row)
    output = {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tolerances": {"atol": ATOL, "rtol": RTOL},
        "synthetic_gate": passed,
        "rows": rows,
    }
    output_path = result_dir / "peer_correctness.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps(output, indent=2, sort_keys=True))
        print(f"wrote {output_path.relative_to(ROOT)}")
    return 0 if passed else 1


def _production_gate(
    dataset_id: str, nfolds: int, lambda_policy: str = "inf"
) -> dict[str, object]:
    """Check finite production output for one synthetic case."""

    peer_input = load_peer_input(dataset_id)
    config = _config(nfolds, lambda_policy)
    try:
        result = fit("ihwkit_numpy_numba", peer_input, config)
        _validate_fit(result, peer_input)
    except Exception as exc:  # noqa: BLE001 - a gate records every fit failure
        return {
            "case_id": f"{dataset_id}_{lambda_policy}_n{nfolds}",
            "method_id": "ihwkit_numpy_numba",
            "dataset_id": dataset_id,
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    return {
        "case_id": f"{dataset_id}_{lambda_policy}_n{nfolds}",
        "method_id": "ihwkit_numpy_numba",
        "dataset_id": dataset_id,
        "status": "ok",
        "rejection_count": result.rejection_count,
        "weight_mean": float(np.mean(result.weights)),
    }


def _oracle_gate(oracle_id: str) -> dict[str, object]:
    """Compare production output with one frozen synthetic R oracle."""

    record = load_oracle(oracle_id)
    peer_input = record.peer_input
    nfolds = int(np.max(peer_input.folds)) + 1 if peer_input.folds is not None else 1
    config = _config(nfolds, "inf")
    try:
        result = fit("ihwkit_numpy_numba", peer_input, config)
        _validate_fit(result, peer_input)
        adjusted_delta = _max_abs(result.adjusted_pvalues, record.adjusted_pvalues)
        weights_delta = _max_abs(result.weights, record.weights)
        rejection_delta = result.rejection_count - record.r_rejections
        status = (
            "ok"
            if (
                rejection_delta == 0
                and np.allclose(
                    result.adjusted_pvalues,
                    record.adjusted_pvalues,
                    atol=ATOL,
                    rtol=RTOL,
                )
                and np.allclose(
                    result.weights,
                    record.weights,
                    atol=ATOL,
                    rtol=RTOL,
                )
            )
            else "fail"
        )
        return {
            "case_id": oracle_id,
            "method_id": "ihwkit_numpy_numba",
            "dataset_id": peer_input.dataset_id,
            "status": status,
            "r_rejections": record.r_rejections,
            "rejection_count": result.rejection_count,
            "rejection_delta": rejection_delta,
            "max_abs_adj_pvalues": adjusted_delta,
            "max_abs_weights": weights_delta,
        }
    except Exception as exc:  # noqa: BLE001 - a gate records every fit failure
        return {
            "case_id": oracle_id,
            "method_id": "ihwkit_numpy_numba",
            "dataset_id": peer_input.dataset_id,
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def _peer_availability(args: argparse.Namespace) -> list[dict[str, object]]:
    """Record explicit status for each comparison method."""

    rows: list[dict[str, object]] = []
    try:
        peer_input = load_peer_input(args.availability_dataset)
    except PeerDataError as exc:
        for method_id in COMPARISON_METHODS:
            rows.append(
                {
                    "case_id": "peer_availability",
                    "method_id": method_id,
                    "dataset_id": args.availability_dataset,
                    "status": "error",
                    "exit_code": 1,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
        return rows
    config = _config(1, "inf")
    for method_id in COMPARISON_METHODS:
        document = run_peer(method_id, peer_input, config)
        rows.append(
            {
                "case_id": "peer_availability",
                "method_id": method_id,
                "dataset_id": args.availability_dataset,
                "status": document["status"],
                "exit_code": document["exit_code"],
                "version": document["version"],
                "error": document["error"],
            }
        )
    return rows


def _airway_diagnostic() -> dict[str, object]:
    """Run airway only as a non-gating local diagnostic."""

    try:
        peer_input = load_peer_input("airway_seed42")
    except PeerDataError as exc:
        return {
            "case_id": "airway_local_diagnostic",
            "dataset_id": "airway_seed42",
            "status": "unavailable",
            "gate": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    row: dict[str, object] = {
        "case_id": "airway_local_diagnostic",
        "dataset_id": peer_input.dataset_id,
        "status": "unavailable",
        "gate": False,
    }
    for nfolds in (1, 5):
        try:
            result = fit("ihwkit_numpy_numba", peer_input, _config(nfolds, "inf"))
            _validate_fit(result, peer_input)
            row[f"nfolds_{nfolds}"] = {
                "status": "ok",
                "rejection_count": result.rejection_count,
            }
        except Exception as exc:  # noqa: BLE001 - a diagnostic records every fit failure
            row[f"nfolds_{nfolds}"] = {
                "status": "error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
    row["status"] = "diagnostic"
    return row


def _config(nfolds: int, lambda_policy: str) -> RunConfig:
    return RunConfig(
        alpha=0.1,
        nbins="auto",
        nfolds=nfolds,
        lambda_policy="auto" if lambda_policy == "auto" else "inf",
        adjustment_type="bh",
        seed=42,
    )


def _validate_fit(result: FitResult, peer_input: PeerInput) -> None:
    if result.adjusted_pvalues.shape != (peer_input.size,):
        raise RuntimeError("adjusted p-value shape does not match input")
    if result.weights is None or result.weights.shape != (peer_input.size,):
        raise RuntimeError("production weights are missing or have the wrong shape")
    if not np.all(np.isfinite(result.adjusted_pvalues)):
        raise RuntimeError("production adjusted p-values are nonfinite")
    if not np.all(np.isfinite(result.weights)):
        raise RuntimeError("production weights are nonfinite")


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--availability-dataset", default="sim_500_seed42")
    parser.add_argument("--result-dir", type=Path, default=Path("tmp/results"))
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
