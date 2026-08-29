"""Run synthetic correctness gates and qualified peer availability checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.data_contract import (
    DataContractError,
    PeerInput,
    load_oracle,
    load_peer_input,
)
from tools.peers.ihwkit_numpy_numba import fit as production_fit
from tools.peers.runner import FitResult, RunConfig

METHOD_SCRIPTS = {
    "ihwkit_numpy_numba": "tools/peers/ihwkit_numpy_numba.py",
    "ihwkit_numpy": "tools/peers/ihwkit_numpy.py",
    "ihwkit_scipy": "tools/peers/ihwkit_scipy.py",
    "pyihw": "tools/peers/pyihw.py",
    "r_ihw": "tools/peers/r_ihw.py",
    "julia_ihw": "tools/peers/julia_ihw.py",
}
COMPARISON_METHOD_SCRIPTS = {
    method_id: path
    for method_id, path in METHOD_SCRIPTS.items()
    if method_id != "ihwkit_numpy_numba"
}
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
    peer_rows = _peer_availability(args, result_dir)
    rows.extend(peer_rows)
    airway_row = _airway_diagnostic()
    rows.append(airway_row)
    output = {
        "schema_version": "peer-correctness-1",
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
        result = production_fit(peer_input, config)
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
    config = _config(nfolds, "inf", oracle_id)
    try:
        result = production_fit(peer_input, config)
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


def _peer_availability(
    args: argparse.Namespace, result_dir: Path
) -> list[dict[str, object]]:
    """Record explicit status for comparison adapters.

    The production adapter is covered by the direct synthetic gates above.
    Running its subprocess adapter here would execute the same production
    path a second time without adding an independent check.
    """

    rows: list[dict[str, object]] = []
    for method_id, script in COMPARISON_METHOD_SCRIPTS.items():
        path = result_dir / f"peer_correctness.{method_id}.json"
        path.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(ROOT / script),
            "--dataset",
            args.availability_dataset,
            "--nbins",
            "auto",
            "--nfolds",
            "1",
            "--lambda-policy",
            "inf",
            "--seed",
            "42",
            "--result",
            str(path),
            "--quiet",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if path.is_file():
            document = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "case_id": "peer_availability",
                    "method_id": method_id,
                    "dataset_id": args.availability_dataset,
                    "status": document.get("status"),
                    "exit_code": document.get("exit_code"),
                    "implementation_version": document.get("implementation_version"),
                    "error": document.get("error"),
                }
            )
        else:
            rows.append(
                {
                    "case_id": "peer_availability",
                    "method_id": method_id,
                    "dataset_id": args.availability_dataset,
                    "status": "error",
                    "exit_code": completed.returncode,
                    "error": {
                        "type": "MissingResultRecord",
                        "message": completed.stderr.strip() or "adapter failed before output",
                    },
                }
            )
    return rows


def _airway_diagnostic() -> dict[str, object]:
    """Run airway only as a non-gating local diagnostic."""

    try:
        peer_input = load_peer_input("airway_seed42")
    except DataContractError as exc:
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
            result = production_fit(peer_input, _config(nfolds, "inf"))
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


def _config(
    nfolds: int, lambda_policy: str, oracle_id: str | None = None
) -> RunConfig:
    return RunConfig(
        alpha=0.1,
        nbins="auto",
        nfolds=nfolds,
        lambda_policy="auto" if lambda_policy == "auto" else "inf",
        adjustment_type="bh",
        seed=42,
        oracle_id=oracle_id,
        include_arrays=False,
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
