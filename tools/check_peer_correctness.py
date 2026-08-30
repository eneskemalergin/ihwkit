"""Run generated correctness gates for the changing production method."""

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

from tools.peer import FitResult, PeerInput, RunConfig, fit, load_peer_input


def main(argv: Sequence[str] | None = None) -> int:
    """Run numerical output checks on generated inputs and production only."""

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
    output = {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "scope": "generated inputs and ihwkit only",
        "correctness_gate": passed,
        "rows": rows,
    }
    output_path = result_dir / "correctness.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps(output, indent=2, sort_keys=True))
        print(f"wrote {output_path.relative_to(ROOT)}")
    return 0 if passed else 1


def _production_gate(dataset_id: str, nfolds: int) -> dict[str, object]:
    """Check finite production output for one synthetic case."""

    peer_input = load_peer_input(dataset_id)
    config = _config(nfolds)
    try:
        result = fit("ihwkit", peer_input, config)
        _validate_fit(result, peer_input)
    except Exception as exc:  # noqa: BLE001 - a gate records every fit failure
        return {
            "case_id": f"{dataset_id}_n{nfolds}",
            "method_id": "ihwkit",
            "dataset_id": dataset_id,
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    return {
        "case_id": f"{dataset_id}_n{nfolds}",
        "method_id": "ihwkit",
        "dataset_id": dataset_id,
        "status": "ok",
        "rejection_count": result.rejection_count,
        "weight_mean": float(np.mean(result.weights)),
    }


def _config(nfolds: int) -> RunConfig:
    return RunConfig(
        alpha=0.1,
        nbins="auto",
        nfolds=nfolds,
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


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("tmp/results"))
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
