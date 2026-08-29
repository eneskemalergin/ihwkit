"""Shared command-line and result handling for peer adapters."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tools.data_contract import DataContractError, PeerInput, load_peer_input

FloatArray = NDArray[np.float64]
LambdaPolicy = Literal["inf", "auto"]
PeerStatus = Literal["ok", "unavailable", "error"]
FitCallable = Callable[["PeerInput", "RunConfig"], "FitResult"]


class AdapterUnavailable(RuntimeError):
    """Raised when a peer runtime or supported API is not available."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Store the normalized settings passed to a peer method."""

    alpha: float
    nbins: int | str
    nfolds: int | None
    lambda_policy: LambdaPolicy
    adjustment_type: str
    seed: int
    oracle_id: str | None
    include_arrays: bool

    def as_document(self) -> dict[str, object]:
        """Return JSON-compatible configuration metadata."""

        return {
            "alpha": self.alpha,
            "nbins": self.nbins,
            "nfolds": self.nfolds,
            "lambda_policy": self.lambda_policy,
            "adjustment_type": self.adjustment_type,
            "seed": self.seed,
            "oracle_id": self.oracle_id,
        }


@dataclass(frozen=True, slots=True)
class FitResult:
    """Store normalized adjusted p-values and optional weights."""

    adjusted_pvalues: FloatArray
    weights: FloatArray | None
    rejection_count: int


def ihw_kwargs(peer_input: PeerInput, config: RunConfig) -> dict[str, object]:
    """Build common ``adjust_ihw`` keyword arguments from a peer input."""

    if config.nfolds is not None:
        nfolds = config.nfolds
    elif peer_input.folds is not None:
        nfolds = int(np.max(peer_input.folds)) + 1
    else:
        nfolds = 5
    kwargs: dict[str, object] = {
        "alpha": config.alpha,
        "nbins": config.nbins,
        "nfolds": nfolds,
        "lambdas": None if config.lambda_policy == "inf" else "auto",
        "adjustment_type": config.adjustment_type,
        "seed": config.seed,
    }
    if peer_input.groups is not None:
        kwargs["groups"] = peer_input.groups
    if peer_input.folds is not None:
        kwargs["folds"] = peer_input.folds
    if peer_input.fold_lambdas is not None:
        kwargs["fold_lambdas"] = peer_input.fold_lambdas
    return kwargs


def fit_result_from_ihw(result: object, alpha: float) -> FitResult:
    """Extract and validate common arrays from an ihw result object."""

    adjusted_value = getattr(result, "adj_pvalues", None)
    weights_value = getattr(result, "weights", None)
    if adjusted_value is None:
        raise RuntimeError("peer result has no adj_pvalues attribute")
    adjusted = np.asarray(adjusted_value, dtype=np.float64)
    weights = (
        None
        if weights_value is None
        else np.asarray(weights_value, dtype=np.float64)
    )
    rejection_count = int(np.sum(adjusted <= alpha))
    return FitResult(adjusted, weights, rejection_count)


def adapter_main(
    method_id: str,
    implementation_version: str,
    fit: FitCallable,
    argv: Sequence[str] | None = None,
) -> int:
    """Run one adapter and emit its explicit result record."""

    parser = _argument_parser(method_id)
    args = parser.parse_args(argv)
    config = RunConfig(
        alpha=args.alpha,
        nbins=args.nbins,
        nfolds=args.nfolds,
        lambda_policy=args.lambda_policy,
        adjustment_type=args.adjustment_type,
        seed=args.seed,
        oracle_id=args.oracle,
        include_arrays=args.include_arrays,
    )
    environment = _environment()
    base = {
        "schema_version": "peer-result-1",
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "method_id": method_id,
        "implementation_version": implementation_version,
        "dataset_id": args.dataset,
        "configuration": config.as_document(),
        "environment": environment,
        "status": "error",
        "exit_code": 1,
        "error": None,
        "rejection_count": None,
        "weights_available": False,
    }
    try:
        peer_input = load_peer_input(args.dataset, oracle_id=args.oracle)
        fit_result = fit(peer_input, config)
        _validate_fit(fit_result, peer_input, config.alpha)
        base.update(
            {
                "status": "ok",
                "exit_code": 0,
                "rejection_count": fit_result.rejection_count,
                "weights_available": fit_result.weights is not None,
            }
        )
        if config.include_arrays:
            base["adjusted_pvalues"] = fit_result.adjusted_pvalues.tolist()
            if fit_result.weights is not None:
                base["weights"] = fit_result.weights.tolist()
    except AdapterUnavailable as exc:
        base.update(
            {
                "status": "unavailable",
                "exit_code": 3,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
    except (DataContractError, RuntimeError, ValueError, OSError) as exc:
        base["error"] = {"type": type(exc).__name__, "message": str(exc)}
    # Adapters may invoke optional external runtimes, so preserve unexpected failures.
    except Exception as exc:  # noqa: BLE001
        base["error"] = {"type": type(exc).__name__, "message": str(exc)}
    _emit(base, args.result, args.quiet)
    return int(base["exit_code"])


def _argument_parser(method_id: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=method_id)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--oracle")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--nbins", type=_parse_nbins, default="auto")
    parser.add_argument("--nfolds", type=int)
    parser.add_argument(
        "--lambda-policy", choices=("inf", "auto"), default="inf"
    )
    parser.add_argument(
        "--adjustment-type", choices=("bh", "bonferroni"), default="bh"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-arrays", action="store_true")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _parse_nbins(value: str) -> int | str:
    if value == "auto":
        return value
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("nbins must be positive")
    return parsed


def _validate_fit(fit_result: FitResult, peer_input: PeerInput, alpha: float) -> None:
    adjusted = fit_result.adjusted_pvalues
    if adjusted.shape != (peer_input.size,):
        raise RuntimeError(
            f"adjusted_pvalues must have shape ({peer_input.size},), got {adjusted.shape}"
        )
    if np.any(~np.isfinite(adjusted)) or np.any(adjusted < 0.0) or np.any(adjusted > 1.0):
        raise RuntimeError("adjusted_pvalues must be finite and lie in [0, 1]")
    expected_rejections = int(np.sum(adjusted <= alpha))
    if fit_result.rejection_count != expected_rejections:
        raise RuntimeError("rejection_count does not match adjusted_pvalues")
    if fit_result.weights is not None:
        weights = fit_result.weights
        if weights.shape != (peer_input.size,):
            raise RuntimeError(
                f"weights must have shape ({peer_input.size},), got {weights.shape}"
            )
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise RuntimeError("weights must be finite and nonnegative")


def _environment() -> dict[str, object]:
    package_versions: dict[str, object] = {}
    for package in ("numpy", "numba", "scipy", "pyihw"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "package_versions": package_versions,
    }


def _emit(document: dict[str, object], path: Path | None, quiet: bool) -> None:
    text = json.dumps(document, sort_keys=True, allow_nan=False)
    if path is None:
        if not quiet:
            print(text)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        if not quiet:
            print(
                f"{document['method_id']} status={document['status']} "
                f"result={path}"
            )
