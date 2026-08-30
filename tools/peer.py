"""Run local and external IHW comparisons through one explicit interface."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.simulators import dense_covariate, ignatiadis

FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.intp]
BooleanArray = NDArray[np.bool_]
LambdaPolicy = Literal["inf", "auto"]
MethodId = Literal[
    "ihwkit",
    "ihwkit_numpy",
    "ihwkit_scipy",
    "pyihw",
    "r_ihw",
]

METHODS: tuple[MethodId, ...] = (
    "ihwkit",
    "ihwkit_numpy",
    "ihwkit_scipy",
    "pyihw",
    "r_ihw",
)
R_SCRIPT = Path(__file__).with_name("peer.R")


@dataclass(frozen=True, slots=True)
class ReferenceSpec:
    """Name one immutable R result inside a dataset reference file."""

    reference_id: str
    dataset_id: str
    relative_path: Path
    prefix: str
    gate: bool
    alpha: float
    nbins: int
    nfolds: int
    lambda_policy: LambdaPolicy
    seed: int


REFERENCE_SPECS: tuple[ReferenceSpec, ...] = (
    ReferenceSpec(
        "sim_5000_inf_n1",
        "sim_5000_seed42",
        Path("bench/data/sim_5000_r_ihw_1_40_0.npz"),
        "inf_n1",
        True,
        0.1,
        3,
        1,
        "inf",
        42,
    ),
    ReferenceSpec(
        "sim_5000_inf_n5",
        "sim_5000_seed42",
        Path("bench/data/sim_5000_r_ihw_1_40_0.npz"),
        "inf_n5",
        True,
        0.1,
        3,
        5,
        "inf",
        42,
    ),
    ReferenceSpec(
        "sim_5000_auto",
        "sim_5000_seed42",
        Path("bench/data/sim_5000_r_ihw_1_40_0.npz"),
        "auto",
        False,
        0.1,
        3,
        5,
        "auto",
        42,
    ),
    ReferenceSpec(
        "airway_inf_n1",
        "airway",
        Path("bench/data/airway_r_ihw_1_40_0.npz"),
        "inf_n1",
        False,
        0.1,
        22,
        1,
        "inf",
        42,
    ),
    ReferenceSpec(
        "airway_inf_n5",
        "airway",
        Path("bench/data/airway_r_ihw_1_40_0.npz"),
        "inf_n5",
        False,
        0.1,
        22,
        5,
        "inf",
        42,
    ),
    ReferenceSpec(
        "airway_auto",
        "airway",
        Path("bench/data/airway_r_ihw_1_40_0.npz"),
        "auto",
        False,
        0.1,
        22,
        5,
        "auto",
        42,
    ),
)
REFERENCE_IDS = tuple(spec.reference_id for spec in REFERENCE_SPECS)
PARITY_GATE_IDS = tuple(spec.reference_id for spec in REFERENCE_SPECS if spec.gate)


class PeerDataError(ValueError):
    """Report an unknown, missing, or malformed peer input."""


@dataclass(frozen=True, slots=True)
class PeerInput:
    """Store one generated input or frozen R comparison partition."""

    dataset_id: str
    source_path: str
    provenance: str
    size: int
    seed: int | None
    pvalues: FloatArray
    covariates: FloatArray
    truth_labels: BooleanArray | None = None
    groups: IntegerArray | None = None
    folds: IntegerArray | None = None
    fold_lambdas: FloatArray | None = None
    reference_id: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceRecord:
    """Store immutable R output and the exact input used to produce it."""

    spec: ReferenceSpec
    peer_input: PeerInput
    metadata: Mapping[str, object]
    r_rejections: int
    adjusted_pvalues: FloatArray
    weights: FloatArray


class PeerUnavailable(RuntimeError):
    """Report that a requested comparison cannot run in this environment."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Store settings shared by the supported IHW comparisons."""

    alpha: float = 0.1
    nbins: int | str = "auto"
    nfolds: int | None = None
    lambda_policy: LambdaPolicy = "inf"
    adjustment_type: str = "bh"
    seed: int = 42

    def as_document(self) -> dict[str, object]:
        """Return the settings as a small JSON-compatible mapping."""

        return {
            "alpha": self.alpha,
            "nbins": self.nbins,
            "nfolds": self.nfolds,
            "lambda_policy": self.lambda_policy,
            "adjustment_type": self.adjustment_type,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class FitResult:
    """Store normalized comparison output and its actual implementation version."""

    adjusted_pvalues: FloatArray
    weights: FloatArray | None
    rejection_count: int
    version: str


@dataclass(frozen=True, slots=True)
class RReferenceResult:
    """Store an R fit plus the partition needed for later Python replay."""

    fit: FitResult
    groups: IntegerArray
    folds: IntegerArray
    fold_lambdas: FloatArray


def load_peer_input(dataset_id: str, *, reference_id: str | None = None) -> PeerInput:
    """Return a named generated input or the input inside a frozen R record.

    Parameters
    ----------
    dataset_id : str
        ``sim_500_seed42``, ``sim_1500_seed42``, ``dense_500_seed42``,
        ``sim_5000_seed42``, ``sim_15000_seed42``, ``sim_50000_seed42`` or
        ``airway``.
    reference_id : str or None, optional
        Frozen R record whose exact inputs and partitions should be returned.

    Returns
    -------
    PeerInput
        Validated one-dimensional arrays and readable provenance.

    Raises
    ------
    PeerDataError
        If the name is unknown or a required local record is unavailable.
    """

    if reference_id is not None:
        record = load_reference(reference_id)
        if record.peer_input.dataset_id != dataset_id:
            raise PeerDataError(
                f"reference {reference_id!r} belongs to "
                f"{record.peer_input.dataset_id!r}, not {dataset_id!r}"
            )
        return record.peer_input
    generated = {
        "sim_500_seed42": (ignatiadis, 500),
        "sim_1500_seed42": (ignatiadis, 1_500),
        "dense_500_seed42": (dense_covariate, 500),
        "sim_5000_seed42": (ignatiadis, 5_000),
        "sim_15000_seed42": (ignatiadis, 15_000),
        "sim_50000_seed42": (ignatiadis, 50_000),
    }
    if dataset_id in generated:
        builder, size = generated[dataset_id]
        draw = builder(size, seed=42, signal_frac=0.15)
        source = f"generated:{builder.__name__}(n={size}, seed=42, signal_frac=0.15)"
        return _validated_input(
            PeerInput(
                dataset_id=dataset_id,
                source_path=source,
                provenance="deterministic synthetic input generated by tools.simulators",
                size=size,
                seed=42,
                pvalues=draw.pvalues,
                covariates=draw.covariates,
                truth_labels=draw.is_null,
            )
        )
    if dataset_id == "airway":
        return _load_reference_input("airway_inf_n5")
    raise PeerDataError(f"unknown dataset {dataset_id!r}")


def load_reference(reference_id: str) -> ReferenceRecord:
    """Load one immutable R result and its exact input and partitions."""

    spec = _reference_spec(reference_id)
    path = ROOT / spec.relative_path
    if not path.is_file():
        raise PeerDataError(f"reference not found: {spec.relative_path}")
    prefix = f"{spec.prefix}_"
    try:
        with np.load(path, allow_pickle=False) as archive:
            pvalues = _archive_vector(archive, "pvalues", np.float64)
            size = pvalues.size
            covariates = _archive_vector(archive, "covariates", np.float64, size)
            groups = _archive_vector(archive, prefix + "groups", np.intp, size)
            folds = _archive_vector(archive, prefix + "folds", np.intp, size)
            adjusted = _archive_vector(
                archive, prefix + "adjusted_pvalues", np.float64, size
            )
            weights = _archive_vector(archive, prefix + "weights", np.float64, size)
            lambdas = _reference_lambdas(archive, prefix, spec.nfolds)
            rejections = _archive_scalar(archive, prefix + "rejections", int)
            metadata = {
                "reference_id": reference_id,
                "dataset_id": _archive_scalar(archive, "dataset_id", str),
                "provenance": _archive_scalar(archive, "provenance", str),
                "source_url": _archive_scalar(archive, "source_url", str),
                "source_license": _archive_scalar(archive, "source_license", str),
                "r_ihw_version": _archive_scalar(archive, "r_ihw_version", str),
                "alpha": float(_archive_scalar(archive, prefix + "alpha", float)),
                "nbins": _archive_scalar(archive, prefix + "nbins", int),
                "nfolds": _archive_scalar(archive, prefix + "nfolds", int),
                "lambda_policy": _archive_scalar(
                    archive, prefix + "lambda_policy", str
                ),
                "seed": _archive_scalar(archive, prefix + "seed", int),
            }
    except (OSError, ValueError, KeyError) as exc:
        raise PeerDataError(
            f"could not read reference {spec.relative_path}: {exc}"
        ) from exc
    _validate_reference_metadata(spec, metadata)
    peer_input = _validated_input(
        PeerInput(
            dataset_id=spec.dataset_id,
            source_path=spec.relative_path.as_posix(),
            provenance=str(metadata["provenance"]),
            size=size,
            seed=spec.seed,
            pvalues=pvalues,
            covariates=covariates,
            groups=groups,
            folds=folds,
            fold_lambdas=lambdas,
            reference_id=reference_id,
        )
    )
    if int(np.max(groups)) + 1 != spec.nbins:
        raise PeerDataError(f"reference {reference_id!r} has unexpected groups")
    if int(np.max(folds)) + 1 != spec.nfolds:
        raise PeerDataError(f"reference {reference_id!r} has unexpected folds")
    if np.any(~np.isfinite(adjusted)) or np.any((adjusted < 0.0) | (adjusted > 1.0)):
        raise PeerDataError(
            f"reference {reference_id!r} adjusted p-values must lie in [0, 1]"
        )
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise PeerDataError(
            f"reference {reference_id!r} weights must be finite and nonnegative"
        )
    if rejections != int(np.sum(adjusted <= spec.alpha)):
        raise PeerDataError(
            f"reference {reference_id!r} rejection count does not match its output"
        )
    return ReferenceRecord(spec, peer_input, metadata, rejections, adjusted, weights)


def fit(method_id: MethodId, peer_input: PeerInput, config: RunConfig) -> FitResult:
    """Run one named comparison and validate its normalized output.

    Parameters
    ----------
    method_id : str
        One of the names in ``METHODS``.
    peer_input : PeerInput
        Normalized p-values, covariates, and optional frozen partitions.
    config : RunConfig
        Shared statistical settings.

    Returns
    -------
    FitResult
        Adjusted p-values, optional weights, rejection count, and version.

    Raises
    ------
    PeerUnavailable
        If an optional runtime, package, or supported API is unavailable.
    RuntimeError
        If the comparison runs but fails or returns invalid output.
    """

    if method_id == "ihwkit":
        result = _fit_production(peer_input, config)
    elif method_id == "ihwkit_numpy":
        result = _fit_legacy(peer_input, config, backend="numpy")
    elif method_id == "ihwkit_scipy":
        result = _fit_legacy(peer_input, config, backend="highs")
    elif method_id == "pyihw":
        result = _fit_pyihw(peer_input, config)
    elif method_id == "r_ihw":
        result = _fit_r(peer_input, config)
    else:
        raise ValueError(f"unknown peer method: {method_id}")
    _validate_fit(result, peer_input, config.alpha)
    return result


def run_peer(
    method_id: MethodId,
    peer_input: PeerInput,
    config: RunConfig,
    *,
    include_arrays: bool = False,
) -> dict[str, object]:
    """Run one comparison and return an explicit success or failure record."""

    document: dict[str, object] = {
        "method_id": method_id,
        "dataset_id": peer_input.dataset_id,
        "configuration": config.as_document(),
        "status": "error",
        "exit_code": 1,
        "version": None,
        "error": None,
        "rejection_count": None,
        "weights_available": False,
    }
    try:
        result = fit(method_id, peer_input, config)
    except PeerUnavailable as exc:
        document.update(
            {
                "status": "unavailable",
                "exit_code": 3,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        return document
    except Exception as exc:  # noqa: BLE001 - peer failures belong in the record
        document["error"] = {"type": type(exc).__name__, "message": str(exc)}
        return document
    document.update(
        {
            "status": "ok",
            "exit_code": 0,
            "version": result.version,
            "rejection_count": result.rejection_count,
            "weights_available": result.weights is not None,
        }
    )
    if include_arrays:
        document["adjusted_pvalues"] = result.adjusted_pvalues.tolist()
        if result.weights is not None:
            document["weights"] = result.weights.tolist()
    return document


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface for one named comparison."""

    args = _argument_parser().parse_args(argv)
    config = RunConfig(
        alpha=args.alpha,
        nbins=args.nbins,
        nfolds=args.nfolds,
        lambda_policy=args.lambda_policy,
        adjustment_type=args.adjustment_type,
        seed=args.seed,
    )
    try:
        peer_input = load_peer_input(args.dataset, reference_id=args.reference)
    except PeerDataError as exc:
        document: dict[str, object] = {
            "method_id": args.method,
            "dataset_id": args.dataset,
            "configuration": config.as_document(),
            "status": "error",
            "exit_code": 1,
            "version": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "rejection_count": None,
            "weights_available": False,
        }
    else:
        document = run_peer(
            args.method,
            peer_input,
            config,
            include_arrays=args.include_arrays,
        )
    _emit(document, args.result, args.quiet)
    return int(document["exit_code"])


def _fit_production(peer_input: PeerInput, config: RunConfig) -> FitResult:
    from ihw import adjust_ihw

    result = adjust_ihw(
        peer_input.pvalues,
        peer_input.covariates,
        **_ihw_kwargs(peer_input, config),
    )
    return _from_ihw_result(result, config.alpha, _version("ihwkit"))


def _fit_legacy(
    peer_input: PeerInput, config: RunConfig, *, backend: Literal["numpy", "highs"]
) -> FitResult:
    if backend == "highs" and not _importable("scipy"):
        raise PeerUnavailable("scipy is not installed")
    from tools import peer_legacy

    kwargs = _ihw_kwargs(peer_input, config)
    kwargs["lp_backend"] = backend
    kwargs["use_numba"] = False
    result = peer_legacy.adjust_ihw(
        peer_input.pvalues,
        peer_input.covariates,
        **kwargs,
    )
    version = f"{peer_legacy.__version__}+{backend}"
    return _from_ihw_result(result, config.alpha, version)


def _fit_pyihw(peer_input: PeerInput, config: RunConfig) -> FitResult:
    if not _importable("pyihw"):
        raise PeerUnavailable("pyihw 0.2.0 is not installed")
    version = _version("pyihw")
    if version != "0.2.0":
        raise PeerUnavailable(f"pyihw 0.2.0 is supported; found {version}")
    if peer_input.groups is not None:
        raise PeerUnavailable("pyihw cannot accept frozen covariate groups")
    import pyihw

    nfolds = 5 if config.nfolds is None else config.nfolds
    lambdas: str | FloatArray
    if config.lambda_policy == "auto":
        lambdas = "auto"
    else:
        lambdas = np.array([np.inf], dtype=np.float64)
    result = pyihw.ihw(
        peer_input.pvalues,
        peer_input.covariates,
        config.alpha,
        nbins=config.nbins,
        nfolds=nfolds,
        lambdas=lambdas,
        adjustment_type=config.adjustment_type,
        folds=peer_input.folds,
        rng=np.random.default_rng(config.seed),
    )
    return _from_ihw_result(result, config.alpha, version)


def _fit_r(peer_input: PeerInput, config: RunConfig) -> FitResult:
    return generate_r_reference(peer_input, config).fit


def generate_r_reference(peer_input: PeerInput, config: RunConfig) -> RReferenceResult:
    """Run R IHW once and return everything required for immutable replay.

    This is the deliberate reference-generation path. Routine correctness,
    parity, validity, robustness, and performance commands do not call it.
    """

    rscript = shutil.which("Rscript")
    if rscript is None:
        raise PeerUnavailable("Rscript is not installed")
    if config.adjustment_type != "bh":
        raise PeerUnavailable("R IHW comparison supports adjustment_type=bh only")
    if peer_input.groups is not None or peer_input.folds is not None:
        raise PeerUnavailable("R IHW comparison cannot accept frozen groups or folds")
    nfolds = 5 if config.nfolds is None else config.nfolds
    nbins = (
        max(1, min(40, peer_input.size // 1500))
        if config.nbins == "auto"
        else config.nbins
    )
    if not isinstance(nbins, int):
        raise TypeError("R IHW comparison could not resolve nbins")
    with tempfile.TemporaryDirectory(prefix="ihwkit_r_peer_") as temporary:
        directory = Path(temporary)
        pvalues_path = directory / "pvalues.txt"
        covariates_path = directory / "covariates.txt"
        output_prefix = directory / "result"
        np.savetxt(pvalues_path, peer_input.pvalues, fmt="%.17g")
        np.savetxt(covariates_path, peer_input.covariates, fmt="%.17g")
        command = [
            rscript,
            "--vanilla",
            str(R_SCRIPT),
            "--pvalues",
            str(pvalues_path),
            "--covariates",
            str(covariates_path),
            "--alpha",
            str(config.alpha),
            "--nbins",
            str(nbins),
            "--nfolds",
            str(nfolds),
            "--lambda-policy",
            config.lambda_policy,
            "--seed",
            str(config.seed),
            "--output-prefix",
            str(output_prefix),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        detail = completed.stderr.strip() or completed.stdout.strip()
        if completed.returncode == 3:
            raise PeerUnavailable(detail or "R package IHW is unavailable")
        if completed.returncode != 0:
            raise RuntimeError(detail or "R IHW comparison failed")
        adjusted = _read_vector(output_prefix.with_suffix(".adj.txt"))
        weights = _read_vector(output_prefix.with_suffix(".weights.txt"))
        rejection_count = int(
            output_prefix.with_suffix(".rejections.txt").read_text(encoding="utf-8")
        )
        version = (
            output_prefix.with_suffix(".version.txt")
            .read_text(encoding="utf-8")
            .strip()
        )
        groups = np.asarray(
            _read_vector(output_prefix.with_suffix(".groups.txt")), dtype=np.intp
        )
        folds = np.asarray(
            _read_vector(output_prefix.with_suffix(".folds.txt")), dtype=np.intp
        )
        fold_lambdas = _read_vector(output_prefix.with_suffix(".lambdas.txt"))
    if not version:
        raise RuntimeError("R IHW comparison did not report its package version")
    fit_result = FitResult(adjusted, weights, rejection_count, version)
    _validate_fit(fit_result, peer_input, config.alpha)
    _validated_input(
        PeerInput(
            dataset_id=peer_input.dataset_id,
            source_path=peer_input.source_path,
            provenance=peer_input.provenance,
            size=peer_input.size,
            seed=peer_input.seed,
            pvalues=peer_input.pvalues,
            covariates=peer_input.covariates,
            groups=groups,
            folds=folds,
            fold_lambdas=fold_lambdas,
        )
    )
    effective_nfolds = _r_output_fold_count(nbins, nfolds)
    if fold_lambdas.shape != (effective_nfolds,):
        raise RuntimeError(f"R fold lambdas must have shape ({effective_nfolds},)")
    if np.any(np.isnan(fold_lambdas)) or np.any(fold_lambdas < 0.0):
        raise RuntimeError("R fold lambdas must be nonnegative and not NaN")
    return RReferenceResult(fit_result, groups, folds, fold_lambdas)


def _r_output_fold_count(nbins: int, requested_nfolds: int) -> int:
    """Return R IHW's effective fold count for its one-bin BH shortcut."""

    return 1 if nbins == 1 else requested_nfolds


def _ihw_kwargs(peer_input: PeerInput, config: RunConfig) -> dict[str, object]:
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


def _from_ihw_result(result: object, alpha: float, version: str) -> FitResult:
    adjusted_value = getattr(result, "adj_pvalues", None)
    weights_value = getattr(result, "weights", None)
    if adjusted_value is None:
        raise RuntimeError("comparison result has no adj_pvalues")
    adjusted = np.asarray(adjusted_value, dtype=np.float64)
    weights = (
        None if weights_value is None else np.asarray(weights_value, dtype=np.float64)
    )
    return FitResult(
        adjusted,
        weights,
        int(np.sum(adjusted <= alpha)),
        version,
    )


def _validate_fit(result: FitResult, peer_input: PeerInput, alpha: float) -> None:
    adjusted = result.adjusted_pvalues
    if adjusted.shape != (peer_input.size,):
        raise RuntimeError(
            f"adjusted p-values must have shape ({peer_input.size},), got {adjusted.shape}"
        )
    if np.any(~np.isfinite(adjusted)) or np.any((adjusted < 0.0) | (adjusted > 1.0)):
        raise RuntimeError("adjusted p-values must be finite and lie in [0, 1]")
    if result.rejection_count != int(np.sum(adjusted <= alpha)):
        raise RuntimeError("rejection count does not match adjusted p-values")
    if result.weights is None:
        return
    if result.weights.shape != (peer_input.size,):
        raise RuntimeError(
            f"weights must have shape ({peer_input.size},), got {result.weights.shape}"
        )
    if np.any(~np.isfinite(result.weights)) or np.any(result.weights < 0.0):
        raise RuntimeError("weights must be finite and nonnegative")


def _load_reference_input(reference_id: str) -> PeerInput:
    record = load_reference(reference_id)
    frozen = record.peer_input
    return _validated_input(
        PeerInput(
            dataset_id=frozen.dataset_id,
            source_path=frozen.source_path,
            provenance=frozen.provenance,
            size=frozen.size,
            seed=frozen.seed,
            pvalues=frozen.pvalues,
            covariates=frozen.covariates,
        )
    )


def _reference_spec(reference_id: str) -> ReferenceSpec:
    for spec in REFERENCE_SPECS:
        if spec.reference_id == reference_id:
            return spec
    raise PeerDataError(f"unknown reference {reference_id!r}")


def _validate_reference_metadata(
    spec: ReferenceSpec, metadata: Mapping[str, object]
) -> None:
    expected: Mapping[str, object] = {
        "reference_id": spec.reference_id,
        "dataset_id": spec.dataset_id,
        "alpha": spec.alpha,
        "nbins": spec.nbins,
        "nfolds": spec.nfolds,
        "lambda_policy": spec.lambda_policy,
        "seed": spec.seed,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise PeerDataError(f"reference {spec.reference_id!r} has unexpected {key}")
    for key in ("provenance", "source_url", "source_license", "r_ihw_version"):
        if not str(metadata.get(key, "")).strip():
            raise PeerDataError(f"reference {spec.reference_id!r} has empty {key}")


def _validated_input(peer_input: PeerInput) -> PeerInput:
    size = peer_input.size
    if peer_input.pvalues.shape != (size,):
        raise PeerDataError(f"pvalues must have shape ({size},)")
    if peer_input.covariates.shape != (size,):
        raise PeerDataError(f"covariates must have shape ({size},)")
    if np.any(~np.isfinite(peer_input.pvalues)) or np.any(
        (peer_input.pvalues < 0.0) | (peer_input.pvalues > 1.0)
    ):
        raise PeerDataError("pvalues must be finite and lie in [0, 1]")
    if np.any(~np.isfinite(peer_input.covariates)):
        raise PeerDataError("covariates must be finite")
    if peer_input.truth_labels is not None and peer_input.truth_labels.shape != (size,):
        raise PeerDataError(f"truth_labels must have shape ({size},)")
    for name, labels in (
        ("groups", peer_input.groups),
        ("folds", peer_input.folds),
    ):
        if labels is None:
            continue
        if labels.shape != (size,):
            raise PeerDataError(f"{name} must have shape ({size},)")
        unique = np.unique(labels)
        if not np.array_equal(unique, np.arange(unique.size, dtype=np.intp)):
            raise PeerDataError(f"{name} labels must be contiguous from zero")
    return peer_input


def _archive_vector(
    archive: np.lib.npyio.NpzFile,
    key: str,
    dtype: type[np.float64 | np.intp],
    size: int | None = None,
) -> FloatArray | IntegerArray:
    value = archive[key]
    array = np.asarray(value, dtype=dtype)
    expected_size = array.size if size is None else size
    if array.ndim != 1 or array.shape != (expected_size,):
        raise PeerDataError(f"{key} must have shape ({expected_size},)")
    return array.copy()


def _archive_scalar(
    archive: np.lib.npyio.NpzFile, key: str, target: type[float | int | str]
) -> float | int | str:
    value = np.asarray(archive[key])
    if value.ndim != 0:
        raise PeerDataError(f"{key} must be scalar")
    return target(value.item())


def _reference_lambdas(
    archive: np.lib.npyio.NpzFile, prefix: str, nfolds: int
) -> FloatArray:
    value = np.asarray(archive[prefix + "fold_lambdas"], dtype=np.float64)
    if value.ndim != 1 or value.shape != (nfolds,):
        raise PeerDataError(f"fold_lambdas must have shape ({nfolds},)")
    if np.any(np.isnan(value)) or np.any(value < 0.0):
        raise PeerDataError("fold_lambdas must be nonnegative and not NaN")
    return value.copy()


@cache
def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "working tree"


@cache
def _importable(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def _read_vector(path: Path) -> FloatArray:
    return np.atleast_1d(np.loadtxt(path, dtype=np.float64))


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reference", choices=REFERENCE_IDS)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--nbins", type=_parse_nbins, default="auto")
    parser.add_argument("--nfolds", type=int)
    parser.add_argument("--lambda-policy", choices=("inf", "auto"), default="inf")
    parser.add_argument("--adjustment-type", choices=("bh", "bonferroni"), default="bh")
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


def _emit(document: dict[str, object], path: Path | None, quiet: bool) -> None:
    text = json.dumps(document, sort_keys=True, allow_nan=False)
    if path is None:
        if not quiet:
            print(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    if not quiet:
        print(f"{document['method_id']} status={document['status']} result={path}")


if __name__ == "__main__":
    raise SystemExit(main())
