"""Load and validate the versioned peer-method input contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.intp]
BooleanArray = NDArray[np.bool_]

MANIFEST_VERSION = 1
PEER_INPUT_SCHEMA_VERSION = "1"
_MANIFEST_PATH = Path("data/manifest.json")


class DataContractError(ValueError):
    """Raised when a manifest, fixture, or oracle violates its contract."""


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Describe one normalized peer input fixture.

    Parameters
    ----------
    dataset_id : str
        Stable identifier used by peer adapters.
    path : str
        Repository-relative path to the fixture.
    kind : str
        Dataset classification such as ``synthetic`` or ``real-local``.
    provenance : str
        Human-readable provenance statement recorded in the manifest.
    size : int
        Number of hypotheses expected in the fixture.
    seed : int or None
        Generation seed when one is known.
    release_eligible : bool
        Whether the fixture is allowed in the comparison set.
    """

    dataset_id: str
    path: str
    kind: str
    provenance: str
    size: int
    seed: int | None
    release_eligible: bool


@dataclass(frozen=True, slots=True)
class OracleSpec:
    """Describe one frozen reference result file.

    Parameters
    ----------
    oracle_id : str
        Stable identifier used by parity checks.
    dataset_id : str
        Identifier of the fixture contained in the oracle.
    path : str
        Repository-relative path to the NumPy oracle file.
    metadata_path : str
        Repository-relative path to the JSON metadata file.
    config : str
        Frozen configuration label such as ``inf_n5`` or ``auto``.
    nfolds : int
        Number of frozen outer folds.
    lambda_policy : str
        Lambda policy used to create the oracle.
    release_eligible : bool
        Whether the oracle is allowed in the comparison set.
    """

    oracle_id: str
    dataset_id: str
    path: str
    metadata_path: str
    config: str
    nfolds: int
    lambda_policy: str
    release_eligible: bool


@dataclass(frozen=True, slots=True)
class Manifest:
    """Represent the validated data manifest."""

    manifest_version: int
    schema_version: str
    last_reviewed: str
    datasets: tuple[DatasetSpec, ...]
    oracles: tuple[OracleSpec, ...]


@dataclass(frozen=True, slots=True)
class PeerInput:
    """Represent one normalized input and optional frozen partition.

    Parameters
    ----------
    dataset_id : str
        Stable dataset identifier.
    schema_version : str
        Version of the normalized peer input schema.
    source_path : str
        Repository-relative path of the input fixture.
    provenance : str
        Provenance statement copied from the manifest.
    size : int
        Number of hypotheses.
    seed : int or None
        Input generation seed when known.
    pvalues : numpy.ndarray
        One-dimensional float64 p-values in the closed unit interval.
    covariates : numpy.ndarray
        One-dimensional finite float64 covariates.
    truth_labels : numpy.ndarray or None
        Optional one-dimensional boolean labels.
    groups : numpy.ndarray or None
        Optional zero-based frozen group labels.
    folds : numpy.ndarray or None
        Optional zero-based frozen fold labels.
    fold_lambdas : numpy.ndarray or None
        Optional frozen lambda value for each fold.
    oracle_id : str or None
        Oracle identifier when a frozen oracle was loaded.
    """

    dataset_id: str
    schema_version: str
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
    oracle_id: str | None = None


@dataclass(frozen=True, slots=True)
class OracleRecord:
    """Represent frozen R outputs and the matching normalized input."""

    peer_input: PeerInput
    metadata: Mapping[str, object]
    r_rejections: int
    adjusted_pvalues: FloatArray
    weights: FloatArray


def repository_root() -> Path:
    """Return the repository root inferred from this module's location."""

    return Path(__file__).resolve().parents[1]


def load_manifest(root: Path | None = None) -> Manifest:
    """Load and validate ``data/manifest.json``.

    Parameters
    ----------
    root : pathlib.Path or None, optional
        Repository root. The module's parent repository is used by default.

    Returns
    -------
    Manifest
        Parsed dataset and oracle specifications.

    Raises
    ------
    DataContractError
        If the manifest is missing or malformed.
    """

    base = repository_root() if root is None else Path(root)
    path = base / _MANIFEST_PATH
    if not path.is_file():
        raise DataContractError(f"manifest not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"could not read manifest {path}: {exc}") from exc
    document = _mapping(raw, "manifest")
    manifest_version = _integer(document.get("manifest_version"), "manifest_version")
    schema_version = _string(document.get("schema_version"), "schema_version")
    last_reviewed = _string(document.get("last_reviewed"), "last_reviewed")
    if manifest_version != MANIFEST_VERSION:
        raise DataContractError(
            f"unsupported manifest_version {manifest_version}; expected {MANIFEST_VERSION}"
        )
    if schema_version != PEER_INPUT_SCHEMA_VERSION:
        raise DataContractError(
            f"unsupported schema_version {schema_version!r}; expected {PEER_INPUT_SCHEMA_VERSION!r}"
        )
    dataset_values = document.get("datasets")
    oracle_values = document.get("oracles")
    if not isinstance(dataset_values, list) or not isinstance(oracle_values, list):
        raise DataContractError("manifest datasets and oracles must be lists")
    datasets = tuple(_dataset_spec(_mapping(value, "dataset")) for value in dataset_values)
    oracles = tuple(_oracle_spec(_mapping(value, "oracle")) for value in oracle_values)
    _ensure_unique((spec.dataset_id for spec in datasets), "dataset")
    _ensure_unique((spec.oracle_id for spec in oracles), "oracle")
    dataset_ids = {spec.dataset_id for spec in datasets}
    if any(spec.dataset_id not in dataset_ids for spec in oracles):
        raise DataContractError("oracle references an unknown dataset")
    return Manifest(
        manifest_version,
        schema_version,
        last_reviewed,
        datasets,
        oracles,
    )


def load_peer_input(
    dataset_id: str,
    *,
    oracle_id: str | None = None,
    root: Path | None = None,
) -> PeerInput:
    """Load one fixture and optionally attach a frozen oracle partition.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier from the manifest.
    oracle_id : str or None, optional
        Oracle identifier whose groups, folds, and lambda values should be attached.
    root : pathlib.Path or None, optional
        Repository root. The module's parent repository is used by default.

    Returns
    -------
    PeerInput
        Float64, one-dimensional arrays and validated optional fields.

    Raises
    ------
    DataContractError
        If the dataset, oracle, file, or array contract is invalid.
    """

    base = repository_root() if root is None else Path(root)
    manifest = load_manifest(base)
    dataset = _find_dataset(manifest, dataset_id)
    fixture_path = _required_path(base, dataset.path, "fixture")
    with np.load(fixture_path, allow_pickle=False) as archive:
        pvalues = _float_vector(archive, "pvalues", dataset.size, "pvalues")
        covariates = _float_vector(archive, "covariates", dataset.size, "covariates")
        truth_labels = _optional_bool_vector(archive, "truth_labels", dataset.size)
    _validate_input_arrays(pvalues, covariates, dataset.size)
    peer_input = PeerInput(
        dataset_id=dataset.dataset_id,
        schema_version=manifest.schema_version,
        source_path=dataset.path,
        provenance=dataset.provenance,
        size=dataset.size,
        seed=dataset.seed,
        pvalues=pvalues,
        covariates=covariates,
        truth_labels=truth_labels,
    )
    if oracle_id is None:
        return peer_input
    oracle = _find_oracle(manifest, oracle_id)
    if oracle.dataset_id != dataset_id:
        raise DataContractError(
            f"oracle {oracle_id!r} belongs to {oracle.dataset_id!r}, not {dataset_id!r}"
        )
    return _attach_oracle(base, peer_input, oracle)


def load_oracle(oracle_id: str, root: Path | None = None) -> OracleRecord:
    """Load one frozen oracle and its normalized input.

    Parameters
    ----------
    oracle_id : str
        Oracle identifier from the manifest.
    root : pathlib.Path or None, optional
        Repository root. The module's parent repository is used by default.

    Returns
    -------
    OracleRecord
        Frozen input partition and R adjusted p-values and weights.

    Raises
    ------
    DataContractError
        If the oracle file is unavailable or violates the manifest contract.
    """

    base = repository_root() if root is None else Path(root)
    manifest = load_manifest(base)
    oracle = _find_oracle(manifest, oracle_id)
    peer_input = load_peer_input(oracle.dataset_id, oracle_id=oracle_id, root=base)
    oracle_path = _required_path(base, oracle.path, "oracle")
    metadata_path = _required_path(base, oracle.metadata_path, "oracle metadata")
    try:
        metadata_value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"could not read oracle metadata {metadata_path}: {exc}") from exc
    metadata = _mapping(metadata_value, "oracle metadata")
    with np.load(oracle_path, allow_pickle=False) as archive:
        adjusted_pvalues = _float_vector(archive, "r_adj_pvalues", peer_input.size, "r_adj_pvalues")
        weights = _float_vector(archive, "r_weights", peer_input.size, "r_weights")
        rejection_values = archive.get("r_rejections")
        if rejection_values is None or rejection_values.ndim != 0:
            raise DataContractError(f"oracle {oracle_id!r} must contain scalar r_rejections")
        r_rejections = int(np.asarray(rejection_values).item())
    if not np.all(np.isfinite(adjusted_pvalues)) or not np.all(np.isfinite(weights)):
        raise DataContractError(f"oracle {oracle_id!r} contains nonfinite outputs")
    return OracleRecord(peer_input, metadata, r_rejections, adjusted_pvalues, weights)


def _dataset_spec(document: dict[str, object]) -> DatasetSpec:
    return DatasetSpec(
        dataset_id=_string(document.get("dataset_id"), "dataset_id"),
        path=_string(document.get("path"), "dataset path"),
        kind=_string(document.get("kind"), "dataset kind"),
        provenance=_string(document.get("provenance"), "dataset provenance"),
        size=_integer(document.get("size"), "dataset size"),
        seed=_optional_integer(document.get("seed"), "dataset seed"),
        release_eligible=_boolean(document.get("release_eligible"), "dataset release_eligible"),
    )


def _oracle_spec(document: dict[str, object]) -> OracleSpec:
    return OracleSpec(
        oracle_id=_string(document.get("oracle_id"), "oracle_id"),
        dataset_id=_string(document.get("dataset_id"), "oracle dataset_id"),
        path=_string(document.get("path"), "oracle path"),
        metadata_path=_string(document.get("metadata_path"), "oracle metadata_path"),
        config=_string(document.get("config"), "oracle config"),
        nfolds=_integer(document.get("nfolds"), "oracle nfolds"),
        lambda_policy=_string(document.get("lambda_policy"), "oracle lambda_policy"),
        release_eligible=_boolean(
            document.get("release_eligible"), "oracle release_eligible"
        ),
    )


def _attach_oracle(root: Path, peer_input: PeerInput, oracle: OracleSpec) -> PeerInput:
    oracle_path = _required_path(root, oracle.path, "oracle")
    with np.load(oracle_path, allow_pickle=False) as archive:
        oracle_pvalues = _float_vector(archive, "pvalues", peer_input.size, "oracle pvalues")
        oracle_covariates = _float_vector(
            archive, "covariates", peer_input.size, "oracle covariates"
        )
        groups = _integer_vector(archive, "groups", peer_input.size, "groups")
        folds = _integer_vector(archive, "folds", peer_input.size, "folds")
        fold_lambdas = _oracle_lambdas(archive, oracle)
    if not np.array_equal(peer_input.pvalues, oracle_pvalues):
        raise DataContractError(f"oracle {oracle.oracle_id!r} pvalues do not match fixture")
    if not np.array_equal(peer_input.covariates, oracle_covariates):
        raise DataContractError(
            f"oracle {oracle.oracle_id!r} covariates do not match fixture"
        )
    _validate_labels(groups, "groups")
    _validate_labels(folds, "folds")
    if int(np.max(folds, initial=-1)) + 1 != oracle.nfolds:
        raise DataContractError(
            f"oracle {oracle.oracle_id!r} folds do not match nfolds={oracle.nfolds}"
        )
    return PeerInput(
        dataset_id=peer_input.dataset_id,
        schema_version=peer_input.schema_version,
        source_path=peer_input.source_path,
        provenance=peer_input.provenance,
        size=peer_input.size,
        seed=peer_input.seed,
        pvalues=peer_input.pvalues,
        covariates=peer_input.covariates,
        truth_labels=peer_input.truth_labels,
        groups=groups,
        folds=folds,
        fold_lambdas=fold_lambdas,
        oracle_id=oracle.oracle_id,
    )


def _oracle_lambdas(
    archive: np.lib.npyio.NpzFile, oracle: OracleSpec
) -> FloatArray | None:
    value = archive.get("fold_lambdas")
    if value is None:
        if oracle.lambda_policy == "inf":
            return np.full(oracle.nfolds, np.inf, dtype=np.float64)
        raise DataContractError(f"oracle {oracle.oracle_id!r} has no fold_lambdas")
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 0 and np.isnan(raw.item()) and oracle.lambda_policy == "inf":
        return np.full(oracle.nfolds, np.inf, dtype=np.float64)
    if raw.ndim == 1 and np.any(np.isnan(raw)):
        if oracle.lambda_policy == "inf" and np.all(np.isnan(raw)):
            return np.full(oracle.nfolds, np.inf, dtype=np.float64)
        if oracle.lambda_policy == "auto":
            return None
    if raw.ndim != 1 or raw.shape[0] != oracle.nfolds:
        raise DataContractError(
            f"oracle {oracle.oracle_id!r} fold_lambdas must have shape ({oracle.nfolds},)"
        )
    if np.any(np.isnan(raw)) or np.any(raw < 0.0):
        raise DataContractError(f"oracle {oracle.oracle_id!r} has invalid fold_lambdas")
    return raw.copy()


def _required_path(root: Path, relative_path: str, label: str) -> Path:
    path = root / relative_path
    if not path.is_file():
        raise DataContractError(f"{label} not found: {path}")
    return path


def _float_vector(
    archive: np.lib.npyio.NpzFile, key: str, size: int, label: str
) -> FloatArray:
    value = archive.get(key)
    if value is None:
        raise DataContractError(f"missing {label} array")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != size:
        raise DataContractError(f"{label} must have shape ({size},)")
    return array.copy()


def _integer_vector(
    archive: np.lib.npyio.NpzFile, key: str, size: int, label: str
) -> IntegerArray:
    value = archive.get(key)
    if value is None:
        raise DataContractError(f"missing {label} array")
    array = np.asarray(value, dtype=np.intp)
    if array.ndim != 1 or array.shape[0] != size:
        raise DataContractError(f"{label} must have shape ({size},)")
    return array.copy()


def _optional_bool_vector(
    archive: np.lib.npyio.NpzFile, key: str, size: int
) -> BooleanArray | None:
    value = archive.get(key)
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != size:
        raise DataContractError(f"{key} must have shape ({size},)")
    if not np.all(np.isin(array, [False, True, 0, 1])):
        raise DataContractError(f"{key} must contain boolean values")
    return np.asarray(array, dtype=np.bool_).copy()


def _validate_input_arrays(
    pvalues: FloatArray, covariates: FloatArray, size: int
) -> None:
    if pvalues.shape != (size,) or covariates.shape != (size,):
        raise DataContractError("input arrays do not match manifest size")
    if np.any(~np.isfinite(pvalues)) or np.any((pvalues < 0.0) | (pvalues > 1.0)):
        raise DataContractError("pvalues must be finite and lie in [0, 1]")
    if np.any(~np.isfinite(covariates)):
        raise DataContractError("covariates must be finite")


def _validate_labels(labels: IntegerArray, label: str) -> None:
    if labels.size == 0 or np.any(labels < 0):
        raise DataContractError(f"{label} must contain nonnegative labels")
    unique = np.unique(labels)
    if not np.array_equal(unique, np.arange(unique.size, dtype=np.intp)):
        raise DataContractError(f"{label} labels must be contiguous from zero")


def _find_dataset(manifest: Manifest, dataset_id: str) -> DatasetSpec:
    for spec in manifest.datasets:
        if spec.dataset_id == dataset_id:
            return spec
    raise DataContractError(f"unknown dataset_id {dataset_id!r}")


def _find_oracle(manifest: Manifest, oracle_id: str) -> OracleSpec:
    for spec in manifest.oracles:
        if spec.oracle_id == oracle_id:
            return spec
    raise DataContractError(f"unknown oracle_id {oracle_id!r}")


def _ensure_unique(values: object, label: str) -> None:
    identifiers = tuple(cast(tuple[str, ...], values))
    if len(set(identifiers)) != len(identifiers):
        raise DataContractError(f"duplicate {label} identifier")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise DataContractError(f"{label} must be an object with string keys")
    return cast(dict[str, object], value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DataContractError(f"{label} must be a nonempty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataContractError(f"{label} must be an integer")
    return value


def _optional_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise DataContractError(f"{label} must be boolean")
    return value


