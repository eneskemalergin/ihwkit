"""Adapt the optional public pyihw package to the peer result contract."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.data_contract import PeerInput
from tools.peers.runner import (
    AdapterUnavailable,
    FitResult,
    RunConfig,
    adapter_main,
    ihw_kwargs,
)

IMPLEMENTATION_VERSION = "pyihw-unpinned-adapter"


def fit(peer_input: PeerInput, config: RunConfig) -> FitResult:
    """Call a supported pyihw function when the package is installed."""

    module = _load_external_module()
    function = _find_function(module)
    if peer_input.groups is not None or peer_input.folds is not None:
        raise AdapterUnavailable(
            "pyihw adapter has no frozen groups and folds contract"
        )
    kwargs = ihw_kwargs(peer_input, config)
    kwargs.pop("groups", None)
    kwargs.pop("folds", None)
    kwargs.pop("fold_lambdas", None)
    try:
        result = function(
            pvalues=peer_input.pvalues,
            covariates=peer_input.covariates,
            **kwargs,
        )
    except TypeError as exc:
        raise AdapterUnavailable(
            f"installed pyihw API is incompatible with the adapter: {exc}"
        ) from exc
    adjusted_value = _result_value(
        result, ("adj_pvalues", "adjusted_pvalues", "adjusted_p")
    )
    if adjusted_value is None:
        raise AdapterUnavailable(
            "installed pyihw result has no recognized adjusted p-value field"
        )
    adjusted = np.asarray(adjusted_value, dtype=np.float64)
    weights_value = _result_value(result, ("weights", "ihw_weights"))
    weights = (
        None
        if weights_value is None
        else np.asarray(weights_value, dtype=np.float64)
    )
    return FitResult(adjusted, weights, int(np.sum(adjusted <= config.alpha)))


def _find_function(module: object) -> Callable[..., object]:
    for name in ("adjust_ihw", "ihw", "independent_hypothesis_weighting"):
        candidate = getattr(module, name, None)
        if callable(candidate):
            return cast(Callable[..., object], candidate)
    raise AdapterUnavailable(
        "installed pyihw package exposes no recognized fitting function"
    )


def _load_external_module() -> object:
    """Import pyihw without allowing this adapter file to shadow it."""

    adapter_directory = Path(__file__).resolve().parent
    original_path = list(sys.path)
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != adapter_directory
    ]
    try:
        if importlib.util.find_spec("pyihw") is None:
            raise AdapterUnavailable(
                "pyihw is not installed; install chi-raag/pyihw before running this peer"
            )
        return importlib.import_module("pyihw")
    finally:
        sys.path[:] = original_path


def _result_value(result: object, names: tuple[str, ...]) -> object | None:
    if isinstance(result, Mapping):
        for name in names:
            if name in result:
                return result[name]
        return None
    for name in names:
        value = getattr(result, name, None)
        if value is not None:
            return value
    return None


if __name__ == "__main__":
    raise SystemExit(adapter_main("pyihw", IMPLEMENTATION_VERSION, fit))
