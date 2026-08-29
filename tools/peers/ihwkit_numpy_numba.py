"""Run the NumPy plus Numba production IHW path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ihw import adjust_ihw
from tools.data_contract import PeerInput
from tools.peers.runner import (
    AdapterUnavailable,
    FitResult,
    RunConfig,
    adapter_main,
    fit_result_from_ihw,
    ihw_kwargs,
)

IMPLEMENTATION_VERSION = "ihwkit-production-numpy-numba"


def fit(peer_input: PeerInput, config: RunConfig) -> FitResult:
    """Fit the single public NumPy plus Numba implementation."""

    if importlib.util.find_spec("numba") is None:
        raise AdapterUnavailable("numba is not installed")
    kwargs = ihw_kwargs(peer_input, config)
    result = adjust_ihw(
        peer_input.pvalues,
        peer_input.covariates,
        **kwargs,
    )
    return fit_result_from_ihw(result, config.alpha)


if __name__ == "__main__":
    raise SystemExit(
        adapter_main("ihwkit_numpy_numba", IMPLEMENTATION_VERSION, fit)
    )
