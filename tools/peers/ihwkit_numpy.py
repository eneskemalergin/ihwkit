"""Run the pinned NumPy-only IHW peer baseline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.data_contract import PeerInput
from tools.peers import legacy_ihw
from tools.peers.runner import (
    FitResult,
    RunConfig,
    adapter_main,
    fit_result_from_ihw,
    ihw_kwargs,
)

IMPLEMENTATION_VERSION = f"{legacy_ihw.__version__}+numpy"


def fit(peer_input: PeerInput, config: RunConfig) -> FitResult:
    """Fit the pinned dense NumPy simplex baseline."""

    kwargs = ihw_kwargs(peer_input, config)
    kwargs["lp_backend"] = "numpy"
    kwargs["use_numba"] = False
    result = legacy_ihw.adjust_ihw(
        peer_input.pvalues,
        peer_input.covariates,
        **kwargs,
    )
    return fit_result_from_ihw(result, config.alpha)


if __name__ == "__main__":
    raise SystemExit(
        adapter_main("ihwkit_numpy", IMPLEMENTATION_VERSION, fit)
    )
