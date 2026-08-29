"""Check production fits against repository-owned data-contract fixtures."""

from __future__ import annotations

import numpy as np

from ihw import adjust_ihw
from tools.data_contract import load_peer_input


def test_production_fit_on_contract_fixture() -> None:
    """The production path returns finite results for a normalized fixture."""

    peer_input = load_peer_input("sim_500_seed42")
    result = adjust_ihw(
        peer_input.pvalues,
        peer_input.covariates,
        0.1,
        nbins=4,
        nfolds=1,
        seed=42,
    )
    assert np.all(np.isfinite(result.weights))
    assert np.all(np.isfinite(result.adj_pvalues))


def test_auto_lambda_on_contract_fixture_is_finite() -> None:
    """The auto-lambda production path remains finite on the larger fixture."""

    peer_input = load_peer_input("sim_5000_seed42")
    result = adjust_ihw(
        peer_input.pvalues,
        peer_input.covariates,
        0.1,
        nbins="auto",
        nfolds=5,
        lambdas="auto",
        seed=42,
    )
    assert result.nbins == 3
    assert result.nfolds == 5
    assert np.all(np.isfinite(result.weights))
    assert np.all(np.isfinite(result.adj_pvalues))
    assert np.sum(result.adj_pvalues <= 0.1) == 155
