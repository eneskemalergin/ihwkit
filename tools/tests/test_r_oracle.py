import numpy as np

from ihw import IHWResult, _p_adjust, _safe_divide, adjust_ihw
from tools.data_contract import OracleRecord, load_oracle

ORACLE_IDS = ("sim_5000_inf_n1", "sim_5000_inf_n5")


def _fit_oracle(record: OracleRecord) -> IHWResult:
    """Fit the production path with partitions from one frozen oracle."""

    peer_input = record.peer_input
    assert peer_input.groups is not None
    assert peer_input.folds is not None
    kwargs: dict[str, object] = {
        "nbins": int(np.max(peer_input.groups)) + 1,
        "nfolds": int(np.max(peer_input.folds)) + 1,
        "groups": peer_input.groups,
        "folds": peer_input.folds,
        "seed": peer_input.seed,
    }
    if peer_input.fold_lambdas is not None:
        kwargs["fold_lambdas"] = peer_input.fold_lambdas
    return adjust_ihw(peer_input.pvalues, peer_input.covariates, 0.1, **kwargs)


def test_production_path_matches_frozen_r_oracles() -> None:
    for oracle_id in ORACLE_IDS:
        record = load_oracle(oracle_id)
        result = _fit_oracle(record)
        np.testing.assert_allclose(
            result.adj_pvalues, record.adjusted_pvalues, atol=1e-8, rtol=1e-6
        )
        np.testing.assert_allclose(
            result.weights, record.weights, atol=1e-8, rtol=1e-6
        )
        assert int(np.sum(result.adj_pvalues <= 0.1)) == record.r_rejections
        np.testing.assert_array_equal(result.folds, record.peer_input.folds)


def test_bh_on_r_weights_matches_frozen_oracles() -> None:
    for oracle_id in ORACLE_IDS:
        record = load_oracle(oracle_id)
        adj = _p_adjust(
            _safe_divide(record.peer_input.pvalues, record.weights), "fdr_bh"
        )
        np.testing.assert_allclose(
            adj, record.adjusted_pvalues, atol=1e-8, rtol=1e-6
        )
        assert int(np.sum(adj <= 0.1)) == record.r_rejections
