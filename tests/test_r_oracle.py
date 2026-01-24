from pathlib import Path

import numpy as np

from ihw import adjust_ihw

ORACLE = Path(__file__).resolve().parent / "fixtures" / "r_inf_n1.npz"
ORACLE_N5 = Path(__file__).resolve().parent / "fixtures" / "r_inf_n5.npz"


def test_python_replay_matches_r_oracle() -> None:
    data = np.load(ORACLE)
    assert set(data.files) == {
        "p",
        "x",
        "groups",
        "adj_pvalues",
        "weights",
        "rejections",
    }
    p = np.asarray(data["p"], dtype=np.float64)
    x = np.asarray(data["x"], dtype=np.float64)
    groups = np.asarray(data["groups"], dtype=np.intp)
    adj = np.asarray(data["adj_pvalues"], dtype=np.float64)
    weights = np.asarray(data["weights"], dtype=np.float64)
    assert p.shape == (2000,)
    assert x.shape == (2000,)
    assert groups.shape == (2000,)
    assert adj.shape == (2000,)
    assert weights.shape == (2000,)
    np.testing.assert_array_equal(np.unique(groups), np.arange(4))
    np.testing.assert_allclose(np.mean(weights), 1.0)
    result = adjust_ihw(p, x, 0.1, nbins=4, nfolds=1, groups=groups, seed=1)
    np.testing.assert_allclose(result.adj_pvalues, adj, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(result.weights, weights, atol=1e-8, rtol=1e-6)
    ihw_rej = int(np.sum(result.adj_pvalues <= 0.1))
    assert ihw_rej == int(data["rejections"])


def test_python_replay_matches_five_fold_r_oracle() -> None:
    data = np.load(ORACLE_N5)
    p = np.asarray(data["p"], dtype=np.float64)
    x = np.asarray(data["x"], dtype=np.float64)
    groups = np.asarray(data["groups"], dtype=np.intp)
    folds = np.asarray(data["folds"], dtype=np.intp)
    result = adjust_ihw(p, x, 0.1, nbins=4, nfolds=5, groups=groups, folds=folds, seed=1)
    np.testing.assert_allclose(result.adj_pvalues, data["adj_pvalues"], atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(result.weights, data["weights"], atol=1e-8, rtol=1e-6)
    ihw_rej = int(np.sum(result.adj_pvalues <= 0.1))
    assert ihw_rej == int(data["rejections"])
    np.testing.assert_array_equal(result.folds, folds)
