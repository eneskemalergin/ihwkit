from pathlib import Path

import numpy as np
import pytest

from ihw import _numba_importable, _p_adjust, _safe_divide, adjust_ihw

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


def test_numpy_is_finite_on_n1_oracle() -> None:
    data = np.load(ORACLE)
    p = np.asarray(data["p"], dtype=np.float64)
    x = np.asarray(data["x"], dtype=np.float64)
    groups = np.asarray(data["groups"], dtype=np.intp)
    result = adjust_ihw(
        p, x, 0.1, nbins=4, nfolds=1, groups=groups, seed=1, lp_backend="numpy"
    )
    assert np.all(np.isfinite(result.weights))
    assert np.all(np.isfinite(result.adj_pvalues))
    np.testing.assert_allclose(np.mean(result.weights), 1.0, atol=1e-8)
    max_abs_adj = float(np.max(np.abs(result.adj_pvalues - data["adj_pvalues"])))
    max_abs_w = float(np.max(np.abs(result.weights - data["weights"])))
    assert np.isfinite(max_abs_adj)
    assert np.isfinite(max_abs_w)


def test_numpy_tracks_highs_on_n1_oracle() -> None:
    data = np.load(ORACLE)
    p = np.asarray(data["p"], dtype=np.float64)
    x = np.asarray(data["x"], dtype=np.float64)
    groups = np.asarray(data["groups"], dtype=np.intp)
    highs = adjust_ihw(
        p, x, 0.1, nbins=4, nfolds=1, groups=groups, seed=1, lp_backend="highs"
    )
    numpy_fit = adjust_ihw(
        p, x, 0.1, nbins=4, nfolds=1, groups=groups, seed=1, lp_backend="numpy"
    )
    np.testing.assert_allclose(numpy_fit.adj_pvalues, highs.adj_pvalues, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(numpy_fit.weights, highs.weights, atol=1e-8, rtol=1e-6)


def test_numpy_is_finite_on_n5_oracle() -> None:
    data = np.load(ORACLE_N5)
    p = np.asarray(data["p"], dtype=np.float64)
    x = np.asarray(data["x"], dtype=np.float64)
    groups = np.asarray(data["groups"], dtype=np.intp)
    folds = np.asarray(data["folds"], dtype=np.intp)
    result = adjust_ihw(
        p,
        x,
        0.1,
        nbins=4,
        nfolds=5,
        groups=groups,
        folds=folds,
        seed=1,
        lp_backend="numpy",
    )
    assert np.all(np.isfinite(result.weights))
    assert np.all(np.isfinite(result.adj_pvalues))
    np.testing.assert_allclose(np.mean(result.weights), 1.0, atol=1e-8)
    np.testing.assert_array_equal(result.folds, folds)
    max_abs_adj = float(np.max(np.abs(result.adj_pvalues - data["adj_pvalues"])))
    max_abs_w = float(np.max(np.abs(result.weights - data["weights"])))
    assert np.isfinite(max_abs_adj)
    assert np.isfinite(max_abs_w)


def test_numpy_numba_is_finite_on_n1_oracle() -> None:
    if not _numba_importable():
        pytest.skip("numba is not installed")
    data = np.load(ORACLE)
    p = np.asarray(data["p"], dtype=np.float64)
    x = np.asarray(data["x"], dtype=np.float64)
    groups = np.asarray(data["groups"], dtype=np.intp)
    off = adjust_ihw(
        p,
        x,
        0.1,
        nbins=4,
        nfolds=1,
        groups=groups,
        seed=1,
        lp_backend="numpy",
        use_numba=False,
    )
    on = adjust_ihw(
        p,
        x,
        0.1,
        nbins=4,
        nfolds=1,
        groups=groups,
        seed=1,
        lp_backend="numpy",
        use_numba=True,
    )
    assert np.all(np.isfinite(on.weights))
    assert np.all(np.isfinite(on.adj_pvalues))
    np.testing.assert_allclose(np.mean(on.weights), 1.0, atol=1e-8)
    np.testing.assert_allclose(on.adj_pvalues, off.adj_pvalues, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(on.weights, off.weights, atol=1e-8, rtol=1e-6)
    max_abs_adj = float(np.max(np.abs(on.adj_pvalues - data["adj_pvalues"])))
    max_abs_w = float(np.max(np.abs(on.weights - data["weights"])))
    assert np.isfinite(max_abs_adj)
    assert np.isfinite(max_abs_w)


def test_numpy_numba_tracks_numpy_on_n5_oracle() -> None:
    if not _numba_importable():
        pytest.skip("numba is not installed")
    data = np.load(ORACLE_N5)
    p = np.asarray(data["p"], dtype=np.float64)
    x = np.asarray(data["x"], dtype=np.float64)
    groups = np.asarray(data["groups"], dtype=np.intp)
    folds = np.asarray(data["folds"], dtype=np.intp)
    off = adjust_ihw(
        p,
        x,
        0.1,
        nbins=4,
        nfolds=5,
        groups=groups,
        folds=folds,
        seed=1,
        lp_backend="numpy",
        use_numba=False,
    )
    on = adjust_ihw(
        p,
        x,
        0.1,
        nbins=4,
        nfolds=5,
        groups=groups,
        folds=folds,
        seed=1,
        lp_backend="numpy",
        use_numba=True,
    )
    assert np.all(np.isfinite(on.weights))
    assert np.all(np.isfinite(on.adj_pvalues))
    np.testing.assert_allclose(on.adj_pvalues, off.adj_pvalues, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(on.weights, off.weights, atol=1e-8, rtol=1e-6)
    max_abs_adj = float(np.max(np.abs(on.adj_pvalues - data["adj_pvalues"])))
    max_abs_w = float(np.max(np.abs(on.weights - data["weights"])))
    assert np.isfinite(max_abs_adj)
    assert np.isfinite(max_abs_w)


def test_highs_still_matches_n1_and_n5_r_oracles() -> None:
    data = np.load(ORACLE)
    p = np.asarray(data["p"], dtype=np.float64)
    x = np.asarray(data["x"], dtype=np.float64)
    groups = np.asarray(data["groups"], dtype=np.intp)
    highs = adjust_ihw(
        p, x, 0.1, nbins=4, nfolds=1, groups=groups, seed=1, lp_backend="highs"
    )
    np.testing.assert_allclose(highs.adj_pvalues, data["adj_pvalues"], atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(highs.weights, data["weights"], atol=1e-8, rtol=1e-6)
    data5 = np.load(ORACLE_N5)
    p5 = np.asarray(data5["p"], dtype=np.float64)
    x5 = np.asarray(data5["x"], dtype=np.float64)
    g5 = np.asarray(data5["groups"], dtype=np.intp)
    f5 = np.asarray(data5["folds"], dtype=np.intp)
    highs5 = adjust_ihw(
        p5, x5, 0.1, nbins=4, nfolds=5, groups=g5, folds=f5, seed=1, lp_backend="highs"
    )
    np.testing.assert_allclose(highs5.adj_pvalues, data5["adj_pvalues"], atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(highs5.weights, data5["weights"], atol=1e-8, rtol=1e-6)


def test_bh_on_r_weights_matches_n1_oracle() -> None:
    data = np.load(ORACLE)
    p = np.asarray(data["p"], dtype=np.float64)
    weights = np.asarray(data["weights"], dtype=np.float64)
    adj = _p_adjust(_safe_divide(p, weights), "fdr_bh")
    np.testing.assert_allclose(adj, data["adj_pvalues"], atol=1e-8, rtol=1e-6)
    assert int(np.sum(adj <= 0.1)) == int(data["rejections"])


def test_bh_on_r_weights_matches_n5_oracle() -> None:
    data = np.load(ORACLE_N5)
    p = np.asarray(data["p"], dtype=np.float64)
    weights = np.asarray(data["weights"], dtype=np.float64)
    adj = _p_adjust(_safe_divide(p, weights), "fdr_bh")
    np.testing.assert_allclose(adj, data["adj_pvalues"], atol=1e-8, rtol=1e-6)
    assert int(np.sum(adj <= 0.1)) == int(data["rejections"])
