import inspect
from math import erf, sqrt

import numpy as np
import pytest

from ihw import (
    IHWValidationError,
    _p_adjust,
    _solve_lp_numpy,
    _thresholds_to_weights,
    adjust_ihw,
)

_P = np.array([0.1, 0.2, 0.3, 0.4])
_X = np.array([1.0, 2.0, 3.0, 4.0])


def _normal_survival(values: np.ndarray) -> np.ndarray:
    return np.fromiter(
        (0.5 * (1.0 - erf(float(value) / sqrt(2.0))) for value in values),
        dtype=np.float64,
        count=values.size,
    )

def test_empty_pvalues_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        adjust_ihw(np.array([]), np.array([]), 0.1)

def test_nan_pvalues_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        adjust_ihw(np.array([0.1, np.nan]), np.array([1.0, 2.0]), 0.1)

def test_inf_pvalues_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        adjust_ihw(np.array([0.1, np.inf]), np.array([1.0, 2.0]), 0.1)

def test_nan_covariates_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        adjust_ihw(np.array([0.1, 0.2]), np.array([1.0, np.nan]), 0.1)

def test_inf_covariates_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        adjust_ihw(np.array([0.1, 0.2]), np.array([1.0, np.inf]), 0.1)

def test_pvalue_below_zero_raises() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        adjust_ihw(np.array([-0.1, 0.2]), np.array([1.0, 2.0]), 0.1)

def test_pvalue_above_one_raises() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        adjust_ihw(np.array([0.1, 1.5]), np.array([1.0, 2.0]), 0.1)

def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        adjust_ihw(_P, np.array([1.0, 2.0]), 0.1)

def test_unknown_adjustment_type_raises() -> None:
    with pytest.raises(ValueError, match="adjustment_type"):
        adjust_ihw(_P, _X, 0.1, adjustment_type="by")

def test_unknown_covariate_type_raises() -> None:
    with pytest.raises(ValueError, match="covariate_type"):
        adjust_ihw(_P, _X, 0.1, covariate_type="cyclic")

def test_validation_error_is_named() -> None:
    with pytest.raises(IHWValidationError, match="empty"):
        adjust_ihw(np.array([]), np.array([]), 0.1)
    with pytest.raises(ValueError, match="empty"):
        adjust_ihw(np.array([]), np.array([]), 0.1)

def test_nfolds_not_positive_raises() -> None:
    with pytest.raises(ValueError, match="nfolds"):
        adjust_ihw(_P, _X, 0.1, nfolds=0)
    with pytest.raises(ValueError, match="nfolds"):
        adjust_ihw(_P, _X, 0.1, nfolds=-3)

def test_nfolds_internal_not_positive_raises() -> None:
    with pytest.raises(ValueError, match="nfolds_internal"):
        adjust_ihw(_P, _X, 0.1, nfolds_internal=0)
    with pytest.raises(ValueError, match="nfolds_internal"):
        adjust_ihw(_P, _X, 0.1, nfolds_internal=-1)

def test_nsplits_internal_not_positive_raises() -> None:
    with pytest.raises(ValueError, match="nsplits_internal"):
        adjust_ihw(_P, _X, 0.1, nsplits_internal=0)
    with pytest.raises(ValueError, match="nsplits_internal"):
        adjust_ihw(_P, _X, 0.1, nsplits_internal=-2)

def test_nbins_not_positive_raises() -> None:
    with pytest.raises(ValueError, match="nbins"):
        adjust_ihw(_P, _X, 0.1, nbins=0)
    with pytest.raises(ValueError, match="nbins"):
        adjust_ihw(_P, _X, 0.1, nbins=-2)

def test_auto_nbins_below_1500() -> None:
    rng = np.random.default_rng(0)
    p_small = rng.uniform(size=1499)
    x_small = rng.uniform(size=1499)
    assert adjust_ihw(p_small, x_small, 0.1, seed=1).nbins == 1
    p_edge = rng.uniform(size=1500)
    x_edge = rng.uniform(size=1500)
    assert adjust_ihw(p_edge, x_edge, 0.1, seed=1).nbins == 1
    p_mid = rng.uniform(size=3000)
    x_mid = rng.uniform(size=3000)
    assert adjust_ihw(p_mid, x_mid, 0.1, seed=1).nbins == 2

def test_pvalue_not_1d_raises() -> None:
    with pytest.raises(ValueError, match="1-d"):
        adjust_ihw(np.ones((4, 1)), _X, 0.1)

def test_covariate_not_1d_raises() -> None:
    with pytest.raises(ValueError, match="1-d"):
        adjust_ihw(_P, np.ones((4, 1)), 0.1)

def test_folds_must_be_1d() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=40)
    x = rng.uniform(size=40)
    folds = rng.integers(0, 5, size=(40, 1))
    with pytest.raises(ValueError, match="1-d"):
        adjust_ihw(p, x, 0.1, nbins=4, folds=folds, seed=1)

def test_out_of_range_fold_labels_raise() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=40)
    x = rng.uniform(size=40)
    folds = np.array([0, 1, 2] * 13 + [0])[:40]
    folds[0] = 4
    with pytest.raises(ValueError, match="folds labels"):
        adjust_ihw(p, x, 0.1, nbins=4, nfolds=5, folds=folds, seed=1)

def test_negative_fold_labels_raise() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=40)
    x = rng.uniform(size=40)
    folds = rng.integers(0, 5, size=40)
    folds[3] = -1
    with pytest.raises(ValueError, match="folds labels"):
        adjust_ihw(p, x, 0.1, nbins=4, nfolds=5, folds=folds, seed=1)

def test_empty_lambdas_raise() -> None:
    with pytest.raises(ValueError, match="lambdas"):
        adjust_ihw(_P, _X, 0.1, nbins=2, lambdas=[])

def test_nan_lambda_raises() -> None:
    with pytest.raises(ValueError, match="lambdas"):
        adjust_ihw(_P, _X, 0.1, nbins=2, lambdas=[np.nan])

def test_negative_lambda_raises() -> None:
    with pytest.raises(ValueError, match="lambdas"):
        adjust_ihw(_P, _X, 0.1, nbins=2, lambdas=[-1.0])

def test_public_api_has_no_backend_switches() -> None:
    parameters = inspect.signature(adjust_ihw).parameters
    assert "lp_backend" not in parameters
    assert "use_numba" not in parameters

def test_default_numba_path_runs() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    result = adjust_ihw(p, x, 0.1, nbins=4, nfolds=1, seed=1)
    assert np.all(np.isfinite(result.weights))
    assert np.all(np.isfinite(result.adj_pvalues))
    np.testing.assert_allclose(np.mean(result.weights), 1.0, atol=1e-8)

def test_single_bin_matches_bh() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    result = adjust_ihw(p, x, 0.1, nbins=1, seed=1)
    np.testing.assert_allclose(result.adj_pvalues, _p_adjust(p, "fdr_bh"))
    np.testing.assert_allclose(result.weights, 1.0)
    assert result.nfolds == 1

def test_bonferroni_vs_bh_with_one_bin() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    bh = adjust_ihw(p, x, 0.1, nbins=1, seed=1)
    bonf = adjust_ihw(p, x, 0.1, nbins=1, adjustment_type="bonferroni", seed=1)
    np.testing.assert_allclose(bh.weights, 1.0)
    np.testing.assert_allclose(bonf.weights, 1.0)
    assert np.all(bonf.adj_pvalues >= bh.adj_pvalues - 1e-12)

def test_bonferroni_adj_p_at_least_bh() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    bh = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    bonf = adjust_ihw(p, x, 0.1, nbins=4, adjustment_type="bonferroni", seed=1)
    assert np.all(bonf.adj_pvalues >= bh.adj_pvalues - 1e-12)

def test_exploratory_uses_one_fold() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    result = adjust_ihw(p, x, 0.1, nbins=4, exploratory=True, seed=1)
    assert result.nfolds == 1

def test_default_uses_five_folds() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    result = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    assert result.nfolds == 5

def test_nominal_covariates_run() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.integers(0, 5, size=80).astype(float)
    result = adjust_ihw(p, x, 0.1, nbins=4, covariate_type="nominal", seed=1)
    assert np.all(np.isfinite(result.weights))
    assert result.penalty == "uniform_deviation"

def test_nominal_uses_unique_levels() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = np.array([0.0] * 8 + [1.0] * 2 + [2.0] * 2 + [3.0] * 68)
    result = adjust_ihw(p, x, 0.1, nbins=4, covariate_type="nominal", seed=1)
    counts = np.bincount(result.groups)
    np.testing.assert_array_equal(np.sort(counts), [2, 2, 8, 68])
    assert result.nbins == 4
    assert result.penalty == "uniform_deviation"

def test_single_nominal_level_is_bh() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = np.full(80, 3.0)
    result = adjust_ihw(p, x, 0.1, nbins=4, covariate_type="nominal", seed=1)
    assert result.nbins == 1
    assert result.nfolds == 1
    np.testing.assert_allclose(result.weights, 1.0)
    np.testing.assert_allclose(result.adj_pvalues, _p_adjust(p, "fdr_bh"))

def test_nfolds_follows_supplied_fold_labels() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=40)
    x = rng.uniform(size=40)
    folds = np.array([0, 1, 2] * 13 + [0])[:40]
    result = adjust_ihw(p, x, 0.1, nbins=4, nfolds=5, folds=folds, seed=1)
    assert result.nfolds == 3
    assert set(result.folds.tolist()) == {0, 1, 2}

def test_default_path_records_inf_fold_lambdas() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    result = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    assert result.fold_lambdas.shape == (5,)
    np.testing.assert_array_equal(result.fold_lambdas, np.inf)

def test_result_metadata_on_a_default_fit() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    result = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    assert result.alpha == 0.1
    assert result.nbins == 4
    assert result.nfolds == 5
    assert result.penalty == "total_variation"
    assert result.fold_lambdas.shape == (5,)
    np.testing.assert_array_equal(result.fold_lambdas, np.inf)

def test_result_includes_bin_counts() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    multi = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    assert multi.m_groups.shape == (4,)
    one = adjust_ihw(p, x, 0.1, nbins=1, seed=1)
    assert one.m_groups.shape == (1,)
    np.testing.assert_array_equal(one.m_groups, [80])

def test_m_groups_matches_the_groups() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    ordinal = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    np.testing.assert_array_equal(
        ordinal.m_groups,
        np.bincount(ordinal.groups, minlength=ordinal.nbins),
    )
    nom_x = np.array([0.0] * 8 + [1.0] * 2 + [2.0] * 2 + [3.0] * 68)
    nominal = adjust_ihw(p, nom_x, 0.1, covariate_type="nominal", seed=1)
    np.testing.assert_array_equal(
        nominal.m_groups,
        np.bincount(nominal.groups, minlength=nominal.nbins),
    )

def test_weights_have_mean_one() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    ordinal = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    np.testing.assert_allclose(np.mean(ordinal.weights), 1.0)
    nom_x = np.array([0.0] * 8 + [1.0] * 2 + [2.0] * 2 + [3.0] * 68)
    nominal = adjust_ihw(p, nom_x, 0.1, covariate_type="nominal", seed=1)
    np.testing.assert_allclose(np.mean(nominal.weights), 1.0)

def test_weighted_pvalues_stay_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    result = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    assert np.all(np.isfinite(result.weighted_pvalues))
    assert np.all(result.weighted_pvalues >= 0.0)
    assert np.all(result.weighted_pvalues <= 1.0)

def test_lambda_zero_gives_unit_weights() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    result = adjust_ihw(p, x, 0.1, nbins=4, lambdas=[0.0], seed=1)
    np.testing.assert_allclose(result.weights, 1.0)
    np.testing.assert_allclose(result.adj_pvalues, _p_adjust(p, "fdr_bh"))

def test_auto_lambda_is_finite_and_may_differ_from_inf() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=500)
    x = rng.uniform(size=500)
    inf = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    auto = adjust_ihw(p, x, 0.1, nbins=4, lambdas="auto", seed=1)
    assert np.all(np.isfinite(auto.weights))
    assert np.all(np.isfinite(auto.adj_pvalues))
    weights_differ = not np.allclose(auto.weights, inf.weights)
    adj_differ = not np.allclose(auto.adj_pvalues, inf.adj_pvalues)
    assert weights_differ or adj_differ

def test_auto_lambda_picks_from_the_grid() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    result = adjust_ihw(p, x, 0.1, nbins=4, lambdas="auto", seed=1)
    allowed = {0.0, 0.5, 1.0, 2.0, 4.0, np.inf}
    for lam in result.fold_lambdas:
        assert float(lam) in allowed

def test_auto_lambda_with_two_inner_splits() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=400)
    x = rng.uniform(size=400)
    result = adjust_ihw(p, x, 0.1, nbins=4, lambdas="auto", nsplits_internal=2, seed=1)
    assert np.all(np.isfinite(result.weights))
    assert np.all(np.isfinite(result.adj_pvalues))

def test_exploratory_ignores_the_auto_lambda_grid() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    result = adjust_ihw(p, x, 0.1, nbins=4, exploratory=True, lambdas="auto", seed=1)
    assert result.nfolds == 1
    np.testing.assert_array_equal(result.fold_lambdas, [np.inf])

def test_preset_bins_lambdas_and_m_groups_run() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    g = np.array([0, 1, 2, 3] * 20)
    mg = np.bincount(g, minlength=4) + 10
    fl = np.full(5, np.inf)
    result = adjust_ihw(
        p,
        x,
        0.1,
        nbins=4,
        groups=g,
        m_groups=mg,
        fold_lambdas=fl,
        seed=1,
    )
    np.testing.assert_array_equal(result.groups, g)
    np.testing.assert_array_equal(result.m_groups, mg)
    np.testing.assert_array_equal(result.fold_lambdas, fl)
    assert np.all(np.isfinite(result.weights))
    assert np.all(np.isfinite(result.adj_pvalues))

def test_preset_groups_skip_binning() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = np.arange(80).astype(float)
    g = np.array([0, 1, 2, 3] * 20)
    result = adjust_ihw(p, x, 0.1, nbins=4, groups=g, seed=1)
    np.testing.assert_array_equal(result.groups, g)
    binned = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    assert not np.array_equal(binned.groups, g)

def test_preset_fold_lambdas_skip_inner_cv(monkeypatch: pytest.MonkeyPatch) -> None:
    import ihw as ihw_mod

    def boom(*args, **kwargs):
        raise AssertionError("inner cv should not run")

    monkeypatch.setattr(ihw_mod, "_select_lambda", boom)
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    fl = np.array([0.0, 0.5, 1.0, 2.0, np.inf])
    result = adjust_ihw(p, x, 0.1, nbins=4, fold_lambdas=fl, lambdas="auto", seed=1)
    np.testing.assert_array_equal(result.fold_lambdas, fl)

def test_inner_folds_redrawn_per_lambda(monkeypatch: pytest.MonkeyPatch) -> None:
    import ihw as ihw_mod

    calls: list[int] = []
    orig = ihw_mod._assign_folds

    def counted(n: int, nfolds: int, rng: np.random.Generator) -> np.ndarray:
        calls.append(nfolds)
        return orig(n, nfolds, rng)

    monkeypatch.setattr(ihw_mod, "_assign_folds", counted)
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    folds = np.zeros(80, dtype=np.intp)
    short = [0.0, np.inf]
    adjust_ihw(p, x, 0.1, nbins=4, folds=folds, lambdas=short, seed=1)
    n_short = len(calls)
    calls.clear()
    long = [0.0, 1.0, np.inf]
    adjust_ihw(p, x, 0.1, nbins=4, folds=folds, lambdas=long, seed=1)
    n_long = len(calls)
    assert n_short == len(short)
    assert n_long == len(long)

def test_m_groups_with_a_subset_of_pvalues() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    g = np.array([0, 1, 2, 3] * 20)
    mg = np.bincount(g, minlength=4) + 50
    keep = np.arange(40)
    result = adjust_ihw(
        p[keep],
        x[keep],
        0.1,
        nbins=4,
        groups=g[keep],
        m_groups=mg,
        seed=1,
    )
    assert np.all(np.isfinite(result.weights))
    assert np.all(np.isfinite(result.adj_pvalues))
    np.testing.assert_array_equal(result.m_groups, mg)

def test_frozen_partition_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    g = np.array([0, 1, 2, 3] * 20)
    folds = np.array([0, 1, 2, 3, 4] * 16)
    a = adjust_ihw(p, x, 0.1, nbins=4, groups=g, folds=folds, rng=np.random.default_rng(1))
    b = adjust_ihw(p, x, 0.1, nbins=4, groups=g, folds=folds, rng=np.random.default_rng(99))
    np.testing.assert_allclose(a.weights, b.weights)
    np.testing.assert_array_equal(a.groups, g)

def test_single_fold_inf_lambda_with_preset_groups() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    g = np.array([0, 1, 2, 3] * 20)
    result = adjust_ihw(p, x, 0.1, nbins=4, nfolds=1, groups=g, seed=1)
    assert result.nfolds == 1
    np.testing.assert_array_equal(result.fold_lambdas, [np.inf])

def test_production_lp_solves_textbook() -> None:
    c = np.array([3.0, 4.0])
    a = np.array([[1.0, 2.0], [3.0, 1.0]])
    b = np.array([8.0, 9.0])
    lb = np.zeros(2)
    ub = np.full(2, np.inf)
    solution = _solve_lp_numpy(c, a, b, lb, ub)
    np.testing.assert_allclose(solution, np.array([2.0, 3.0]), atol=1e-8)
    np.testing.assert_allclose(c @ solution, 18.0, atol=1e-8)

def test_production_lp_respects_box_bounds() -> None:
    c = np.array([1.0, 1.0])
    a = np.zeros((0, 2))
    b = np.zeros(0)
    lb = np.zeros(2)
    ub = np.array([1.0, 2.0])
    solution = _solve_lp_numpy(c, a, b, lb, ub)
    np.testing.assert_allclose(solution, np.array([1.0, 2.0]), atol=1e-8)

def test_production_lp_respects_random_box_lps() -> None:
    rng = np.random.default_rng(1)
    for _ in range(12):
        n = 8
        m = 12
        a = rng.normal(size=(m, n))
        x_true = rng.uniform(0.0, 2.0, size=n)
        b = a @ x_true + rng.uniform(0.1, 1.0, size=m)
        c = rng.normal(size=n)
        lb = np.zeros(n)
        ub = np.full(n, 2.0)
        solution = _solve_lp_numpy(c, a, b, lb, ub)
        assert c @ solution >= c @ x_true - 1e-7
        assert np.max(a @ solution - b) < 1e-7
        assert np.all(solution >= -1e-8)
        assert np.all(solution <= 2.0 + 1e-8)

def test_production_lp_handles_column_scale() -> None:
    c = np.array([1.0, 1e6])
    a = np.array([[1.0, 1e6], [1e-6, 1.0]])
    b = np.array([2.0, 2.0])
    lb = np.zeros(2)
    ub = np.full(2, np.inf)
    solution = _solve_lp_numpy(c, a, b, lb, ub)
    assert np.all(np.isfinite(solution))
    assert np.max(a @ solution - b) < 1e-6
    np.testing.assert_allclose(c @ solution, 2.0, atol=1e-6, rtol=1e-6)

def test_numpy_lp_infeasible_raises() -> None:
    c = np.array([1.0])
    a = np.array([[1.0], [-1.0]])
    b = np.array([-1.0, -1.0])
    lb = np.zeros(1)
    ub = np.array([10.0])
    with pytest.raises(RuntimeError, match="did not solve"):
        _solve_lp_numpy(c, a, b, lb, ub)

def test_zero_weight_denom_raises() -> None:
    with pytest.raises(RuntimeError, match="weight denom"):
        _thresholds_to_weights(np.array([0.1, 0.2, 0.0]), np.array([0, 0, 10]))

def test_weight_lp_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import ihw as ihw_mod

    def fail(*args: object, **kwargs: object) -> np.ndarray:
        raise RuntimeError("weight LP did not solve: stub infeasibility")

    monkeypatch.setattr(ihw_mod, "_solve_lp_numpy", fail)
    rng = np.random.default_rng(0)
    p = rng.uniform(size=40)
    x = rng.uniform(size=40)
    with pytest.raises(RuntimeError, match="weight LP did not solve"):
        adjust_ihw(p, x, 0.1, nbins=4, seed=1)

def test_successful_lp_does_not_fall_back_to_uniform() -> None:
    rng = np.random.default_rng(7)
    n = 80
    cov = rng.uniform(0.0, 3.0, size=n)
    signals = rng.binomial(1, 0.12, size=n).astype(bool)
    z = rng.normal(loc=signals * cov)
    p = _normal_survival(z)
    result = adjust_ihw(p, cov, 0.1, nbins=4, seed=1)
    assert np.all(np.isfinite(result.weights))
    assert not np.allclose(result.weights, 1.0)

def test_bin_ties_follow_the_seed_not_the_fold_rng() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = np.round(rng.uniform(size=80), decimals=1)
    a = adjust_ihw(p, x, 0.1, nbins=4, rng=np.random.default_rng(1), seed=1)
    b = adjust_ihw(p, x, 0.1, nbins=4, rng=np.random.default_rng(2), seed=1)
    np.testing.assert_array_equal(a.groups, b.groups)
    assert not np.array_equal(a.folds, b.folds)

def test_tied_covariates_return_finite_result() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = np.full(80, 2.5)
    result = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    assert np.all(np.isfinite(result.weights))
    assert np.all(np.isfinite(result.adj_pvalues))

def test_pvalue_zero_and_one_are_legal() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(size=80)
    p[0] = 0.0
    p[-1] = 1.0
    x = rng.uniform(size=80)
    result = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    assert np.all(np.isfinite(result.weights))
    assert np.all(np.isfinite(result.adj_pvalues))

def test_default_path_has_no_nan_weights_or_adj_p() -> None:
    rng = np.random.default_rng(2)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    result = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    assert not np.any(np.isnan(result.weights))
    assert not np.any(np.isnan(result.adj_pvalues))

def test_tiny_pvalues_are_allowed() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    p[0] = 1e-30
    p[1] = 1e-21
    x = rng.uniform(size=80)
    result = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    assert np.all(np.isfinite(result.weights))
    assert np.all(np.isfinite(result.adj_pvalues))

def test_constant_pvalues() -> None:
    p = np.full(80, 0.5)
    x = np.linspace(0.0, 1.0, 80)
    result = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    assert np.all(np.isfinite(result.weights))
    assert np.all(np.isfinite(result.adj_pvalues))

def test_uniform_null_fdr_is_conservative() -> None:
    alpha = 0.1
    n = 2000
    rates = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        p = rng.uniform(size=n)
        x = rng.uniform(size=n)
        result = adjust_ihw(p, x, alpha, nbins=4, seed=1)
        rates.append(float(np.mean(result.adj_pvalues <= alpha)))
    assert max(rates) < alpha

def test_nominal_null_fdr_is_conservative() -> None:
    alpha = 0.1
    n = 2000
    labels = np.array([0.0] * 200 + [1.0] * 300 + [2.0] * 500 + [3.0] * 1000)
    rates = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        p = rng.uniform(size=n)
        result = adjust_ihw(p, labels, alpha, covariate_type="nominal", seed=1)
        rates.append(float(np.mean(result.adj_pvalues <= alpha)))
    assert max(rates) < alpha

def test_ihw_beats_bh_when_covariate_is_informative() -> None:
    rng = np.random.default_rng(7)
    n = 2000
    alpha = 0.1
    cov = rng.uniform(0.0, 3.0, size=n)
    signals = rng.binomial(1, 0.12, size=n).astype(bool)
    z = rng.normal(loc=signals * cov)
    p = _normal_survival(z)
    result = adjust_ihw(p, cov, alpha, nbins=4, seed=1)
    bh_adj = _p_adjust(p, "fdr_bh")
    ihw_rej = int(np.sum(result.adj_pvalues <= alpha))
    bh_rej = int(np.sum(bh_adj <= alpha))
    assert ihw_rej >= bh_rej
    assert np.all(np.isfinite(result.weights))
    assert np.all(result.weights >= 0.0)

def test_ihw_beats_bh_on_a_mixture_sim() -> None:
    rng = np.random.default_rng(4)
    n = 2000
    alpha = 0.1
    cov = rng.uniform(0.0, 1.0, size=n)
    pi_alt = 0.02 + 0.35 * cov
    is_alt = rng.uniform(size=n) < pi_alt
    z = rng.normal(loc=np.where(is_alt, 1.5 + 1.5 * cov, 0.0))
    p = _normal_survival(z)
    result = adjust_ihw(p, cov, alpha, nbins=4, seed=1)
    bh_adj = _p_adjust(p, "fdr_bh")
    ihw_rej = int(np.sum(result.adj_pvalues <= alpha))
    bh_rej = int(np.sum(bh_adj <= alpha))
    assert ihw_rej >= bh_rej

def test_power_vs_bh_on_more_seeds() -> None:
    alpha = 0.1
    n = 2000
    for sim_seed in (3, 11, 23):
        rng = np.random.default_rng(sim_seed)
        cov = rng.uniform(0.0, 3.0, size=n)
        signals = rng.binomial(1, 0.12, size=n).astype(bool)
        z = rng.normal(loc=signals * cov)
        p = _normal_survival(z)
        result = adjust_ihw(p, cov, alpha, nbins=4, seed=1)
        bh_adj = _p_adjust(p, "fdr_bh")
        ihw_rej = int(np.sum(result.adj_pvalues <= alpha))
        bh_rej = int(np.sum(bh_adj <= alpha))
        assert ihw_rej >= bh_rej
