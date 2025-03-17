import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ihw import _p_adjust, _thresholds_to_weights, adjust_ihw

_P = np.array([0.1, 0.2, 0.3, 0.4])
_X = np.array([1.0, 2.0, 3.0, 4.0])


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


def test_single_bin_matches_bh() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    result = adjust_ihw(p, x, 0.1, nbins=1, seed=1)
    np.testing.assert_allclose(result.adj_pvalues, _p_adjust(p, "fdr_bh"))
    np.testing.assert_allclose(result.weights, 1.0)
    assert result.nfolds == 1


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


def test_nfolds_not_positive_raises() -> None:
    with pytest.raises(ValueError, match="nfolds"):
        adjust_ihw(_P, _X, 0.1, nfolds=0)
    with pytest.raises(ValueError, match="nfolds"):
        adjust_ihw(_P, _X, 0.1, nfolds=-3)


def test_nbins_not_positive_raises() -> None:
    with pytest.raises(ValueError, match="nbins"):
        adjust_ihw(_P, _X, 0.1, nbins=0)
    with pytest.raises(ValueError, match="nbins"):
        adjust_ihw(_P, _X, 0.1, nbins=-2)


def test_pvalue_not_1d_raises() -> None:
    with pytest.raises(ValueError, match="1-d"):
        adjust_ihw(np.ones((4, 1)), _X, 0.1)


def test_covariate_not_1d_raises() -> None:
    with pytest.raises(ValueError, match="1-d"):
        adjust_ihw(_P, np.ones((4, 1)), 0.1)


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


def test_ihw_beats_bh_when_covariate_is_informative() -> None:
    rng = np.random.default_rng(7)
    n = 2000
    alpha = 0.1
    cov = rng.uniform(0.0, 3.0, size=n)
    signals = rng.binomial(1, 0.12, size=n).astype(bool)
    z = rng.normal(loc=signals * cov)
    p = 1.0 - norm.cdf(z)
    result = adjust_ihw(p, cov, alpha, nbins=4, seed=1)
    bh_adj = _p_adjust(p, "fdr_bh")
    ihw_rej = int(np.sum(result.adj_pvalues <= alpha))
    bh_rej = int(np.sum(bh_adj <= alpha))
    assert ihw_rej >= bh_rej
    assert np.all(np.isfinite(result.weights))
    assert np.all(result.weights >= 0.0)


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


def test_lambda_zero_gives_unit_weights() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    result = adjust_ihw(p, x, 0.1, nbins=4, lambdas=[0.0], seed=1)
    np.testing.assert_allclose(result.weights, 1.0)
    np.testing.assert_allclose(result.adj_pvalues, _p_adjust(p, "fdr_bh"))


def test_weight_lp_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import ihw as ihw_mod

    class Failed:
        success = False
        x = None
        message = "stub infeasibility"

    monkeypatch.setattr(ihw_mod, "linprog", lambda *args, **kwargs: Failed())
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
    p = 1.0 - norm.cdf(z)
    result = adjust_ihw(p, cov, 0.1, nbins=4, seed=1)
    assert np.all(np.isfinite(result.weights))
    assert not np.allclose(result.weights, 1.0)


def test_nominal_covariates_run() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.integers(0, 5, size=80).astype(float)
    result = adjust_ihw(p, x, 0.1, nbins=4, covariate_type="nominal", seed=1)
    assert np.all(np.isfinite(result.weights))
    assert result.penalty == "uniform_deviation"


def test_bonferroni_adj_p_at_least_bh() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    bh = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    bonf = adjust_ihw(p, x, 0.1, nbins=4, adjustment_type="bonferroni", seed=1)
    assert np.all(bonf.adj_pvalues >= bh.adj_pvalues - 1e-12)


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


def test_bin_ties_follow_the_passed_rng() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = np.round(rng.uniform(size=80), decimals=1)
    a = adjust_ihw(p, x, 0.1, nbins=4, rng=np.random.default_rng(1), seed=1)
    b = adjust_ihw(p, x, 0.1, nbins=4, rng=np.random.default_rng(2), seed=1)
    assert not np.array_equal(a.groups, b.groups)


def test_empty_lambdas_raise() -> None:
    with pytest.raises(ValueError, match="lambdas"):
        adjust_ihw(_P, _X, 0.1, nbins=2, lambdas=[])


def test_nan_lambda_raises() -> None:
    with pytest.raises(ValueError, match="lambdas"):
        adjust_ihw(_P, _X, 0.1, nbins=2, lambdas=[np.nan])


def test_negative_lambda_raises() -> None:
    with pytest.raises(ValueError, match="lambdas"):
        adjust_ihw(_P, _X, 0.1, nbins=2, lambdas=[-1.0])


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


def test_zero_weight_denom_raises() -> None:
    with pytest.raises(RuntimeError, match="weight denom"):
        _thresholds_to_weights(np.array([0.1, 0.2, 0.0]), np.array([0, 0, 10]))


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


def test_weights_have_mean_one() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=80)
    x = rng.uniform(size=80)
    ordinal = adjust_ihw(p, x, 0.1, nbins=4, seed=1)
    np.testing.assert_allclose(np.mean(ordinal.weights), 1.0)
    nom_x = np.array([0.0] * 8 + [1.0] * 2 + [2.0] * 2 + [3.0] * 68)
    nominal = adjust_ihw(p, nom_x, 0.1, covariate_type="nominal", seed=1)
    np.testing.assert_allclose(np.mean(nominal.weights), 1.0)


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
