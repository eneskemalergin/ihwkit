import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ihw import _p_adjust, adjust_ihw

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
    folds = rng.integers(0, 5, size=40)
    folds[0] = 5
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
