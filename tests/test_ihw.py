import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ihw import adjust_ihw

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
