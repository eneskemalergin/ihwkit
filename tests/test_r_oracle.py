from pathlib import Path

import numpy as np
import pytest

from ihw import adjust_ihw

ORACLE = Path(__file__).resolve().parent / "fixtures" / "r_inf_n1.npz"


@pytest.mark.skipif(not ORACLE.is_file(), reason="r_inf_n1.npz is absent")
def test_python_replay_matches_r_oracle_when_present() -> None:
    data = np.load(ORACLE)
    p = np.asarray(data["p"], dtype=np.float64)
    x = np.asarray(data["x"], dtype=np.float64)
    groups = np.asarray(data["groups"], dtype=np.intp)
    result = adjust_ihw(p, x, 0.1, nbins=4, nfolds=1, groups=groups, seed=1)
    np.testing.assert_allclose(result.adj_pvalues, data["adj_pvalues"], atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(result.weights, data["weights"], atol=1e-8, rtol=1e-6)
    ihw_rej = int(np.sum(result.adj_pvalues <= 0.1))
    assert ihw_rej == int(data["rejections"])
