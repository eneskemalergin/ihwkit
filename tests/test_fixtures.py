from pathlib import Path

import numpy as np

from ihw import adjust_ihw

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sim_n2000_seed1.npz"


def test_ihw_on_the_simulation_fixture() -> None:
    data = np.load(FIXTURE)
    result = adjust_ihw(data["p"], data["x"], 0.1, nbins=4, nfolds=1, seed=1)
    assert np.all(np.isfinite(result.weights))
    assert np.all(np.isfinite(result.adj_pvalues))
    np.testing.assert_allclose(np.mean(result.weights), 1.0)
