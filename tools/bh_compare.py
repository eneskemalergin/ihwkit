"""Compare BH with IHW on a small synthetic draw."""

import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ihw import _p_adjust, adjust_ihw


def _normal_survival(values: np.ndarray) -> np.ndarray:
    """Return the standard normal survival function for a one-dimensional array."""

    return np.fromiter(
        (0.5 * (1.0 - erf(float(value) / sqrt(2.0))) for value in values),
        dtype=np.float64,
        count=values.size,
    )


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
print(f"bh {bh_rej}")
print(f"ihw {ihw_rej}")
