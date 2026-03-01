#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ihw import adjust_ihw

FIXTURE = ROOT / "tests" / "fixtures" / "sim_n2000_seed1.npz"
data = np.load(FIXTURE)
p = np.asarray(data["p"], dtype=np.float64)
x = np.asarray(data["x"], dtype=np.float64)
alpha = 0.1
n_reps = 5


def run(backend: str):
    return adjust_ihw(p, x, alpha, nbins=4, nfolds=1, seed=1, lp_backend=backend)


highs = run("highs")
numpy_fit = run("numpy")


def median_wall(backend: str) -> float:
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        run(backend)
        times.append(time.perf_counter() - t0)
    times.sort()
    return float(times[n_reps // 2])


print(
    f"fixture {FIXTURE.relative_to(ROOT)} n={p.shape[0]} nfolds=1 nbins=4 lambda=inf reps={n_reps}"
)
print(
    f"highs median_s {median_wall('highs'):.6f} rejections {int(np.sum(highs.adj_pvalues <= alpha))}"
)
print(
    f"numpy median_s {median_wall('numpy'):.6f} rejections {int(np.sum(numpy_fit.adj_pvalues <= alpha))}"
)
print(f"max_abs_weights {float(np.max(np.abs(highs.weights - numpy_fit.weights))):.6e}")
print(
    f"max_abs_adj_pvalues {float(np.max(np.abs(highs.adj_pvalues - numpy_fit.adj_pvalues))):.6e}"
)
print(f"mean_weight_highs {float(np.mean(highs.weights)):.6f}")
print(f"mean_weight_numpy {float(np.mean(numpy_fit.weights)):.6f}")
