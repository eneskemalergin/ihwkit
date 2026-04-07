#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ihw import adjust_ihw

TMP = ROOT / "tmp" / "bench_sim.npz"
FALLBACK = ROOT / "tests" / "fixtures" / "sim_n2000_seed1.npz"
path = TMP if TMP.is_file() else FALLBACK
data = np.load(path)
p = np.asarray(data["p"], dtype=np.float64)
x = np.asarray(data["x"], dtype=np.float64)
alpha = 0.1
n_reps = 5
nfolds = 1


def run(backend: str):
    return adjust_ihw(
        p, x, alpha, nbins=4, nfolds=nfolds, seed=1, lp_backend=backend
    )


run("highs")
run("numpy")


def median_wall(backend: str):
    times = []
    last = None
    for _ in range(n_reps):
        t0 = time.perf_counter()
        last = run(backend)
        times.append(time.perf_counter() - t0)
    times.sort()
    return float(times[n_reps // 2]), last


highs_med, highs = median_wall("highs")
numpy_med, numpy_fit = median_wall("numpy")
src = path.relative_to(ROOT)
print(f"sim {src} n={p.shape[0]} nfolds={nfolds} nbins=4 lambda=inf reps={n_reps}")
print(
    f"highs median_s {highs_med:.6f} rejections {int(np.sum(highs.adj_pvalues <= alpha))}"
)
print(
    f"numpy median_s {numpy_med:.6f} rejections {int(np.sum(numpy_fit.adj_pvalues <= alpha))}"
)
