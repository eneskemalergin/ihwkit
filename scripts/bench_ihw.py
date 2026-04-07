#!/usr/bin/env python3
import csv
import resource
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
rss_unit = "kilobytes"


def run(backend: str, nfolds: int):
    return adjust_ihw(
        p, x, alpha, nbins=4, nfolds=nfolds, seed=1, lp_backend=backend
    )


def median_wall(backend: str, nfolds: int):
    times = []
    last = None
    for _ in range(n_reps):
        t0 = time.perf_counter()
        last = run(backend, nfolds)
        times.append(time.perf_counter() - t0)
    times.sort()
    return float(times[n_reps // 2]), last


def rss_max():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


src = path.relative_to(ROOT)
print(f"sim {src} n={p.shape[0]} nbins=4 lambda=inf reps={n_reps} rss_unit={rss_unit}")
rows = []
for nfolds in (1, 5):
    run("highs", nfolds)
    run("numpy", nfolds)
    for backend in ("highs", "numpy"):
        med, fit = median_wall(backend, nfolds)
        rej = int(np.sum(fit.adj_pvalues <= alpha))
        rss = rss_max()
        print(
            f"nfolds={nfolds} {backend} median_s {med:.6f} rejections {rej} rss_max {rss} {rss_unit}"
        )
        rows.append(
            {
                "nfolds": nfolds,
                "backend": backend,
                "median_s": f"{med:.6f}",
                "rejections": rej,
                "rss_max": rss,
                "rss_unit": rss_unit,
            }
        )

csv_path = ROOT / "tmp" / "bench_last.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", newline="") as fh:
    writer = csv.DictWriter(
        fh, fieldnames=["nfolds", "backend", "median_s", "rejections", "rss_max", "rss_unit"]
    )
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {csv_path.relative_to(ROOT)}")
