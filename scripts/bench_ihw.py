#!/usr/bin/env python3
import csv
import resource
import subprocess
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


def run(backend: str, nfolds: int, pvalues=None, covariates=None, groups=None, folds=None):
    pv = p if pvalues is None else pvalues
    xv = x if covariates is None else covariates
    kw = {}
    if groups is not None:
        kw["groups"] = groups
    if folds is not None:
        kw["folds"] = folds
    return adjust_ihw(
        pv, xv, alpha, nbins=4, nfolds=nfolds, seed=1, lp_backend=backend, **kw
    )


def median_wall(
    backend: str, nfolds: int, pvalues=None, covariates=None, groups=None, folds=None
):
    times = []
    last = None
    for _ in range(n_reps):
        t0 = time.perf_counter()
        last = run(backend, nfolds, pvalues, covariates, groups, folds)
        times.append(time.perf_counter() - t0)
    times.sort()
    return float(times[n_reps // 2]), last


def rss_max():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def run_r_bench():
    script = ROOT / "scripts" / "r_ihw_bench.R"
    try:
        proc = subprocess.run(
            ["Rscript", "--vanilla", str(script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
    except FileNotFoundError:
        print("r skip Rscript not found")
        return []
    except subprocess.TimeoutExpired:
        print("r skip Rscript timed out")
        return []
    parsed = []
    printed = False
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("r "):
            continue
        print(line)
        printed = True
        if line.startswith("r skip"):
            continue
        toks = line.split()
        if len(toks) < 10 or toks[1] != "sim":
            continue
        nfolds = int(toks[4].split("=", 1)[1])
        median_s = toks[6]
        rejections = int(toks[8])
        parsed.append(
            {
                "nfolds": nfolds,
                "backend": "r",
                "median_s": median_s,
                "rejections": rejections,
                "rss_max": "",
                "rss_unit": "",
            }
        )
    if printed:
        return parsed
    if proc.returncode != 0:
        print("r skip Rscript failed")
        return []
    print("r skip no r timing line")
    return []


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

rows.extend(run_r_bench())

csv_path = ROOT / "tmp" / "bench_last.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", newline="") as fh:
    writer = csv.DictWriter(
        fh, fieldnames=["nfolds", "backend", "median_s", "rejections", "rss_max", "rss_unit"]
    )
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {csv_path.relative_to(ROOT)}")

rng = np.random.default_rng(21)
p_null = rng.uniform(size=p.shape[0])
x_null = rng.uniform(size=p.shape[0])
print(f"uniform_null n={p_null.shape[0]} independent_cov nbins=4 lambda=inf reps={n_reps}")
for nfolds in (1, 5):
    run("highs", nfolds, p_null, x_null)
    run("numpy", nfolds, p_null, x_null)
    for backend in ("highs", "numpy"):
        med, fit = median_wall(backend, nfolds, p_null, x_null)
        rej = int(np.sum(fit.adj_pvalues <= alpha))
        print(f"uniform_null nfolds={nfolds} {backend} median_s {med:.6f} rejections {rej}")

n1_path = ROOT / "tests" / "fixtures" / "r_inf_n1.npz"
n1 = np.load(n1_path)
p1 = np.asarray(n1["p"], dtype=np.float64)
x1 = np.asarray(n1["x"], dtype=np.float64)
g1 = np.asarray(n1["groups"], dtype=np.intp)
adj1 = np.asarray(n1["adj_pvalues"], dtype=np.float64)
w1 = np.asarray(n1["weights"], dtype=np.float64)
print(f"oracle n1 n={p1.shape[0]} frozen_groups nbins=4 lambda=inf reps={n_reps}")
run("highs", 1, p1, x1, g1)
run("numpy", 1, p1, x1, g1)
for backend in ("highs", "numpy"):
    med, fit = median_wall(backend, 1, p1, x1, g1)
    rej = int(np.sum(fit.adj_pvalues <= alpha))
    max_adj = float(np.max(np.abs(fit.adj_pvalues - adj1)))
    max_w = float(np.max(np.abs(fit.weights - w1)))
    tag = "" if backend == "highs" else " informational"
    print(
        f"oracle n1 {backend} median_s {med:.6f} rejections {rej} "
        f"max_abs_adj_vs_r {max_adj:.6e} max_abs_weights_vs_r {max_w:.6e}{tag}"
    )
