#!/usr/bin/env python3
import csv
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ihw import _numba_importable, adjust_ihw

TMP = ROOT / "tmp" / "bench_sim.npz"
FALLBACK = ROOT / "tests" / "fixtures" / "sim_n2000_seed1.npz"
path = TMP if TMP.is_file() else FALLBACK
data = np.load(path)
p = np.asarray(data["p"], dtype=np.float64)
x = np.asarray(data["x"], dtype=np.float64)
alpha = 0.1
n_reps = 5
rss_unit = "kilobytes"


def backend_for_label(label: str):
    if label == "scipy":
        return "highs", False
    if label == "numpy":
        return "numpy", False
    if label == "numpy_numba":
        return "numpy", True
    raise ValueError(f"unknown bench label {label!r}")


def run(
    backend: str,
    nfolds: int,
    pvalues=None,
    covariates=None,
    groups=None,
    folds=None,
    use_numba=False,
):
    pv = p if pvalues is None else pvalues
    xv = x if covariates is None else covariates
    kw = {"use_numba": use_numba}
    if groups is not None:
        kw["groups"] = groups
    if folds is not None:
        kw["folds"] = folds
    return adjust_ihw(
        pv, xv, alpha, nbins=4, nfolds=nfolds, seed=1, lp_backend=backend, **kw
    )


def median_wall(
    backend: str,
    nfolds: int,
    pvalues=None,
    covariates=None,
    groups=None,
    folds=None,
    use_numba=False,
):
    times = []
    last = None
    for _ in range(n_reps):
        t0 = time.perf_counter()
        last = run(
            backend, nfolds, pvalues, covariates, groups, folds, use_numba
        )
        times.append(time.perf_counter() - t0)
    times.sort()
    return float(times[n_reps // 2]), last


def rss_max():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def mixture_sim(n, seed=4):
    rng = np.random.default_rng(seed)
    cov = rng.uniform(0.0, 1.0, size=n)
    pi_alt = 0.02 + 0.35 * cov
    is_alt = rng.uniform(size=n) < pi_alt
    z = rng.normal(loc=np.where(is_alt, 1.5 + 1.5 * cov, 0.0))
    pmix = 1.0 - norm.cdf(z)
    return pmix, cov


def sim_arrays(kind: str):
    if kind == "sim":
        return p, x, None, None
    if kind == "null":
        rng = np.random.default_rng(21)
        n = p.shape[0]
        return rng.uniform(size=n), rng.uniform(size=n), None, None
    raise ValueError(f"unknown bench kind {kind!r}")


def child_main(label: str, nfolds: int, kind: str) -> None:
    if label == "numpy_numba" and not _numba_importable():
        print(f"child skip {label} numba is not installed")
        return
    pv, xv, groups, folds = sim_arrays(kind)
    backend, use_numba = backend_for_label(label)
    t0 = time.perf_counter()
    run(backend, nfolds, pv, xv, groups, folds, use_numba)
    warmup_s = time.perf_counter() - t0
    med, fit = median_wall(backend, nfolds, pv, xv, groups, folds, use_numba)
    rej = int(np.sum(fit.adj_pvalues <= alpha))
    rss = rss_max()
    print(
        f"child {label} kind={kind} nfolds={nfolds} median_s {med:.6f} "
        f"rejections {rej} rss_max {rss} {rss_unit} warmup_s {warmup_s:.6f}"
    )


def isolated_python(label: str, nfolds: int, kind: str = "sim"):
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "bench_ihw.py"),
            "--child",
            label,
            str(nfolds),
            kind,
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        print(line)
        if line.startswith("child skip"):
            return None
        if not line.startswith("child "):
            continue
        toks = line.split()
        median_s = toks[5]
        rejections = int(toks[7])
        rss = toks[9]
        warmup_s = toks[12] if len(toks) > 12 else ""
        return {
            "nfolds": nfolds,
            "backend": label,
            "median_s": median_s,
            "rejections": rejections,
            "rss_max": rss,
            "rss_unit": rss_unit,
            "warmup_s": warmup_s,
        }
    if proc.returncode != 0:
        err = proc.stderr.strip().splitlines()
        tail = err[-1] if err else "child failed"
        print(f"child skip {label} {tail}")
    return None


def run_r_bench(nfolds=None):
    script = ROOT / "scripts" / "r_ihw_bench.R"
    cmd = ["Rscript", "--vanilla", str(script)]
    if nfolds is not None:
        cmd.append(str(nfolds))
    try:
        proc = subprocess.run(
            cmd,
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
        nf = int(toks[4].split("=", 1)[1])
        median_s = toks[6]
        rejections = int(toks[8])
        rss = toks[11] if len(toks) > 11 else ""
        unit = toks[12] if len(toks) > 12 else ""
        parsed.append(
            {
                "nfolds": nf,
                "backend": "r",
                "median_s": median_s,
                "rejections": rejections,
                "rss_max": rss,
                "rss_unit": unit,
                "warmup_s": "",
            }
        )
    if printed:
        return parsed
    if proc.returncode != 0:
        print("r skip Rscript failed")
        return []
    print("r skip no r timing line")
    return []


def python_labels():
    labels = ["scipy", "numpy"]
    if _numba_importable():
        labels.append("numpy_numba")
    else:
        print("numpy_numba skip numba is not installed")
    return labels


def csv_fields():
    return [
        "nfolds",
        "backend",
        "median_s",
        "rejections",
        "rss_max",
        "rss_unit",
        "warmup_s",
    ]


def main() -> None:
    src = path.relative_to(ROOT)
    print(
        f"sim {src} n={p.shape[0]} nbins=4 lambda=inf reps={n_reps} "
        f"rss_unit={rss_unit} backends=r,scipy,numpy,numpy_numba"
    )
    rows = []
    labels = python_labels()
    for nfolds in (1, 5):
        for label in labels:
            row = isolated_python(label, nfolds, "sim")
            if row is not None:
                rows.append(row)
        rows.extend(run_r_bench(nfolds))

    csv_path = ROOT / "tmp" / "bench_last.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path.relative_to(ROOT)}")

    rng = np.random.default_rng(21)
    p_null = rng.uniform(size=p.shape[0])
    x_null = rng.uniform(size=p.shape[0])
    print(
        f"uniform_null n={p_null.shape[0]} independent_cov nbins=4 lambda=inf reps={n_reps}"
    )
    for nfolds in (1, 5):
        run("highs", nfolds, p_null, x_null, use_numba=False)
        run("numpy", nfolds, p_null, x_null, use_numba=False)
        for backend in ("highs", "numpy"):
            med, fit = median_wall(
                backend, nfolds, p_null, x_null, use_numba=False
            )
            rej = int(np.sum(fit.adj_pvalues <= alpha))
            print(
                f"uniform_null nfolds={nfolds} {backend} median_s {med:.6f} rejections {rej}"
            )

    n1_path = ROOT / "tests" / "fixtures" / "r_inf_n1.npz"
    n1 = np.load(n1_path)
    p1 = np.asarray(n1["p"], dtype=np.float64)
    x1 = np.asarray(n1["x"], dtype=np.float64)
    g1 = np.asarray(n1["groups"], dtype=np.intp)
    adj1 = np.asarray(n1["adj_pvalues"], dtype=np.float64)
    w1 = np.asarray(n1["weights"], dtype=np.float64)
    print(f"oracle n1 n={p1.shape[0]} frozen_groups nbins=4 lambda=inf reps={n_reps}")
    run("highs", 1, p1, x1, g1, use_numba=False)
    run("numpy", 1, p1, x1, g1, use_numba=False)
    for backend in ("highs", "numpy"):
        med, fit = median_wall(backend, 1, p1, x1, g1, use_numba=False)
        rej = int(np.sum(fit.adj_pvalues <= alpha))
        max_adj = float(np.max(np.abs(fit.adj_pvalues - adj1)))
        max_w = float(np.max(np.abs(fit.weights - w1)))
        tag = "" if backend == "highs" else " informational"
        print(
            f"oracle n1 {backend} median_s {med:.6f} rejections {rej} "
            f"max_abs_adj_vs_r {max_adj:.6e} max_abs_weights_vs_r {max_w:.6e}{tag}"
        )

    n5_path = ROOT / "tests" / "fixtures" / "r_inf_n5.npz"
    n5 = np.load(n5_path)
    p5 = np.asarray(n5["p"], dtype=np.float64)
    x5 = np.asarray(n5["x"], dtype=np.float64)
    g5 = np.asarray(n5["groups"], dtype=np.intp)
    f5 = np.asarray(n5["folds"], dtype=np.intp)
    adj5 = np.asarray(n5["adj_pvalues"], dtype=np.float64)
    w5 = np.asarray(n5["weights"], dtype=np.float64)
    print(
        f"oracle n5 n={p5.shape[0]} frozen_groups_folds nbins=4 lambda=inf reps={n_reps}"
    )
    run("highs", 5, p5, x5, g5, f5, use_numba=False)
    run("numpy", 5, p5, x5, g5, f5, use_numba=False)
    for backend in ("highs", "numpy"):
        med, fit = median_wall(backend, 5, p5, x5, g5, f5, use_numba=False)
        rej = int(np.sum(fit.adj_pvalues <= alpha))
        max_adj = float(np.max(np.abs(fit.adj_pvalues - adj5)))
        max_w = float(np.max(np.abs(fit.weights - w5)))
        tag = "" if backend == "highs" else " informational"
        print(
            f"oracle n5 {backend} median_s {med:.6f} rejections {rej} "
            f"max_abs_adj_vs_r {max_adj:.6e} max_abs_weights_vs_r {max_w:.6e}{tag}"
        )

    p_mix, x_mix = mixture_sim(2000)
    print(f"mixture n=2000 informative_pi nbins=4 lambda=inf reps={n_reps}")
    for nfolds in (1, 5):
        run("highs", nfolds, p_mix, x_mix, use_numba=False)
        run("numpy", nfolds, p_mix, x_mix, use_numba=False)
        fits = {}
        for backend in ("highs", "numpy"):
            med, fit = median_wall(
                backend, nfolds, p_mix, x_mix, use_numba=False
            )
            fits[backend] = fit
            rej = int(np.sum(fit.adj_pvalues <= alpha))
            print(
                f"mixture n=2000 nfolds={nfolds} {backend} median_s {med:.6f} rejections {rej}"
            )
        max_adj = float(
            np.max(np.abs(fits["highs"].adj_pvalues - fits["numpy"].adj_pvalues))
        )
        print(
            f"mixture n=2000 nfolds={nfolds} max_abs_adj_highs_numpy {max_adj:.6e}"
        )

    if TMP.is_file():
        p_mix8, x_mix8 = mixture_sim(8000)
        print(f"mixture n=8000 informative_pi nbins=4 lambda=inf reps={n_reps}")
        for nfolds in (1, 5):
            run("highs", nfolds, p_mix8, x_mix8, use_numba=False)
            run("numpy", nfolds, p_mix8, x_mix8, use_numba=False)
            fits = {}
            for backend in ("highs", "numpy"):
                med, fit = median_wall(
                    backend, nfolds, p_mix8, x_mix8, use_numba=False
                )
                fits[backend] = fit
                rej = int(np.sum(fit.adj_pvalues <= alpha))
                print(
                    f"mixture n=8000 nfolds={nfolds} {backend} median_s {med:.6f} rejections {rej}"
                )
            max_adj = float(
                np.max(
                    np.abs(fits["highs"].adj_pvalues - fits["numpy"].adj_pvalues)
                )
            )
            print(
                f"mixture n=8000 nfolds={nfolds} max_abs_adj_highs_numpy {max_adj:.6e}"
            )
    else:
        print("mixture n=8000 skip tmp/bench_sim.npz is not present")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--child":
        kind = sys.argv[4] if len(sys.argv) > 4 else "sim"
        child_main(sys.argv[2], int(sys.argv[3]), kind)
    else:
        main()
