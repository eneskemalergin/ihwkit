# ihwkit

Independent Hypothesis Weighting. A local NumPy + SciPy module: covariate-weighted FDR via Grenander estimation, a linear program (`scipy.optimize.linprog`, HiGHS), and weighted Benjamini-Hochberg.

Needs NumPy and SciPy. Install from the repository root with `pip install -e .` (or `uv pip install -e .`), then `from ihw import adjust_ihw`. `pip install -e ".[numba]"` adds an optional Numba extra; the default path does not require it. Without installing, from the repository root:

```python
import sys

sys.path.insert(0, "src")

import numpy as np
from ihw import adjust_ihw

p = np.random.default_rng(0).uniform(size=5000)
cov = np.random.default_rng(1).uniform(size=5000)

result = adjust_ihw(p, cov, alpha=0.1, seed=1)
result.adj_pvalues
result.weights
```

Default fit is five-fold cross-validation with λ = ∞ (no inner λ search). `exploratory=True` uses a single fold and is for inspection, not claimed FDR control. `lambdas="auto"` runs the Bioconductor-style λ grid with nested CV; that path uses HiGHS, not SYMPHONY.

`nbins="auto"` is `max(1, min(40, n // 1500))`, so n < 1500 is single-bin BH unless `nbins` is set. If the weight LP fails to solve, `adjust_ihw` raises `RuntimeError`.

`covariate_type="nominal"` uses each unique covariate value as a group (not quantile bins). `rng` (or `seed` when `rng` is omitted) drives both bin-tie permutation and fold assignment. `IHWResult.fold_lambdas` is the λ chosen for each fold (all `inf` on the default path). `IHWResult.m_groups` is the per-bin hypothesis count used in the weight LP and in BH `n_tests`.

Pass `groups=`, `folds=`, `fold_lambdas=`, and `m_groups=` to freeze the partition and regularization. Preset `groups=` skips covariate binning. Preset `fold_lambdas=` skips the inner λ search. Preset `m_groups=` is the filtered-p path: BH uses `sum(m_groups)` even when only a subset of p-values is observed.

`IHWResult` also holds `weighted_pvalues`, `groups`, `folds`, `nbins`, `nfolds`, `m_groups`, and `fold_lambdas`.

Default here is λ=∞ and nfolds=5. R IHW defaults to `lambdas="auto"`. The stored comparable file is `tests/fixtures/r_inf_n1.npz` (λ=∞, nfolds=1, frozen `groups=`). Unfrozen `seed` is not claimed to match R.

The default weight LP is HiGHS (`lp_backend="highs"`). `lp_backend="numpy"` is an experimental dense tableau simplex and may differ from R. SciPy is still required for the default path. `scripts/compare_lp_backends.py` runs both backends on `tests/fixtures/sim_n2000_seed1.npz` (nfolds=1 and 5, λ=∞): median of 5 wall times, rejection counts, max-abs of weights and adj-p. Replay of the stored R oracles stays on HiGHS.

Larger Python benches: `scripts/make_bench_sim.py` writes an n=8000 informative sim to `tmp/bench_sim.npz` (gitignored). `scripts/bench_ihw.py` loads that file if present, otherwise the n=2000 fixture. It prints median of 5 wall times for HiGHS and numpy at nfolds=1 and 5, max RSS (`resource.getrusage`, kilobytes on Linux), a uniform-null control with an independent covariate, frozen n1/n5 oracle quality+time (`groups=` / `folds=`, adj-p and weight max-abs versus R; numpy versus R is informational), and an S19-style mixture at n=2000 (n=8000 if the tmp sim is present). Numbers go to stdout and `tmp/bench_last.csv`. Default fit remains λ=∞ HiGHS.

Optional R wall time uses the same p/x as the loaded sim: `Rscript --vanilla scripts/r_ihw_bench.R` (λ=Inf, nfolds 1 and 5, median of 5). If IHW is not installed it prints a skip line and exits 0. `scripts/bench_ihw.py` calls that helper and prints the `r sim ...` lines beside the Python backends; `backend=r` rows are appended to the csv when R ran. A local venv is enough for the Python backends. renv is only needed if you want the R column. Python still runs when R is missing. Replay of the stored R oracles stays on HiGHS.
