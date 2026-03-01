# ihwkit

Independent Hypothesis Weighting. A local NumPy + SciPy module: covariate-weighted FDR via Grenander estimation, a linear program (`scipy.optimize.linprog`, HiGHS), and weighted Benjamini-Hochberg.

Needs NumPy and SciPy. Install from the repository root with `pip install -e .` (or `uv pip install -e .`), then `from ihw import adjust_ihw`. Without installing, from the repository root:

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

The default weight LP is HiGHS (`lp_backend="highs"`). `lp_backend="numpy"` is an experimental dense simplex and may differ from HiGHS or from R. SciPy is still required for the default path. `scripts/compare_lp_backends.py` runs both backends on `tests/fixtures/sim_n2000_seed1.npz` (nfolds=1, λ=∞): median of 5 wall times, rejection counts, max-abs of weights and adj-p. Replay of the stored R oracles stays on HiGHS.
