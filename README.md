# ihwkit

Independent Hypothesis Weighting. A local NumPy + SciPy module: covariate-weighted FDR via Grenander estimation, a linear program (`scipy.optimize.linprog`, HiGHS), and weighted Benjamini–Hochberg.

Needs NumPy and SciPy. From the repository root:

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

`IHWResult` also holds `weighted_pvalues`, `groups`, `folds`, `nbins`, and `nfolds`.
