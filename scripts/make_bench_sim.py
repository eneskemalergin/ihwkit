#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "tmp" / "bench_sim.npz"
out.parent.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(11)
n = 8000
cov = rng.uniform(0.0, 3.0, size=n)
signals = rng.binomial(1, 0.12, size=n).astype(bool)
z = rng.normal(loc=signals * cov)
p = 1.0 - norm.cdf(z)
np.savez(out, p=p, x=cov)
print(f"wrote {out.relative_to(ROOT)} n={n}")
