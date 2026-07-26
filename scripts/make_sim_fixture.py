#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)


def write_n2000() -> Path:
    out = FIXTURES / "sim_n2000_seed1.npz"
    rng = np.random.default_rng(1)
    np.savez(out, p=rng.uniform(size=2000), x=rng.uniform(size=2000))
    print(f"wrote {out.relative_to(ROOT)} n=2000")
    return out


def write_n5000() -> Path:
    out = FIXTURES / "sim_n5000_seed42.npz"
    rng = np.random.default_rng(42)
    n = 5000
    cov = rng.uniform(0.0, 3.0, size=n)
    signals = rng.binomial(1, 0.12, size=n).astype(np.bool_)
    z = rng.normal(loc=signals * cov)
    p = 1.0 - norm.cdf(z)
    np.savez(out, p=p.astype(np.float64), x=cov.astype(np.float64))
    print(f"wrote {out.relative_to(ROOT)} n={n}")
    return out


which = sys.argv[1] if len(sys.argv) > 1 else "n2000"
if which == "n2000":
    write_n2000()
elif which == "n5000":
    write_n5000()
elif which == "all":
    write_n2000()
    write_n5000()
else:
    raise SystemExit("usage: make_sim_fixture.py [n2000|n5000|all]")
