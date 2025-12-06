#!/usr/bin/env python3
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "tests" / "fixtures" / "sim_n2000_seed1.npz"
out.parent.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)
np.savez(out, p=rng.uniform(size=2000), x=rng.uniform(size=2000))
