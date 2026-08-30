# ihwkit

`ihwkit` is a small NumPy implementation of Independent Hypothesis Weighting for covariate-weighted multiple testing.

Version 0.1 deliberately offers one production method:

- five-fold cross-weighting by default;
- the unregularized allocation, also called infinite lambda in the IHW literature;
- Benjamini-Hochberg or Bonferroni adjustment;
- ordinal covariates with automatic equal-frequency grouping;
- a NumPy-only runtime with no solver, JIT, or backend choice.

Finite regularization is not a hidden optional path in 0.1. It remains future work requiring explicit statistical and numerical acceptance criteria.

## Install and use

The supported runtime is Python 3.12 or newer with NumPy:

```bash
python -m pip install -e .
```

The public entry point is `adjust_ihw`:

```python
import numpy as np

from ihw import adjust_ihw

rng = np.random.default_rng(0)
pvalues = rng.uniform(size=5_000)
covariates = rng.uniform(size=5_000)

result = adjust_ihw(pvalues, covariates, alpha=0.1, seed=1)
result.adj_pvalues
result.weights
```

The result also contains weighted p-values, group and fold assignments, the requested alpha, effective bin and fold counts, covariate and adjustment types, and full-family group counts.

`nbins="auto"` selects `max(1, min(40, n // 1500))`. Set it explicitly for a small example that needs more than one group. `exploratory=True` learns and applies weights on one fold for inspection; it is not the confirmatory default. Frozen `groups` and `folds` support direct replay, while `m_groups` supports a filtered subset whose full family counts are known.

Nominal covariates and Bonferroni adjustment are implemented and covered by structural tests. The current parity and full simulation evidence is much broader for ordinal covariates with BH, so the benchmark report does not imply equal validation for every accepted option.

## Statistical boundary

IHW needs a covariate that is informative about power while satisfying the required null-independence conditions. A data-derived covariate is not suitable merely because it predicts small p-values. Cross-weighting keeps each hypothesis out of the data used to learn its weight, but it does not repair an invalid covariate, arbitrary dependence, or invalid upstream p-values.

The learned weights depend on the requested `alpha`. Consequently, `adj_pvalues` answer the decision problem for that fitted alpha; one fit should not be presented as an alpha-free q-value curve.

The current simulation study covers named independent-null and mixture scenarios. It does not prove universal FDR control, and it does not yet cover arbitrary dependence, discrete p-values, or every filtered-family design. The limits and failed or unavailable fits remain visible in [`bench/REPORT.md`](bench/REPORT.md).

## Benchmark and peer comparisons

`python -m bench` keeps correctness, fixed R parity, statistical validity, robustness, speed, and process memory as separate questions. There is no combined winner score.

The comparison methods are tool-owned, not alternate public backends:

- **`ihwkit`:** the production NumPy method;
- **`ihwkit_scipy`:** the retained dense SciPy/HiGHS implementation;
- **`pyihw`:** the reviewed public PyIHW 0.2.0 interface;
- **`r_ihw`:** Bioconductor IHW, used to create the fixed R references.

The old NumPy peer was removed because production is now the optimized NumPy implementation; timing both labels compared the method with a slower copy of itself.

Useful commands are:

```bash
python -m bench matrix
python -m bench correctness
python -m bench parity
python -m bench robustness
python -m bench validity --quick
python tools/peer.py --method ihwkit --dataset sim_5000_seed42
```

`bench/data/` contains the synthetic and existing local airway arrays needed for fixed R comparison. Routine benchmarks never download data or rerun R. `python -m bench references` lists the records, and the explicit `--refresh DATASET` form recomputes R results from the arrays already present. There is intentionally no manifest, checksum, or detached identity machinery; [`bench/data/README.md`](bench/data/README.md) records the readable ownership and limitations.

For process measurements, install [`zebrac`](https://github.com/eneskemalergin/zebrac) and run:

```bash
python -m bench performance --dataset sim_5000_seed42 --duration 5000 --warmup 3 --min-samples 10 --max-samples 10
```

The full optional comparison environment is:

```bash
uv run --no-project --with pytest --with numpy --with scipy --with pyihw==0.2.0 --with matplotlib python -m bench study
```

Peer timing is retained in `bench/peer-performance.json` because unchanged external methods do not need to be remeasured on every production edit. Use `--refresh-peers` when a peer, runtime, machine, or protocol actually changes.

## Development

Run the package tests with:

```bash
uv run --no-project --with pytest --with numpy pytest -q tests
```

Run the complete repository gate with:

```bash
uv run --no-project --with pytest --with numpy --with scipy pytest -q
```

The installable implementation remains one file, `src/ihw.py`. Package tests live in `tests/`; peer and benchmark checks live in `tools/tests/`; public benchmark evidence lives in `bench/`.
