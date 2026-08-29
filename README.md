# ihwkit

Independent Hypothesis Weighting for covariate-weighted multiple testing.

## Runtime

The installable library has one implementation: NumPy arrays and required Numba kernels. The weight optimization uses the dense simplex solver in `src/ihw.py`. SciPy is not an installable runtime dependency and is not a hidden fallback.

The supported runtime is Python 3.12 or newer with NumPy and Numba. Install the project from the repository root in an active environment:

```bash
python -m pip install -e .
```

The public entry point is `adjust_ihw`:

```python
import numpy as np

from ihw import adjust_ihw

rng = np.random.default_rng(0)
pvalues = rng.uniform(size=5000)
covariates = rng.uniform(size=5000)

result = adjust_ihw(pvalues, covariates, alpha=0.1, seed=1)
result.adj_pvalues
result.weights
```

The default fit uses five outer folds and infinite lambda, so no inner lambda search runs. `exploratory=True` uses one fold for weight inspection and is not a testing guarantee. `lambdas="auto"` enables the built-in lambda grid and nested cross-validation.

`nbins="auto"` selects `max(1, min(40, n // 1500))`. Set `nbins` explicitly when a small fixture needs more than one group. Invalid input, partition, and option values raise `IHWValidationError`. A failed or nonfinite weight optimization raises `RuntimeError`.

The public function has no solver or JIT switches. `lp_backend` and `use_numba` are not accepted parameters.

## Data and peer methods

`data/manifest.json` records fixture paths, sizes, seeds, provenance, and release eligibility. The development loader is `tools/data_contract.py`; it normalizes `pvalues` and `covariates` to one-dimensional float64 arrays and validates optional frozen groups, folds, and lambda values.

The comparison implementations are tool-owned and are not part of the installable API:

- **`ihwkit_numpy_numba`:** the production library path.
- **`ihwkit_numpy`:** a pinned pre-transition dense NumPy simplex baseline.
- **`ihwkit_scipy`:** a pinned pre-transition SciPy HiGHS baseline.
- **`pyihw`:** an adapter for the public pyihw package, reported unavailable when the package or supported API is absent.
- **`r_ihw`:** an adapter for native R IHW.
- **`julia_ihw`:** an adapter for the preliminary Julia package, reported unavailable when Julia or the package is absent.

Run a correctness and availability gate with:

```bash
python tools/check_peer_correctness.py
```

Synthetic lambda-infinity replay uses the frozen R oracle cases. Rejection counts, adjusted p-values, weights, and error status are separate checks. A close rejection count alone is not a parity claim.

Airway files are local diagnostics only. Their provenance and licensing are unresolved, so `.gitignore` excludes them and they never block a synthetic release gate.

## Benchmarks

`/home/eke/bin/zebrac` version 0.6.2 is the selected Linux benchmark binary. Benchmark metadata records its path, reported version, date, command, and runtime details.

Run a small process-level comparison with:

```bash
python tools/benchmark_zebrac.py --dataset sim_5000_seed42 --duration 5000 --warmup 3 --min-samples 10 --max-samples 10
```

The production command runs first, followed by available peer adapters. Raw zebrac JSON and adapter metadata are written under ignored `tmp/results/`. The measurement includes process startup, fixture loading, and any JIT initialization performed by an adapter. Keep cold process measurements separate from any future warmed algorithm measurement.

## Development

Run package-only tests with a temporary pytest runner when pytest is not installed in the minimal runtime environment:

```bash
uv run --no-project --with pytest --with numpy --with numba pytest -q tests
```

Run the repository-wide gate, including tool-owned data and adapter checks, with:

```bash
uv run --no-project --with pytest --with numpy --with numba pytest -q
```

`tests/` contains tests for the installable `ihw` package only. The repository-wide test command also collects tool-owned checks under `tools/tests/`. `tools/` contains local utilities, statistical evidence workflows, data-contract code, and peer adapters. `data/` owns fixtures and oracle records. The installable module remains `src/ihw.py`.
