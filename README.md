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

## Peer comparisons

`tools/peer.py` is the single local interface for comparison methods. Ordinary synthetic inputs are generated from readable names, sizes, and seeds. The only stored comparison data are two self-contained R records under `tools/fixtures/`, containing the exact inputs, partitions, and outputs used by frozen parity replay.

The supported comparison methods are tool-owned and are not part of the installable API:

- **`ihwkit_numpy_numba`:** the production library path.
- **`ihwkit_numpy`:** a pinned pre-transition dense NumPy simplex baseline.
- **`ihwkit_scipy`:** a pinned pre-transition SciPy HiGHS baseline.
- **`pyihw`:** the reviewed public `pyihw` 0.2.0 API, using its `rng` and lambda conventions explicitly.
- **`r_ihw`:** native R IHW, with the installed IHW version read from the completed run.

Run one method directly with:

```bash
python tools/peer.py --method ihwkit_numpy_numba --dataset sim_5000_seed42
```

Run a correctness and availability gate with:

```bash
python tools/check_peer_correctness.py
```

Synthetic lambda-infinity replay uses the frozen R oracle cases. Rejection counts, adjusted p-values, weights, and error status are separate checks. A close rejection count alone is not a parity claim.

Airway files are local diagnostics only. Their provenance and licensing are unresolved, so `.gitignore` excludes them and they never block a synthetic release gate.

## Benchmarks

Install [`zebrac`](https://github.com/eneskemalergin/zebrac) and make the executable available on `PATH`. The benchmark runner records the resolved executable path, reported version, date, command, and runtime details for each local result.

Run a small process-level comparison with:

```bash
python tools/benchmark_zebrac.py --dataset sim_5000_seed42 --duration 5000 --warmup 3 --min-samples 10 --max-samples 10
```

The production command runs first, followed by available peer methods. Raw zebrac JSON and readable comparison metadata are written under ignored `tmp/results/`. The measurement includes process startup, input generation or loading, and any JIT initialization performed by a method. Keep cold process measurements separate from any future warmed algorithm measurement.

## Development

Run package-only tests with a temporary pytest runner when pytest is not installed in the minimal runtime environment:

```bash
uv run --no-project --with pytest --with numpy --with numba pytest -q tests
```

Run the repository-wide gate, including generated inputs, frozen R replay, and peer checks, with:

```bash
uv run --no-project --with pytest --with numpy --with numba pytest -q
```

`tests/` contains tests for the installable `ihw` package only. The repository-wide test command also collects the consolidated peer checks under `tools/tests/`. `tools/` contains local utilities, simulations, the peer interface, frozen R records, and statistical evidence workflows. Ignored `data/` is only for local diagnostics such as airway. The installable module remains `src/ihw.py`.
