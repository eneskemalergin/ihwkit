# ihwkit

Independent Hypothesis Weighting for covariate-weighted multiple testing.

## Runtime

The installable library has one implementation: NumPy arrays and required Numba kernels. The default infinite-lambda fit solves the separable Grenander allocation directly; finite-lambda fits retain the dense simplex solver in `src/ihw.py`. SciPy is not an installable runtime dependency and is not a hidden fallback.

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

`tools/peer.py` is the single local interface for comparison methods. Ordinary synthetic inputs are generated from readable names, sizes, and seeds. `bench/data/` contains one self-contained R reference file per retained data shape: the synthetic `n=5000` shape and the existing local airway p-value/base-mean shape. Each file stores the input once and the partitions and R outputs for one-fold infinity, five-fold infinity, and five-fold automatic lambda.

The supported comparison methods are tool-owned and are not part of the installable API:

- **`ihwkit_numpy_numba`:** the production library path.
- **`ihwkit_numpy`:** the NumPy-only reference, with the same direct default allocation and a retained dense finite-lambda simplex.
- **`ihwkit_scipy`:** a pinned pre-transition SciPy HiGHS baseline.
- **`pyihw`:** the reviewed public `pyihw` 0.2.0 API, using its `rng` and lambda conventions explicitly.
- **`r_ihw`:** native R IHW, with the installed IHW version read from the completed run.

Run one method directly with:

```bash
python tools/peer.py --method ihwkit_numpy_numba --dataset sim_5000_seed42
```

Run the generated production correctness gate with:

```bash
python tools/check_peer_correctness.py
```

Synthetic lambda-infinity replay uses the fixed R reference cases. Rejection counts, adjusted p-values, weights, and error status are separate checks. A close rejection count alone is not a parity claim.

There is no root `data/` tree, manifest, checksum, detached metadata file, or benchmark download step. [`bench/data/README.md`](bench/data/README.md) records the human-readable provenance and the known limitation of the existing airway export. Airway remains diagnostic and never blocks a synthetic release gate.

## Benchmarks

The local `bench` entry point keeps correctness, parity, statistical validity, robustness, and performance as separate questions. The full study runs those tracks in a visible order and reports them separately; it never reduces them to a combined score.

Show the current evidence matrix with:

```bash
python -m bench matrix
```

Run the cheap tracks directly:

```bash
python -m bench correctness
python -m bench parity
python -m bench robustness
python -m bench validity --quick
```

`python -m bench references` lists the immutable R records without running R. An explicit `--refresh DATASET` reruns R IHW against the arrays already stored for that dataset; routine benchmarks never refresh references.

The robustness command currently returns nonzero because the finite-lambda airway replay still reports LP infeasibility. The direct default path now passes both airway infinite-lambda replays and the generated `mixture_mild`, `n=3000`, seed-2034 case. The command still writes the complete report, including the independently passing weighted-BH checks; failures never become uniform-weight fallbacks.

The current comparative report is [`bench/REPORT.md`](bench/REPORT.md). Its light/dark summary figures use horizontal FDR and power intervals, tolerance-scaled fixed-reference parity, an absolute cost matrix, and a peer-to-ihwkit ratio matrix. Numerical robustness and collapsible detailed tables remain beside the visual summaries. Missing, failed, and unavailable methods remain visible.

The validity runner writes a row for every attempted fit plus a compact summary under ignored `tmp/results/`. Global-null FDR is measured as the probability of any rejection, not as the fraction of hypotheses rejected. The quick run is a wiring smoke test, not calibration evidence.

For process-level performance, install [`zebrac`](https://github.com/eneskemalergin/zebrac) and make the executable available on `PATH`. The runner records the resolved executable path, reported version, date, command, and runtime details for each local result.

Run a small process-level comparison with:

```bash
python -m bench performance --dataset sim_5000_seed42 --duration 5000 --warmup 3 --min-samples 10 --max-samples 10
```

The default measures only `ihwkit_numpy_numba`, because that is the implementation expected to change. Request comparisons explicitly with `--methods`, for example `--methods ihwkit_numpy_numba r_ihw`. Raw zebrac JSON and readable comparison metadata are written under ignored `tmp/results/`. The measurement includes process startup, input generation or loading, and any JIT initialization performed by a method. The full study reports these complete-process measurements separately from repeated fit calls after warmup.

Run the full benchmark and regenerate the report with the optional peer and plotting environment:

```bash
uv run --no-project --with pytest --with numpy --with numba --with scipy --with pyihw==0.2.0 --with matplotlib python -m bench study
```

`bench/peer-performance.json` retains the dated machine-local measurements for comparison methods that normally do not change. The default full study remeasures production and reuses that readable table. Use `--refresh-peers` deliberately when peer code, versions, the machine, or the measurement protocol changes. Fixed R parity outputs remain in `bench/data/` and are never regenerated by the study command.

## Development

Run package-only tests with a temporary pytest runner when pytest is not installed in the minimal runtime environment:

```bash
uv run --no-project --with pytest --with numpy --with numba pytest -q tests
```

Run the repository-wide gate, including generated inputs, frozen R replay, and peer checks, with:

```bash
uv run --no-project --with pytest --with numpy --with numba pytest -q
```

`tests/` contains tests for the installable `ihw` package only. The repository-wide test command also collects the consolidated tool and benchmark checks under `tools/tests/`. `bench/` is the single human-facing evidence entry point and owns retained benchmark data. `tools/` contains focused implementations, simulations, and the peer interface. The installable module remains `src/ihw.py`.
