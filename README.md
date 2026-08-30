<!-- markdownlint-disable MD033 MD041 -->

<h1 align="center">ihwkit</h1>

<p align="center">
  <strong>Independent Hypothesis Weighting with a small, NumPy-only runtime.</strong>
</p>

<p align="center">
  <a href="#install"><img src="https://img.shields.io/badge/python-3.12%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.12+"></a>
  <a href="CHANGELOG.md#010---2026-08-29"><img src="https://img.shields.io/badge/version-0.1.0-8B5CF6?style=flat-square" alt="Version 0.1.0"></a>
  <img src="https://img.shields.io/badge/runtime-NumPy%20only-4D77CF?style=flat-square&amp;logo=numpy&amp;logoColor=white" alt="NumPy-only runtime">
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-release%20notes-7C3AED?style=flat-square" alt="Changelog"></a>
  <a href="bench/REPORT.md"><img src="https://img.shields.io/badge/benchmark-report-C17D10?style=flat-square" alt="Benchmark report"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-4B9D6E?style=flat-square" alt="MIT License"></a>
</p>

<p align="center">
  Cross-weighted multiple testing through one NumPy implementation, with no generic optimization solver or JIT dependency.
</p>

---

`ihwkit` learns hypothesis weights from an informative covariate, then applies weighted Benjamini-Hochberg or Bonferroni adjustment. The installable implementation is one Python module with one runtime dependency: NumPy.

Version 0.1 keeps the production scope deliberately narrow:

- Five-fold cross-weighting by default.
- The unregularized allocation, also called infinite lambda in the IHW literature.
- Benjamini-Hochberg or Bonferroni adjustment.
- Ordinal covariates with automatic equal-frequency grouping and support for nominal covariates.
- Frozen groups and folds for direct replay, plus full-family group counts for filtered analyses.
- One installed NumPy implementation rather than a backend selector.

Finite regularization is outside the 0.1 scope. Adding it requires separate statistical, numerical, and performance acceptance criteria.

## Install

A PyPI release is planned after the next small set of additions passes the source, wheel, clean-install, and supported-Python checks. Until then, install from the GitHub source:

```bash
git clone https://github.com/eneskemalergin/ihwkit.git
cd ihwkit
python -m pip install .
```

Use `python -m pip install -e .` for editable development. The distribution is named `ihwkit`; the intentionally small import module is `ihw`.

## Use

```python
import numpy as np

from ihw import adjust_ihw

rng = np.random.default_rng(0)
n = 5_000
power_covariate = rng.uniform(size=n)
nonnull = rng.random(n) < 0.1
pvalues = rng.uniform(size=n)

# Null p-values stay independent of the covariate; alternatives strengthen with it.
pvalues[nonnull] = rng.beta(0.8 - 0.6 * power_covariate[nonnull], 1.0)

result = adjust_ihw(pvalues, power_covariate, alpha=0.1, seed=1)
discoveries = np.flatnonzero(result.adj_pvalues <= result.alpha)

result.weights
result.adj_pvalues
discoveries
```

The result also contains weighted p-values, group and fold assignments, the requested alpha, effective bin and fold counts, covariate and adjustment types, and full-family group counts.

`nbins="auto"` selects `max(1, min(40, n // 1500))`. Set it explicitly for a small example that needs more than one group. `exploratory=True` learns and applies weights on one fold for inspection; it is not the confirmatory default. Frozen `groups` and `folds` support direct replay, while `m_groups` supports a filtered subset whose full family counts are known.

Benchmark evidence is centered on ordinal covariates with BH. Nominal covariates and Bonferroni adjustment have structural coverage, but should not be read as equally validated.

## Statistical boundary

IHW needs a covariate that is informative about power while satisfying the required null-independence conditions. A data-derived covariate is not suitable merely because it predicts small p-values. Cross-weighting keeps each hypothesis out of the data used to learn its weight, but it does not repair an invalid covariate, arbitrary dependence, or invalid upstream p-values.

The learned weights depend on the requested `alpha`. Consequently, `adj_pvalues` answer the decision problem for that fitted alpha; one fit should not be presented as an alpha-free q-value curve.

The current simulation study covers named independent-null and mixture scenarios. It does not prove universal FDR control, and it does not cover arbitrary dependence, discrete p-values, or filtered-family designs beyond structural checks. Limits, failed fits, and unavailable peers remain visible in the benchmark report.

## Public evidence

[`bench/REPORT.md`](bench/REPORT.md) keeps correctness, fixed R parity, statistical validity, numerical robustness, speed, and process memory as separate questions. It computes no combined winner score and does not turn an unavailable or failed fit into a successful result.

<p align="center">
  <a href="bench/REPORT.md">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="bench/figures/01-statistical-evidence-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="bench/figures/01-statistical-evidence-light.svg">
      <img src="bench/figures/01-statistical-evidence-light.svg" alt="Fixed R parity error, empirical FDR intervals, and paired power differences for the current ihwkit benchmark study" width="100%">
    </picture>
  </a>
</p>
<p align="center"><sub>Fixed-reference parity, named FDR screens, and paired power comparisons. Open the report for estimands, denominators, failures, absolute timings, process memory, peer ratios, and limitations.</sub></p>

The current recorded study shows:

- Fixed synthetic R replays pass the declared full-vector parity tolerance.
- Across the six named Monte Carlo screens, no 95% FDR interval lies wholly above the nominal 0.10 level.
- Three of four paired power intervals favor ihwkit over BH; the dense-covariate scenario shows a real power loss rather than being hidden by an overall score.
- ihwkit ranks first for warmed fit time, complete-process time, and peak RSS at the reported 5,000- and 50,000-hypothesis scales.

These are results for the recorded scenarios, host, versions, and scales, not universal guarantees. The report compares ihwkit with a distinct SciPy/HiGHS implementation, pyihw 0.2.0, R IHW 1.40.0, and unweighted BH where each comparison is meaningful.

<p align="center">
  <a href="bench/REPORT.md#absolute-compute-cost">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="bench/figures/02-process-cost-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="bench/figures/02-process-cost-light.svg">
      <img src="bench/figures/02-process-cost-light.svg" alt="Absolute warmed-fit time, complete-process time, and peak RSS across the current ihwkit scaling study" width="100%">
    </picture>
  </a>
</p>
<p align="center"><sub>Absolute machine-local warmed-fit time, complete-process time, and peak RSS. Lower is better; open the report for measurements, dispersion, versions, and scope.</sub></p>

From a repository checkout, `python -m bench matrix` lists every evidence track. The NumPy-only checks are directly runnable:

```bash
python -m bench correctness
python -m bench parity
python -m bench robustness
python -m bench validity --quick
```

Fixed-reference benchmarks use the two self-contained records in [`bench/data/`](bench/data/README.md); they do not download data or rerun R. The full report documents optional peers, complete-process measurement with zebrac, the retained timing environment, and explicit reference refreshes.

## Coming next

Near-term work, subject to the same evidence gates:

- Broader validation for dependence, discrete p-values, nominal covariates, and filtered families.
- More real omics benchmark shapes, beginning with proteomic, genomic, and donor-level single-cell cases.
- Table-first diagnostics, plus a measured Rust core and Python-binding experiment.

## Development

Run the package tests with:

```bash
uv run --no-project --with pytest --with numpy pytest -q tests
```

Run the complete repository gate with:

```bash
uv run --no-project --with pytest --with numpy --with scipy pytest -q
```

The installed product is `src/ihw.py`. Package tests live in `tests/`; `bench/` owns the public evidence and fixed records; `tools/` owns simulations and peer adapters. These repository-only files do not enter the wheel.

## License

MIT. See [LICENSE](LICENSE).

---

<p align="center"><em>
First frost on the ridge:<br>
a quiet slope draws the weight,<br>
spurious seeds fall.
</em></p>
