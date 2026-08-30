<!-- markdownlint-disable MD024 -->

# Changelog

All notable user-visible changes to ihwkit are documented here. Each release begins with a short summary, followed by the shipped interface, verification, and known limits.

## [0.1.0] - 2026-08-29

This is the initial release of ihwkit: a one-module, NumPy-only implementation of unregularized Independent Hypothesis Weighting. It provides one production path and a public evidence report that keeps implementation parity, statistical behavior, numerical robustness, speed, and process memory separate.

### Added

- `adjust_ihw` with five-fold cross-weighting by default, unregularized weight allocation, and weighted Benjamini-Hochberg or Bonferroni adjustment.
- Ordinal covariates with automatic equal-frequency grouping, nominal covariates, frozen groups and folds for replay, and full-family group counts for filtered analyses.
- `IHWResult` with adjusted and weighted p-values, learned weights, partitions, effective configuration, and family counts.
- Boundary validation for aligned one-dimensional finite inputs, p-values in `[0, 1]`, valid alpha, integer-valued partitions, contiguous labels, and compatible family counts.
- A repository benchmark suite and [benchmark report](bench/REPORT.md) covering fixed R IHW parity, named FDR and power simulations, numerical stress cases, warmed fit time, complete-process time, and peak RSS.
- Two self-contained [fixed R reference records](bench/data/README.md): one deterministic synthetic case used by the parity gate and one airway-shaped diagnostic record.

### Verified

- The complete repository gate passes 85 tests across the package, fixed references, peer adapters, and benchmark reporting support.
- Fixed synthetic replays agree with R IHW 1.40.0 in rejection decisions, adjusted p-values, and learned weights within the declared tolerances in the benchmark report.
- The recorded simulation study reports fit failures separately and retains the dense-covariate power loss alongside the scenarios that favor ihwkit.
- The recorded performance study separates warmed fitting from process startup and reports peak RSS without combining them into one score.

### Known limitations

- Version 0.1 implements only the unregularized allocation. Finite regularization is not part of the public API.
- The broadest current empirical evidence is for ordinal covariates with BH under the named simulation designs. Nominal covariates and Bonferroni adjustment have narrower coverage.
- Learned weights depend on the requested alpha, so one fitted adjusted-p-value vector is not an alpha-free q-value curve.
- The current study does not establish FDR control under arbitrary dependence, discrete p-values, invalid covariates, or filtered-family designs beyond structural checks.
