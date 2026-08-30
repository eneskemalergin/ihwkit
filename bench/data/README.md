# Benchmark data

This directory contains only the arrays and fixed R outputs that cannot be honestly regenerated during every benchmark run. Each compressed NumPy file stores its p-values and covariates once, followed by the R groups, folds, weights, adjusted p-values, and rejection count for one-fold and five-fold unregularized IHW 1.40.0.

- `sim_5000_r_ihw_1_40_0.npz` is the deterministic local Ignatiadis-style simulation at `n=5000`, seed 42, and signal fraction 0.15.
- `airway_r_ihw_1_40_0.npz` is the existing local airway DESeq2 p-value and base-mean export with 33,469 rows. The experiment is GEO [GSE52778](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE52778), and the Bioconductor airway package is LGPL. The exact historical DESeq2 version used for this export was not retained, so this is a real-data shape and not a claim that the upstream DESeq2 analysis is completely reproducible.

The records are directly inspectable with `python -m bench references`:

| Reference         | Folds | R rejections | Release gate |
| ----------------- | ----: | -----------: | :----------: |
| `sim_5000_inf_n1` |     1 |          163 |     yes      |
| `sim_5000_inf_n5` |     5 |          159 |     yes      |
| `airway_inf_n1`   |     1 |        4,957 |      no      |
| `airway_inf_n5`   |     5 |        4,892 |      no      |

The `inf` fragment is retained in the reference name because it states the R configuration that created the historical record. The public ihwkit 0.1 API has no lambda option; this unregularized allocation is its only method.

No benchmark command downloads data. Routine correctness, parity, validity, robustness, and performance commands also do not run R. The explicit `python -m bench references --refresh DATASET` command recomputes that dataset's two R results from the arrays already in its file. Review numerical changes before committing them, and use a new readable filename before adopting a different R IHW version.

There is intentionally no manifest, checksum, or detached metadata file. The human-readable source notes are here, and the small configuration values are stored with the arrays and checked by the loader.
