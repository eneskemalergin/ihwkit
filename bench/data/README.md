# Benchmark data

This directory contains the small amount of data that a benchmark cannot
honestly regenerate during every run: inputs and outputs for fixed comparison
with R IHW 1.40.0. There is one compressed NumPy file per data shape. Each file
contains the p-values and covariate once, followed by the R groups, folds,
selected lambdas, weights, adjusted p-values, and rejection count for the
named configurations. The recorded seed controls both R's process-wide random
stream and IHW's fold seed when a record is deliberately refreshed.

- `sim_5000_r_ihw_1_40_0.npz` is the repository's deterministic Ignatiadis-style
  simulation at `n=5000`, seed 42, and signal fraction 0.15.
- `airway_r_ihw_1_40_0.npz` uses the existing local airway DESeq2 p-value and
  base-mean export. It has 33,469 rows. Its experiment is GEO
  [GSE52778](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE52778); the
  Bioconductor airway data package is LGPL. The exact historical DESeq2 version
  used for this already-existing export was not retained, so the file must not
  be presented as a fully reproducible DESeq2 analysis.

The current R results are directly inspectable through `python -m bench
references` and the replay reports:

| Reference | Folds | Lambda policy or selected values | R rejections | Gate |
|---|---:|---|---:|:---:|
| `sim_5000_inf_n1` | 1 | infinity | 163 | yes |
| `sim_5000_inf_n5` | 5 | infinity in every fold | 159 | yes |
| `sim_5000_auto` | 5 | infinity, 1.5, 0.75, infinity, infinity | 158 | no |
| `airway_inf_n1` | 1 | infinity | 4,957 | no |
| `airway_inf_n5` | 5 | infinity in every fold | 4,892 | no |
| `airway_auto` | 5 | 2.75, 2.75, 2.75, 11, 22 | 4,887 | no |

The former local synthetic auto result did not record the R process seed used
for its inner random choices. The consolidated result was intentionally
regenerated after setting both the R process seed and IHW seed to 42; its
selected lambdas and 158 rejections replace the ambiguous older result.

No benchmark command downloads data. Routine commands also do not run R. They
test the changing Python implementation against these fixed results. Running
`python -m bench references` lists the records; adding `--refresh DATASET`
explicitly recomputes that dataset's three R results from the arrays already in
its file. Refresh only after an intentional reference-configuration change and
review the numerical changes before committing them. To adopt a different R IHW
version, first change the readable file name and `REFERENCE_SPECS`; the refresh
command refuses to put a different version into a file named for 1.40.0.

There is intentionally no manifest, checksum, or detached metadata file. The
human-readable source notes and configuration values live inside each data file
and are checked directly by the loader.
