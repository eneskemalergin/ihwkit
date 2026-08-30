<!-- markdownlint-disable MD033 MD041 -->

# ihwkit benchmark report

Recorded: 2026-08-30T02:01:17+00:00

This is the current measurement baseline for correctness, R parity, statistical behavior, numerical robustness, speed, and process memory. It is a presentation of evidence, not a combined winner score. Failed and unavailable fits remain visible.

## Method labels used throughout

| label               | implementation                                             | role in this report                                    |
| ------------------- | ---------------------------------------------------------- | ------------------------------------------------------ |
| **ihwkit**          | current installable NumPy + Numba method                   | subject under evaluation                               |
| **NumPy reference** | NumPy-only direct default plus dense finite-lambda simplex | correctness and scaling reference, not installable API |
| **SciPy/HiGHS**     | retained implementation using SciPy's HiGHS solver         | numerical and performance reference; never a fallback  |
| **pyihw**           | public pyihw 0.2.0 package                                 | external Python comparison                             |
| **R IHW**           | Bioconductor IHW 1.40.0                                    | fixed parity authority and external timing comparison  |
| **BH**              | unweighted Benjamini-Hochberg                              | statistical baseline only                              |

`ihwkit` always means the production method in figures and tables. Longer internal method identifiers appear only in raw JSON.

## Results at a glance

| question              |                           result | judgement                                                              |
| --------------------- | -------------------------------: | ---------------------------------------------------------------------- |
| repository tests      |                               ok | 97 passed in 2.76s                                                     |
| generated correctness |                             pass | finite output and structural invariants on named cases                 |
| fixed R parity        |                             pass | strong agreement inside the declared fixed synthetic envelope          |
| numerical robustness  |              1 failed diagnostic | current release blocker; no fallback concealed the failures            |
| validity study        |             0 failed ihwkit fits | FDR and power remain conditional on successful fits                    |
| FDR screen            |  0/6 intervals wholly above 0.10 | no clear inflation in these named scenarios; not a universal guarantee |
| paired power          |  3/4 intervals wholly above zero | gains in three scenarios and a loss in the dense-covariate scenario    |
| performance           | 20 process rows + 20 warmed rows | current direct-default measurements; no universal speed claim          |

### Direct reading

- **Numerical agreement is strong where it is defined.** The fixed five-fold synthetic replay agrees with R IHW in rejection decisions and full output vectors at errors far below the declared tolerance.
- **Numerical reliability is not finished.** The remaining failed diagnostics stay visible in the detailed table. They block a blanket robustness claim, but no longer describe the default infinite-lambda path as failing.
- **The development-scale statistical result is encouraging but conditional.** This is 1,000 replicates per null scenario and 200 per alternative scenario; 0 ihwkit failures are reported separately instead of converted to zero discoveries.
- **Performance is mixed and size-dependent.** n=5000: median warmed-fit rank 1/5 (6.7x faster than the next measured method); median process-time rank 5/5 and median RSS rank 5/5; n=50000: median warmed-fit rank 1/5 (5.7x faster than the next measured method); median process-time rank 4/5 and median RSS rank 5/5.
- **The peer timing environment matches this study.** Peer measurements are reused only while that human-readable environment and protocol remain applicable.

## Statistical evidence

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/01-statistical-evidence-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="figures/01-statistical-evidence-light.svg">
  <img src="figures/01-statistical-evidence-light.svg" alt="Fixed-reference errors, empirical FDR intervals, and paired power differences" width="100%">
</picture>

The FDR panel shows empirical estimates and 95% Monte Carlo intervals against nominal alpha 0.10. The paired-power panel reports ihwkit minus BH in percentage points; positive values favor ihwkit. The parity panel expresses maximum vector differences as a percentage of the absolute tolerance, so 100% is the declared boundary and farther left is better. Failed ihwkit fits keep their own denominator and never become zero-discovery runs.

<details>
<summary>Simulation designs and statistical estimands</summary>

Every method receives the same truth-labelled draw at alpha 0.10. BH is unweighted; ihwkit uses five folds, automatic bin count, and infinite lambda. For each scenario:

- **Global null:** n=3,000; 1,000 attempts per method. All hypotheses are null; p-values and a continuous uniform covariate are independent.
- **Tied null:** n=3,000; 1,000 attempts per method. All hypotheses are null; p-values are uniform and independent of a rounded, skewed log-normal covariate with ties.
- **Mild mixture:** n=3,000; 200 attempts per method. Ten percent are alternatives; their one-sided normal signal increases with a continuous uniform covariate.
- **Sparse mixture:** n=3,000; 200 attempts per method. Five percent are alternatives under the same covariate-dependent signal model as Mild mixture.
- **Ignatiadis:** n=3,000; 200 attempts per method. Twelve percent are alternatives in the retained Ignatiadis-style covariate-dependent normal model.
- **Dense covariate:** n=3,000; 200 attempts per method. Fifteen percent are alternatives; a wide log-normal covariate drives stronger high-end signals.

FDR is the mean replicate false-discovery proportion. Under either all-null scenario this equals the probability of at least one rejection, and the report uses a Wilson interval; other FDR intervals use the mean plus or minus 1.96 Monte Carlo standard errors, bounded to [0, 1]. Power is the fraction of true alternatives rejected. Paired differences compare ihwkit with BH on the same successful draw; unsuccessful ihwkit fits retain their own denominator and are excluded from the paired estimand.

</details>

## Absolute compute cost

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/02-process-cost-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="figures/02-process-cost-light.svg">
  <img src="figures/02-process-cost-light.svg" alt="Absolute warmed-fit time, complete-process time, and peak RSS at 5k, 15k, and 50k hypotheses" width="100%">
</picture>

Each row is one explicit hypothesis-family size; point position is the sample mean and horizontal timing whiskers are +/- one sample standard deviation. Warmed Python fits remain inside one benchmark process after input construction. R IHW still includes serialization, the adapter, and an R process launch. Complete-process measurements launch a fresh command and therefore include startup, imports, deterministic input generation, solver work, and Numba initialization. Peak RSS is whole-process memory.

The main scaling figures use exactly 5k, 15k, and 50k hypotheses, shown as explicit axis labels. The one-bin n=500 startup floor remains in the detailed tables but is excluded from the scaling figures.

## Relative compute cost

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/03-warm-fit-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="figures/03-warm-fit-light.svg">
  <img src="figures/03-warm-fit-light.svg" alt="Peer-to-ihwkit ratios for warmed-fit time, complete-process time, and peak RSS" width="100%">
</picture>

Every endpoint divides a peer mean by the ihwkit mean on the same input. The orange 1x line is equal measured cost: timing points left of it favor the peer, while points to the right favor ihwkit; memory points to the left use less RSS than ihwkit. Lines expose the magnitude and direction of each comparison without replacing the absolute measurements above.

## Detailed statistical results

| scenario        | method | ok/attempted |   FDR (95% interval) | power | mean rejections |
| --------------- | ------ | -----------: | -------------------: | ----: | --------------: |
| Global null     | BH     |    1000/1000 | 0.082 [0.067, 0.101] |       |             0.1 |
| Global null     | ihwkit |    1000/1000 | 0.082 [0.067, 0.101] |       |             0.1 |
| Tied null       | BH     |    1000/1000 | 0.082 [0.067, 0.101] |       |             0.1 |
| Tied null       | ihwkit |    1000/1000 | 0.088 [0.072, 0.107] |       |             0.1 |
| Mild mixture    | BH     |      200/200 | 0.095 [0.088, 0.101] | 0.138 |            46.0 |
| Mild mixture    | ihwkit |      200/200 | 0.095 [0.089, 0.101] | 0.173 |            57.6 |
| Sparse mixture  | BH     |      200/200 | 0.103 [0.091, 0.115] | 0.091 |            15.5 |
| Sparse mixture  | ihwkit |      200/200 | 0.095 [0.086, 0.105] | 0.119 |            20.2 |
| Ignatiadis      | BH     |      200/200 | 0.084 [0.079, 0.089] | 0.155 |            61.2 |
| Ignatiadis      | ihwkit |      200/200 | 0.085 [0.080, 0.090] | 0.190 |            75.0 |
| Dense covariate | BH     |      200/200 | 0.084 [0.081, 0.086] | 0.540 |           265.6 |
| Dense covariate | ihwkit |      200/200 | 0.084 [0.081, 0.086] | 0.536 |           263.7 |

Paired differences use only replicates where both BH and production succeeded. They improve precision for method comparison but do not erase production failures.

| scenario        | paired reps | FDP difference (MCSE) | power difference (MCSE) |
| --------------- | ----------: | --------------------: | ----------------------: |
| Global null     |        1000 |      +0.0000 (0.0025) |                         |
| Tied null       |        1000 |      +0.0060 (0.0035) |                         |
| Mild mixture    |         200 |      +0.0004 (0.0030) |        +0.0348 (0.0015) |
| Sparse mixture  |         200 |      -0.0074 (0.0060) |        +0.0286 (0.0020) |
| Ignatiadis      |         200 |      +0.0010 (0.0025) |        +0.0348 (0.0014) |
| Dense covariate |         200 |      -0.0003 (0.0006) |        -0.0037 (0.0007) |

## Performance interpretation

The default infinite-lambda path solves the separable Grenander allocation directly. Finite-lambda fits still use the general dense LP, so this optimization does not erase their numerical failures or imply a universal solver claim.

- **Measured position:** n=5000: median warmed-fit rank 1/5 (6.7x faster than the next measured method); median process-time rank 5/5 and median RSS rank 5/5; n=50000: median warmed-fit rank 1/5 (5.7x faster than the next measured method); median process-time rank 4/5 and median RSS rank 5/5.
- **Complete process:** Import, JIT, input construction, and fitting are intentionally combined because that is what a new command experiences.
- **Scope contrast:** process divided by warmed-fit time is descriptive, not a pure startup decomposition. A large factor says that fit-only timing cannot explain command latency; it does not assign the difference to one component.
- **Remaining numerical boundary:** the finite-lambda airway failure stays visible and blocks a blanket robustness claim.

### Representative measurements at 5k hypotheses

| method          | warm median | process median | process / warm | process peak RSS | status |
| --------------- | ----------: | -------------: | -------------: | ---------------: | :----: |
| ihwkit          |    6.044 ms |     965.660 ms |         159.8x |         183.5 MB |   ok   |
| NumPy reference |   55.563 ms |     211.285 ms |           3.8x |          43.6 MB |   ok   |
| SciPy/HiGHS     |   64.497 ms |     484.135 ms |           7.5x |          84.2 MB |   ok   |
| pyihw           |   40.450 ms |     466.796 ms |          11.5x |          85.4 MB |   ok   |
| R IHW           |  465.416 ms |     616.047 ms |           1.3x |          92.8 MB |   ok   |

<details>
<summary>Protocol and comparability details</summary>

The label key at the start of this report defines every implementation. SciPy/HiGHS is a retained comparison and never a production dependency or fallback. BH is a truth-labelled simulation baseline, not a solver-performance peer. pyihw participates in generated-input timing but cannot accept the fixed covariate groups required for stored-vector parity.

All timed IHW lanes use alpha 0.1, five folds, automatic bin count, infinite lambda, seed 42, 3 warmups, and 10 measured samples. Exact parity instead uses the fixed R groups, folds, and fold lambdas stored with the reference.
The NumPy reference uses the same direct infinite-lambda allocation without Numba and is measured at every displayed size.

The n=500 automatic configuration has one bin. R IHW correctly reduces that case to ordinary BH with one effective fold-lambda value; the adapter preserves that result. Because this is a one-bin shortcut dominated by startup, it remains in the detailed tables but not the main scaling figures.

</details>

<details>
<summary>Fixed-reference parity details</summary>

Tolerance: absolute 1e-8 and relative 1e-6.

| method          |    status   | R rej | method rej | max adjusted-p difference | max weight difference | decision agreement |
| --------------- | :---------: | ----: | ---------: | ------------------------: | --------------------: | -----------------: |
| ihwkit          |     pass    |   159 |        159 |                  1.89e-15 |              4.00e-15 |           100.000% |
| NumPy reference |     pass    |   159 |        159 |                  1.89e-15 |              4.00e-15 |           100.000% |
| SciPy/HiGHS     |     pass    |   159 |        159 |                  2.55e-15 |              4.00e-15 |           100.000% |
| pyihw           | unavailable |   159 |            |                           |                       |                    |

The release gate itself contains one-fold and five-fold synthetic replays. This expanded table shows the five-fold reference across compatible local solvers. An unavailable fixed-partition API is not counted as a parity failure.

</details>

<details>
<summary>Correctness and robustness details</summary>

#### Generated correctness

| case                    | status | rejections | mean weight | error |
| ----------------------- | :----: | ---------: | ----------: | ----- |
| sim_500_seed42_inf_n1   |   ok   |          6 |    1.000000 |       |
| dense_500_seed42_inf_n1 |   ok   |         45 |    1.000000 |       |
| sim_5000_seed42_inf_n1  |   ok   |        163 |    1.000000 |       |
| sim_50000_seed42_inf_n1 |   ok   |       1777 |    1.000000 |       |
| sim_50000_seed42_inf_n5 |   ok   |       1694 |    1.000000 |       |
| sim_5000_auto_native    |   ok   |        155 |    1.000000 |       |

#### Fixed and generated robustness

| case                               | production | R rejections | BH on R weights | error                                                      |
| ---------------------------------- | :--------: | -----------: | :-------------: | ---------------------------------------------------------- |
| sim_5000_inf_n1                    |    pass    |          163 |       pass      |                                                            |
| sim_5000_inf_n5                    |    pass    |          159 |       pass      |                                                            |
| sim_5000_auto                      |    pass    |          158 |       pass      |                                                            |
| airway_inf_n1                      |    pass    |         4957 |       pass      |                                                            |
| airway_inf_n5                      |    pass    |         4892 |       pass      |                                                            |
| airway_auto                        |    fail    |         4887 |       pass      | RuntimeError: weight LP did not solve: infeasible solution |
| mixture_mild_n3000_seed2034_inf_n5 |     ok     |              |       n/a       |                                                            |

#### Seed-2034 solver comparison

| method          | status | rejections | error |
| --------------- | :----: | ---------: | ----- |
| ihwkit          |   ok   |         51 |       |
| NumPy reference |   ok   |         51 |       |
| SciPy/HiGHS     |   ok   |         51 |       |
| pyihw           |   ok   |         49 |       |

Weighted BH using stored R weights passes on every airway row. The remaining finite-lambda `airway_auto` error occurs before final p-value adjustment, while both infinite-lambda airway fits now pass. Successful peer fits show that a production failure is not evidence that the mathematical LP is infeasible.

</details>

<details>
<summary>All complete-process measurements</summary>

|     n | bins | method          | samples | median wall |  mean wall |   wall SD | median RSS | status |
| ----: | ---: | --------------- | ------: | ----------: | ---------: | --------: | ---------: | :----: |
|   500 |    1 | ihwkit          |      10 |  313.647 ms | 311.761 ms |  7.274 ms |   110.5 MB |   ok   |
|   500 |    1 | NumPy reference |      10 |  147.459 ms | 148.592 ms |  5.235 ms |    42.5 MB |   ok   |
|   500 |    1 | SciPy/HiGHS     |      10 |  150.116 ms | 149.833 ms |  3.800 ms |    42.4 MB |   ok   |
|   500 |    1 | pyihw           |      10 |  425.602 ms | 426.831 ms | 12.257 ms |    81.3 MB |   ok   |
|   500 |    1 | R IHW           |      10 |  527.626 ms | 529.259 ms |  7.140 ms |    85.7 MB |   ok   |
|  5000 |    3 | ihwkit          |      10 |  965.660 ms | 970.222 ms | 32.546 ms |   183.5 MB |   ok   |
|  5000 |    3 | NumPy reference |      10 |  211.285 ms | 212.060 ms |  5.691 ms |    43.6 MB |   ok   |
|  5000 |    3 | SciPy/HiGHS     |      10 |  484.135 ms | 479.663 ms | 11.614 ms |    84.2 MB |   ok   |
|  5000 |    3 | pyihw           |      10 |  466.796 ms | 471.502 ms | 15.723 ms |    85.4 MB |   ok   |
|  5000 |    3 | R IHW           |      10 |  616.047 ms | 618.213 ms | 21.866 ms |    92.8 MB |   ok   |
| 15000 |   10 | ihwkit          |      10 |  971.800 ms | 973.333 ms | 42.294 ms |   185.1 MB |   ok   |
| 15000 |   10 | NumPy reference |      10 |  318.490 ms | 318.187 ms |  3.277 ms |    44.5 MB |   ok   |
| 15000 |   10 | SciPy/HiGHS     |      10 |  596.740 ms | 596.157 ms | 11.830 ms |    85.8 MB |   ok   |
| 15000 |   10 | pyihw           |      10 |  530.559 ms | 529.690 ms | 10.623 ms |    86.8 MB |   ok   |
| 15000 |   10 | R IHW           |      10 |  703.509 ms | 705.273 ms | 11.464 ms |   107.5 MB |   ok   |
| 50000 |   33 | ihwkit          |      10 |     1.041 s |    1.040 s | 22.259 ms |   190.9 MB |   ok   |
| 50000 |   33 | NumPy reference |      10 |  722.021 ms | 725.399 ms | 15.424 ms |    50.8 MB |   ok   |
| 50000 |   33 | SciPy/HiGHS     |      10 |  996.118 ms |    1.000 s | 13.754 ms |    91.5 MB |   ok   |
| 50000 |   33 | pyihw           |      10 |  771.769 ms | 772.734 ms |  7.144 ms |    93.0 MB |   ok   |
| 50000 |   33 | R IHW           |      10 |     1.059 s |    1.062 s |  9.696 ms |   141.7 MB |   ok   |

</details>

<details>
<summary>All warmed-fit measurements</summary>

|     n | bins | method          | samples | median wall |  mean wall |    wall SD | status |
| ----: | ---: | --------------- | ------: | ----------: | ---------: | ---------: | :----: |
|   500 |    1 | ihwkit          |      10 |  562.283 us | 561.886 us |   7.061 us |   ok   |
|   500 |    1 | NumPy reference |      10 |  203.601 us | 205.237 us |   5.909 us |   ok   |
|   500 |    1 | SciPy/HiGHS     |      10 |  199.066 us | 201.928 us |   5.618 us |   ok   |
|   500 |    1 | pyihw           |      10 |  554.768 us | 555.793 us |   5.094 us |   ok   |
|   500 |    1 | R IHW           |      10 |  372.842 ms | 392.955 ms |  40.075 ms |   ok   |
|  5000 |    3 | ihwkit          |      10 |    6.044 ms |   6.059 ms |  44.105 us |   ok   |
|  5000 |    3 | NumPy reference |      10 |   55.563 ms |  55.974 ms |   1.722 ms |   ok   |
|  5000 |    3 | SciPy/HiGHS     |      10 |   64.497 ms |  65.330 ms |   2.227 ms |   ok   |
|  5000 |    3 | pyihw           |      10 |   40.450 ms |  41.837 ms |   3.959 ms |   ok   |
|  5000 |    3 | R IHW           |      10 |  465.416 ms | 470.021 ms |  22.308 ms |   ok   |
| 15000 |   10 | ihwkit          |      10 |   17.258 ms |  17.285 ms |  85.186 us |   ok   |
| 15000 |   10 | NumPy reference |      10 |  164.120 ms | 165.520 ms |   4.447 ms |   ok   |
| 15000 |   10 | SciPy/HiGHS     |      10 |  178.962 ms | 180.178 ms |   4.412 ms |   ok   |
| 15000 |   10 | pyihw           |      10 |  109.066 ms | 108.828 ms |   2.040 ms |   ok   |
| 15000 |   10 | R IHW           |      10 |  561.194 ms | 564.694 ms |  16.663 ms |   ok   |
| 50000 |   33 | ihwkit          |      10 |   58.633 ms |  58.661 ms | 280.810 us |   ok   |
| 50000 |   33 | NumPy reference |      10 |  554.357 ms | 552.390 ms |   9.714 ms |   ok   |
| 50000 |   33 | SciPy/HiGHS     |      10 |  569.839 ms | 570.915 ms |  17.964 ms |   ok   |
| 50000 |   33 | pyihw           |      10 |  333.134 ms | 334.035 ms |   2.418 ms |   ok   |
| 50000 |   33 | R IHW           |      10 |  909.196 ms | 920.787 ms |  30.432 ms |   ok   |

</details>

<details>
<summary>Environment, retained data, and rerun commands</summary>

Peer timing recorded: 2026-08-30T01:51:38+00:00

- **platform:** Linux-7.1.10-200.fc44.x86_64-x86_64-with-glibc2.43
- **cpu:** AMD Ryzen 9 3950X 16-Core Processor
- **logical_cpus:** 32
- **python:** 3.14.7
- **numpy:** 2.5.2
- **numba:** 0.67.0
- **scipy:** 1.18.1
- **pyihw:** 0.2.0
- **zebrac:** zebrac 0.6.2

The retained datasets are deterministic generated draws and the two self-contained files in `bench/data`: one n=5000 synthetic shape and one airway p-value/base-mean shape. No benchmark downloads data; `bench/data/README.md` records their human-readable source and derivation notes.

Run the complete study, reusing the dated peer timing table:

```bash
uv run --no-project --with pytest --with numpy --with numba --with scipy --with pyihw==0.2.0 --with matplotlib python -m bench study
```

Refresh unchanged peers only when their code, runtime, machine, or benchmark protocol changes:

```bash
uv run --no-project --with pytest --with numpy --with numba --with scipy --with pyihw==0.2.0 --with matplotlib python -m bench study --refresh-peers
```

Render the report again without rerunning measurements:

```bash
uv run --no-project --with numpy --with numba --with matplotlib python -m bench report
```

</details>

## Limits and next datasets

- The current real-data evidence is one airway shape. It is a useful numerical stress case, not broad real-data coverage.
- No reviewed local single-cell export is present, so real single-cell calibration, realistic count-based power, and donor-stability panels remain unmeasured.
- The statistical simulations assume the named data-generating mechanisms. They do not establish FDR control under arbitrary dependence, discrete p-values, or invalid covariates.
- Peak RSS is whole-process memory. It cannot be interpreted as solver allocation alone.
- Machine-local performance is evidence for this environment, not a universal speed or memory guarantee.

The simulation structure follows the ADEMP framework from [Morris, White, and Crowther (2019)](https://doi.org/10.1002/sim.8086). The separation of datasets, metrics, failures, and method scope follows the benchmarking guidance of [Weber et al. (2019)](https://doi.org/10.1186/s13059-019-1738-8). The report layout borrows z-fasta's useful pattern of correctness-first execution, absolute and ratio plots, and collapsible detailed evidence while intentionally using a much smaller local implementation.
