# Benchmarks

Protocol: per-design multi-conductor capacitance extraction (2 excitations,
shared assembly/setup/predictions), verified relative residual <= 1e-8 on the
target discretization. Machine: Apple M4 Pro, 24 GB. All numbers reproducible
via `scripts/run_all_benchmarks.sh` (see docs/DATA.md for dataset recipes).

## Tier 1: forward model vs practitioner alternatives (real Q3D data)

| model | cap MAPE iid | cap MAPE out-of-hull | g err iid | chi err iid |
|---|---|---|---|---|
| nearest-design lookup | 1.75% | 3.50% | 4.03% | 28.72% |
| kNN-5 | 1.11% | 3.96% | 2.17% | 13.58% |
| linear | 3.02% | 3.50% | 7.79% | 20.70% |
| random forest | 2.19% | 4.12% | 5.01% | 15.48% |
| **Ansatz GBR** | 0.31% | 3.51% | 0.70% | 1.71% |

## Tier 2: routed solver, mean wall-clock per verified design solve (s)

| n | sparse direct | AMG-CG (pyamg) | surrogate + CG | surrogate + MG-PCG | surrogate + AMG-CG | Ansatz router | oracle |
|---|---|---|---|---|---|---|---|
| 255 | 0.0051 | 0.0142 | 0.0838 | 0.1434 | 0.0400 | 0.0051 | 0.0051 |
| 511 | 0.0198 | 0.0398 | 0.4169 | 0.6458 | 0.1057 | 0.0198 | 0.0198 |
| 1023 | 0.0982 | 0.1585 | 3.1313 | 2.8422 | 0.2398 | 0.0982 | 0.0982 |
| 2047 | 0.5011 | 0.7079 | 32.0903 | 14.2311 | 0.8383 | 0.5011 | 0.5011 |

- Router within **1.000x** of the per-instance oracle overall; decision overhead 0.011 ms.
- Overall speedup vs best fixed pipeline: **1.00x** (worst per-instance ratio vs best fixed: 1.00).
- Verification failures across all benchmark cells: **0** (every returned field meets tolerance; failures would fall back to direct).

Figures: `paper/figs/`. HINTS numbers are reported at n<=511 where its
fixed schedule terminates within budget; it is dominated at every size.