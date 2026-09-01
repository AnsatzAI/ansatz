# Option A: 3D frequency-domain / eigenmode EM (Palace-class)

The 2D benchmark taught us where value cannot live (single-solve 2D SPD
problems: sparse direct wins, see BENCHMARKS.md). This document tracks the 3D
program, where the measured pain is real: SQuADDS reports resonator eigenmode
simulations at "up to an hour or more" per design (16-core Threadripper) and
full chips at "many hours or days"; every quantum hardware team runs these
loops through Ansys HFSS or AWS Palace.

## Why 3D flips the ensemble

- Direct factorization: O(N^2) flops / O(N^{4/3}) memory in 3D — dies around
  10^7 unknowns. The memory wall is the binding constraint, not flops.
- Eigenmode analysis requires shift-invert inner solves of *indefinite*
  curl-curl systems — the regime where AMG is weakest.
- Many-query structure is native, twice: geometry sweeps (frequency
  allocation / yield) and frequency sweeps (S-parameters).

The learned components therefore attack multipliers, not replacements:
eigensolver **target/shift prediction** (Tier-1 geometry -> frequency model),
**config routing** (cheapest verified (order, refinement) pair),
eigenvector **warm-starts** (roadmap; requires a Palace fork), and learned
**ROM frequency sweeps** (roadmap).

## Local baseline (Apple M4 Pro, 12 cores, Palace v0.13+ built from source)

Palace's shipped transmon example (transmon + quarter-wave CPW readout +
feedline on sapphire, coarse mesh):

| quantity | value |
|---|---|
| wall-clock (6 MPI ranks, end-to-end) | **206.9 s** |
| f0 (qubit-like mode) | 4.0991 GHz |
| f1 (readout mode) | 5.6033 GHz |
| Q0 | 18,553 |

This matches the documented reference results (~4.1 / 5.6 GHz, Q ~ 18.5k),
validating the toolchain end-to-end.

## Dev-scale experiments (this machine)

1. **Shift sensitivity** — eigensolver `Target` sweep around the true mode.
   Quantifies the wall-clock value of a Tier-1 frequency predictor.
   Results: see `runs/em3d_results.parquet` (table below once complete).
2. **Config frontier** — (FEM order, uniform refinement) cost/accuracy
   frontier vs the reference config. The routing table for
   cheapest-verified-config selection, verified via refinement-delta checks
   on the eigenfrequency.

## Compute plan

- **This Mac (dev tier)**: pipeline development, ~10^5–10^6 DOF solves at
  1–10 min each, overnight sweeps of O(100) solves. Sufficient for method
  validation and the dev-scale results above.
- **Campaign tier (10^7–10^8 DOF, full chips)**: AWS HPC instances
  (hpc7a/c7a-class, 100s of GB RAM, MPI) rented per benchmark campaign
  (est. $100s–low $1,000s per campaign), or university cluster time.
  Geometry-swept training-set generation (DeviceLayout.jl / SQDMetal
  parameterizations) belongs to this tier as well.
