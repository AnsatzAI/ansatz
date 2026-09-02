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

## Dev-scale experiments (this machine, measured)

**1. Shift sensitivity — the missed-mode cliff.** Eigensolver `Target` sweep
at fixed config (true modes: 4.099 / 5.603 GHz):

| Target (GHz) | wall-clock (s) | modes found |
|---|---|---|
| 2.0–4.05 (below f0) | 188–212 | 4.099, 5.603 ✔ |
| 4.5 (above f0) | **617 (3.0x)** | **4.099 GHz qubit mode MISSED** |
| 5.0 (above f0) | **532 (2.6x)** | **4.099 GHz qubit mode MISSED** |

A target above a mode both triples the cost and silently drops that mode.
Tier-1 frequency prediction is therefore a **correctness feature**: keep the
shift on the safe side (predicted f0 minus margin), and verify the returned
mode count. This is the 3D analogue of residual verification.

**2. Config frontier — cheap configs fail verification.** On this mesh
family, order-1 elements miss the qubit mode entirely (17.7 s but f0 off by
28%; refined order-1 is slow *and* wrong at 1,174 s). Order-2 coarse
(207 s) is the verified workhorse. Routing insight: there is no cheaper
verified config on this family — the wall-clock win must come from the
*design loop*, which motivates the advantage demo below.

**3. Advantage demo — MEASURED.** Verified inverse design: hit
(f_qubit, f_readout) = (4.20, 5.45) GHz within 25 MHz, final design verified
by a full Palace eigenmode solve in both arms, identical solver fidelity.

| arm | wall-clock | Palace solves | verified miss | within tol? |
|---|---|---|---|---|
| **Ansatz** (surrogate inversion + verified solve + one physics correction) | **34.9 min** | **2** | **11.8 MHz** | **yes** |
| Classical (Nelder-Mead over geometry, full solve per evaluation) | 215.6 min | 12 | 269.7 MHz | no |

**6.2x faster and 22.8x closer to spec, with the classical arm failing to
reach tolerance within its 12-solve budget.** The Ansatz arm's final answer
is a real Palace solve — the surrogate only chose where to point it.

Two robustness lessons became product features along the way:
- **Prediction-informed eigensolver targeting.** The lossy readout mode
  (Q ~ 8k) frequently fails to converge and silently drops from the output
  when the shift target is far away (62/101 campaign solves at a fixed
  far target) or too close to the qubit mode. Setting the target from the
  predicted f0 (a few hundred MHz below), with N=6 and a larger Krylov
  subspace, recovered it reliably — mode-completeness verification catches
  any remaining dropouts.
- **Correction steps need physics, not tree gradients.** Finite differences
  on gradient-boosted trees are unusable (piecewise-constant); the working
  correction is quarter-wave scaling for the resonator plus a local
  data-fit slope for the pad, damped with a trust region.

## Tier-1 3D forward model (bespoke data, LOO-validated)

101 usable solves (12.9 h overnight campaign): f0 **0.88% MAPE / 36 MHz
MAE**; f1 (39 clean-label rows) **0.93% / 54 MHz** — vs kNN at 2.3%/95 MHz
and 3.6%/215 MHz. Physics sanity: corr(total_length, f1) = -0.992,
corr(cap_length, f0) = -0.926.

## Bespoke data campaign

Public data does not cover parameterized 3D full-wave eigenmodes, so the
training set is generated in-house: `scripts/em3d_campaign.py` sweeps
(cap_length, cap_gap, total_length, l_claw, n_meander_turns) through
DeviceLayout.jl meshing and Palace eigenmode solves (~6 min/variant at dev
tier; 100-variant overnight run on this machine). Records: geometry, f0, f1,
Q factors, mesh/solve wall-clock. Resumable by tag.

## Compute plan

- **This Mac (dev tier)**: pipeline development, ~10^5–10^6 DOF solves at
  1–10 min each, overnight sweeps of O(100) solves. Sufficient for method
  validation and the dev-scale results above.
- **Campaign tier (10^7–10^8 DOF, full chips)**: AWS HPC instances
  (hpc7a/c7a-class, 100s of GB RAM, MPI) rented per benchmark campaign
  (est. $100s–low $1,000s per campaign), or university cluster time.
  Geometry-swept training-set generation (DeviceLayout.jl / SQDMetal
  parameterizations) belongs to this tier as well.
