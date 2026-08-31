# Architecture

`ansatz` accelerates the electromagnetic solve loop inside superconducting
qubit design. This document describes the system; see `BENCHMARKS.md` for
measured results and `DATA.md` for dataset provenance.

## The problem

Transmon design iterates on planar geometry (a cross-shaped island + readout
claw, embedded in a ground plane) until the derived Hamiltonian parameters —
charging energy E_C, qubit frequency f_q, anharmonicity α, readout coupling g,
dispersive shift χ — hit targets. Each iteration requires a multi-conductor
capacitance extraction: one electrostatic (Laplace) solve per conductor, then a
lumped-oscillator-model (LOM) reduction of the capacitance matrix. In industry
this runs on Ansys Q3D at minutes per design; the design loop runs hundreds to
thousands of such solves.

## The two-tier answer

**Tier 1 — instant forward model (in-distribution).** A gradient-boosted
regressor trained on 1,934 real Ansys Q3D capacitance matrices (SQuADDS
database) maps geometry parameters directly to the capacitance matrix at
microsecond latency. On held-out designs it is ~5x more accurate than
nearest-design lookup (0.31% vs 1.75% capacitance MAPE), and ~6x/17x more
accurate on downstream g/χ. Outside the database hull its accuracy degrades
like every data-driven method — which is why Tier 2 exists.

**Tier 2 — routed, verified field solves (everywhere).** For designs outside
the data hull (or whenever a verified answer is required), we solve the PDE —
but route each solve through the cheapest path that reaches a *verified*
relative residual:

```
                      ┌────────────────────────────────────────────┐
   design params ──►  │ CostModelRouter: per-pipeline wall-clock   │
   grid size, tol     │ regressors (GBM), argmin over:             │
                      │  direct    assemble + sparse LU (K RHS)    │
                      │  amg_cg    assemble + SA-AMG setup + CG    │
                      │  surr_cg   U-Net init + matrix-free CG     │
                      │  surr_amg  U-Net init + AMG-CG polish      │
                      │  hints     fixed-schedule hybrid (baseline)│
                      └───────────────┬────────────────────────────┘
                                      ▼
                      EscalationMonitor: if the observed residual
                      slope projects a worse finish than the best
                      alternative, switch once, warm-started.
                                      ▼
                      Residual verification at tol (else: direct fallback)
```

Key properties:

- **Verification is unconditional.** Every returned field satisfies
  ||b − Au|| / ||b|| ≤ tol on the target discretization. The surrogate can
  only save time, never corrupt results.
- **Costs are honest.** Assembly, AMG setup, factorization, and surrogate
  inference are all charged to the pipeline that incurs them; per-design state
  (assembly, LU, hierarchy, predictions) is shared across the K excitations of
  a capacitance sweep, as a practitioner would.
- **The router is boring on purpose.** Gradient-boosted trees on 10 cheap
  features (7 geometry parameters, log grid size, free-node fraction, log
  tolerance): microsecond decisions, retrainable on a customer's instance mix
  in seconds, no GPU dependency at decision time.
- **The escalation monitor bounds regret.** If the cost model mispredicts,
  the monitored solve escalates to the predicted-best alternative warm-started;
  worst case is bounded by (observed segment + alternative time) rather than an
  unbounded stall.

## Why routing (and not one method)

No single pipeline dominates across resolutions:

- Small grids (n≈255, ~6k unknowns): sparse direct wins outright — assembly
  and factorization are milliseconds; any learned path pays surrogate
  inference for nothing.
- Large grids (n≥1023, 10^5–10^6 unknowns): assembly + factorization grow
  superlinearly; a good surrogate init + matrix-free CG (zero assembly, zero
  setup) reaches tolerance far sooner; AMG-CG sits between.
- The crossover point depends on geometry (free-node fraction, gap widths),
  tolerance, and hardware — i.e., it is a *learned* function, not a threshold
  you hand-tune.

The router learns exactly this map, and the same infrastructure extends to 3D
solvers (AWS Palace) and eigenmode analyses on the roadmap.

## Module map

```
src/ansatz/
  geometry/   XmonDesign rasterizer + samplers anchored to SQuADDS ranges
  pde/        LaplaceProblem (matrix-free + assembled), capacitance, LOM
  solvers/    Jacobi/RBGS/SOR, GMG V-cycle, CG blocks, AMG (pyamg), direct
  surrogate/  FieldUNet, shard dataset, trainer, cross-resolution inference
  router/     CostModelRouter, EscalationMonitor, executable pipelines
  bench/      harness (time-to-verified-tolerance), baselines, HINTS
```

## Physics notes

- 2D plan-view electrostatics with unit-square normalization; conductors are
  embedded Dirichlet regions; the outer boundary is grounded. Capacitance
  entries come from discrete Gauss-law charges; an effective-permittivity /
  thickness calibration maps per-unit-length values to farads. v0's realism
  claim rests on (a) geometry distributions anchored to experimentally
  validated designs and (b) Tier 1 being trained on *real Q3D* matrices; the
  routed-solver benchmarks are self-consistent on the target discretization.
- LOM follows Shanto et al. (Quantum 8, 1465) / Koch et al. (PRA 76, 042319):
  exact Cooper-pair-box diagonalization, exact lumped g, beyond-RWA χ.
- GMG with embedded thin gaps is fundamentally limited by coarse-grid
  representation (documented factor ≤ 0.7/cycle); Galerkin (algebraic)
  coarsening handles it properly, which is why AMG is the heavy op.
