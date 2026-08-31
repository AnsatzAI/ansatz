# ansatz

**Learned solver routing for superconducting qubit design.**

`ansatz` accelerates the electromagnetic simulation loop inside superconducting
chip design — capacitance extraction and the lumped-oscillator quantities
derived from it (charging energy, anharmonicity, couplings) — by routing each
solve through the cheapest path that reaches a *verified* tolerance:

- **Neural surrogates** predict solution fields from geometry in milliseconds.
- **A learned router** decides, per solve and per iteration, whether the
  surrogate's output can be cheaply corrected to tolerance (smoothers, CG,
  matrix-free multigrid) or whether the solve should fall back to the classical
  path (algebraic multigrid, sparse direct).
- **Residual verification** means the result is never trusted on faith: every
  accepted field satisfies the discretized PDE to the tolerance you asked for.
  Worst case, you pay a few milliseconds of overhead and get exactly the
  classical answer.

Status: v0 targets 2D planar electrostatics on geometry distributions anchored
to experimentally validated design databases. See `docs/` for architecture and
benchmarks, and `paper/` for the technical report.

## Headline results (v0, honest edition)

- **Forward model** (trained on 1,934 real Ansys Q3D matrices, SQuADDS DB):
  **0.31%** capacitance MAPE held-out vs 1.75% for nearest-design lookup;
  downstream **0.70% g / 1.71% χ** vs 4.0%/28.7%. Microsecond latency.
- **Routed solver**: **1.000× the per-instance oracle** (100% correct pipeline
  selection on 735 held-out + OOD designs, 11 µs/decision) — geometric-mean
  **2.1× faster than pyamg AMG-CG**, 5–30× faster than committing to any
  surrogate pipeline, **~900× faster than a fixed HINTS schedule**, and never
  more than 0.4% behind the best fixed choice on any instance.
- **Zero verification failures in 5,930 measured solves.** In 2D, the router
  correctly learns that sparse direct dominates — the negative result for
  neural pipelines is reported, not hidden. The 3D backend (where direct
  factorization stops scaling) is the roadmap item that flips the ensemble.

Full protocol and tables: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Install

```bash
pip install -e .
```

## Layout

```
src/ansatz/
  geometry/   parameterized planar transmon/coupler layouts
  pde/        discretization, capacitance extraction, lumped-oscillator model
  solvers/    smoothers, geometric MG, Krylov, AMG, direct — as composable ops
  surrogate/  neural field predictors (initializers + mid-solve correctors)
  router/     learned routing policy with verified fallback
  bench/      time-to-tolerance benchmarking vs practitioner baselines
```

## License

Apache-2.0. © Ansatz AI — https://useansatz.ai
