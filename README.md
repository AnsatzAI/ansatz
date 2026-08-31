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
