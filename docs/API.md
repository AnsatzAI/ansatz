# API reference (v0)

## Geometry

```python
from ansatz.geometry.transmon import XmonDesign, build_problem_masks
from ansatz.geometry.sampler import sample_designs, DEFAULT_RANGES, ood_ranges

design = XmonDesign(cross_length=310, claw_length=160, ground_spacing=10)  # um
conductors, ground = build_problem_masks(design, n=511)   # ([cross, claw], ground)
```

`XmonDesign` fields (um): `cross_length`, `cross_width`, `cross_gap`,
`claw_length`, `claw_width`, `claw_gap`, `ground_spacing`, `domain`.

## PDE core

```python
from ansatz.pde.laplace import LaplaceProblem
p = LaplaceProblem(n, fixed_mask, fixed_values)   # (n,n) bool / float arrays
p.apply_operator(u); p.residual(u, b); p.residual_norm(u)   # matrix-free
A, b, idx = p.assemble()                                     # CSR free-node system
```

## Capacitance + LOM

```python
from ansatz.pde.capacitance import capacitance_matrix
from ansatz.pde.lom import hamiltonian_from_cap
C, fields = capacitance_matrix(solve, n, [cross, claw])      # solve: problem -> field
hp = hamiltonian_from_cap(c_cross_gnd_fF, c_cross_claw_fF, lj_nh=10, f_r_hz=6.116e9)
hp.f_q, hp.alpha, hp.e_c, hp.g, hp.chi                        # Hz
```

## Solver ops (composable; all keep Dirichlet nodes clamped)

```python
from ansatz.solvers import DampedJacobi, RedBlackGaussSeidel, sor, \
    MultigridVCycle, ConjugateGradientBlock, DirectSolver
from ansatz.solvers.krylov import MGPreconditionedCG
from ansatz.solvers.amg import AMGContext, AMGVCycleOp
u = op(problem, u, b=None)   # every op shares this signature
```

## Surrogate

```python
from ansatz.surrogate.infer import FieldSurrogate
s = FieldSurrogate("runs/unet_255.pt")     # MPS/CPU auto
fields = s.predict([p1, p2])               # batched; cross-resolution transfer
```

Training: `python -m ansatz.surrogate.train --data data/fields --n 255 --epochs 24`.

## Router

```python
from ansatz.router.pipelines import DesignSolveContext, run_pipeline
from ansatz.router.policy import CostModelRouter, instance_features

ctx = DesignSolveContext(n, fixed_mask, conductors, surrogate=s)
fields, info = run_pipeline("surr_mgcg", ctx, tol=1e-8)
# info: t_total, residuals (verified), escalated, pipeline

router = CostModelRouter.load("runs/router.pkl")
decision = router.decide(instance_features(params, n, free_frac, tol))
fields, info = run_pipeline(decision.pipeline, ctx, tol=1e-8,
                            alternatives=decision.predicted_costs)
```

Pipelines: `direct`, `amg_cg`, `surr_cg`, `surr_mgcg`, `surr_amg`, `hints`.
Every pipeline verifies the residual at `tol` and falls back to `direct` on
failure — routing never changes correctness.

## Benchmarks

```bash
bash scripts/run_all_benchmarks.sh     # full suite -> runs/*.parquet + router
python scripts/make_figures.py         # figures + docs/BENCHMARKS.md
```
