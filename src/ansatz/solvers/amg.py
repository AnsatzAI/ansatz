"""Algebraic multigrid ops (pyamg) with setup as an explicit, chargeable step.

Embedded-Dirichlet geometries defeat naive geometric coarsening (thin gaps
close, thin conductors vanish), which is precisely what Galerkin coarse
operators handle. The trade is a per-geometry assembly + setup cost — so
whether to build the hierarchy at all is a routing decision, not a default.
"""

from __future__ import annotations

import time

import numpy as np
import pyamg

from ..pde.laplace import LaplaceProblem


class AMGContext:
    """Assembled system + AMG hierarchy for one problem (reusable across ops)."""

    def __init__(self, problem: LaplaceProblem, method: str = "sa"):
        t0 = time.perf_counter()
        self.a, self.b, self.idx = problem.assemble()
        self.t_assemble = time.perf_counter() - t0
        t0 = time.perf_counter()
        if method == "sa":
            self.ml = pyamg.smoothed_aggregation_solver(self.a)
        else:
            self.ml = pyamg.ruge_stuben_solver(self.a)
        self.t_setup = time.perf_counter() - t0
        self.problem = problem

    def field_to_vec(self, u: np.ndarray) -> np.ndarray:
        return u[self.idx >= 0][np.argsort(self.idx[self.idx >= 0])]

    def vec_to_field(self, x: np.ndarray) -> np.ndarray:
        return self.problem.to_grid(x, self.idx)


class AMGVCycleOp:
    """One AMG V-cycle from the current iterate; requires a built context."""

    def __init__(self, ctx: AMGContext, cycles: int = 1):
        self.ctx = ctx
        self.cycles = cycles
        self.name = f"amg_v(k={cycles})"

    def __call__(self, problem: LaplaceProblem, u, b=None):
        rhs = self.ctx.b
        if b is not None:
            rhs = rhs + b[self.ctx.idx >= 0]
        x = u[~problem.fixed_mask]
        x = self.ctx.ml.solve(
            rhs, x0=x, maxiter=self.cycles, cycle="V", tol=1e-30, accel=None
        )
        return self.ctx.vec_to_field(x)
