"""Krylov blocks operating on grid fields (matrix-free CG / MG-PCG)."""

from __future__ import annotations

import numpy as np

from ..pde.laplace import LaplaceProblem


class MGPreconditionedCG:
    """PCG with one geometric V-cycle as the preconditioner.

    Fully matrix-free: no assembly, no setup. The V-cycle approximately solves
    A z = r with homogeneous Dirichlet data on the conductor set, which is SPD
    as a preconditioner for the free-node system.
    """

    def __init__(self, iters: int = 10, pre: int = 1, post: int = 1):
        from .multigrid import MultigridVCycle

        self.iters = iters
        self._mg = MultigridVCycle(pre=pre, post=post)
        self.name = f"mgpcg(k={iters})"

    def _apply_m(self, problem: LaplaceProblem, r: np.ndarray) -> np.ndarray:
        hom = LaplaceProblem(
            n=problem.n,
            fixed_mask=problem.fixed_mask,
            fixed_values=np.zeros_like(problem.fixed_values),
        )
        z = self._mg(hom, np.zeros_like(r), r)
        z[problem.fixed_mask] = 0.0
        return z

    def __call__(self, problem: LaplaceProblem, u, b=None):
        free = ~problem.fixed_mask
        u = u.copy()
        r = problem.residual(u, b)
        z = self._apply_m(problem, r)
        p = z.copy()
        rz = float(np.sum(r * z))
        for _ in range(self.iters):
            if rz <= 0:
                break
            ap = problem.apply_operator(p)
            denom = float(np.sum(p * ap))
            if denom <= 0:
                break
            alpha = rz / denom
            u[free] += alpha * p[free]
            r -= alpha * ap
            if float(np.sum(r * r)) < 1e-32:
                break
            z = self._apply_m(problem, r)
            rz_new = float(np.sum(r * z))
            p = z + (rz_new / rz) * p
            p[~free] = 0.0
            rz = rz_new
        return u


class ConjugateGradientBlock:
    """Run `iters` CG iterations from the current iterate (warm-startable)."""

    def __init__(self, iters: int = 10):
        self.iters = iters
        self.name = f"cg(k={iters})"

    def __call__(self, problem: LaplaceProblem, u, b=None):
        free = ~problem.fixed_mask
        r = problem.residual(u, b)
        p = r.copy()
        rs = float(np.sum(r * r))
        if rs == 0.0:
            return u
        u = u.copy()
        for _ in range(self.iters):
            ap = problem.apply_operator(p)
            denom = float(np.sum(p * ap))
            if denom <= 0:
                break
            alpha = rs / denom
            u[free] += alpha * p[free]
            r -= alpha * ap
            rs_new = float(np.sum(r * r))
            if rs_new < 1e-32:
                break
            p = r + (rs_new / rs) * p
            p[~free] = 0.0
            rs = rs_new
        return u
