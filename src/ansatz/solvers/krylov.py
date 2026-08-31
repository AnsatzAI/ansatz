"""Krylov blocks operating on grid fields (matrix-free CG)."""

from __future__ import annotations

import numpy as np

from ..pde.laplace import LaplaceProblem


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
