"""HINTS baseline: fixed-schedule hybrid iteration (Zhang et al., Nat. Mach.
Intell. 2024, Algorithm S3).

Every n_r-th iteration applies a neural corrector to the residual equation
A(delta) = r; all other iterations are classical relaxation sweeps. This is the
canonical fixed-schedule hybrid the learned router should beat: same surrogate,
same smoothers, no routing intelligence.
"""

from __future__ import annotations

import time

import numpy as np

from ..pde.laplace import LaplaceProblem
from ..solvers.smoothers import RedBlackGaussSeidel


def solve_hints(
    problem: LaplaceProblem,
    corrector,
    tol: float = 1e-8,
    n_r: int = 16,
    max_iters: int = 4000,
    smoother=None,
) -> tuple[np.ndarray, dict]:
    """`corrector(problem, residual) -> error-field estimate` (clamped free nodes)."""
    smoother = smoother or RedBlackGaussSeidel(sweeps=1)
    t0 = time.perf_counter()
    u = problem.initial_guess()
    it = 0
    corrections = 0
    while problem.residual_norm(u) > tol and it < max_iters:
        if it % n_r == 0 and it > 0:
            r = problem.residual(u)
            delta = corrector(problem, r)
            u = np.where(problem.fixed_mask, u, u + delta)
            corrections += 1
        else:
            u = smoother(problem, u)
        it += 1
    t = time.perf_counter() - t0
    return u, {
        "t_total": t,
        "iters": it,
        "corrections": corrections,
        "residual": problem.residual_norm(u),
        "converged": problem.residual_norm(u) <= tol,
    }
