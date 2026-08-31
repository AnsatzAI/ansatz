"""Geometric multigrid V-cycle on masked grids.

Coarsening keeps Dirichlet structure by marking a coarse node fixed when any of
its fine-grid parents is fixed (conservative). Fixed values are restricted by
injection. Because errors vanish at fixed nodes on every level, corrections are
prolongated only into free nodes.
"""

from __future__ import annotations

import numpy as np

from ..pde.laplace import LaplaceProblem
from .smoothers import RedBlackGaussSeidel


def _restrict_full_weighting(r: np.ndarray) -> np.ndarray:
    n = r.shape[0]
    nc = (n - 1) // 2
    fi = 2 * np.arange(nc)[:, None] + 1
    fj = 2 * np.arange(nc)[None, :] + 1
    c = 4.0 * r[fi, fj]
    c += 2.0 * (r[fi - 1, fj] + r[fi + 1, fj] + r[fi, fj - 1] + r[fi, fj + 1])
    c += r[fi - 1, fj - 1] + r[fi - 1, fj + 1] + r[fi + 1, fj - 1] + r[fi + 1, fj + 1]
    return c / 16.0


def _coarsen_mask(mask: np.ndarray) -> np.ndarray:
    n = mask.shape[0]
    nc = (n - 1) // 2
    fi = 2 * np.arange(nc)[:, None] + 1
    fj = 2 * np.arange(nc)[None, :] + 1
    out = mask[fi, fj].copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            out |= mask[fi + di, fj + dj]
    return out


def _prolong_bilinear(c: np.ndarray, n: int) -> np.ndarray:
    nc = c.shape[0]
    padded = np.zeros((nc + 2, nc + 2))
    padded[1:-1, 1:-1] = c
    f = np.zeros((n, n))
    fi = 2 * np.arange(nc) + 1
    f[np.ix_(fi, fi)] = c
    ei = 2 * np.arange(nc + 1)
    f[np.ix_(ei, fi)] = 0.5 * (padded[:-1, 1:-1] + padded[1:, 1:-1])
    f[np.ix_(fi, ei)] = 0.5 * (padded[1:-1, :-1] + padded[1:-1, 1:])
    f[np.ix_(ei, ei)] = 0.25 * (
        padded[:-1, :-1] + padded[:-1, 1:] + padded[1:, :-1] + padded[1:, 1:]
    )
    return f


class MultigridVCycle:
    """One V(pre, post) cycle. Requires n = 2^k - 1 grids."""

    def __init__(self, pre: int = 2, post: int = 2, coarsest: int = 7):
        self.pre = pre
        self.post = post
        self.coarsest = coarsest
        self.name = f"mg_v({pre},{post})"
        self._smoother = RedBlackGaussSeidel(sweeps=1)

    def __call__(self, problem: LaplaceProblem, u, b=None):
        if b is None:
            b = np.zeros_like(u)
        return self._vcycle(problem, u, b)

    def _vcycle(self, problem: LaplaceProblem, u, b):
        smooth_pre = RedBlackGaussSeidel(sweeps=self.pre)
        smooth_post = RedBlackGaussSeidel(sweeps=self.post)
        u = smooth_pre(problem, u, b)
        if problem.n <= self.coarsest:
            return RedBlackGaussSeidel(sweeps=30)(problem, u, b)

        r = problem.residual(u, b)
        rc = 4.0 * _restrict_full_weighting(r)  # h^2 scaling: (2h)^2/h^2
        mask_c = _coarsen_mask(problem.fixed_mask)
        rc[mask_c] = 0.0
        nc = (problem.n - 1) // 2
        prob_c = LaplaceProblem(
            n=nc, fixed_mask=mask_c, fixed_values=np.zeros((nc, nc))
        )
        ec = self._vcycle(prob_c, np.zeros((nc, nc)), rc)
        e = _prolong_bilinear(ec, problem.n)
        u = np.where(problem.fixed_mask, u, u + e)
        return smooth_post(problem, u, b)
