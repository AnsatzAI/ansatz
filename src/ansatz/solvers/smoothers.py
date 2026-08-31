"""Vectorized stationary smoothers on the masked grid.

Red-black orderings make Gauss-Seidel/SOR fully vectorizable with NumPy while
preserving their convergence behavior on the 5-point stencil. All smoothers
keep Dirichlet (conductor) nodes clamped.
"""

from __future__ import annotations

import numpy as np

from ..pde.laplace import LaplaceProblem


def _neighbor_sum(u: np.ndarray) -> np.ndarray:
    s = np.zeros_like(u)
    s[1:, :] += u[:-1, :]
    s[:-1, :] += u[1:, :]
    s[:, 1:] += u[:, :-1]
    s[:, :-1] += u[:, 1:]
    return s


class DampedJacobi:
    def __init__(self, omega: float = 0.8, sweeps: int = 1):
        self.omega = omega
        self.sweeps = sweeps
        self.name = f"jacobi(w={omega},k={sweeps})"

    def __call__(self, problem: LaplaceProblem, u, b=None):
        free = ~problem.fixed_mask
        for _ in range(self.sweeps):
            rhs = _neighbor_sum(u)
            if b is not None:
                rhs = rhs + b
            u = np.where(free, (1 - self.omega) * u + self.omega * 0.25 * rhs, u)
        return u


class RedBlackGaussSeidel:
    def __init__(self, sweeps: int = 1, omega: float = 1.0):
        self.sweeps = sweeps
        self.omega = omega
        self.name = f"rbgs(w={omega},k={sweeps})" if omega != 1.0 else f"rbgs(k={sweeps})"

    def __call__(self, problem: LaplaceProblem, u, b=None):
        n = problem.n
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        parity = (ii + jj) % 2
        free = ~problem.fixed_mask
        u = u.copy()
        for _ in range(self.sweeps):
            for color in (0, 1):
                upd = free & (parity == color)
                rhs = _neighbor_sum(u)
                if b is not None:
                    rhs = rhs + b
                gs = 0.25 * rhs
                u[upd] = (1 - self.omega) * u[upd] + self.omega * gs[upd]
        return u


def sor(omega: float = 1.9, sweeps: int = 1) -> RedBlackGaussSeidel:
    op = RedBlackGaussSeidel(sweeps=sweeps, omega=omega)
    op.name = f"sor(w={omega},k={sweeps})"
    return op
