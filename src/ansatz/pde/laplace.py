"""Finite-difference electrostatics on a uniform grid.

The computational domain is the unit square with a grounded outer boundary
(chip ground shield). Conductors are Dirichlet regions embedded in the grid.
We expose both an assembled CSR system (for AMG/Krylov/direct baselines) and
matrix-free grid operators (for smoothers, geometric multigrid, and the
surrogate/router loop, which all work on 2D fields).

Convention: `u` is an (N, N) array of interior nodes with spacing h = 1/(N+1).
Nodes where `fixed_mask` is True are clamped to `fixed_values` and are not
unknowns. The PDE is Laplace's equation; sources are supported through `b`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class LaplaceProblem:
    """Discretized Dirichlet problem for one excitation pattern."""

    n: int                      # grid is n x n interior nodes
    fixed_mask: np.ndarray      # (n, n) bool, True at conductor nodes
    fixed_values: np.ndarray    # (n, n) float, potential at fixed nodes (0 elsewhere)

    def __post_init__(self) -> None:
        assert self.fixed_mask.shape == (self.n, self.n)
        assert self.fixed_values.shape == (self.n, self.n)

    @property
    def h(self) -> float:
        return 1.0 / (self.n + 1)

    def clamp(self, u: np.ndarray) -> np.ndarray:
        """Return u with fixed nodes overwritten by their Dirichlet values."""
        out = u.copy()
        out[self.fixed_mask] = self.fixed_values[self.fixed_mask]
        return out

    def initial_guess(self) -> np.ndarray:
        return self.clamp(np.zeros((self.n, self.n)))

    def apply_operator(self, u: np.ndarray) -> np.ndarray:
        """Matrix-free A @ u on the h^2-scaled 5-point stencil.

        Result is only meaningful at free nodes; fixed nodes are returned as 0.
        Neighbors outside the domain contribute the grounded boundary value 0.
        """
        au = 4.0 * u
        au[1:, :] -= u[:-1, :]
        au[:-1, :] -= u[1:, :]
        au[:, 1:] -= u[:, :-1]
        au[:, :-1] -= u[:, 1:]
        au[self.fixed_mask] = 0.0
        return au

    def residual(self, u: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
        """r = b - A u at free nodes (zero at fixed nodes). u must be clamped."""
        r = -self.apply_operator(u)
        if b is not None:
            r += np.where(self.fixed_mask, 0.0, b)
        return r

    def residual_norm(self, u: np.ndarray, b: np.ndarray | None = None) -> float:
        """Relative residual: ||b - A u||_2 / ||b_eff||_2.

        For pure Dirichlet drive (b = 0) the natural scale is the RHS produced
        by moving boundary data to the right-hand side, which equals A u0 for
        u0 = clamped zero field; we use ||A u_d|| with u_d = fixed values only.
        """
        r = self.residual(u, b)
        scale = self._rhs_scale if hasattr(self, "_rhs_scale") else None
        if scale is None:
            ud = np.where(self.fixed_mask, self.fixed_values, 0.0)
            scale = float(np.linalg.norm(self.apply_operator(ud)))
            if b is not None:
                scale = float(np.hypot(scale, np.linalg.norm(b[~self.fixed_mask])))
            self._rhs_scale = max(scale, 1e-300)
        return float(np.linalg.norm(r)) / self._rhs_scale

    # ---------------- assembled form ----------------

    def assemble(self) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """Assemble the free-node system A x = b (h^2-scaled).

        Returns (A, b, free_index) where free_index maps grid position to
        unknown index (-1 at fixed nodes).
        """
        n = self.n
        free = ~self.fixed_mask
        idx = -np.ones((n, n), dtype=np.int64)
        idx[free] = np.arange(int(free.sum()))

        rows, cols, vals = [], [], []
        b = np.zeros(int(free.sum()))

        fi, fj = np.nonzero(free)
        center = idx[fi, fj]
        rows.append(center)
        cols.append(center)
        vals.append(np.full(center.shape, 4.0))

        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = fi + di, fj + dj
            inside = (ni >= 0) & (ni < n) & (nj >= 0) & (nj < n)
            # neighbor inside and free: off-diagonal entry
            nf = inside.copy()
            nf[inside] = free[ni[inside], nj[inside]]
            rows.append(center[nf])
            cols.append(idx[ni[nf], nj[nf]])
            vals.append(np.full(int(nf.sum()), -1.0))
            # neighbor inside and fixed: move to RHS
            nd = inside.copy()
            nd[inside] = ~free[ni[inside], nj[inside]]
            np.add.at(b, center[nd], self.fixed_values[ni[nd], nj[nd]])
            # neighbor outside: grounded boundary contributes 0

        a = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(b), len(b)),
        )
        return a, b, idx

    def to_grid(self, x: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Scatter a free-node vector back to a clamped (n, n) field."""
        u = np.where(self.fixed_mask, self.fixed_values, 0.0)
        u[idx >= 0] = x[idx[idx >= 0]]
        return u
