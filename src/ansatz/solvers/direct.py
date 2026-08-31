"""Sparse direct solve (UMFPACK/SuperLU via scipy) — accuracy reference."""

from __future__ import annotations

from scipy.sparse.linalg import splu

from ..pde.laplace import LaplaceProblem


class DirectSolver:
    name = "direct(splu)"

    def __call__(self, problem: LaplaceProblem, u=None, b=None):
        a, rhs, idx = problem.assemble()
        if b is not None:
            rhs = rhs + b[idx >= 0]
        x = splu(a.tocsc()).solve(rhs)
        return problem.to_grid(x, idx)


def solve_direct(problem: LaplaceProblem):
    return DirectSolver()(problem)
