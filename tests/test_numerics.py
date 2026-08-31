"""Correctness tests for the discretization, solvers, and capacitance extraction."""

import numpy as np
import pytest

from ansatz.pde.capacitance import capacitance_matrix
from ansatz.pde.laplace import LaplaceProblem
from ansatz.solvers import (
    ConjugateGradientBlock,
    MultigridVCycle,
    RedBlackGaussSeidel,
    solve_direct,
)


def _empty_problem(n):
    return LaplaceProblem(
        n=n, fixed_mask=np.zeros((n, n), bool), fixed_values=np.zeros((n, n))
    )


def test_manufactured_solution_direct():
    """-Lap u = f with u = sin(pi x) sin(pi y): direct solve matches analytic."""
    n = 63
    p = _empty_problem(n)
    h = p.h
    x = np.arange(1, n + 1) * h
    xx, yy = np.meshgrid(x, x, indexing="ij")
    u_exact = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    f = 2 * np.pi**2 * u_exact
    b = h * h * f  # operator is h^2-scaled
    a, rhs, idx = p.assemble()
    import scipy.sparse.linalg as spla

    u = p.to_grid(spla.splu(a.tocsc()).solve(rhs + b[idx >= 0]), idx)
    assert np.max(np.abs(u - u_exact)) < 2e-3  # O(h^2) discretization error


def test_mg_converges_with_conductors():
    """GMG with fat-mask coarsening: modest but monotone convergence.

    Embedded thin gaps limit the coarse correction; the documented contract is
    factor <= 0.7 per cycle, stable (AMG is the heavy op for fast convergence).
    """
    n = 127
    mask = np.zeros((n, n), bool)
    mask[40:50, 30:90] = True
    mask[70:80, 30:90] = True
    vals = np.zeros((n, n))
    vals[40:50, 30:90] = 1.0
    p = LaplaceProblem(n=n, fixed_mask=mask, fixed_values=vals)
    u = p.initial_guess()
    mg = MultigridVCycle()
    rs = [p.residual_norm(u)]
    for _ in range(15):
        u = mg(p, u)
        rs.append(p.residual_norm(u))
    factors = [rs[i + 1] / rs[i] for i in range(2, 14)]
    assert max(factors) < 0.7
    assert rs[-1] < 1e-3


def test_amg_cycles_converge_fast():
    from ansatz.solvers.amg import AMGContext, AMGVCycleOp

    n = 127
    mask = np.zeros((n, n), bool)
    mask[40:50, 30:90] = True
    mask[70:80, 30:90] = True
    vals = np.zeros((n, n))
    vals[40:50, 30:90] = 1.0
    p = LaplaceProblem(n=n, fixed_mask=mask, fixed_values=vals)
    ctx = AMGContext(p)
    op = AMGVCycleOp(ctx)
    u = p.initial_guess()
    for _ in range(12):
        u = op(p, u)
    assert p.residual_norm(u) < 1e-8


def test_mg_matches_direct():
    n = 63
    mask = np.zeros((n, n), bool)
    mask[25:30, 15:45] = True
    vals = np.where(mask, 1.0, 0.0)
    p = LaplaceProblem(n=n, fixed_mask=mask, fixed_values=vals)
    u_direct = solve_direct(p)
    u = p.initial_guess()
    mg = MultigridVCycle()
    for _ in range(60):
        u = mg(p, u)
    assert np.max(np.abs(u - u_direct)) < 1e-6


def test_cg_matches_direct():
    n = 63
    mask = np.zeros((n, n), bool)
    mask[25:30, 15:45] = True
    vals = np.where(mask, 1.0, 0.0)
    p = LaplaceProblem(n=n, fixed_mask=mask, fixed_values=vals)
    u_direct = solve_direct(p)
    u = ConjugateGradientBlock(iters=800)(p, p.initial_guess())
    assert np.max(np.abs(u - u_direct)) < 1e-6


def test_smoother_reduces_residual():
    n = 63
    mask = np.zeros((n, n), bool)
    mask[20:25, 20:44] = True
    vals = np.where(mask, 1.0, 0.0)
    p = LaplaceProblem(n=n, fixed_mask=mask, fixed_values=vals)
    u = p.initial_guess()
    r0 = p.residual_norm(u)
    u = RedBlackGaussSeidel(sweeps=20)(p, u)
    assert p.residual_norm(u) < r0


def test_capacitance_symmetry_and_sign():
    n = 127
    m1 = np.zeros((n, n), bool)
    m2 = np.zeros((n, n), bool)
    m1[50:56, 30:98] = True
    m2[72:78, 30:98] = True
    c, _ = capacitance_matrix(solve_direct, n, [m1, m2], eps_eff=1.0, thickness_eff=1.0)
    assert c[0, 1] == pytest.approx(c[1, 0])
    assert c[0, 0] > 0 and c[1, 1] > 0
    assert c[0, 1] < 0  # mutual capacitance is negative in the Maxwell matrix
    # off-diagonal magnitude bounded by self-capacitance
    assert -c[0, 1] < c[0, 0]
