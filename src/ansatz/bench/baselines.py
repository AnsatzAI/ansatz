"""Practitioner baselines with a common `solve(problem) -> field` interface.

These are the tools a superconducting-design engineer would actually reach for
on sparse electrostatic systems: pyamg smoothed aggregation / Ruge-Stuben AMG
(with CG acceleration), plain preconditioned CG, sparse direct factorization,
geometric multigrid iterated to tolerance, and a HINTS-style fixed hybrid
schedule once a surrogate is available. Each returns (field, info) with
wall-clock breakdown so setup cost is charged honestly.
"""

from __future__ import annotations

import time

import numpy as np
import pyamg
import scipy.sparse.linalg as spla

from ..pde.laplace import LaplaceProblem
from ..solvers.multigrid import MultigridVCycle


def solve_pyamg(
    problem: LaplaceProblem,
    tol: float = 1e-8,
    method: str = "sa",
    accel: str = "cg",
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    a, b, idx = problem.assemble()
    t_assemble = time.perf_counter() - t0

    t0 = time.perf_counter()
    if method == "sa":
        ml = pyamg.smoothed_aggregation_solver(a)
    else:
        ml = pyamg.ruge_stuben_solver(a)
    t_setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    x = ml.solve(b, tol=tol, accel=accel)
    t_solve = time.perf_counter() - t0

    u = problem.to_grid(x, idx)
    return u, {
        "t_assemble": t_assemble,
        "t_setup": t_setup,
        "t_solve": t_solve,
        "t_total": t_assemble + t_setup + t_solve,
        "residual": problem.residual_norm(u),
    }


def solve_cg_ilu(problem: LaplaceProblem, tol: float = 1e-8) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    a, b, idx = problem.assemble()
    t_assemble = time.perf_counter() - t0

    t0 = time.perf_counter()
    ilu = spla.spilu(a.tocsc(), drop_tol=1e-4, fill_factor=10)
    m = spla.LinearOperator(a.shape, ilu.solve)
    t_setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    x, info = spla.cg(a, b, rtol=tol, M=m, maxiter=2000)
    t_solve = time.perf_counter() - t0

    u = problem.to_grid(x, idx)
    return u, {
        "t_assemble": t_assemble,
        "t_setup": t_setup,
        "t_solve": t_solve,
        "t_total": t_assemble + t_setup + t_solve,
        "cg_info": info,
        "residual": problem.residual_norm(u),
    }


def solve_direct_timed(problem: LaplaceProblem) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    a, b, idx = problem.assemble()
    t_assemble = time.perf_counter() - t0
    t0 = time.perf_counter()
    x = spla.splu(a.tocsc()).solve(b)
    t_solve = time.perf_counter() - t0
    u = problem.to_grid(x, idx)
    return u, {
        "t_assemble": t_assemble,
        "t_setup": 0.0,
        "t_solve": t_solve,
        "t_total": t_assemble + t_solve,
        "residual": problem.residual_norm(u),
    }


def solve_gmg(
    problem: LaplaceProblem, tol: float = 1e-8, max_cycles: int = 60
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    op = MultigridVCycle()
    u = problem.initial_guess()
    cycles = 0
    while problem.residual_norm(u) > tol and cycles < max_cycles:
        u = op(problem, u)
        cycles += 1
    t = time.perf_counter() - t0
    return u, {
        "t_assemble": 0.0,
        "t_setup": 0.0,
        "t_solve": t,
        "t_total": t,
        "cycles": cycles,
        "residual": problem.residual_norm(u),
    }
