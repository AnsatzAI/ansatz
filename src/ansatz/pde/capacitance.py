"""Multi-conductor capacitance extraction from electrostatic solves.

For K conductors we solve K Dirichlet problems (conductor k at 1 V, all others
and the outer shield at 0 V). The Maxwell capacitance matrix follows from the
induced charge on each conductor, computed as the discrete flux of the
solution field through each conductor's surface.

Units: the 2D solve yields capacitance per unit length in units of eps0*eps_eff;
`C_SCALE` converts to farads via an effective thickness, calibrated once against
reference 3D extractions. All downstream lumped-oscillator quantities take the
capacitance matrix in farads.
"""

from __future__ import annotations

import numpy as np

from .laplace import LaplaceProblem

EPS0 = 8.8541878128e-12  # F/m


def conductor_charge(problem: LaplaceProblem, u: np.ndarray, mask: np.ndarray) -> float:
    """Discrete Gauss-law charge on `mask` (in units of eps0*eps_eff per length).

    Sums (u_free_neighbor - u_conductor) over all edges from conductor nodes to
    free nodes; h cancels between edge length and finite-difference gradient.
    """
    q = 0.0
    n = problem.n
    ci, cj = np.nonzero(mask)
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni, nj = ci + di, cj + dj
        inside = (ni >= 0) & (ni < n) & (nj >= 0) & (nj < n)
        free = inside.copy()
        free[inside] = ~problem.fixed_mask[ni[inside], nj[inside]]
        q += float(np.sum(u[ni[free], nj[free]] - u[ci[free], cj[free]]))
    return -q


def capacitance_matrix(
    solve,
    n: int,
    conductor_masks: list[np.ndarray],
    eps_eff: float = 6.45,
    thickness_eff: float = 1.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Maxwell capacitance matrix via one solve per conductor.

    `solve` is any callable mapping LaplaceProblem -> clamped solution field,
    which is exactly the interface the routed solver, pyamg baseline, and
    surrogate all implement. Returns (C, fields) with C in farads given the
    effective-thickness calibration.
    """
    k = len(conductor_masks)
    fixed = np.zeros((n, n), dtype=bool)
    for m in conductor_masks:
        fixed |= m

    c = np.zeros((k, k))
    fields = []
    for a in range(k):
        values = np.where(conductor_masks[a], 1.0, 0.0)
        problem = LaplaceProblem(n=n, fixed_mask=fixed, fixed_values=values)
        u = solve(problem)
        fields.append(u)
        for b in range(k):
            q_b = conductor_charge(problem, u, conductor_masks[b])
            c[a, b] = EPS0 * eps_eff * thickness_eff * q_b
    return 0.5 * (c + c.T), fields
