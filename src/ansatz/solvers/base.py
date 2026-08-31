"""Composable solver operators.

Every solver action — a smoother sweep, a multigrid V-cycle, a block of CG
iterations, a surrogate correction — implements `SolverOp`. The router picks a
sequence of ops; the harness charges each op its measured wall-clock cost. Ops
are pure functions of (problem, u, b) so hybrid schedules compose freely.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..pde.laplace import LaplaceProblem


class SolverOp(Protocol):
    name: str

    def __call__(
        self, problem: LaplaceProblem, u: np.ndarray, b: np.ndarray | None = None
    ) -> np.ndarray:
        """Return an improved clamped iterate."""
        ...
