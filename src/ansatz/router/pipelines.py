"""Executable solve pipelines with shared per-design context and verification.

A `DesignSolveContext` owns everything shareable across the K excitations of a
capacitance sweep (assembly, factorizations, AMG hierarchies, surrogate batch
predictions) so each pipeline is charged the honest amortized cost a
practitioner would pay. Every pipeline returns fields whose relative residual
is verified <= tol; iterative pipelines carry an escalation monitor.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse.linalg as spla

from ..pde.laplace import LaplaceProblem
from ..solvers.amg import AMGContext
from ..solvers.krylov import ConjugateGradientBlock
from .policy import EscalationMonitor


class DesignSolveContext:
    """Shared state for one design (fixed mask) across excitations."""

    def __init__(self, n: int, fixed_mask: np.ndarray, conductor_masks, surrogate=None):
        self.n = n
        self.fixed_mask = fixed_mask
        self.conductor_masks = conductor_masks
        self.surrogate = surrogate
        self._assembled = None
        self._lu = None
        self._amg = None
        self._predictions = None

    def problems(self) -> list[LaplaceProblem]:
        out = []
        for m in self.conductor_masks:
            vals = np.where(m, 1.0, 0.0)
            out.append(
                LaplaceProblem(n=self.n, fixed_mask=self.fixed_mask, fixed_values=vals)
            )
        return out

    def assembled(self):
        if self._assembled is None:
            p0 = self.problems()[0]
            a, _, idx = p0.assemble()
            self._assembled = (a, idx)
        return self._assembled

    def lu(self):
        if self._lu is None:
            a, _ = self.assembled()
            self._lu = spla.splu(a.tocsc())
        return self._lu

    def amg(self):
        if self._amg is None:
            self._amg = AMGContext(self.problems()[0])
        return self._amg

    def predictions(self) -> list[np.ndarray]:
        if self._predictions is None:
            self._predictions = self.surrogate.predict(self.problems())
        return self._predictions


def _finish_iter(problem, u, tol, op, max_blocks=400, budget_s=60.0,
                 monitor=None, on_escalate=None):
    """Drive `op` blocks until verified tol, escalation, or time budget."""
    t0 = time.perf_counter()
    for _ in range(max_blocks):
        r = problem.residual_norm(u)
        if r <= tol:
            return u, False
        elapsed = time.perf_counter() - t0
        if elapsed > budget_s:
            return u, False  # verification below will trigger fallback
        if monitor is not None:
            target = monitor.observe(elapsed, r, tol)
            if target is not None and on_escalate is not None:
                return on_escalate(problem, u), True
        u = op(problem, u)
    return u, False


def run_pipeline(name: str, ctx: DesignSolveContext, tol: float = 1e-8,
                 alternatives: dict | None = None) -> tuple[list[np.ndarray], dict]:
    """Run one pipeline for all excitations; returns (fields, info)."""
    t0 = time.perf_counter()
    problems = ctx.problems()
    escalated = False

    if name == "direct":
        lu = ctx.lu()
        _, idx = ctx.assembled()
        fields = []
        for p in problems:
            _, b, _ = p.assemble()
            fields.append(p.to_grid(lu.solve(b), idx))

    elif name == "amg_cg":
        amg = ctx.amg()
        fields = []
        for p in problems:
            _, b, _ = p.assemble()
            x = amg.ml.solve(b, tol=tol * 1e-2, accel="cg", maxiter=400)
            fields.append(p.to_grid(x, amg.idx))

    elif name in ("surr_cg", "surr_mgcg", "surr_amg"):
        from ..solvers.krylov import MGPreconditionedCG

        preds = ctx.predictions()
        fields = []
        for p, u0 in zip(problems, preds):
            u = p.clamp(u0)
            if name in ("surr_cg", "surr_mgcg"):
                op = (ConjugateGradientBlock(iters=40) if name == "surr_cg"
                      else MGPreconditionedCG(iters=5))
                mon = EscalationMonitor(alternatives) if alternatives else None

                def esc(pp, uu):
                    amg = ctx.amg()
                    x = uu[~pp.fixed_mask]
                    x = amg.ml.solve(_rhs(pp), x0=x, tol=tol * 1e-2, accel="cg",
                                     maxiter=400)
                    return pp.to_grid(x, amg.idx)

                u, esc_flag = _finish_iter(p, u, tol, op, monitor=mon,
                                           on_escalate=esc)
                escalated |= esc_flag
            else:
                amg = ctx.amg()
                x = u[~p.fixed_mask]
                x = amg.ml.solve(_rhs(p), x0=x, tol=tol * 1e-2, accel="cg", maxiter=400)
                u = p.to_grid(x, amg.idx)
            fields.append(u)

    elif name == "hints":
        fields = []
        for k, p in enumerate(problems):
            u, _ = _hints_with_state(p, ctx, k, tol)
            fields.append(u)
    else:
        raise ValueError(name)

    residuals = [p.residual_norm(u) for p, u in zip(problems, fields)]
    ok = all(r <= tol * 1.01 for r in residuals)
    if not ok:
        # guaranteed fallback: exact direct solve (verification failure path)
        lu = ctx.lu()
        _, idx = ctx.assembled()
        fields = []
        for p in problems:
            _, b, _ = p.assemble()
            fields.append(p.to_grid(lu.solve(b), idx))
        residuals = [p.residual_norm(u) for p, u in zip(problems, fields)]
        escalated = True

    return fields, {
        "t_total": time.perf_counter() - t0,
        "residuals": residuals,
        "escalated": escalated,
        "pipeline": name,
    }


def _rhs(problem: LaplaceProblem) -> np.ndarray:
    _, b, _ = problem.assemble()
    return b


def _hints_with_state(problem, ctx, k, tol, n_r: int = 16):
    from ..solvers.smoothers import RedBlackGaussSeidel

    state = {"u": problem.initial_guess()}

    def corrector(p, r):
        u_pred = ctx.predictions()[k]
        return np.where(p.fixed_mask, 0.0, p.clamp(u_pred) - state["u"])

    smoother = RedBlackGaussSeidel(sweeps=1)
    u = state["u"]
    it = 0
    while problem.residual_norm(u) > tol and it < 3000:
        if it % n_r == 0 and it > 0:
            state["u"] = u
            delta = corrector(problem, problem.residual(u))
            u = np.where(problem.fixed_mask, u, u + delta)
        else:
            u = smoother(problem, u)
        it += 1
    return u, {"iters": it}
