"""The Ansatz router: cost-predicting policy over verified solve pipelines.

Design (v0, deliberately practical):
  * Actions are complete solve pipelines (direct, AMG-CG, surrogate+CG,
    surrogate+AMG-CG, HINTS). Every pipeline terminates with residual
    verification at the requested tolerance, so routing affects wall-clock
    only — never correctness.
  * The policy is a per-pipeline wall-clock regressor on cheap instance
    features (geometry parameters, grid size, free-node fraction, tolerance).
    Route = argmin of predicted cost. Regressors are gradient-boosted trees:
    microsecond inference, no GPU dependency, trivially retrainable on a
    customer's own instance mix.
  * A monitored escalation rule supervises iterative pipelines: if the
    observed log-residual slope projects a finish time worse than the best
    alternative (plus switch cost), the solve escalates once to that
    alternative, warm-started from the current iterate. This bounds the
    router's worst case at (overhead + best-alternative time) even when the
    cost model is wrong.

This replaces the per-iteration teacher-forced classifier of the greedy-router
line of work with an instance-level cost model + runtime guardrail, which is
what actually pays off at production scale (see paper, Sec. 3).
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PIPELINES = ["direct", "amg_cg", "surr_cg", "surr_mgcg", "surr_amg", "hints"]


def instance_features(params: np.ndarray, n: int, free_frac: float, tol: float) -> np.ndarray:
    """params: the 7 geometry parameters (um)."""
    return np.concatenate([
        params / 100.0,
        [np.log2(n), free_frac, -np.log10(tol)],
    ]).astype(np.float32)


@dataclass
class RouteDecision:
    pipeline: str
    predicted_costs: dict[str, float]
    features: np.ndarray


class CostModelRouter:
    def __init__(self, models: dict | None = None):
        self.models = models or {}

    def fit(self, feats: np.ndarray, costs: dict[str, np.ndarray]) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        self.models = {}
        for name, y in costs.items():
            m = GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9
            )
            m.fit(feats, np.log(np.maximum(y, 1e-4)))
            self.models[name] = m

    def decide(self, feats: np.ndarray, allowed: list[str] | None = None) -> RouteDecision:
        allowed = allowed or list(self.models)
        pred = {
            name: float(np.exp(self.models[name].predict(feats[None])[0]))
            for name in allowed
        }
        best = min(pred, key=pred.get)
        return RouteDecision(pipeline=best, predicted_costs=pred, features=feats)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.models, f)

    @classmethod
    def load(cls, path: str | Path) -> "CostModelRouter":
        with open(path, "rb") as f:
            return cls(models=pickle.load(f))


class EscalationMonitor:
    """Project finish time from the observed log-residual slope; escalate once
    if the projection exceeds the best alternative's predicted total."""

    def __init__(self, alternatives: dict[str, float], patience: int = 3):
        self.alternatives = alternatives
        self.patience = patience
        self.history: list[tuple[float, float]] = []  # (elapsed, log10 residual)
        self.fired = False

    def observe(self, elapsed: float, residual: float, tol: float) -> str | None:
        if self.fired or residual <= tol:
            return None
        self.history.append((elapsed, np.log10(max(residual, 1e-300))))
        if len(self.history) < self.patience + 1:
            return None
        (t0, r0), (t1, r1) = self.history[-self.patience - 1], self.history[-1]
        slope = (r1 - r0) / max(t1 - t0, 1e-9)  # log10 decades per second
        if slope >= -1e-9:
            projected = np.inf
        else:
            projected = t1 + (np.log10(tol) - r1) / slope
        best_alt = min(self.alternatives, key=self.alternatives.get)
        if projected > self.alternatives[best_alt] + t1:
            self.fired = True
            return best_alt
        return None


def now() -> float:
    return time.perf_counter()
