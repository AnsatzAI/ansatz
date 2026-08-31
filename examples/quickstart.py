"""End-to-end quickstart: design -> routed verified solve -> Hamiltonian params.

Run from the repo root (after training or downloading surrogate weights):
    python examples/quickstart.py --weights runs/unet_255.pt
Without weights it falls back to classical-only routing (still verified).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ansatz.geometry.transmon import XmonDesign, build_problem_masks
from ansatz.pde.capacitance import capacitance_matrix
from ansatz.pde.lom import hamiltonian_from_cap
from ansatz.router.pipelines import DesignSolveContext, run_pipeline
from ansatz.router.policy import CostModelRouter, instance_features

PARAM_KEYS = ["cross_length", "cross_width", "cross_gap",
              "claw_length", "claw_width", "claw_gap", "ground_spacing"]


def main(weights: str | None, router_path: str | None, n: int = 511):
    design = XmonDesign(cross_length=310, claw_length=160, ground_spacing=10)
    conductors, ground = build_problem_masks(design, n)
    fixed = ground | conductors[0] | conductors[1]

    surrogate = None
    if weights and Path(weights).exists():
        from ansatz.surrogate.infer import FieldSurrogate

        surrogate = FieldSurrogate(weights)
        surrogate.warmup()

    if router_path and Path(router_path).exists():
        router = CostModelRouter.load(router_path)
        feats = instance_features(
            np.array([getattr(design, k) for k in PARAM_KEYS], dtype=np.float32),
            n, float((~fixed).mean()), 1e-8,
        )
        allowed = list(router.models)
        if surrogate is None:
            allowed = [p for p in allowed if not p.startswith("surr") and p != "hints"]
        decision = router.decide(feats, allowed=allowed)
        pipeline = decision.pipeline
        print("router decision:", pipeline,
              {k: f"{v*1e3:.1f}ms" for k, v in decision.predicted_costs.items()})
    else:
        pipeline = "surr_mgcg" if surrogate else "direct"
        print("no router model; using", pipeline)

    ctx = DesignSolveContext(n, fixed, conductors, surrogate=surrogate)
    fields, info = run_pipeline(pipeline, ctx, tol=1e-8)
    print(f"solved in {info['t_total']*1e3:.1f} ms, "
          f"max residual {max(info['residuals']):.2e}, escalated={info['escalated']}")

    def solve(problem):
        for m, f in zip(conductors, fields):
            if np.array_equal(problem.fixed_values > 0, m):
                return f
        raise RuntimeError

    c, _ = capacitance_matrix(solve, n, conductors)
    # 2D per-unit-length values; scale is calibrated for absolute farads —
    # relative design comparisons and LOM trends are meaningful as-is.
    hp = hamiltonian_from_cap(c[0, 0] * 1e15, -c[0, 1] * 1e15)
    print(f"C (fF/unit): C_q={c[0,0]*1e15:.2f} C_c={-c[0,1]*1e15:.3f}")
    print(f"LOM: f_q={hp.f_q/1e9:.3f} GHz  alpha={hp.alpha/1e6:.1f} MHz  "
          f"g={hp.g/1e6:.1f} MHz  chi={hp.chi/1e6:.3f} MHz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/unet_255.pt")
    ap.add_argument("--router", default="runs/router.pkl")
    ap.add_argument("--n", type=int, default=511)
    a = ap.parse_args()
    main(a.weights, a.router, a.n)
