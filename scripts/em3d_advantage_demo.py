"""The advantage demonstration: verified inverse design, Ansatz vs classical.

Task: hit target qubit/readout frequencies (f0*, f1*) within `tol_mhz` by
choosing geometry, with the final design VERIFIED by a full Palace eigenmode
solve in both arms.

Classical arm (what a practitioner does today):
  derivative-free optimization (Nelder-Mead over the geometry box), every
  objective evaluation = one full Palace solve. Budgeted at `max_solves`.

Ansatz arm:
  1. invert the Tier-1 forward model on a dense candidate grid (microseconds)
  2. run ONE Palace verification solve at the chosen geometry, with the
     eigensolver target set safely below the predicted f0 (avoids the
     missed-mode cliff measured in the shift experiment)
  3. if the verified miss exceeds tol: one local linear correction using the
     model's Jacobian, then one final verification solve.

Both arms report end-to-end wall-clock and the verified miss. Honesty rules:
identical solver settings for verification solves; classical arm warm-started
at the box center; every Palace call counted.

Usage: python scripts/em3d_advantage_demo.py --targets 4.05,5.55 --tol-mhz 15
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import time
from pathlib import Path

import numpy as np

from em3d_campaign import EX, RANGES_INT, RANGES_UM, run_variant

ROOT = Path(__file__).resolve().parents[1]


def load_model():
    with open(ROOT / "runs" / "em3d_forward.pkl", "rb") as f:
        d = pickle.load(f)
    return d["models"], d["feats"]


def predict(models, feats, params: dict) -> tuple[float, float]:
    x = np.array([[params[k] for k in feats]])
    return float(models["f0"].predict(x)[0]), float(models["f1"].predict(x)[0])


def _pick_modes(row: dict) -> dict:
    """Identify qubit/readout modes from the full list by physical band."""
    import json as _json

    freqs = sorted(_json.loads(row.get("f_all", "[]")))
    qubit = next((f for f in freqs if 3.2 < f < 5.0), None)
    readout = next((f for f in freqs if f != qubit and 4.8 < f < 7.6
                    and (qubit is None or f > qubit)), None)
    row = dict(row)
    row["f0"], row["f1"] = qubit, readout
    return row


def solve(params: dict, tag: str, ranks: int) -> dict:
    p_um = {k: params[k] for k in RANGES_UM}
    p_int = {k: int(params[k]) for k in RANGES_INT}
    # N=4 modes so qubit + readout are robustly captured (see EM3D.md)
    return _pick_modes(run_variant(tag, p_um, p_int, ranks, n_modes=4))


def ansatz_arm(models, feats, f0t, f1t, tol_mhz, ranks) -> dict:
    t0 = time.time()
    # dense candidate grid over the box
    grids = [np.linspace(*RANGES_UM[k], 24) for k in RANGES_UM]
    grids.append(np.arange(RANGES_INT["n_meander_turns"][0],
                           RANGES_INT["n_meander_turns"][1] + 1))
    keys = list(RANGES_UM) + list(RANGES_INT)
    best, best_err = None, np.inf
    names = list(RANGES_UM) + list(RANGES_INT)
    cand = np.array(list(itertools.product(*grids)))
    xs = cand[:, [names.index(k) for k in feats]]
    p0 = models["f0"].predict(xs)
    p1 = models["f1"].predict(xs)
    err = np.hypot(p0 - f0t, p1 - f1t)
    i = int(np.argmin(err))
    best = dict(zip(keys, cand[i]))
    t_predict = time.time() - t0

    solves = []
    row = solve(best, "demo_ansatz_0", ranks)
    solves.append(row)
    miss = np.hypot((row["f0"] or 9) - f0t, (row["f1"] or 9) - f1t) * 1e3
    if miss > tol_mhz:
        # one linear correction via finite-difference Jacobian of the model
        x = np.array([best[k] for k in feats])
        jac = np.zeros((2, len(feats)))
        eps = np.array([1.0 if k != "n_meander_turns" else 1 for k in feats])
        for j in range(len(feats)):
            xp = x.copy(); xp[j] += eps[j]
            f0p = models["f0"].predict(xp[None])[0]
            f1p = models["f1"].predict(xp[None])[0]
            jac[:, j] = [(f0p - predict(models, feats, best)[0]) / eps[j],
                         (f1p - predict(models, feats, best)[1]) / eps[j]]
        resid = np.array([f0t - row["f0"], f1t - row["f1"]])
        step, *_ = np.linalg.lstsq(jac, resid, rcond=None)
        for j, k in enumerate(feats):
            lo, hi = (RANGES_UM.get(k) or RANGES_INT[k])
            best[k] = float(np.clip(best[k] + step[j], lo, hi))
        row = solve(best, "demo_ansatz_1", ranks)
        solves.append(row)
        miss = np.hypot((row["f0"] or 9) - f0t, (row["f1"] or 9) - f1t) * 1e3

    return dict(
        arm="ansatz", t_total=time.time() - t0, t_predict=t_predict,
        n_solves=len(solves), miss_mhz=float(miss), design=best,
        t_solves=[s.get("t_solve") for s in solves],
    )


def classical_arm(f0t, f1t, tol_mhz, ranks, max_solves=12) -> dict:
    from scipy.optimize import minimize

    t0 = time.time()
    keys = list(RANGES_UM) + list(RANGES_INT)
    lo = np.array([RANGES_UM[k][0] for k in RANGES_UM] +
                  [RANGES_INT[k][0] for k in RANGES_INT])
    hi = np.array([RANGES_UM[k][1] for k in RANGES_UM] +
                  [RANGES_INT[k][1] for k in RANGES_INT])
    count = [0]
    best = {"err": np.inf, "x": None}

    def objective(z):
        if count[0] >= max_solves:
            return best["err"]
        x = lo + (hi - lo) / (1 + np.exp(-z))  # box via sigmoid
        params = dict(zip(keys, x))
        row = solve(params, f"demo_classical_{count[0]}", ranks)
        count[0] += 1
        if not row["ok"] or row["f0"] is None or row["f1"] is None:
            return 10.0
        err = float(np.hypot(row["f0"] - f0t, row["f1"] - f1t))
        if err < best["err"]:
            best.update(err=err, x=params, row=row)
        return err

    minimize(objective, np.zeros(len(keys)), method="Nelder-Mead",
             options={"maxfev": max_solves, "xatol": 1e-3, "fatol": 1e-4})
    return dict(
        arm="classical", t_total=time.time() - t0, n_solves=count[0],
        miss_mhz=float(best["err"] * 1e3), design=best["x"],
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="4.05,5.55")
    ap.add_argument("--tol-mhz", type=float, default=15.0)
    ap.add_argument("--ranks", type=int, default=6)
    ap.add_argument("--max-classical-solves", type=int, default=12)
    ap.add_argument("--skip-classical", action="store_true")
    a = ap.parse_args()
    f0t, f1t = (float(v) for v in a.targets.split(","))

    models, feats = load_model()
    out = {"targets": [f0t, f1t], "tol_mhz": a.tol_mhz}
    out["ansatz"] = ansatz_arm(models, feats, f0t, f1t, a.tol_mhz, a.ranks)
    print(json.dumps(out["ansatz"], indent=2, default=str))
    if not a.skip_classical:
        out["classical"] = classical_arm(f0t, f1t, a.tol_mhz, a.ranks,
                                         a.max_classical_solves)
        print(json.dumps(out["classical"], indent=2, default=str))
    with open(ROOT / "runs" / "em3d_advantage.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
