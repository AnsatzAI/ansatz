"""Rung 2 advantage demo: verified frequency PLANNING on the two-qubit cell.

Task: hit four mode targets (f_q1*, f_q2*, f_r1*, f_r2*) simultaneously —
the unit frequency-crowding problem — with the final design verified by a
full Palace solve of the two-qubit cell. Success = worst per-mode miss
within tol.

Ansatz arm: invert the 4-target forward model by random search over the
parameter box (millisecond-scale), one verified solve, at most one decoupled
physics correction (1/L scaling for resonators, local cap-length slopes for
qubits), final verified solve.

Classical arm: Nelder-Mead over the 8 continuous parameters, one full
two-qubit Palace solve per evaluation, fixed solve budget.

Usage:
  python scripts/em3d_two_demo.py --targets 4.00,4.35,5.45,6.40 --tol-mhz 40
  python scripts/em3d_two_demo.py ... --skip-classical
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from em3d_two_campaign import RANGES2, run_variant_two, sample_params2

ROOT = Path(__file__).resolve().parents[1]
TARGET_KEYS = ["f_q1", "f_q2", "f_r1", "f_r2"]


def load_model():
    with open(ROOT / "runs" / "em3d_two_forward.pkl", "rb") as f:
        d = pickle.load(f)
    return d["models"], d["feats"]


def predict_all(models, feats, p: dict) -> dict:
    x = np.array([[p[k] for k in feats]])
    return {t: float(models[t].predict(x)[0]) for t in TARGET_KEYS}


def worst_miss_mhz(meas: dict, targets: dict) -> float:
    return max(abs((meas[t] or 99.0) - targets[t]) for t in TARGET_KEYS) * 1e3


def ansatz_arm(models, feats, targets, tol_mhz, ranks, seed: int = 0) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    cand = [sample_params2(rng) for _ in range(200_000)]
    xs = np.array([[c[k] for k in feats] for c in cand])
    errs = np.zeros(len(cand))
    for t in TARGET_KEYS:
        errs += (models[t].predict(xs) - targets[t]) ** 2
    best = cand[int(np.argmin(errs))]
    t_predict = time.time() - t0

    pred = predict_all(models, feats, best)
    row = run_variant_two("demo2_ansatz_0", best, ranks,
                          target_ghz=pred["f_q1"] - 0.3)
    n_solves = 1
    meas = {t: row.get(t) for t in TARGET_KEYS}
    miss = worst_miss_mhz(meas, targets) if row["ok"] else 9e9

    if miss > tol_mhz and row["ok"]:
        df = pd.read_parquet(ROOT / "runs" / "em3d_two_dataset.parquet")
        df = df[df.ok == True]  # noqa: E712
        damp = 0.8
        for i in ("1", "2"):
            # resonator: quarter-wave 1/L scaling
            tl = best[f"total_length_{i}"] * (
                1 + damp * (meas[f"f_r{i}"] / targets[f"f_r{i}"] - 1.0))
            lo, hi = RANGES2[f"total_length_{i}"]
            best[f"total_length_{i}"] = float(np.clip(tl, lo, hi))
            # qubit: local slope of f_q vs cap_length
            col = f"cap_length_{i}"
            near = df.iloc[(df[col] - best[col]).abs().argsort()[:20]]
            slope = np.polyfit(near[col], near[f"f_q{i}"], 1)[0]
            dcap = damp * (targets[f"f_q{i}"] - meas[f"f_q{i}"]) / slope
            lo, hi = RANGES2[col]
            tr = 0.12 * (hi - lo)
            best[col] = float(np.clip(best[col] + np.clip(dcap, -tr, tr), lo, hi))
        if best["cap_length_1"] < best["cap_length_2"] + 40:
            best["cap_length_1"] = best["cap_length_2"] + 40
        pred = predict_all(models, feats, best)
        row = run_variant_two("demo2_ansatz_1", best, ranks,
                              target_ghz=min(pred["f_q1"],
                                             meas["f_q1"] or 9) - 0.3)
        n_solves += 1
        meas = {t: row.get(t) for t in TARGET_KEYS}
        miss = worst_miss_mhz(meas, targets) if row["ok"] else 9e9

    return dict(arm="ansatz", t_total=time.time() - t0, t_predict=t_predict,
                n_solves=n_solves, worst_miss_mhz=float(miss),
                measured=meas, design=best)


def classical_arm(targets, tol_mhz, ranks, max_solves=10) -> dict:
    from scipy.optimize import minimize

    t0 = time.time()
    keys = list(RANGES2)
    lo = np.array([RANGES2[k][0] for k in keys])
    hi = np.array([RANGES2[k][1] for k in keys])
    count = [0]
    best = {"miss": np.inf, "x": None}

    def objective(z):
        if count[0] >= max_solves:
            return best["miss"]
        x = lo + (hi - lo) / (1 + np.exp(-z))
        p = dict(zip(keys, x))
        if p["cap_length_1"] < p["cap_length_2"] + 40:
            p["cap_length_1"] = p["cap_length_2"] + 40
        row = run_variant_two(f"demo2_classical_{count[0]}", p, ranks)
        count[0] += 1
        if not row["ok"]:
            return 9e9
        meas = {t: row.get(t) for t in TARGET_KEYS}
        miss = worst_miss_mhz(meas, targets)
        if miss < best["miss"]:
            best.update(miss=miss, x=p, meas=meas)
        return miss

    minimize(objective, np.zeros(len(keys)), method="Nelder-Mead",
             options={"maxfev": max_solves, "xatol": 1e-3, "fatol": 1e-2})
    return dict(arm="classical", t_total=time.time() - t0,
                n_solves=count[0], worst_miss_mhz=float(best["miss"]),
                measured=best.get("meas"), design=best["x"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="4.00,4.35,5.45,6.40")
    ap.add_argument("--tol-mhz", type=float, default=40.0)
    ap.add_argument("--ranks", type=int, default=6)
    ap.add_argument("--max-classical-solves", type=int, default=10)
    ap.add_argument("--skip-classical", action="store_true")
    ap.add_argument("--skip-ansatz", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-name", default="em3d_two_advantage.json")
    a = ap.parse_args()
    tvals = [float(v) for v in a.targets.split(",")]
    targets = dict(zip(TARGET_KEYS, tvals))

    out_path = ROOT / "runs" / a.out_name
    out = {"targets": targets, "tol_mhz": a.tol_mhz}
    if out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("targets") == targets:
            out.update({k: v for k, v in prev.items()
                        if k in ("classical", "ansatz")})
    models, feats = load_model()
    if not a.skip_ansatz:
        out["ansatz"] = ansatz_arm(models, feats, targets, a.tol_mhz, a.ranks,
                                   seed=a.seed)
        print(json.dumps(out["ansatz"], indent=2, default=str))
    if not a.skip_classical:
        out["classical"] = classical_arm(targets, a.tol_mhz, a.ranks,
                                         a.max_classical_solves)
        print(json.dumps(out["classical"], indent=2, default=str))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
