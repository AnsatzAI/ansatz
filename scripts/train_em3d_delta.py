"""Rung 1 multi-fidelity model: coarse-model prediction + learned delta.

f_fine(params) ~ f_coarse_model(params) + delta(params)
With K ~ 12 fine solves, delta is fit as (a) constant offset and (b) linear
in params; pick by LOO error. Writes runs/em3d_delta.pkl + results json.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["cap_length", "cap_gap", "total_length", "l_claw", "n_meander_turns"]


def fit_delta(x, d):
    """Return dict of candidate delta models with LOO MAE (GHz)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut

    out = {}
    # constant offset
    loo_c = np.array([
        np.mean(d[tr]) for tr, te in LeaveOneOut().split(x)
    ])
    out["const"] = {
        "loo_mae": float(np.mean(np.abs(loo_c - d))),
        "model": ("const", float(np.mean(d))),
    }
    # linear in params
    preds = np.zeros_like(d)
    for tr, te in LeaveOneOut().split(x):
        m = LinearRegression().fit(x[tr], d[tr])
        preds[te] = m.predict(x[te])
    out["linear"] = {
        "loo_mae": float(np.mean(np.abs(preds - d))),
        "model": ("linear", LinearRegression().fit(x, d)),
    }
    return out


def main() -> None:
    fine = pd.read_parquet(ROOT / "runs" / "em3d_dataset_fine.parquet")
    fine = fine[fine.ok == True].reset_index(drop=True)  # noqa: E712
    with open(ROOT / "runs" / "em3d_forward.pkl", "rb") as f:
        fwd = pickle.load(f)

    results = {"k": int(len(fine))}
    saved = {}
    for tgt, fine_col in (("f0", "f0"), ("f1", "f1")):
        sub = fine[fine[fine_col].notna()].copy()
        if tgt == "f1":
            sub = sub[(sub.f1 > 4.2) & (sub.f1 < 7.6)]
        if len(sub) < 5:
            print(f"{tgt}: only {len(sub)} fine labels; skipping")
            continue
        x = sub[FEATS].values
        coarse_pred = fwd["models"][tgt].predict(x)
        d = sub[fine_col].values - coarse_pred
        cands = fit_delta(x, d)
        best = min(cands, key=lambda k: cands[k]["loo_mae"])
        results[tgt] = {
            "n": int(len(sub)),
            "raw_coarse_vs_fine_mae_mhz":
                float(np.mean(np.abs(d)) * 1e3),
            "delta_mean_mhz": float(np.mean(d) * 1e3),
            **{f"loo_{k}_mae_mhz": v["loo_mae"] * 1e3 for k, v in cands.items()},
            "chosen": best,
        }
        saved[tgt] = cands[best]["model"]
        print(f"{tgt}: coarse-vs-fine MAE {results[tgt]['raw_coarse_vs_fine_mae_mhz']:.1f} MHz "
              f"| delta({best}) LOO {cands[best]['loo_mae']*1e3:.1f} MHz")

    with open(ROOT / "runs" / "em3d_delta.pkl", "wb") as f:
        pickle.dump(saved, f)
    with open(ROOT / "runs" / "em3d_delta_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
