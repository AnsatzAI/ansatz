"""Tier-1 3D forward model: geometry params -> (f0, f1, Q) from campaign data.

Small-data regime (~100 bespoke Palace solves), so: gradient-boosted trees +
leave-one-out cross-validation for an honest error estimate, plus a kNN
baseline. Writes runs/em3d_forward.pkl + runs/em3d_forward_results.json.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs" / "em3d_dataset.parquet"

FEATS = ["cap_length", "cap_gap", "total_length", "claw_length", "meander_turn_count"]
TARGETS = ["f0", "f1"]


def main() -> None:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import LeaveOneOut
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet(DATA)
    df = df[df.ok & df.f0.notna() & df.f1.notna()].reset_index(drop=True)
    print(f"{len(df)} usable solves")
    x = df[FEATS].values
    results = {"n": int(len(df))}

    for tgt in TARGETS:
        y = df[tgt].values
        loo_pred = np.zeros_like(y)
        knn_pred = np.zeros_like(y)
        for tr, te in LeaveOneOut().split(x):
            m = GradientBoostingRegressor(n_estimators=300, max_depth=2,
                                          learning_rate=0.05)
            m.fit(x[tr], y[tr])
            loo_pred[te] = m.predict(x[te])
            sc = StandardScaler().fit(x[tr])
            k = KNeighborsRegressor(n_neighbors=min(5, len(tr)),
                                    weights="distance")
            k.fit(sc.transform(x[tr]), y[tr])
            knn_pred[te] = k.predict(sc.transform(x[te]))
        for name, pred in [("gbr", loo_pred), ("knn", knn_pred)]:
            mape = float(np.mean(np.abs(pred - y) / y) * 100)
            mae_mhz = float(np.mean(np.abs(pred - y)) * 1e3)
            results[f"{tgt}/{name}"] = {"loo_mape_pct": mape, "loo_mae_mhz": mae_mhz}
            print(f"{tgt}/{name}: LOO MAPE {mape:.3f}%  MAE {mae_mhz:.1f} MHz")

    models = {}
    for tgt in TARGETS:
        m = GradientBoostingRegressor(n_estimators=300, max_depth=2,
                                      learning_rate=0.05)
        m.fit(x, df[tgt].values)
        models[tgt] = m
    with open(ROOT / "runs" / "em3d_forward.pkl", "wb") as f:
        pickle.dump({"models": models, "feats": FEATS}, f)
    with open(ROOT / "runs" / "em3d_forward_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
