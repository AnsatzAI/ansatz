"""Two-qubit-cell forward model: 8 geometry params -> 4 mode frequencies.

LOO-validated GBR per target vs kNN baseline; writes runs/em3d_two_forward.pkl.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["cap_length_1", "cap_length_2", "cap_gap_1", "cap_gap_2",
         "total_length_1", "total_length_2", "l_claw_1", "l_claw_2"]
TARGETS = ["f_q1", "f_q2", "f_r1", "f_r2"]


def main() -> None:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import LeaveOneOut
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet(ROOT / "runs" / "em3d_two_dataset.parquet")
    df = df[df.ok == True].dropna(subset=TARGETS).reset_index(drop=True)  # noqa: E712
    print(f"{len(df)} usable two-qubit solves")
    x = df[FEATS].values
    results = {"n": int(len(df))}
    models = {}
    for tgt in TARGETS:
        y = df[tgt].values
        loo = np.zeros_like(y)
        knn = np.zeros_like(y)
        for tr, te in LeaveOneOut().split(x):
            m = GradientBoostingRegressor(n_estimators=300, max_depth=2,
                                          learning_rate=0.05)
            m.fit(x[tr], y[tr])
            loo[te] = m.predict(x[te])
            sc = StandardScaler().fit(x[tr])
            kn = KNeighborsRegressor(n_neighbors=min(5, len(tr)),
                                     weights="distance").fit(
                sc.transform(x[tr]), y[tr])
            knn[te] = kn.predict(sc.transform(x[te]))
        for name, pred in (("gbr", loo), ("knn", knn)):
            results[f"{tgt}/{name}"] = {
                "loo_mape_pct": float(np.mean(np.abs(pred - y) / y) * 100),
                "loo_mae_mhz": float(np.mean(np.abs(pred - y)) * 1e3),
            }
        print(f"{tgt}: gbr LOO MAE "
              f"{results[f'{tgt}/gbr']['loo_mae_mhz']:.1f} MHz | knn "
              f"{results[f'{tgt}/knn']['loo_mae_mhz']:.1f} MHz")
        m = GradientBoostingRegressor(n_estimators=300, max_depth=2,
                                      learning_rate=0.05)
        m.fit(x, y)
        models[tgt] = m

    with open(ROOT / "runs" / "em3d_two_forward.pkl", "wb") as f:
        pickle.dump({"models": models, "feats": FEATS}, f)
    with open(ROOT / "runs" / "em3d_two_forward_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
