"""Forward surrogate on real SQuADDS Q3D data: geometry -> capacitance matrix.

Compares model classes against the practitioner alternatives (nearest-design
lookup, linear fit) on (a) a random in-distribution split and (b) an
out-of-hull extrapolation split (largest 12% of cross_length held out), which
is where database interpolation is known to degrade. Also reports downstream
Hamiltonian-parameter errors through the LOM pipeline.

Writes runs/forward_results.json and runs/forward_gbr.pkl.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ansatz.pde.lom import hamiltonian_from_cap

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["cross_length", "claw_length", "ground_spacing"]
TARGETS = [
    "sim_cross_to_ground",
    "sim_claw_to_ground",
    "sim_cross_to_claw",
    "sim_claw_to_claw",
    "sim_ground_to_ground",
]


def mape(y, p):
    return float(np.mean(np.abs(p - y) / np.abs(y)) * 100)


def eval_downstream(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Hamiltonian-parameter % errors via LOM (columns follow TARGETS order)."""
    errs = {k: [] for k in ("f_q", "alpha", "g", "chi")}
    for t, p in zip(y_true, y_pred):
        ht = hamiltonian_from_cap(t[0], t[2])
        hp = hamiltonian_from_cap(p[0], p[2])
        for k in errs:
            tv, pv = getattr(ht, k), getattr(hp, k)
            errs[k].append(abs(pv - tv) / abs(tv) * 100)
    return {k: float(np.mean(v)) for k, v in errs.items()}


def run_split(df, train_idx, test_idx, tag, results):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    xtr = df.loc[train_idx, FEATS].values
    xte = df.loc[test_idx, FEATS].values
    ytr = df.loc[train_idx, TARGETS].values
    yte = df.loc[test_idx, TARGETS].values

    sc = StandardScaler().fit(xtr)

    models = {
        "nn_lookup": KNeighborsRegressor(n_neighbors=1).fit(sc.transform(xtr), ytr),
        "knn5": KNeighborsRegressor(n_neighbors=5, weights="distance").fit(
            sc.transform(xtr), ytr
        ),
        "linear": LinearRegression().fit(xtr, ytr),
        "rf": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0).fit(
            xtr, ytr
        ),
    }
    gbr = [
        GradientBoostingRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.05, random_state=0
        ).fit(xtr, ytr[:, j])
        for j in range(len(TARGETS))
    ]

    preds = {name: (m.predict(sc.transform(xte)) if "nn" in name or "knn" in name
                    else m.predict(xte))
             for name, m in models.items()}  # noqa: PLC0206
    preds["gbr"] = np.stack([g.predict(xte) for g in gbr], axis=1)

    for name, p in preds.items():
        per_target = {t.replace("sim_", ""): mape(yte[:, j], p[:, j])
                      for j, t in enumerate(TARGETS)}
        results[f"{tag}/{name}"] = {
            "cap_mape": per_target,
            "cap_mape_mean": float(np.mean(list(per_target.values()))),
            "downstream": eval_downstream(yte, p),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
    return gbr, sc


def main():
    df = pd.read_parquet(ROOT / "data" / "squadds_qubit_cap.parquet").dropna(
        subset=FEATS + TARGETS
    ).reset_index(drop=True)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(df))
    n_test = int(0.15 * len(df))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    results: dict = {}
    gbr, sc = run_split(df, train_idx, test_idx, "iid", results)

    # out-of-hull split: hold out the largest cross_length designs
    thresh = df.cross_length.quantile(0.88)
    hull_test = df.index[df.cross_length > thresh].to_numpy()
    hull_train = df.index[df.cross_length <= thresh].to_numpy()
    run_split(df, hull_train, hull_test, "outhull", results)

    out = ROOT / "runs"
    out.mkdir(exist_ok=True)
    with open(out / "forward_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out / "forward_gbr.pkl", "wb") as f:
        pickle.dump({"models": gbr, "scaler": sc, "feats": FEATS, "targets": TARGETS}, f)

    for k, v in results.items():
        print(f"{k:22s} capMAPE={v['cap_mape_mean']:5.2f}%  "
              f"g_err={v['downstream']['g']:5.2f}%  chi_err={v['downstream']['chi']:5.2f}%")


if __name__ == "__main__":
    main()
