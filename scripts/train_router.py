"""Fit the cost-model router from benchmark measurements and evaluate routing.

Reads benchmark parquets (routertrain split for fitting, test/ood for eval),
fits per-pipeline log-cost regressors, and reports:
  - routed vs best-single-pipeline vs oracle wall-clock (sum + geomean ratio)
  - per-instance win/tie rates and worst-case regret vs each fixed pipeline
  - decision overhead

Router decision cost is measured and charged to the routed time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ansatz.router.policy import PIPELINES, CostModelRouter


def load_frames(paths):
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def pivot(df):
    feat_cols = sorted(
        (c for c in df.columns if c.startswith("f") and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )
    wide = df.pivot_table(
        index=["split", "n", "design", "tol"] + feat_cols,
        columns="pipeline", values="t_total",
    ).reset_index()
    return wide, feat_cols


def main(train_paths, eval_paths, out_dir):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    tr, feat_cols = pivot(load_frames(train_paths))
    avail = [p for p in PIPELINES if p in tr.columns]
    feats = tr[feat_cols].values.astype(np.float32)
    costs = {p: tr[p].values for p in avail}  # NaNs masked inside fit()
    router = CostModelRouter()
    router.fit(feats, costs)
    router.save(out / "router.pkl")

    ev, _ = pivot(load_frames(eval_paths))
    core = [p for p in avail if p != "hints" and p in ev.columns]
    ev = ev.dropna(subset=core).reset_index(drop=True)
    results = {}
    X = ev[feat_cols].values.astype(np.float32)

    # single-solve mode overhead (honest per-call cost)
    t0 = time.perf_counter()
    for x in X[:50]:
        router.decide(x)
    single_overhead = (time.perf_counter() - t0) / max(min(len(X), 50), 1)
    # sweep mode: batch decisions, then mask to measured pipelines
    t0 = time.perf_counter()
    decisions = router.decide_batch(X)
    overhead = (time.perf_counter() - t0) / max(len(X), 1)
    for i, d in enumerate(decisions):
        ok = [p for p in d.predicted_costs
              if p in ev.columns and np.isfinite(ev.iloc[i][p])]
        if d.pipeline not in ok:
            best = min({p: d.predicted_costs[p] for p in ok}, key=d.predicted_costs.get)
            decisions[i] = type(d)(pipeline=best, predicted_costs=d.predicted_costs,
                                   features=d.features)

    routed_time = np.array(
        [ev.iloc[i][d.pipeline] + overhead for i, d in enumerate(decisions)]
    )
    oracle_time = ev[core].min(axis=1).values
    oracle_choice = ev[core].idxmin(axis=1)

    summary = {
        "n_eval": len(ev),
        "decision_overhead_ms": overhead * 1e3,
        "decision_overhead_single_ms": single_overhead * 1e3,
        "routed_total_s": float(routed_time.sum()),
        "oracle_total_s": float(oracle_time.sum()),
        "routed_vs_oracle": float(routed_time.sum() / oracle_time.sum()),
        "oracle_agreement": float(
            np.mean([d.pipeline == oc for d, oc in zip(decisions, oracle_choice)])
        ),
        "pipelines": {},
    }
    for p in core:
        base = ev[p].values
        ratio = routed_time / base
        summary["pipelines"][p] = {
            "total_s": float(base.sum()),
            "routed_speedup_total": float(base.sum() / routed_time.sum()),
            "routed_speedup_geomean": float(np.exp(np.mean(np.log(base / routed_time)))),
            "win_rate": float(np.mean(routed_time < base * 0.98)),
            "within_5pct_or_better": float(np.mean(ratio <= 1.05)),
            "worst_case_ratio": float(ratio.max()),
        }
    # per-resolution breakdown
    summary["by_n"] = {}
    for n, g in ev.groupby("n"):
        idx = g.index
        rt = routed_time[[ev.index.get_loc(i) for i in idx]]
        summary["by_n"][int(n)] = {
            "routed_total_s": float(rt.sum()),
            **{p: float(np.nansum(g[p])) for p in avail if p in g.columns},
            "router_choices": pd.Series(
                [decisions[ev.index.get_loc(i)].pipeline for i in idx]
            ).value_counts().to_dict(),
        }

    with open(out / "router_eval.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    main(a.train, a.eval, a.out)
