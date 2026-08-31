"""CLI benchmark runner. Example:
python scripts/benchmark.py --split test --n 511 --tol 1e-8 --out runs/bench_test_511.parquet
"""

from __future__ import annotations

import argparse

from ansatz.bench.harness import run_benchmark
from ansatz.surrogate.infer import FieldSurrogate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fields")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=255)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--weights", default="runs/unet_255.pt")
    ap.add_argument("--pipelines", default="direct,amg_cg,surr_cg,surr_amg,hints")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    surrogate = FieldSurrogate(a.weights)
    surrogate.warmup()
    df = run_benchmark(
        a.data, a.split, a.n, surrogate, tol=a.tol,
        pipelines=a.pipelines.split(","), limit=a.limit, out=a.out,
    )
    ok = df.dropna(subset=["t_total"])
    print(ok.groupby("pipeline").t_total.describe()[["count", "mean", "50%"]])
    print("verification failures:", int((ok.max_residual > a.tol * 1.01).sum()))
