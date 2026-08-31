"""Time-to-verified-tolerance benchmark over dataset shards.

Each (design, resolution, tolerance, pipeline) cell is measured with a fresh
solve context, so every pipeline pays its true end-to-end cost including
assembly, setup, surrogate inference, and verification. Results stream to a
parquet for router fitting and reporting.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..router.pipelines import DesignSolveContext, run_pipeline
from ..router.policy import instance_features
from ..surrogate.data import _unpack


def iter_designs(root: str | Path, split: str, n: int):
    files = sorted(Path(root).glob(f"{split}_{n}_*.npz"))
    for f in files:
        with np.load(f) as z:
            params = z["params"]
            for d in range(params.shape[0]):
                cross = _unpack(z[f"cross_{d}"], n)
                claw = _unpack(z[f"claw_{d}"], n)
                ground = _unpack(z[f"ground_{d}"], n)
                yield params[d], [cross, claw], ground


def run_benchmark(
    root: str | Path,
    split: str,
    n: int,
    surrogate,
    tol: float = 1e-8,
    pipelines: list[str] | None = None,
    limit: int | None = None,
    out: str | Path | None = None,
) -> pd.DataFrame:
    pipelines = pipelines or ["direct", "amg_cg", "surr_cg", "surr_amg", "hints"]
    rows = []
    for i, (params, conductors, ground) in enumerate(iter_designs(root, split, n)):
        if limit is not None and i >= limit:
            break
        fixed = ground | conductors[0] | conductors[1]
        free_frac = float((~fixed).mean())
        feats = instance_features(params, n, free_frac, tol)
        for name in pipelines:
            ctx = DesignSolveContext(n, fixed, conductors, surrogate=surrogate)
            try:
                fields, info = run_pipeline(name, ctx, tol=tol)
                rows.append(
                    dict(
                        split=split, n=n, design=i, tol=tol, pipeline=name,
                        t_total=info["t_total"],
                        max_residual=float(max(info["residuals"])),
                        escalated=bool(info["escalated"]),
                        **{f"f{j}": float(v) for j, v in enumerate(feats)},
                    )
                )
            except Exception as e:  # noqa: BLE001 — record failures, keep sweeping
                rows.append(
                    dict(
                        split=split, n=n, design=i, tol=tol, pipeline=name,
                        t_total=np.nan, max_residual=np.nan, escalated=False,
                        error=str(e)[:200],
                        **{f"f{j}": float(v) for j, v in enumerate(feats)},
                    )
                )
        if (i + 1) % 20 == 0:
            print(f"[{split} n={n}] {i + 1} designs", flush=True)
            if out is not None:
                pd.DataFrame(rows).to_parquet(out)
    df = pd.DataFrame(rows)
    if out is not None:
        df.to_parquet(out)
    return df
