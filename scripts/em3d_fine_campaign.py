"""Rung 1: production-fidelity (order-3 + AMR-2) solves for multi-fidelity learning.

Re-solves K geometries from the existing coarse campaign at sign-off fidelity.
Each fine solve uses a safe eigensolver target derived from the geometry's
KNOWN coarse qubit frequency (coarse_f0 - 0.3 GHz) — prediction-informed
targeting applied to data generation itself. Writes coarse/fine pairs to
runs/em3d_dataset_fine.parquet (resumable).

Usage: python scripts/em3d_fine_campaign.py --k 12 --ranks 6
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from em3d_campaign import RANGES_INT, RANGES_UM, run_variant

ROOT = Path(__file__).resolve().parents[1]
COARSE = ROOT / "runs" / "em3d_dataset.parquet"
OUT = ROOT / "runs" / "em3d_dataset_fine.parquet"


def main(k: int, ranks: int) -> None:
    df = pd.read_parquet(COARSE)
    df = df[(df.ok == True) & (df.f0 > 3.2) & (df.f0 < 5.0)  # noqa: E712
            & df.tag.str.startswith("s5_")].reset_index(drop=True)
    # spread the K picks across the design space (every len/k-th by cap_length)
    df = df.sort_values("cap_length").reset_index(drop=True)
    picks = df.iloc[:: max(len(df) // k, 1)].head(k)

    done = set()
    if OUT.exists():
        done = set(pd.read_parquet(OUT).tag)

    t0 = time.time()
    for _, r in picks.iterrows():
        tag = f"fine_{r.tag}"
        if tag in done:
            continue
        p_um = {key: float(r[key]) for key in RANGES_UM}
        p_int = {key: int(r[key]) for key in RANGES_INT}
        row = run_variant(tag, p_um, p_int, ranks,
                          n_modes=6, target_ghz=float(r.f0) - 0.3, max_size=60,
                          solver_order=3, amr_iterations=2, timeout_s=14400)
        row["coarse_tag"] = r.tag
        row["coarse_f0"] = float(r.f0)
        row["coarse_f1"] = float(r.f1) if pd.notna(r.f1) else None
        out = pd.DataFrame([row])
        if OUT.exists():
            out = pd.concat([pd.read_parquet(OUT), out], ignore_index=True)
        out.to_parquet(OUT)
        print(f"{tag}: ok={row['ok']} f0={row.get('f0')} "
              f"(coarse {r.f0:.4f}) t={row.get('t_solve', 0):.0f}s "
              f"[{(time.time() - t0) / 60:.0f} min]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--ranks", type=int, default=6)
    a = ap.parse_args()
    main(a.k, a.ranks)
