"""Fetch the SQuADDS database (MIT-licensed) and extract the qubit cap-matrix table.

Writes data/squadds_qubit_cap.parquet with design parameters (um) and Q3D
capacitance-matrix entries (fF), which anchor our geometry sampler ranges and
calibrate the 2D model against 3D extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)


def main() -> None:
    ds = load_dataset("SQuADDS/SQuADDS_DB", "qubit-TransmonCross-cap_matrix")
    df = ds["train"].to_pandas()
    print("rows:", len(df))

    rows = []
    for _, r in df.iterrows():
        try:
            opts = r["design"]["design_options"]
            if isinstance(opts, str):
                opts = json.loads(opts)
            sim = r["sim_results"]
            if isinstance(sim, str):
                sim = json.loads(sim)
            rec = {}
            for k in ("cross_length", "cross_width", "cross_gap"):
                rec[k] = _um(opts.get(k))
            claw = opts.get("connection_pads", {}).get("readout", {})
            for k in ("claw_length", "claw_width", "claw_gap", "ground_spacing",
                      "claw_cpw_length", "claw_cpw_width"):
                rec[k] = _um(claw.get(k))
            for k, v in sim.items():
                if isinstance(v, (int, float)):
                    rec[f"sim_{k}"] = float(v)
            rows.append(rec)
        except Exception:  # noqa: BLE001, S112 — skip malformed rows
            continue
    flat = pd.DataFrame(rows).dropna(axis=1, how="all")
    print("flat columns:", list(flat.columns))
    print(flat.describe().T)
    flat.to_parquet(OUT / "squadds_qubit_cap.parquet")


def _um(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower().replace("um", "").replace("µm", "")
    try:
        return float(s)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
