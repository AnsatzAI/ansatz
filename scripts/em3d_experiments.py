"""Option-A local experiments on the Palace transmon example (dev scale).

Experiments (all end-to-end wall-clock, honest):
  baseline  — shipped coarse config as-is
  shift     — eigensolver Target sweep around the true mode: quantifies what a
              Tier-1 frequency predictor buys via shift-invert quality
  frontier  — (order, refinement) cost/accuracy frontier vs the reference
              config: the routing table for cheapest-verified-config selection

Usage:
  python scripts/em3d_experiments.py --exp baseline
  python scripts/em3d_experiments.py --exp shift
  python scripts/em3d_experiments.py --exp frontier
Results append to runs/em3d_results.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ansatz.em3d.config import load_config, make_variant, write_config
from ansatz.em3d.runner import run_palace

ROOT = Path(__file__).resolve().parents[1]
EX = Path.home() / "Documents/Personal/ansatz/palace/examples/transmon"
BASE = EX / "transmon_coarse.json"
OUT = ROOT / "runs"


def record(rows: list[dict]) -> None:
    OUT.mkdir(exist_ok=True)
    path = OUT / "em3d_results.parquet"
    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path)
    print(df.tail(len(rows)).to_string())


def run_variant(tag: str, ranks: int, **kw) -> dict:
    base = load_config(BASE)
    out_dir = f"postpro/ansatz/{tag}"
    cfg = make_variant(base, out_dir, **kw)
    cfg_path = EX / f"ansatz_{tag}.json"
    write_config(cfg, cfg_path)
    res = run_palace(cfg_path, ranks=ranks)
    row = dict(
        tag=tag, ranks=ranks, t_wall=res.t_wall, returncode=res.returncode,
        n_dof=res.n_dof,
        f0=res.freqs_ghz[0] if res.freqs_ghz else None,
        f1=res.freqs_ghz[1] if len(res.freqs_ghz) > 1 else None,
        q0=res.q_factors[0] if res.q_factors else None,
        **{k: v for k, v in kw.items() if v is not None},
    )
    if res.returncode != 0:
        print(res.log_tail[-800:])
    return row


def main(exp: str, ranks: int) -> None:
    if exp == "baseline":
        record([run_variant("baseline", ranks)])
    elif exp == "shift":
        rows = []
        for tgt in (2.0, 3.0, 3.5, 3.9, 4.05, 4.5, 5.0):
            rows.append(run_variant(f"shift_{tgt}", ranks, target_ghz=tgt))
            record(rows[-1:])
    elif exp == "frontier":
        rows = []
        for order in (1, 2, 3):
            for ref in (0, 1):
                if order == 3 and ref == 1:
                    continue  # memory guard at dev scale
                rows.append(
                    run_variant(f"frontier_o{order}_r{ref}", ranks,
                                order=order, refinement=ref)
                )
                record(rows[-1:])
    else:
        raise SystemExit(f"unknown exp {exp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ranks", type=int, default=6)
    a = ap.parse_args()
    main(a.exp, a.ranks)
