"""Bespoke 3D data campaign: parameterized DeviceLayout meshes -> Palace solves.

Public data does not cover parameterized 3D full-wave eigenmodes, so this is
our own generator. Each variant: sample geometry params -> Julia/DeviceLayout
mesh + config -> Palace eigenmode solve -> record (params, f0, f1, Q, EPR,
wall-clock). Resumable: completed tags are skipped.

Dev-tier ranges are centered on the shipped example's defaults; the campaign
tier (wider ranges, finer meshes, more ranks) runs the same script on rented
compute. Run AFTER any timing-sensitive experiments (contention pollutes them).

Usage:
  python scripts/em3d_campaign.py --n 100 --ranks 6 --seed 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ansatz.em3d.config import load_config, make_variant, write_config
from ansatz.em3d.runner import run_palace

ROOT = Path(__file__).resolve().parents[1]
EX = Path.home() / "Documents/Personal/ansatz/palace/examples/transmon"
OUT = ROOT / "runs" / "em3d_dataset.parquet"

# geometry ranges (um unless noted) centered on the example defaults;
# names must match SingleTransmon.generate_transmon kwargs.
RANGES_UM = {
    "cap_length": (450.0, 800.0),      # transmon pad length (default 620)
    "cap_gap": (20.0, 40.0),           # pad-to-pad junction gap (default 30)
    "total_length": (4000.0, 5800.0),  # readout resonator length (default 5000)
    "l_claw": (90.0, 160.0),           # readout claw length (default 121)
}
RANGES_INT = {
    "n_meander_turns": (4, 6),
}


def sample_params(rng: np.random.Generator) -> tuple[dict, dict]:
    p_um = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in RANGES_UM.items()}
    p_int = {k: int(rng.integers(lo, hi + 1)) for k, (lo, hi) in RANGES_INT.items()}
    return p_um, p_int


def done_tags() -> set[str]:
    if OUT.exists():
        return set(pd.read_parquet(OUT).tag)
    return set()


def append(row: dict) -> None:
    df = pd.DataFrame([row])
    if OUT.exists():
        df = pd.concat([pd.read_parquet(OUT), df], ignore_index=True)
    df.to_parquet(OUT)


def run_variant(tag: str, p_um: dict, p_int: dict, ranks: int,
                n_modes: int | None = None,
                target_ghz: float = 3.0) -> dict:
    spec = {"tag": tag, "solver_order": 2, "params_um": p_um, "params_int": p_int}
    spec_path = EX / f"spec_{tag}.json"
    spec_path.write_text(json.dumps(spec))

    t0 = time.perf_counter()
    gen = subprocess.run(
        ["julia", "--project", str(ROOT / "scripts" / "gen_variant.jl"),
         str(spec_path)],
        cwd=EX, capture_output=True, text=True, timeout=1800,
    )
    t_mesh = time.perf_counter() - t0
    if gen.returncode != 0 or "GENERATED" not in gen.stdout:
        return dict(tag=tag, ok=False, stage="mesh", t_mesh=t_mesh,
                    err=(gen.stderr or gen.stdout)[-500:], **p_um, **p_int)

    # normalize output dir + solver knobs through our variant machinery;
    # target 3.0 GHz sits safely below the family's lowest qubit mode
    # (shift-cliff finding: targets above a mode skip it and run 2.5-3x slower)
    cfg = load_config(EX / f"ansatz_{tag}.json")
    cfg = make_variant(cfg, f"postpro/ansatz/{tag}", target_ghz=target_ghz,
                       n_modes=n_modes)
    cfg_path = write_config(cfg, EX / f"ansatz_{tag}.json")
    res = run_palace(cfg_path, ranks=ranks)
    row = dict(
        tag=tag, ok=(res.returncode == 0 and len(res.freqs_ghz) > 0),
        stage="solve", t_mesh=t_mesh, t_solve=res.t_wall,
        f0=res.freqs_ghz[0] if res.freqs_ghz else None,
        f1=res.freqs_ghz[1] if len(res.freqs_ghz) > 1 else None,
        q0=res.q_factors[0] if res.q_factors else None,
        q1=res.q_factors[1] if len(res.q_factors) > 1 else None,
        f_all=json.dumps(res.freqs_ghz),
        **p_um, **p_int,
    )
    if not row["ok"]:
        row["err"] = res.log_tail[-500:]
    # clean up large artifacts, keep the record small
    (EX / f"spec_{tag}.json").unlink(missing_ok=True)
    mesh = EX / "mesh" / f"{tag}.msh2"
    mesh.unlink(missing_ok=True)
    return row


def main(n: int, ranks: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    skip = done_tags()
    t_start = time.time()
    for i in range(n):
        tag = f"s{seed}_{i:04d}"
        p_um, p_int = sample_params(rng)  # advance rng even when skipping
        if tag in skip:
            continue
        row = run_variant(tag, p_um, p_int, ranks)
        append(row)
        el = time.time() - t_start
        print(f"[{i + 1}/{n}] {tag} ok={row['ok']} "
              f"f0={row.get('f0')} t={row.get('t_solve', 0):.0f}s "
              f"({el / 60:.0f} min elapsed)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ranks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    main(a.n, a.ranks, a.seed)
