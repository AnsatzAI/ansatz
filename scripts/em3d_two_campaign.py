"""Rung 2 bespoke data campaign: two-transmon unit cell (shared feedline).

The industrially-shaped object: two qubit+readout pairs on one feedline —
the minimal frequency-planning/crowding problem. Parameter ordering makes
mode attribution unambiguous by construction:
  cap_length_1 > cap_length_2  =>  f_q1 < f_q2
  total_length_1 > total_length_2  =>  f_r1 < f_r2
with disjoint readout bands. Records (params, f_q1, f_q2, f_r1, f_r2, Qs).

Usage: python scripts/em3d_two_campaign.py --n 60 --ranks 6
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
OUT = ROOT / "runs" / "em3d_two_dataset.parquet"

RANGES2 = {
    "cap_length_1": (560.0, 800.0),
    "cap_length_2": (460.0, 620.0),   # resampled to keep cl1 >= cl2 + 40
    "cap_gap_1": (22.0, 38.0),
    "cap_gap_2": (22.0, 38.0),
    "total_length_1": (4800.0, 5400.0),  # f_r1 ~ 5.2-5.8 GHz
    "total_length_2": (4000.0, 4600.0),  # f_r2 ~ 6.1-7.0 GHz
    "l_claw_1": (95.0, 150.0),
    "l_claw_2": (95.0, 150.0),
}
QUBIT_BAND = (3.2, 5.05)
READOUT_BAND = (5.05, 7.4)


def sample_params2(rng: np.random.Generator) -> dict:
    while True:
        p = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in RANGES2.items()}
        if p["cap_length_1"] >= p["cap_length_2"] + 40.0:
            return p


def identify_modes(freqs: list[float]) -> dict | None:
    """Sorted freqs -> {f_q1, f_q2, f_r1, f_r2} or None if incomplete."""
    fs = sorted(freqs)
    qubits = [f for f in fs if QUBIT_BAND[0] < f < QUBIT_BAND[1]]
    reads = [f for f in fs if READOUT_BAND[0] < f < READOUT_BAND[1]]
    if len(qubits) < 2 or len(reads) < 2:
        return None
    return {"f_q1": qubits[0], "f_q2": qubits[1],
            "f_r1": reads[0], "f_r2": reads[1]}


def run_variant_two(tag: str, p: dict, ranks: int,
                    solver_order: int = 2, amr_iterations: int = 0,
                    target_ghz: float = 3.0, timeout_s: float = 7200.0,
                    n_modes: int | None = None, max_size: int | None = None,
                    tol: float | None = None) -> dict:
    spec = {"tag": tag, "solver_order": solver_order,
            "amr_iterations": amr_iterations,
            "params_um": p, "params_int": {"n_meander_turns": 5}}
    spec_path = EX / f"spec_{tag}.json"
    spec_path.write_text(json.dumps(spec))
    t0 = time.perf_counter()
    gen = subprocess.run(
        ["julia", "--project",
         str(ROOT / "scripts" / "gen_two_transmon.jl"), str(spec_path)],
        cwd=EX, capture_output=True, text=True, timeout=3600, check=False,
    )
    t_mesh = time.perf_counter() - t0
    if gen.returncode != 0 or "GENERATED" not in gen.stdout:
        return dict(tag=tag, ok=False, stage="mesh", t_mesh=t_mesh,
                    err=(gen.stderr or gen.stdout)[-500:], **p)

    cfg = load_config(EX / f"ansatz_{tag}.json")
    cfg = make_variant(cfg, f"postpro/ansatz/{tag}", target_ghz=target_ghz,
                       n_modes=n_modes, max_size=max_size, tol=tol)
    cfg_path = write_config(cfg, EX / f"ansatz_{tag}.json")
    res = run_palace(cfg_path, ranks=ranks, timeout_s=timeout_s)
    modes = identify_modes(res.freqs_ghz) if res.returncode == 0 else None
    row = dict(
        tag=tag, ok=(modes is not None), stage="solve",
        t_mesh=t_mesh, t_solve=res.t_wall,
        solver_order=solver_order, amr_iterations=amr_iterations,
        f_all=json.dumps(res.freqs_ghz), **(modes or {}), **p,
    )
    if modes is None:
        row["err"] = res.log_tail[-400:]
    (EX / f"spec_{tag}.json").unlink(missing_ok=True)
    (EX / "mesh" / f"{tag}.msh2").unlink(missing_ok=True)
    return row


def main(n: int, ranks: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    done = set(pd.read_parquet(OUT).tag) if OUT.exists() else set()
    t0 = time.time()
    for i in range(n):
        tag = f"two_s{seed}_{i:04d}"
        p = sample_params2(rng)
        if tag in done:
            continue
        # campaign labels: cheaper eigensolver settings (frequencies are
        # converged well before 1e-8; demos verify at strict settings)
        row = run_variant_two(tag, p, ranks, target_ghz=3.4,
                              n_modes=6, max_size=60, tol=1e-6)
        df = pd.DataFrame([row])
        if OUT.exists():
            df = pd.concat([pd.read_parquet(OUT), df], ignore_index=True)
        df.to_parquet(OUT)
        print(f"[{i+1}/{n}] {tag} ok={row['ok']} "
              f"fq=({row.get('f_q1')},{row.get('f_q2')}) "
              f"fr=({row.get('f_r1')},{row.get('f_r2')}) "
              f"t={row.get('t_solve', 0):.0f}s "
              f"[{(time.time()-t0)/60:.0f}m]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--ranks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(a.n, a.ranks, a.seed)
