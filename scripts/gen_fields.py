"""Generate the field-solve dataset for surrogate training and solver benchmarks.

For each sampled design: rasterize, solve the two conductor excitations to
machine precision with an amortized sparse direct factorization, and shard the
results. Splits: train/val/test from DEFAULT_RANGES, plus an OOD split from
widened ranges. Reproducible via fixed seeds.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from ansatz.geometry.sampler import sample_designs, ood_ranges
from ansatz.geometry.transmon import build_problem_masks
from ansatz.pde.laplace import LaplaceProblem

OUT = Path(__file__).resolve().parents[1] / "data" / "fields"


def solve_design(design, n: int):
    conductors, ground = build_problem_masks(design, n)
    fixed = ground.copy()
    for m in conductors:
        fixed |= m
    fields = []
    # assemble once (structure shared across excitations), factor once
    base = LaplaceProblem(n=n, fixed_mask=fixed, fixed_values=np.zeros((n, n)))
    a, _, idx = base.assemble()
    lu = spla.splu(a.tocsc())
    for k in range(len(conductors)):
        vals = np.where(conductors[k], 1.0, 0.0)
        p = LaplaceProblem(n=n, fixed_mask=fixed, fixed_values=vals)
        _, b, _ = p.assemble()  # RHS depends on excitation
        u = p.to_grid(lu.solve(b), idx)
        fields.append(u.astype(np.float32))
    return conductors, ground, fields


def pack(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.astype(np.uint8))


def generate(split: str, m: int, seed: int, n: int, ranges=None, shard_size: int = 250):
    rng = np.random.default_rng(seed)
    designs = sample_designs(m, rng=rng, ranges=ranges)
    OUT.mkdir(parents=True, exist_ok=True)
    shard, meta = [], []
    si = 0
    t0 = time.time()
    for i, d in enumerate(designs):
        conductors, ground, fields = solve_design(d, n)
        shard.append(
            dict(
                cross=pack(conductors[0]),
                claw=pack(conductors[1]),
                ground=pack(ground),
                u_cross=fields[0],
                u_claw=fields[1],
            )
        )
        meta.append([getattr(d, k) for k in PARAM_KEYS])
        if len(shard) == shard_size or i == m - 1:
            path = OUT / f"{split}_{n}_{si:03d}.npz"
            np.savez_compressed(
                path,
                params=np.array(meta[-len(shard):], dtype=np.float32),
                **{f"{key}_{j}": s[key] for j, s in enumerate(shard) for key in s},
                n=n,
            )
            print(f"[{split}] shard {si} ({i + 1}/{m}) "
                  f"{(time.time() - t0) / (i + 1):.2f}s/design", flush=True)
            shard = []
            si += 1


PARAM_KEYS = [
    "cross_length", "cross_width", "cross_gap",
    "claw_length", "claw_width", "claw_gap", "ground_spacing",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=255)
    ap.add_argument("--train", type=int, default=2200)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--ood", type=int, default=150)
    args = ap.parse_args()

    generate("train", args.train, seed=1, n=args.n)
    generate("val", args.val, seed=2, n=args.n)
    generate("test", args.test, seed=3, n=args.n)
    generate("ood", args.ood, seed=4, n=args.n, ranges=ood_ranges(1.3))
    print("done")
