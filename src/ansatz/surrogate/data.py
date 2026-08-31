"""Shard-backed dataset for field-surrogate training.

Each sample: input channels [drive, fixed_mask] and target potential field for
one conductor excitation. The continuum solution is resolution-independent, so
the network trains at its native resolution and transfers across grids via
resample-and-correct (see surrogate.infer).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _unpack(bits: np.ndarray, n: int) -> np.ndarray:
    return np.unpackbits(bits, count=n * n).reshape(n, n).astype(bool)


class FieldShards(Dataset):
    """Eagerly materialized in RAM: masks stay packed as uint8 bits (compact),
    unpacked per item (fast, ~100us); targets stored float32."""

    def __init__(self, root: str | Path, split: str, n: int):
        self.files = sorted(Path(root).glob(f"{split}_{n}_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no shards for {split}_{n} under {root}")
        self.n = n
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for f in self.files:
            with np.load(f) as z:
                m = z["params"].shape[0]
                for d in range(m):
                    cross = z[f"cross_{d}"]
                    claw = z[f"claw_{d}"]
                    ground = z[f"ground_{d}"]
                    self.samples.append((cross, claw, ground, z[f"u_cross_{d}"], 0))
                    self.samples.append((cross, claw, ground, z[f"u_claw_{d}"], 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        cross_b, claw_b, ground_b, target, exc = self.samples[i]
        n = self.n
        cross = _unpack(cross_b, n)
        claw = _unpack(claw_b, n)
        ground = _unpack(ground_b, n)
        fixed = cross | claw | ground
        driven = cross if exc == 0 else claw
        drive = np.where(driven, 1.0, 0.0).astype(np.float32)
        x = np.stack([drive, fixed.astype(np.float32)])
        free = (~fixed).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(np.ascontiguousarray(target)),
            torch.from_numpy(free),
        )
