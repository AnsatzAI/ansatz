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
    def __init__(self, root: str | Path, split: str, n: int):
        self.files = sorted(Path(root).glob(f"{split}_{n}_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no shards for {split}_{n} under {root}")
        self.n = n
        self.index: list[tuple[int, int, int]] = []  # (file_idx, design_idx, exc)
        self._counts = []
        for fi, f in enumerate(self.files):
            with np.load(f) as z:
                m = z["params"].shape[0]
            self._counts.append(m)
            for d in range(m):
                self.index.append((fi, d, 0))
                self.index.append((fi, d, 1))
        self._cache: tuple[int, dict] | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _shard(self, fi: int) -> dict:
        if self._cache is None or self._cache[0] != fi:
            with np.load(self.files[fi]) as z:
                self._cache = (fi, {k: z[k] for k in z.files})
        return self._cache[1]

    def __getitem__(self, i: int):
        fi, d, exc = self.index[i]
        z = self._shard(fi)
        n = self.n
        cross = _unpack(z[f"cross_{d}"], n)
        claw = _unpack(z[f"claw_{d}"], n)
        ground = _unpack(z[f"ground_{d}"], n)
        fixed = cross | claw | ground
        driven = cross if exc == 0 else claw
        drive = np.where(driven, 1.0, 0.0).astype(np.float32)
        target = z[f"u_cross_{d}"] if exc == 0 else z[f"u_claw_{d}"]
        x = np.stack([drive, fixed.astype(np.float32)])
        free = (~fixed).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(np.ascontiguousarray(target)),
            torch.from_numpy(free),
        )
