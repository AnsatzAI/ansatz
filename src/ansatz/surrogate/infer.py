"""Surrogate inference with cross-resolution transfer.

The U-Net predicts the continuum potential at its native training resolution;
for other grids we downsample the geometry channels, predict, and bilinearly
upsample — the routed correction loop then removes the resample error (and
verifies the result, so transfer costs iterations, never correctness).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..pde.laplace import LaplaceProblem
from .unet import FieldUNet


class FieldSurrogate:
    def __init__(self, weights: str, width: int = 32, depth: int = 4,
                 native_n: int = 255, device: str | None = None):
        self.device = torch.device(
            device or ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.model = FieldUNet(width=width, depth=depth).to(self.device).eval()
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.native_n = native_n

    @torch.no_grad()
    def predict(self, problems: list[LaplaceProblem]) -> list[np.ndarray]:
        """Batched prediction; returns clamped fields at each problem's grid."""
        n0 = self.native_n
        xs = []
        for p in problems:
            drive = torch.from_numpy(
                np.where(p.fixed_mask, p.fixed_values, 0.0).astype(np.float32)
            )
            fixed = torch.from_numpy(p.fixed_mask.astype(np.float32))
            x = torch.stack([drive, fixed])[None]
            if p.n != n0:
                x = F.interpolate(x, size=(n0, n0), mode="area")
                x[:, 1] = (x[:, 1] > 0.35).float()
                x[:, 0] = x[:, 0] * x[:, 1]
            xs.append(x)
        batch = torch.cat(xs).to(self.device)
        pred = self.model(batch).cpu()[:, None]
        out = []
        for p, u0 in zip(problems, pred):
            u = u0
            if p.n != n0:
                u = F.interpolate(u[None], size=(p.n, p.n), mode="bilinear",
                                  align_corners=False)[0]
            field = u[0].numpy().astype(np.float64)
            out.append(p.clamp(field))
        return out

    def warmup(self, n: int = 255) -> None:
        p = LaplaceProblem(n=n, fixed_mask=np.zeros((n, n), bool),
                           fixed_values=np.zeros((n, n)))
        self.predict([p])
