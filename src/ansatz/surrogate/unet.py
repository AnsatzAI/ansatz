"""Field-prediction U-Net.

Input channels (n x n):
  0: Dirichlet drive  — fixed potential where fixed, 0 at free nodes
  1: fixed-node mask  — 1 where Dirichlet, 0 where free
Output: predicted potential field (clamped to drive at fixed nodes by caller).

The same network serves two roles:
  * initializer  — predict u from the drive pattern (residual-verified)
  * corrector    — predict the error field from a residual pattern mid-solve
                   (HINTS-style low-mode damping); the residual is fed through
                   channel 0 with an amplitude-normalized scale.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.GELU(),
    )


class FieldUNet(nn.Module):
    def __init__(self, width: int = 32, depth: int = 4, in_ch: int = 2):
        super().__init__()
        self.depth = depth
        widths = [width * (2**i) for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        cin = in_ch
        for w in widths[:-1]:
            self.enc.append(_block(cin, w))
            cin = w
        self.pool = nn.MaxPool2d(2)
        self.mid = _block(widths[-2], widths[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.up.append(nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2))
            self.dec.append(_block(2 * widths[i], widths[i]))
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        pad = (-n) % (2**self.depth)
        if pad:
            x = nn.functional.pad(x, (0, pad, 0, pad))
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.mid(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        out = self.head(x)[..., :n, :n]
        return out.squeeze(1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
