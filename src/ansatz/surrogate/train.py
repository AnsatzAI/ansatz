"""Train the field U-Net (masked MSE on free nodes, MPS-accelerated)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import FieldShards
from .unet import FieldUNet, count_params


def masked_mse(pred, target, free):
    diff = (pred - target) * free
    return (diff**2).sum() / free.sum().clamp(min=1.0)


def train(
    data_root: str,
    out: str,
    n: int = 255,
    epochs: int = 30,
    batch: int = 16,
    lr: float = 3e-4,
    width: int = 32,
    depth: int = 4,
    init: str | None = None,
):
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tr = FieldShards(data_root, "train", n)
    va = FieldShards(data_root, "val", n)
    dl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0)
    dv = DataLoader(va, batch_size=batch, num_workers=0)

    model = FieldUNet(width=width, depth=depth).to(dev)
    if init:
        model.load_state_dict(torch.load(init, map_location=dev))
    print(f"params: {count_params(model)/1e6:.2f}M device: {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = float("inf")
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot, steps = 0.0, 0
        for x, y, free in dl:
            x, y, free = x.to(dev), y.to(dev), free.to(dev)
            opt.zero_grad()
            loss = masked_mse(model(x), y, free)
            loss.backward()
            opt.step()
            tot += float(loss)
            steps += 1
        sched.step()
        model.eval()
        vtot, vsteps = 0.0, 0
        with torch.no_grad():
            for x, y, free in dv:
                x, y, free = x.to(dev), y.to(dev), free.to(dev)
                vtot += float(masked_mse(model(x), y, free))
                vsteps += 1
        vloss = vtot / max(vsteps, 1)
        print(
            f"ep {ep:03d} train {tot/max(steps,1):.3e} val {vloss:.3e} "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )
        if vloss < best:
            best = vloss
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out)
    print(f"best val {best:.3e} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fields")
    ap.add_argument("--out", default="runs/unet_255.pt")
    ap.add_argument("--n", type=int, default=255)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--init", default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    train(a.data, a.out, n=a.n, epochs=a.epochs, batch=a.batch,
          width=a.width, depth=a.depth, init=a.init, lr=a.lr)
