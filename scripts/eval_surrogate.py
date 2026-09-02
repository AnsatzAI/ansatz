"""Evaluate surrogate field quality: relative L2 error and post-prediction
residual across resolutions (native 255 + transferred)."""

from __future__ import annotations

import argparse

import numpy as np

from ansatz.bench.harness import iter_designs
from ansatz.pde.laplace import LaplaceProblem
from ansatz.surrogate.infer import FieldSurrogate


def main(weights: str, data: str = "data/fields"):
    if "," in weights:
        from ansatz.surrogate.infer import MultiResSurrogate
        by_n = {int(p.split("=")[0]): p.split("=")[1] for p in weights.split(",")}
        s = MultiResSurrogate(by_n)
    else:
        s = FieldSurrogate(weights)
    s.warmup()
    for n in (255, 511, 1023):
        errs, res = [], []
        for i, (params, conductors, ground) in enumerate(iter_designs(data, "test", n)):
            if i >= 40:
                break
            fixed = ground | conductors[0] | conductors[1]
            for k, m in enumerate(conductors):
                vals = np.where(m, 1.0, 0.0)
                p = LaplaceProblem(n=n, fixed_mask=fixed, fixed_values=vals)
                pred = s.predict([p])[0]
                res.append(p.residual_norm(pred))
            # compare against stored exact fields
        # exact comparison via shards
        from ansatz.surrogate.data import FieldShards

        try:
            ds = FieldShards(data, "test", n)

            for i in range(min(80, len(ds))):
                x, y, _free = ds[i]
                fixedm = x[1].numpy().astype(bool)
                p = LaplaceProblem(n=n, fixed_mask=fixedm,
                                   fixed_values=x[0].numpy().astype(np.float64))
                pred = s.predict([p])[0]
                yv = y.numpy()
                num = np.linalg.norm((pred - yv)[~fixedm])
                den = max(np.linalg.norm(yv[~fixedm]), 1e-30)
                errs.append(num / den)
        except FileNotFoundError:
            pass
        print(f"n={n}: relL2 {np.mean(errs):.4f} (med {np.median(errs):.4f}) | "
              f"pred residual {np.mean(res):.3e} (med {np.median(res):.3e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/unet_255.pt")
    a = ap.parse_args()
    main(a.weights)
