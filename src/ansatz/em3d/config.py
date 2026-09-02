"""Palace config templating for parameterized eigenmode studies.

Wraps a base Palace JSON config (e.g. the shipped transmon example) and
exposes the solver knobs that matter for routing studies:

  order       — FEM polynomial order (dominant DOF/cost knob)
  target_ghz  — eigensolver shift/target frequency (shift-invert quality knob)
  n_modes     — number of requested modes
  tol         — eigensolver tolerance
  refinement  — uniform serial refinement levels (mesh density knob)

Each variant gets its own output directory so runs never collide.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def make_variant(
    base: dict,
    out_dir: str,
    order: int | None = None,
    target_ghz: float | None = None,
    n_modes: int | None = None,
    tol: float | None = None,
    refinement: int | None = None,
    save_fields: bool = False,
    max_size: int | None = None,
) -> dict:
    cfg = copy.deepcopy(base)
    cfg["Problem"]["Output"] = out_dir
    # field export is large and slow; keep off for sweeps
    cfg["Problem"].setdefault("OutputFormats", {})["GridFunction"] = bool(save_fields)
    eig = cfg["Solver"].setdefault("Eigenmode", {})
    if save_fields is False:
        eig["Save"] = 0
    if order is not None:
        cfg["Solver"]["Order"] = int(order)
    if target_ghz is not None:
        eig["Target"] = float(target_ghz)
    if n_modes is not None:
        eig["N"] = int(n_modes)
    if tol is not None:
        eig["Tol"] = float(tol)
    if max_size is not None:
        eig["MaxSize"] = int(max_size)
    if refinement is not None:
        model = cfg.setdefault("Model", {})
        ref = model.setdefault("Refinement", {})
        ref["UniformLevels"] = int(refinement)
    return cfg


def write_config(cfg: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1)
    return path
