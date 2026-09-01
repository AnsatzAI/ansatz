"""Timed Palace execution + output parsing.

Palace is invoked as `palace -np <ranks> <config.json>` with the working
directory set next to the config (mesh paths in shipped examples are
relative). Outputs parsed from the postpro directory:

  eig.csv               — mode frequencies (Re/Im, GHz) and Q factors
  domain-E.csv          — energy-participation ratios (when present)

Wall-clock is measured end-to-end (the honest number a design loop pays);
the Palace log's own timing table and DOF counts are captured when present.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PALACE_BIN = os.environ.get(
    "PALACE_BIN",
    str(Path.home() / "Documents/Personal/ansatz/palace-install/bin/palace"),
)


@dataclass
class PalaceResult:
    t_wall: float
    returncode: int
    freqs_ghz: list[float] = field(default_factory=list)
    q_factors: list[float] = field(default_factory=list)
    n_dof: int | None = None
    log_tail: str = ""
    out_dir: str = ""


def run_palace(
    config_path: str | Path,
    ranks: int = 6,
    workdir: str | Path | None = None,
    timeout_s: float = 3600.0,
) -> PalaceResult:
    config_path = Path(config_path)
    workdir = Path(workdir) if workdir else config_path.parent
    t0 = time.perf_counter()
    proc = subprocess.run(
        [PALACE_BIN, "-np", str(ranks), str(config_path)],
        cwd=workdir, capture_output=True, text=True, timeout=timeout_s,
    )
    t_wall = time.perf_counter() - t0

    log = proc.stdout + "\n" + proc.stderr
    n_dof = None
    m = re.search(r"unknowns[^\d]*([\d,]+)", log, re.IGNORECASE)
    if m:
        n_dof = int(m.group(1).replace(",", ""))

    with open(config_path) as f:
        import json

        out_rel = json.load(f)["Problem"]["Output"]
    out_dir = (workdir / out_rel).resolve()

    freqs, qs = [], []
    eig_csv = out_dir / "eig.csv"
    if eig_csv.exists():
        df = pd.read_csv(eig_csv, skipinitialspace=True)
        fcol = next((c for c in df.columns if "Re" in c and "f" in c), None)
        qcol = next((c for c in df.columns if "Q" in c), None)
        if fcol is not None:
            freqs = [float(v) for v in df[fcol]]
        if qcol is not None:
            qs = [float(v) for v in df[qcol]]

    return PalaceResult(
        t_wall=t_wall, returncode=proc.returncode, freqs_ghz=freqs,
        q_factors=qs, n_dof=n_dof, log_tail=log[-2000:], out_dir=str(out_dir),
    )
