"""Design-space sampling anchored to validated design ranges.

Default ranges follow the SQuADDS qubit-claw design family (see docs/DATA.md
for provenance). `sample_designs` draws i.i.d. designs for dataset generation;
`ood_ranges` widens every interval for out-of-distribution stress tests.
"""

from __future__ import annotations

import numpy as np

from .transmon import XmonDesign

# (low, high) in um — anchored to the SQuADDS qubit+claw family (observed DB
# ranges for the varied params; DB-fixed params get symmetric spreads around
# their standard values). See docs/DATA.md for provenance.
DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "cross_length": (90.0, 420.0),   # DB range
    "cross_width": (20.0, 40.0),     # DB fixed at 30
    "cross_gap": (20.0, 40.0),       # DB fixed at 30
    "claw_length": (70.0, 400.0),    # DB range
    "claw_width": (10.0, 25.0),      # DB fixed at 15
    "claw_gap": (4.0, 12.0),         # DB fixed at 5.1
    "ground_spacing": (4.0, 12.0),   # DB range 4.1-10
}


def sample_designs(
    m: int,
    rng: np.random.Generator | None = None,
    ranges: dict[str, tuple[float, float]] | None = None,
    domain: float = 1100.0,
) -> list[XmonDesign]:
    rng = rng or np.random.default_rng()
    ranges = ranges or DEFAULT_RANGES
    out = []
    for _ in range(m):
        params = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in ranges.items()}
        out.append(XmonDesign(domain=domain, **params))
    return out


def ood_ranges(scale: float = 1.25) -> dict[str, tuple[float, float]]:
    """Widen each range about its midpoint by `scale` (for OOD evaluation)."""
    out = {}
    for k, (lo, hi) in DEFAULT_RANGES.items():
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * scale
        out[k] = (max(mid - half, 2.0), mid + half)
    return out
