"""Parameterized planar transmon layouts (plan view).

The v0 device family mirrors the xmon-cross + claw-coupler geometry used by
experimentally validated design databases (SQuADDS / qiskit-metal
TransmonCross): a cross-shaped island in an etched pocket of the ground plane,
with a C-shaped readout claw wrapping one arm. All lengths in micrometers.

Rasterization emits boolean masks on the interior grid:
  cross_mask  — qubit island (driven conductor)
  claw_mask   — readout claw (sense conductor)
  ground_mask — ground plane (everything outside the etch pocket)
Free (unknown) nodes are the etched vacuum gaps between conductors.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class XmonDesign:
    cross_length: float = 220.0   # arm length from center, um
    cross_width: float = 30.0     # arm width, um
    cross_gap: float = 30.0       # etch gap around cross, um
    claw_length: float = 200.0    # length of claw fingers, um
    claw_width: float = 15.0      # claw metal width, um
    claw_gap: float = 10.0        # etch gap around claw metal, um
    ground_spacing: float = 10.0  # cross-etch to claw-etch separation, um
    domain: float = 1100.0        # simulated window, um

    def to_dict(self) -> dict:
        return asdict(self)

    def rasterize(self, n: int) -> dict[str, np.ndarray]:
        """Rasterize to (n, n) interior-node masks."""
        px = self.domain / (n + 1)

        def um2px(x: float) -> int:
            return max(int(round(x / px)), 1)

        c = n // 2
        arm = um2px(self.cross_length)
        w = max(um2px(self.cross_width), 2)
        gap = max(um2px(self.cross_gap), 2)
        hw = w // 2

        cross = np.zeros((n, n), bool)
        cross[c - arm : c + arm + 1, c - hw : c + hw + 1] = True
        cross[c - hw : c + hw + 1, c - arm : c + arm + 1] = True

        # etch pocket: cross dilated by its gap (chebyshev distance, rectangular)
        pocket = _dilate_rect(cross, gap)

        # claw wraps the top arm tip: base bar + two fingers descending
        clw = max(um2px(self.claw_width), 2)
        clg = max(um2px(self.claw_gap), 2)
        gsp = max(um2px(self.ground_spacing), 2)
        cl_len = um2px(self.claw_length)

        tip_row = c - arm  # top arm tip row index
        base_top = tip_row - gap - gsp - clg - clw
        base_bot = base_top + clw
        half_span = hw + gap + gsp + clg + clw

        claw = np.zeros((n, n), bool)
        if base_top > 1:
            claw[base_top:base_bot, c - half_span : c + half_span + 1] = True
            fin_end = min(base_bot + cl_len, n - 2)
            claw[base_bot:fin_end, c - half_span : c - half_span + clw] = True
            claw[base_bot:fin_end, c + half_span - clw + 1 : c + half_span + 1] = True

        claw_pocket = _dilate_rect(claw, clg)
        etch = pocket | claw_pocket
        ground = ~etch
        # conductors carve out of the etch region
        ground &= ~(cross | claw)

        return {"cross": cross, "claw": claw & ~cross, "ground": ground}


def _dilate_rect(mask: np.ndarray, r: int) -> np.ndarray:
    """Dilation by a (2r+1) square structuring element via cumulative shifts."""
    out = mask.copy()
    cur = mask
    for _ in range(r):
        nxt = cur.copy()
        nxt[1:, :] |= cur[:-1, :]
        nxt[:-1, :] |= cur[1:, :]
        nxt[:, 1:] |= cur[:, :-1]
        nxt[:, :-1] |= cur[:, 1:]
        cur = nxt
    out |= cur
    return out


def build_problem_masks(design: XmonDesign, n: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Return ([cross, claw] conductor masks, ground mask) for capacitance runs."""
    m = design.rasterize(n)
    return [m["cross"], m["claw"]], m["ground"]
