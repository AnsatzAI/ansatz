"""Lumped-oscillator model: capacitance matrix -> transmon Hamiltonian parameters.

Follows the SQuADDS/qiskit-metal quasi-lumped pipeline (Shanto et al., Quantum
8, 1465 (2024), Sec. 2 and App. A/B; Koch et al., PRA 76, 042319 (2007)):

  C_q = |cross_to_ground| + |cross_to_claw|      (qubit shunt capacitance)
  C_c = |cross_to_claw|                          (qubit-readout coupling cap)
  E_C = e^2 / (2 C) / h                          (charging energy, Hz)
  qubit spectrum from exact Cooper-pair-box diagonalization in charge basis
  g   from the exact lumped formula with det C = (C_q+C_c)(C_r+C_c) - C_c^2
  chi from 2nd-order dispersive theory beyond RWA (paper Eq. 9; we use the
      paper's sign convention for the counter-rotating term)

No qiskit-metal / scqubits dependency: the CPB spectrum is a 61x61 symmetric
tridiagonal eigenproblem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

E_CHARGE = 1.602176634e-19  # C
H_PLANCK = 6.62607015e-34   # J s
PHI0 = 2.067833848e-15      # Wb, flux quantum


def transmon_spectrum(ej_hz: float, ec_hz: float, ncut: int = 30) -> np.ndarray:
    """Lowest eigenfrequencies (Hz) of H = 4 E_C n^2 - E_J cos(phi), ng = 0."""
    n = np.arange(-ncut, ncut + 1)
    diag = 4.0 * ec_hz * n**2
    off = -0.5 * ej_hz * np.ones(2 * ncut)
    from scipy.linalg import eigh_tridiagonal

    vals = eigh_tridiagonal(diag, off, select="i", select_range=(0, 3))[0]
    return vals


def ej_from_lj(lj_h: float) -> float:
    """Josephson energy (Hz) from junction inductance (H)."""
    return PHI0**2 / (4 * np.pi**2 * lj_h) / H_PLANCK


def charging_energy(c_farad: float) -> float:
    """E_C (Hz) from capacitance (F)."""
    return E_CHARGE**2 / (2.0 * c_farad) / H_PLANCK


@dataclass
class HamiltonianParams:
    f_q: float      # qubit 0-1 frequency, Hz
    alpha: float    # anharmonicity, Hz (negative)
    e_c: float      # charging energy, Hz
    g: float        # qubit-readout coupling, Hz (linear frequency)
    chi: float      # dispersive shift, Hz


def hamiltonian_from_cap(
    cross_to_ground_ff: float,
    cross_to_claw_ff: float,
    lj_nh: float = 10.0,
    f_r_hz: float = 6.116e9,
    z_c: float = 50.0,
    resonator: str = "quarter",
) -> HamiltonianParams:
    """Full LOM evaluation from the two relevant cap-matrix entries (fF)."""
    c_q = (abs(cross_to_ground_ff) + abs(cross_to_claw_ff)) * 1e-15
    c_c = abs(cross_to_claw_ff) * 1e-15

    ej = ej_from_lj(lj_nh * 1e-9)

    m = 4.0 if resonator == "quarter" else 2.0
    w_r = 2 * np.pi * f_r_hz
    c_r = np.pi / (m * w_r * z_c)

    det_c = (c_q + c_c) * (c_r + c_c) - c_c**2
    c_q_eff = det_c / (c_r + c_c)
    ec = charging_energy(c_q_eff)

    levels = transmon_spectrum(ej, ec)
    f_q = levels[1] - levels[0]
    alpha = (levels[2] - levels[1]) - (levels[1] - levels[0])

    # exact lumped coupling (SQuADDS transmon_cross implementation)
    hbar = H_PLANCK / (2 * np.pi)
    g_j = (
        (c_c / np.sqrt(c_q + c_c))
        * np.sqrt(hbar * w_r * E_CHARGE**2 / det_c)
        * (ej / (8 * ec)) ** 0.25
    )
    g = g_j / H_PLANCK  # Hz (linear frequency)

    delta = f_r_hz - f_q
    sigma = f_r_hz + f_q
    chi = 2 * g**2 * (alpha / (delta * (delta - alpha)) + alpha / (sigma * (sigma + alpha)))

    return HamiltonianParams(f_q=f_q, alpha=alpha, e_c=ec, g=g, chi=chi)
