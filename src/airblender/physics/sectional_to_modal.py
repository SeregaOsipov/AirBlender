"""Analytical Ackermann (1998) lognormal moment helpers for MADE/SORGAM modal aerosols."""
from __future__ import annotations

import math
import numpy as np

DRY_AIR_MOLAR_MASS_G_MOL = 28.97

# Initial MADE geometric mean diameters [m] and geometric standard deviations for i, j, c modes.
MADE_MODES_DGS_M = (0.01e-6, 0.07e-6, 1.0e-6)
MADE_MODES_SIGMAS = (1.7, 2.0, 2.5)


def compute_made_number_factor(aer_rho_kg_m3: float, dg_m: float, sigma_g: float) -> float:
    """
    Compute M0 / mass factor [# kg^-1 per ug kg^-1] (or [# m^-3 per ug m^-3]) following
    Ackermann et al. (1998) and WRF-Chem `module_aerosols_sorgam.F`.
    """
    aer_fac = 1.0e-6 * 6.0 / (math.pi * aer_rho_kg_m3)
    return aer_fac / ((dg_m ** 3) * math.exp(4.5 * math.log(sigma_g) ** 2))


def get_wrf_made_modpar(sigmas, m0s, m3s):
    """Compute modal median diameters capped at `dgmin = 1e-9 m` per MADE formulation."""
    dgs = ()
    dgmin = 1.0e-9
    for sg, m3, m0 in zip(sigmas, m3s, m0s):
        es36 = np.exp(0.125 * np.log(sg) ** 2) ** 36
        dg = (m3 / (m0 * es36)) ** (1.0 / 3.0)
        if isinstance(dg, np.ndarray):
            dg = np.where(dg < dgmin, dgmin, dg)
        else:
            dg = dg.where(dg >= dgmin, dgmin)
        dgs += (dg,)
    return dgs
