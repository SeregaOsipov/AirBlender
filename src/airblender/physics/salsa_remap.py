"""Conservative dry-modal WRF-Chem MADE/SORGAM+GOCART -> PALM SALSA sectional remapper.

Maps 41 WRF-Chem modal aerosol mass fields (ug/kg dry air) and 3 modal number
concentrations (nu0, ac0, corn in #/kg dry air) onto 10 sectional SALSA bins
(3..50 nm, 50 nm..10 um) and 7 dry chemical components (SO4, OC, BC, DU, SS, NO, NH).
Stoichiometrically pairs fine Na/Cl into sea salt (SS), excludes coarse anthropogenic PM
(`antha`), and validates mass/number ledger conservation (`absolute_mass_number_v1`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

COMPONENTS = ("SO4", "OC", "BC", "DU", "SS", "NO", "NH")
TARGET_DENSITIES = np.array([1830.0, 2000.0, 2000.0, 2650.0, 2165.0, 1479.0, 1530.0], dtype=np.float64)

BASE_DENSITIES = {
    "so4a": 1800.0,
    "nh4a": 1800.0,
    "no3a": 1800.0,
    "naa": 2200.0,
    "cla": 2200.0,
    "caa": 2600.0,
    "ka": 2600.0,
    "mga": 2600.0,
    "p25": 2200.0,
    "ec": 2200.0,
    "orgpa": 1000.0,
}
BASE_DENSITIES.update({f"{o}{k}": 1000.0 for o in ("asoa", "bsoa") for k in range(1, 5)})
SOURCE_DENSITIES = {a + m: density for a, density in BASE_DENSITIES.items() for m in ("i", "j")}
SOURCE_DENSITIES.update(soila=2600.0, seas=2200.0, antha=2200.0)
SOURCE_COMPONENTS = tuple(SOURCE_DENSITIES)

DIRECT_MAP = {"so4a": "SO4", "nh4a": "NH", "no3a": "NO", "ec": "BC", "orgpa": "OC"}
DIRECT_MAP.update({f"{o}{k}": "OC" for o in ("asoa", "bsoa") for k in range(1, 5)})

SCHEMA = "absolute_mass_number_v1"


@dataclass(frozen=True)
class RemapPolicy:
    sigmas: tuple[float, float, float] = (1.7, 2.0, 2.5)
    material_mass_kg_m3: float = 1e-15
    unresolved_tolerance_kg_m3: float = 1e-23
    full_small_bin_species: bool = True


@dataclass
class RemapResult:
    number: np.ndarray
    mass: np.ndarray
    edges_m: np.ndarray
    mode_diameters_m: np.ndarray
    source_mass: np.ndarray
    below_mass: np.ndarray
    above_mass: np.ndarray
    imported_by_source: np.ndarray
    excluded_mass: np.ndarray
    unresolved_mass: np.ndarray
    below_number: np.ndarray
    above_number: np.ndarray
    source_number: np.ndarray
    issues: dict[str, int]

    def validate_budgets(self) -> None:
        accounted = (
            self.imported_by_source.sum(axis=-2)
            + self.excluded_mass.sum(axis=-2)
            + self.unresolved_mass.sum(axis=-2)
            + self.below_mass
            + self.above_mass
        )
        if not np.allclose(accounted, self.source_mass, rtol=2e-13, atol=1e-30):
            raise ValueError("SALSA source component mass ledger does not close")
        if not np.allclose(
            self.number.sum(axis=-1) + self.below_number.sum(axis=-1) + self.above_number.sum(axis=-1),
            self.source_number.sum(axis=-1),
            rtol=2e-13,
            atol=1e-25,
        ):
            raise ValueError("SALSA source number ledger does not close")
        if not np.allclose(
            self.mass.sum(axis=-1),
            self.imported_by_source.sum(axis=-1),
            rtol=2e-13,
            atol=1e-30,
        ):
            raise ValueError("SALSA chemical aggregation does not conserve imported mass")


def target_edges() -> np.ndarray:
    edges = np.concatenate((np.geomspace(3e-9, 50e-9, 4), np.geomspace(50e-9, 10e-6, 8)[1:]))
    edges[[0, 3, -1]] = [3e-9, 50e-9, 10e-6]
    return edges


def dry_air_density(p: np.ndarray, pb: np.ndarray, perturbation_theta: np.ndarray, qv: np.ndarray) -> np.ndarray:
    p, pb, theta, qv = np.broadcast_arrays(p, pb, perturbation_theta, qv)
    pressure = p + pb
    if (
        not all(np.all(np.isfinite(x)) for x in (p, pb, theta, qv))
        or np.any(pressure <= 0)
        or np.any(qv < 0)
        or np.any(theta + 300.0 <= 0)
    ):
        raise ValueError("Invalid pressure, dry potential temperature, or water mixing ratio")
    temperature = (theta + 300.0) * (pressure / 100000.0) ** (2.0 / 7.0)
    return pressure / (287.0 * temperature * (1.0 + qv * 461.6 / 287.0))


def _fractions(
    dg: np.ndarray, sigma: float, edges: np.ndarray, moment: int, active: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.log(sigma)
    z = (np.log(edges / dg[..., None]) - moment * s * s) / s
    lo, hi = z[..., :-1], z[..., 1:]
    bins = np.where(lo > 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
    bins = np.where(active[..., None], np.maximum(bins, 0.0), 0.0)
    return bins, np.where(active, ndtr(z[..., 0]), 0.0), np.where(active, ndtr(-z[..., -1]), 0.0)


def remap_modes(
    component_mixing_ratios: dict[str, np.ndarray],
    number_mixing_ratios: dict[str, np.ndarray],
    dry_air_density_arr: np.ndarray,
    *,
    edges_m: np.ndarray | None = None,
    policy: RemapPolicy | None = None,
) -> RemapResult:
    policy = policy or RemapPolicy()
    if len(policy.sigmas) != 3 or any(not np.isfinite(s) or s <= 1.0 for s in policy.sigmas):
        raise ValueError("Three geometric standard deviations greater than one are required")
    edges = target_edges() if edges_m is None else np.asarray(edges_m, dtype=float)
    if edges.shape != (11,) or not np.all(np.isfinite(edges)) or np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("Ten increasing positive target diameter intervals are required")

    missing = set(SOURCE_COMPONENTS) - set(component_mixing_ratios)
    if missing or set(number_mixing_ratios) != {"nu0", "ac0", "corn"}:
        raise ValueError(f"Missing WRF source aerosol fields: {sorted(missing)}")

    arrays = [np.asarray(component_mixing_ratios[n], dtype=float) for n in SOURCE_COMPONENTS]
    nums = [np.asarray(number_mixing_ratios[n], dtype=float) for n in ("nu0", "ac0", "corn")]
    arrays = np.broadcast_arrays(*arrays, *nums, np.asarray(dry_air_density_arr, dtype=float))
    shape = arrays[0].shape
    raw = {n: np.maximum(0.0, arrays[i]) for i, n in enumerate(SOURCE_COMPONENTS)}
    numbers = [np.maximum(0.0, arr) for arr in arrays[len(SOURCE_COMPONENTS) : -1]]
    rho = arrays[-1]
    if not np.all(np.isfinite(rho)) or np.any(rho <= 0):
        raise ValueError("Nonfinite or nonpositive dry-air density")

    clean_numbers = []
    for mode, (suffix, _) in enumerate(zip(("i", "j", "c"), policy.sigmas)):
        names = [n for n in SOURCE_COMPONENTS if n.endswith(suffix)] if suffix != "c" else ["soila", "seas", "antha"]
        qmass = sum(raw[n] for n in names)
        qn = numbers[mode]
        if np.any((qmass * rho * 1e-9 > policy.material_mass_kg_m3) & (qn <= 0.0)):
            raise ValueError(f"Material aerosol mass with zero mode number in WRF mode '{suffix}'")
        qn = np.where((qmass <= 0.0) & (qn > 0.0), 0.0, qn)
        for n in names:
            raw[n] = np.where((qmass > 0.0) & (qn <= 0.0), 0.0, raw[n])
        clean_numbers.append(qn)

    mass = np.zeros(shape + (10, 7), dtype=np.float64)
    number = np.zeros(shape + (10,), dtype=np.float64)
    source_mass = np.stack([raw[n] * rho * 1e-9 for n in SOURCE_COMPONENTS], axis=-1)
    below = np.zeros_like(source_mass)
    above = np.zeros_like(source_mass)
    imported = np.zeros(shape + (10, len(SOURCE_COMPONENTS)), dtype=np.float64)
    excluded = np.zeros_like(imported)
    unresolved = np.zeros_like(imported)
    below_n = np.zeros(shape + (3,), dtype=np.float64)
    above_n = np.zeros_like(below_n)
    source_n = np.stack([n * rho for n in clean_numbers], axis=-1)
    diameters = np.zeros(shape + (3,), dtype=np.float64)
    issues: dict[str, int] = {"source_number_floor": 0}

    for mode, (suffix, sigma) in enumerate(zip(("i", "j", "c"), policy.sigmas)):
        names = [n for n in SOURCE_COMPONENTS if n.endswith(suffix)] if suffix != "c" else ["soila", "seas", "antha"]
        qmass = sum(raw[n] for n in names)
        qnumber = clean_numbers[mode]
        active = (qmass > 0.0) & (qnumber > 0.0)
        volume = sum(raw[n] * 1e-9 / SOURCE_DENSITIES[n] for n in names)
        dg = np.where(
            active,
            (6.0 * volume / (np.pi * np.where(qnumber > 0.0, qnumber, 1.0) * np.exp(4.5 * np.log(sigma) ** 2)))
            ** (1.0 / 3.0),
            1e-9,
        )
        diameters[..., mode] = np.where(active, dg, 0.0)
        nf, nl, nh = _fractions(dg, sigma, edges, 0, active)
        mf, ml, mh = _fractions(dg, sigma, edges, 3, active)
        number += qnumber[..., None] * rho[..., None] * nf
        below_n[..., mode] = qnumber * rho * nl
        above_n[..., mode] = qnumber * rho * nh
        issues["source_number_floor"] += int(
            np.count_nonzero((qnumber <= 1e-12) & (qmass * rho * 1e-9 > policy.material_mass_kg_m3))
        )
        if suffix != "c":
            pair = np.minimum(raw["naa" + suffix] / 22.989769, raw["cla" + suffix] / 35.45)
        for name in names:
            si = SOURCE_COMPONENTS.index(name)
            c = source_mass[..., si]
            bins = c[..., None] * mf
            below[..., si] = c * ml
            above[..., si] = c * mh
            if name == "antha":
                excluded[..., :, si] = bins
                continue
            target = {"soila": "DU", "seas": "SS"}.get(name)
            accepted = bins
            if suffix != "c":
                base = name[:-1]
                target = DIRECT_MAP.get(base)
                if base in ("naa", "cla"):
                    mw = 22.989769 if base == "naa" else 35.45
                    accepted = (pair * mw * rho * 1e-9)[..., None] * mf
                    target = "SS"
                elif target is None:
                    accepted = np.zeros_like(bins)
            unresolved[..., :, si] = np.maximum(bins - accepted, 0.0)
            if target:
                imported[..., :, si] = accepted
                mass[..., :, COMPONENTS.index(target)] += accepted

    issues["unresolved_mass"] = int(
        np.count_nonzero(unresolved.sum(axis=(-1, -2)) > policy.unresolved_tolerance_kg_m3)
    )
    restricted = mass[..., :3, [2, 3, 4, 5, 6]].sum(axis=(-1, -2))
    issues["unsupported_small_bins"] = (
        0
        if policy.full_small_bin_species
        else int(np.count_nonzero(restricted > policy.unresolved_tolerance_kg_m3))
    )
    volume_out = (mass / TARGET_DENSITIES).sum(axis=-1)
    effective = (6.0 * volume_out / (np.pi * np.maximum(number, 1e-300))) ** (1.0 / 3.0)
    material = mass.sum(axis=-1) > policy.material_mass_kg_m3
    mismatch = material & ((effective < edges[:-1] * (1.0 - 1e-8)) | (effective > edges[1:] * (1.0 + 1e-8)))
    issues["target_bin_volume"] = int(np.count_nonzero(mismatch))
    issues["massless_particles"] = int(np.count_nonzero((number > 1.0) & (mass.sum(axis=-1) <= 1e-23)))

    result = RemapResult(
        number=number,
        mass=mass,
        edges_m=edges.copy(),
        mode_diameters_m=diameters,
        source_mass=source_mass,
        below_mass=below,
        above_mass=above,
        imported_by_source=imported,
        excluded_mass=excluded,
        unresolved_mass=unresolved,
        below_number=below_n,
        above_number=above_n,
        source_number=source_n,
        issues=issues,
    )
    result.validate_budgets()
    return result
