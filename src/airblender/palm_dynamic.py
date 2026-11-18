"""Standalone WRF-Chem (`wrfout_d02_*` / `wrfout_d01_*`) -> PALM-4U `PIDS_DYNAMIC` generator.

Supports all four PALM-4U modes:
1. `meteorology` (31 fields): Meteorology + surface pressure forcing.
2. `phstatp` (55 fields): Meteorology + PHSTATP gas chemistry (`NO`, `NO2`, `O3`, `PM10`).
3. `phstatp-salsa` (115 fields, `absolute_mass_number_v1`): Full 7-component
   (`SO4, OC, BC, DU, SS, NO, NH`) x 10-bin WRF MADE/SORGAM+GOCART -> SALSA remapping.
4. `legacy_3spec` (97 fields): Historical 3-component (`OC, DU, SS`) `mass_fracs_a` mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from netCDF4 import Dataset
import numpy as np
from pyproj import CRS, Geod, Transformer

from airblender.physics import salsa_remap
from airblender.utils.config_io import load_yaml_config, resolve_path

R_DRY = 287.05
G_GRAV = 9.81
SPECIES = ("NO", "NO2", "O3", "PM10")
AUXILIARY_GASES = ("H2SO4", "HNO3", "NH3", "OCNV", "OCSV")
QC_LABELS = ("unresolved", "antha_excluded", "floored_source")
PALM_MODES = ("meteorology", "phstatp", "phstatp-salsa", "legacy_3spec")

LEGACY_3SPEC_DMID_M = np.array(
    [
        4.79468456e-9,
        1.22472675e-8,
        3.12841813e-8,
        7.30010274e-8,
        1.55613467e-7,
        3.31715886e-7,
        7.07107052e-7,
        1.50731324e-6,
        3.21308440e-6,
        6.84921382e-6,
    ],
    dtype=np.float64,
)
MADE_MODES = (
    ("nu0", 0.01e-6, 1.70),
    ("ac0", 0.07e-6, 2.00),
    ("corn", 1.00e-6, 2.50),
)


def compute_salsa_bin_weights(
    nbin: tuple[int, int] = (3, 7),
    reglim: tuple[float, float, float] = (3.0e-9, 5.0e-8, 1.0e-5),
    modes: tuple[tuple[str, float, float], ...] = MADE_MODES,
) -> np.ndarray:
    """Compute legacy (N_modes, N_bins) modal-to-sectional number weight matrix."""
    edges_1 = np.geomspace(reglim[0], reglim[1], nbin[0] + 1)
    edges_2 = np.geomspace(reglim[1], reglim[2], nbin[1] + 1)
    lows = np.concatenate([edges_1[:-1], edges_2[:-1]])
    highs = np.concatenate([edges_1[1:], edges_2[1:]])

    weights = np.zeros((len(modes), len(lows)), dtype=np.float64)
    for m_idx, (_, dpg, sigmag) in enumerate(modes):
        log_sigma = math.log10(sigmag)
        for ib in range(len(lows)):
            d1_base = lows[ib]
            d2_base = highs[ib]
            delta_d = 0.1 * (d2_base - d1_base)
            acc = 0.0
            for sub in range(10):
                d1 = d1_base + sub * delta_d
                d2 = d1 + delta_d
                dmidi = 0.5 * (d1 + d2)
                deltadp = math.log10(d2 / d1)
                acc += (
                    deltadp
                    / (math.sqrt(2.0 * math.pi) * log_sigma)
                    * math.exp(-(math.log10(dmidi / dpg) ** 2) / (2.0 * log_sigma**2))
                )
            weights[m_idx, ib] = acc
        row_sum = weights[m_idx].sum()
        if row_sum > 0.0:
            weights[m_idx] /= row_sum
    return weights


def palm_vertical_coordinates(
    nz: int,
    dz: float = 3.0,
    stretch_start: float = 51.0,
    stretch_factor: float = 1.20,
    dz_max: float = 30.0,
) -> tuple[list[float], list[float]]:
    """Port of PALM `init_grid.f90` single-region vertical stretching."""
    snapped = math.trunc((stretch_start - dz / 2.0) / dz) * dz + dz / 2.0
    zu = [0.0, dz / 2.0]
    spacing = dz
    for _ in range(2, nz + 2):
        if zu[-1] >= snapped:
            spacing = min(dz_max, spacing * stretch_factor)
        zu.append(zu[-1] + spacing)
    zw = [0.0] + [(zu[k] + zu[k + 1]) / 2.0 for k in range(1, nz + 1)]
    zw.append(zw[-1] + 2.0 * (zu[nz + 1] - zw[-1]))
    return zu, zw


@dataclass
class TargetGrid:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    stretch_start: float
    stretch_factor: float
    dz_max: float
    origin_z: float
    origin: dict[str, float]
    geographic: Any
    wind_angle_deg: Any


def load_target_grid_from_static(static_path: Path, domain_cfg: dict[str, Any]) -> TargetGrid:
    """Build PALM `TargetGrid` from the static driver NetCDF4 file and domain YAML config."""
    nx = int(domain_cfg.get("nx", 199))
    ny = int(domain_cfg.get("ny", 199))
    nz = int(domain_cfg.get("nz", 48))
    dx = float(domain_cfg.get("dx", 10.0))
    dy = float(domain_cfg.get("dy", 10.0))
    dz = float(domain_cfg.get("dz", 3.0))
    stretch_start = float(domain_cfg.get("dz_stretch_level", 51.0))
    stretch_factor = float(domain_cfg.get("dz_stretch_factor", 1.20))
    dz_max = float(domain_cfg.get("dz_max", 30.0))

    with Dataset(static_path, "r") as d:
        origin = {
            k: float(d.getncattr(k))
            for k in ("origin_x", "origin_y", "origin_z", "origin_lat", "origin_lon", "rotation_angle")
        }
        crs = CRS.from_user_input(d["crs"].crs_wkt)
        transform = Transformer.from_crs(crs, 4326, always_xy=True)
        geod = Geod(ellps="WGS84")

        def geographic(x: np.ndarray, y: np.ndarray):
            return transform.transform(np.asarray(x) + origin["origin_x"], np.asarray(y) + origin["origin_y"])

        def angle(x: np.ndarray, y: np.ndarray):
            lon, lat = geographic(x, y)
            lon2, lat2 = geographic(np.asarray(x) + 1.0, np.asarray(y))
            azimuth, _, _ = geod.inv(lon, lat, lon2, lat2)
            return 90.0 - np.asarray(azimuth)

    return TargetGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        dy=dy,
        dz=dz,
        stretch_start=stretch_start,
        stretch_factor=stretch_factor,
        dz_max=dz_max,
        origin_z=origin["origin_z"],
        origin=origin,
        geographic=geographic,
        wind_angle_deg=angle,
    )


def driver_coordinates(g: TargetGrid) -> tuple[dict[str, np.ndarray], list[float], list[float]]:
    zu, zw = palm_vertical_coordinates(g.nz, g.dz, g.stretch_start, g.stretch_factor, g.dz_max)
    coords = {
        "x": (np.arange(g.nx + 1) + 0.5) * g.dx,
        "xu": np.arange(1, g.nx + 1) * g.dx,
        "y": (np.arange(g.ny + 1) + 0.5) * g.dy,
        "yv": np.arange(1, g.ny + 1) * g.dy,
        "z": np.array(zu[1 : g.nz + 1], dtype=np.float64),
        "zw": np.array(zw[1 : g.nz], dtype=np.float64),
    }
    return coords, zu, zw


def sampling_targets(g: TargetGrid):
    """Yield all 31 meteorological sampling geometries for PALM LOD2 dynamic forcing."""
    c, zu, zw = driver_coordinates(g)
    maxx = (g.nx + 1) * g.dx
    maxy = (g.ny + 1) * g.dy

    def spec(name: str, comp: str, x, y, z_agl, scope: str):
        x_b, y_b = np.broadcast_arrays(x, y)
        z_arr = np.asarray(z_agl, dtype=np.float64)
        return {
            "name": name,
            "component": comp,
            "x": x_b,
            "y": y_b,
            "z_agl": z_arr,
            "z_msl": z_arr + g.origin_z,
            "time_scope": scope,
        }

    for comp in ("u", "v", "w", "pt", "qv"):
        xc = c["xu"] if comp == "u" else c["x"]
        yc = c["yv"] if comp == "v" else c["y"]
        zc = c["zw"] if comp == "w" else c["z"]
        x, y = np.meshgrid(xc, yc)
        yield spec(f"init_atmosphere_{comp}", comp, x, y, zc, "initialization")
        for side in ("left", "right", "south", "north", "top"):
            if side == "left":
                xx = np.zeros_like(yc) if comp == "u" else np.full_like(yc, -g.dx / 2.0)
                yy = yc
                zz = zc
            elif side == "right":
                xx = np.full_like(yc, maxx if comp == "u" else maxx + g.dx / 2.0)
                yy = yc
                zz = zc
            elif side == "south":
                xx = xc
                yy = np.zeros_like(xc) if comp == "v" else np.full_like(xc, -g.dy / 2.0)
                zz = zc
            elif side == "north":
                xx = xc
                yy = np.full_like(xc, maxy if comp == "v" else maxy + g.dy / 2.0)
                zz = zc
            else:
                xx, yy = x, y
                zz = [zw[g.nz] if comp == "w" else zu[g.nz + 1]]
            yield spec(f"ls_forcing_{side}_{comp}", comp, xx, yy, zz, "all")

    x, y = np.meshgrid(c["x"], c["y"])
    yield spec("surface_forcing_surface_pressure", "pressure_pa", x, y, [0.0], "all")


def _wrf_lcc_grid(path: Path) -> tuple[CRS, Transformer, float, float, int, int, float, float]:
    with Dataset(path, "r") as d:
        attrs = {
            k: float(d.getncattr(k))
            for k in ("TRUELAT1", "TRUELAT2", "MOAD_CEN_LAT", "STAND_LON", "CEN_LAT", "CEN_LON", "DX", "DY")
        }
        nx = len(d.dimensions["west_east"])
        ny = len(d.dimensions["south_north"])

    crs = CRS.from_proj4(
        "+proj=lcc +lat_1={TRUELAT1} +lat_2={TRUELAT2} +lat_0={MOAD_CEN_LAT} "
        "+lon_0={STAND_LON} +a=6370000 +b=6370000 +units=m".format(**attrs)
    )
    forward = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
    cx, cy = forward.transform(attrs["CEN_LON"], attrs["CEN_LAT"])
    x0 = cx - (nx - 1) * attrs["DX"] / 2.0
    y0 = cy - (ny - 1) * attrs["DY"] / 2.0
    return crs, forward, x0, y0, nx, ny, attrs["DX"], attrs["DY"]


def _grid_encloses_palm(wrf_file: Path, target_grid: TargetGrid) -> tuple[bool, tuple[int, int, int, int], float, int, int]:
    _, forward, x0, y0, nx, ny, dx, _ = _wrf_lcc_grid(wrf_file)
    xs_m = np.array([-target_grid.dx, (target_grid.nx + 2) * target_grid.dx], dtype=np.float64)
    ys_m = np.array([-target_grid.dy, (target_grid.ny + 2) * target_grid.dy], dtype=np.float64)
    gx, gy = np.meshgrid(xs_m, ys_m)
    lon, lat = target_grid.geographic(gx, gy)
    wx, wy = forward.transform(lon, lat)
    cols = (np.asarray(wx) - x0) / dx
    rows = (np.asarray(wy) - y0) / dx
    encloses = bool(
        np.all(cols >= 0.0)
        and np.all(cols <= nx - 1.0)
        and np.all(rows >= 0.0)
        and np.all(rows <= ny - 1.0)
    )
    i0 = max(0, int(math.floor(float(cols.min()))) - 2)
    i1 = min(nx, int(math.ceil(float(cols.max()))) + 3)
    j0 = max(0, int(math.floor(float(rows.min()))) - 2)
    j1 = min(ny, int(math.ceil(float(rows.max()))) + 3)
    return encloses, (j0, j1, i0, i1), dx, nx, ny


def _discover_wrf_source(
    run_cfg: dict[str, Any],
    cycle_id: str,
    target_grid: TargetGrid,
    start_dt: datetime,
    forecast_hours: float,
    wrf_dir_override: str | Path | None = None,
) -> tuple[list[Path], Path, str, tuple[int, int, int, int], float]:
    """Select best matching WRF domain (`d02` preferred, `d01` fallback) enclosing space and time."""
    end_seconds = forecast_hours * 3600.0
    prefer = str(run_cfg.get("prefer_wrf_domain", "d02")).lower()
    dom_order = [prefer] + [d for d in ("d02", "d01") if d != prefer]

    base_dirs: list[Path] = []
    if wrf_dir_override:
        base_dirs.append(Path(wrf_dir_override))
    env_wrf_dir = os.environ.get("OPER_FORECAST_WRF_SOURCE_DIR", "").strip()
    if env_wrf_dir and Path(env_wrf_dir) not in base_dirs:
        base_dirs.append(Path(env_wrf_dir))
    wrf_base = resolve_path(run_cfg.get("wrf_source_dir"), "run.wrf_source_dir", required=not bool(base_dirs))
    if wrf_base and wrf_base not in base_dirs:
        base_dirs.append(wrf_base)
    for fb in run_cfg.get("wrf_fallback_dirs", []) or []:
        fb_p = resolve_path(fb, "run.wrf_fallback_dirs", required=False)
        if fb_p and fb_p not in base_dirs:
            base_dirs.append(fb_p)

    candidate_dirs: list[Path] = []
    for base in base_dirs:
        for sub in (base / "output" / cycle_id, base / "run", base):
            if sub.is_dir() and sub not in candidate_dirs:
                candidate_dirs.append(sub)

    diagnostics: list[str] = []
    for cand_dir in candidate_dirs:
        for dom in dom_order:
            found = sorted(f for f in cand_dir.glob(f"wrfout_{dom}_*") if f.is_file() or f.is_symlink())
            if not found:
                continue
            encloses, bbox_slice, wrf_dx, nx, ny = _grid_encloses_palm(found[0], target_grid)
            if not encloses:
                diagnostics.append(f"{cand_dir}/wrfout_{dom}_* does not enclose PALM bounds")
                continue

            leads: list[float] = []
            for p in found:
                with Dataset(p, "r") as d:
                    for idx in range(len(d.dimensions["Time"])):
                        raw_ts = d["Times"][idx].tobytes().decode("ascii").strip()
                        valid_dt = datetime.strptime(raw_ts, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
                        leads.append((valid_dt - start_dt).total_seconds())
            if not leads:
                diagnostics.append(f"{cand_dir}/wrfout_{dom}_* has zero time records")
                continue
            min_lead, max_lead = min(leads), max(leads)
            if min_lead > 0.0 or max_lead < end_seconds:
                diagnostics.append(
                    f"{cand_dir}/wrfout_{dom}_* lead window [{min_lead / 3600:.1f}h, {max_lead / 3600:.1f}h] "
                    f"does not cover [0h, {forecast_hours:.1f}h]"
                )
                continue

            print(
                f"[PALM-Prep] Selected WRF {dom.upper()} forcing in {cand_dir} "
                f"({nx}x{ny}, dx={wrf_dx:.0f}m, crop y[{bbox_slice[0]}:{bbox_slice[1]}], x[{bbox_slice[2]}:{bbox_slice[3]}])"
            )
            return found, cand_dir, dom, bbox_slice, wrf_dx

    detail = "; ".join(diagnostics) if diagnostics else f"checked {base_dirs}"
    raise FileNotFoundError(f"Could not find suitable wrfout_d02/d01 files for cycle {cycle_id}: {detail}")


def _enforce_aerosol_compatibility(
    num_arr: np.ndarray, mass_arr: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure exact dry bin mean volume / number consistency expected by PALM `salsa_absolute_compatibility`."""
    n = np.maximum(0.0, np.asarray(num_arr, dtype=np.float64)).copy()
    m = np.maximum(0.0, np.asarray(mass_arr, dtype=np.float64)).copy()
    tot_m = m.sum(axis=-1)
    empty = (n <= 0.0) | (tot_m <= 1.0e-23)
    n[empty] = 0.0
    m[empty, :] = 0.0

    tiny_mass = (n > 1.0) & (tot_m <= 1.0e-23)
    if np.any(tiny_mass):
        n[tiny_mass] = 0.0
        m[tiny_mass, :] = 0.0

    vol = (m / salsa_remap.TARGET_DENSITIES).sum(axis=-1)
    active = tot_m > 1.0e-15
    if np.any(active):
        d_eff = np.cbrt(6.0 * np.maximum(vol, 0.0) / (np.pi * np.maximum(n, 1.0e-300)))
        lo = edges[:-1] * (1.0 + 1.0e-9)
        hi = edges[1:] * (1.0 - 1.0e-9)
        bad_lo = active & (d_eff < lo)
        bad_hi = active & (d_eff > hi)
        if np.any(bad_lo):
            n[bad_lo] = 6.0 * vol[bad_lo] / (np.pi * (np.broadcast_to(lo, n.shape)[bad_lo] ** 3))
        if np.any(bad_hi):
            n[bad_hi] = 6.0 * vol[bad_hi] / (np.pi * (np.broadcast_to(hi, n.shape)[bad_hi] ** 3))
    return n, m


class WRFCoupledReader:
    """Reads operational WRF-Chem output (sub-cropped around PALM) and samples met + chemistry."""

    def __init__(
        self,
        wrf_files: list[Path],
        start_dt: datetime,
        forecast_hours: float,
        palm_mode: str = "phstatp-salsa",
        chem_cfg: dict[str, Any] | None = None,
        crop_bounds: tuple[int, int, int, int] | None = None,
        dynamic_interval_s: float = 3600.0,
        vertical_adaptation: str = "universal",
    ):
        if not wrf_files:
            raise FileNotFoundError("No wrfout files provided for PALM dynamic driver generation")

        self.start_dt = start_dt
        self.forecast_hours = float(forecast_hours)
        self.palm_mode = palm_mode
        self.chem_cfg = chem_cfg or {}
        self.vertical_adaptation = str(vertical_adaptation).lower()
        end_seconds = self.forecast_hours * 3600.0

        expected_chem_opt = int(self.chem_cfg.get("wrf_chem_opt", 100))
        if self.palm_mode != "meteorology":
            palm_mechanism = str(self.chem_cfg.get("mechanism", "phstatp")).lower()
            if palm_mechanism != "phstatp":
                raise ValueError(f"Unsupported PALM chemistry mechanism '{palm_mechanism}'; expected 'phstatp'")

        default_gas_map = {
            "NO": {"wrf_var": "no", "scale": 1.0},
            "NO2": {"wrf_var": "no2", "scale": 1.0},
            "O3": {"wrf_var": "o3", "scale": 1.0},
            "PM10": {"wrf_var": "PM10", "scale": 1.0e-9},
        }
        self.gas_mapping = self.chem_cfg.get("gas_mapping", default_gas_map)

        salsa_map = self.chem_cfg.get("salsa_mapping", {})
        nbin_cfg = tuple(int(x) for x in salsa_map.get("nbin", (3, 7)))
        reglim_cfg = tuple(float(x) for x in salsa_map.get("reglim_m", (3.0e-9, 5.0e-8, 1.0e-5)))
        modes_cfg = salsa_map.get("modes")
        self.modes = (
            tuple((str(m["wrf_var"]), float(m["dpg_m"]), float(m["sigmag"])) for m in modes_cfg)
            if modes_cfg
            else MADE_MODES
        )
        self.salsa_weights = compute_salsa_bin_weights(nbin=nbin_cfg, reglim=reglim_cfg, modes=self.modes)
        self.mass_species_vars = salsa_map.get(
            "mass_species_vars",
            {"OC": ["orgpaj", "orgpai", "p25j", "p25i", "ecj", "eci", "so4aj", "so4ai"], "DU": ["soila"], "SS": ["seas"]},
        )
        self.legacy_aerosol_floor = float(self.chem_cfg.get("aerosol_floor_m3", 1.0))
        fb_cfg = self.chem_cfg.get("fallback_fractions") or salsa_map.get("fallback_fractions", (0.3333, 0.3333, 0.3334))
        self.legacy_fallback_fractions = tuple(float(x) for x in fb_cfg)

        records_by_time: dict[datetime, tuple[Path, int]] = {}
        for path in sorted(wrf_files):
            with Dataset(path, "r") as d:
                if self.palm_mode != "meteorology" and "CHEM_OPT" in d.ncattrs():
                    actual_chem_opt = int(d.getncattr("CHEM_OPT"))
                    if actual_chem_opt != expected_chem_opt:
                        raise ValueError(
                            f"WRF-Chem CHEM_OPT mismatch in {path}: got {actual_chem_opt}, expected {expected_chem_opt}"
                        )
                for idx in range(len(d.dimensions["Time"])):
                    raw_ts = d["Times"][idx].tobytes().decode("ascii").strip()
                    valid_dt = datetime.strptime(raw_ts, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
                    lead_s = (valid_dt - self.start_dt).total_seconds()
                    if 0.0 <= lead_s <= end_seconds:
                        records_by_time[valid_dt] = (path, idx)

        if not records_by_time:
            raise ValueError(f"No WRF records within [{self.start_dt.isoformat()}, +{self.forecast_hours}h]")

        sorted_recs = [(dt, p, idx) for dt, (p, idx) in sorted(records_by_time.items())]
        if dynamic_interval_s > 0.0 and len(sorted_recs) > 2:
            filtered = []
            for dt, p, idx in sorted_recs:
                lead_s = (dt - self.start_dt).total_seconds()
                rem = lead_s % dynamic_interval_s
                if lead_s == 0.0 or abs(lead_s - end_seconds) < 1e-6 or min(rem, dynamic_interval_s - rem) < 1.0:
                    filtered.append((dt, p, idx))
            if len(filtered) >= 2:
                sorted_recs = filtered
        self.records = sorted_recs
        self.lead_seconds = np.array([(dt - self.start_dt).total_seconds() for dt, _, _ in self.records], dtype=np.float64)
        if self.lead_seconds[0] != 0.0:
            raise ValueError(f"WRF records do not start at 0s (first={self.lead_seconds[0]}s)")
        if self.lead_seconds[-1] < end_seconds:
            raise ValueError(f"WRF records end at {self.lead_seconds[-1] / 3600:.2f}h < {self.forecast_hours:.2f}h")

        crs, forward, full_x0, full_y0, full_nx, full_ny, dx, dy = _wrf_lcc_grid(self.records[0][1])
        self.crs = crs
        self.forward = forward
        self.inverse = Transformer.from_crs(self.crs, self.crs.geodetic_crs, always_xy=True)
        self.geod = Geod(a=6370000, b=6370000)
        self.dx = dx
        self.dy = dy

        if crop_bounds is not None:
            j0, j1, i0, i1 = crop_bounds
        else:
            j0, j1, i0, i1 = 0, full_ny, 0, full_nx
        self._j0, self._j1, self._i0, self._i1 = j0, j1, i0, i1
        self._ys = slice(j0, j1)
        self._xs = slice(i0, i1)
        self._ys_stag = slice(j0, j1 + 1)
        self._xs_stag = slice(i0, i1 + 1)
        self.nx = i1 - i0
        self.ny = j1 - j0
        self.x0 = full_x0 + i0 * self.dx
        self.y0 = full_y0 + j0 * self.dy

        xx, yy = np.meshgrid(self.x0 + np.arange(self.nx) * self.dx, self.y0 + np.arange(self.ny) * self.dy)
        lon, lat = self.inverse.transform(xx, yy)
        lon2, lat2 = self.inverse.transform(xx + 1.0, yy)
        azimuth, _, _ = self.geod.inv(lon, lat, lon2, lat2)
        self.angle = np.deg2rad(90.0 - np.asarray(azimuth))

        self.source_receipts: dict[str, Any] = {}
        self.max_unresolved_fraction = 0.0
        self.max_floored_mass = 0.0
        self._cache: dict[int, dict[str, np.ndarray]] = {}

    def get_frame(self, index: int) -> dict[str, np.ndarray]:
        if index in self._cache:
            return self._cache[index]
        if len(self._cache) >= 2:
            del self._cache[next(iter(self._cache))]

        _, path, rec = self.records[index]
        ys, xs = self._ys, self._xs
        with Dataset(path, "r") as d:
            def rd(var_name: str, default_zero: bool = False) -> np.ndarray:
                if var_name not in d.variables:
                    if default_zero:
                        return np.zeros((1,), dtype=np.float64)
                    raise KeyError(f"Missing required variable '{var_name}' in {path}")
                v = d.variables[var_name]
                dims = v.dimensions
                if dims == ("Time", "bottom_top", "south_north", "west_east") or dims == (
                    "Time",
                    "bottom_top_stag",
                    "south_north",
                    "west_east",
                ):
                    return np.asarray(v[rec, :, ys, xs], dtype=np.float64)
                if dims == ("Time", "bottom_top", "south_north", "west_east_stag"):
                    return np.asarray(v[rec, :, ys, self._xs_stag], dtype=np.float64)
                if dims == ("Time", "bottom_top", "south_north_stag", "west_east"):
                    return np.asarray(v[rec, :, self._ys_stag, xs], dtype=np.float64)
                if dims == ("Time", "south_north", "west_east"):
                    return np.asarray(v[rec, ys, xs], dtype=np.float64)
                raise ValueError(f"Unexpected dimensions {dims} on variable '{var_name}' in {path}")

            t_pert = rd("T")
            p_pert = rd("P")
            pb = rd("PB")
            qv = np.maximum(0.0, rd("QVAPOR"))
            ph = rd("PH")
            phb = rd("PHB")
            hgt = rd("HGT")
            u_stag = rd("U")
            v_stag = rd("V")
            w_stag = rd("W")

            p = p_pert + pb
            temperature_k = (t_pert + 300.0) * (p / 100000.0) ** (2.0 / 7.0)
            pt = temperature_k * (100000.0 / p) ** (287.0 / 1005.0)

            z_w_msl = (ph + phb) / G_GRAV
            z_u_msl = 0.5 * (z_w_msl[:-1] + z_w_msl[1:])
            z_w_agl = np.maximum(0.0, z_w_msl - hgt[None, :, :])
            z_u_agl = np.maximum(0.5, z_u_msl - hgt[None, :, :])

            u = 0.5 * (u_stag[..., :-1] + u_stag[..., 1:])
            v = 0.5 * (v_stag[:, :-1, :] + v_stag[:, 1:, :])
            east = u * np.cos(self.angle) - v * np.sin(self.angle)
            north = u * np.sin(self.angle) + v * np.cos(self.angle)

            frame: dict[str, np.ndarray] = {
                "z_u_agl": z_u_agl,
                "z_w_agl": z_w_agl,
                "z_u_msl": z_u_msl,
                "z_w_msl": z_w_msl,
                "pressure_pa": p,
                "pt": pt,
                "temperature_k": temperature_k,
                "qv": qv,
                "u_ms": east,
                "v_ms": north,
                "w": w_stag,
            }

            if self.palm_mode != "meteorology":
                for sp_name in SPECIES:
                    sp_spec = self.gas_mapping.get(sp_name, {"wrf_var": sp_name.lower(), "scale": 1.0})
                    wrf_vname = str(sp_spec.get("wrf_var", sp_name.lower()))
                    scale = float(sp_spec.get("scale", 1.0))
                    frame[sp_name] = np.maximum(0.0, rd(wrf_vname)) * scale

            if self.palm_mode == "phstatp-salsa":
                rho_dry = (
                    1.0 / np.maximum(rd("ALT"), 1.0e-12)
                    if "ALT" in d.variables
                    else salsa_remap.dry_air_density(p_pert, pb, t_pert, qv)
                )
                comp = {name: np.maximum(0.0, rd(name)) for name in salsa_remap.SOURCE_COMPONENTS}
                nums = {name: np.maximum(0.0, rd(name)) for name in ("nu0", "ac0", "corn")}
                res = salsa_remap.remap_modes(
                    comp,
                    nums,
                    rho_dry,
                    policy=salsa_remap.RemapPolicy(full_small_bin_species=True),
                )
                floored_bin = np.zeros_like(res.number)
                for mi, suffix in enumerate(("i", "j", "c")):
                    names = (
                        [n for n in salsa_remap.SOURCE_COMPONENTS if n.endswith(suffix)]
                        if suffix != "c"
                        else ["soila", "seas", "antha"]
                    )
                    m_idxs = [salsa_remap.SOURCE_COMPONENTS.index(n) for n in names]
                    flags = nums[("nu0", "ac0", "corn")[mi]] <= 1.0e-12
                    floored_bin += np.where(flags[..., None], res.imported_by_source[..., m_idxs].sum(axis=-1), 0.0)

                frame["aerosol"] = res.number
                frame["aerosol_mass"] = res.mass
                frame["unresolved_aerosol_mass"] = res.unresolved_mass.sum(axis=-1)
                frame["antha_excluded_aerosol_mass"] = res.excluded_mass.sum(axis=-1)
                frame["floored_source_aerosol_mass"] = floored_bin

                self.source_receipts[str(index)] = {
                    "issues": res.issues,
                    "source_mass_sum": res.source_mass.sum(axis=(0, 1, 2)).tolist(),
                    "imported_mass_sum": res.imported_by_source.sum(axis=(0, 1, 2, 3)).tolist(),
                    "excluded_mass_sum": res.excluded_mass.sum(axis=(0, 1, 2, 3)).tolist(),
                    "unresolved_mass_sum": res.unresolved_mass.sum(axis=(0, 1, 2, 3)).tolist(),
                }
            elif self.palm_mode == "legacy_3spec":
                p_dry = p / (1.0 + qv / 0.622)
                rho_dry = p_dry / (R_DRY * temperature_k)
                mode_arrays = [np.maximum(0.0, rd(m_var)) * rho_dry for m_var, _, _ in self.modes]
                modes_m3 = np.stack(mode_arrays, axis=-1)
                aerosol_bins = np.maximum(1.0, np.tensordot(modes_m3, self.salsa_weights, axes=([-1], [0])))

                def sum_vars(var_list: list[str]) -> np.ndarray:
                    acc = np.zeros_like(rho_dry)
                    for vn in var_list:
                        if vn in d.variables:
                            acc = acc + np.maximum(0.0, rd(vn))
                    return acc

                oc_m = sum_vars(list(self.mass_species_vars.get("OC", [])))
                du_m = sum_vars(list(self.mass_species_vars.get("DU", ["soila"])))
                ss_m = sum_vars(list(self.mass_species_vars.get("SS", ["seas"])))
                tot_m = oc_m + du_m + ss_m
                valid_m = tot_m > 1.0e-12
                safe_t = np.where(valid_m, tot_m, 1.0)
                fb_oc, fb_du, fb_ss = self.legacy_fallback_fractions
                f_oc = np.where(valid_m, oc_m / safe_t, fb_oc)
                f_du = np.where(valid_m, du_m / safe_t, fb_du)
                f_ss = np.clip(np.where(valid_m, 1.0 - (f_oc + f_du), fb_ss), 0.0, 1.0)
                f_sum = f_oc + f_du + f_ss
                frame["aerosol"] = aerosol_bins
                frame["mass_fracs_a"] = np.stack([f_oc / f_sum, f_du / f_sum, f_ss / f_sum], axis=-1)

        self._cache[index] = frame
        return frame

    def sample(
        self,
        index: int,
        component: str,
        longitude: np.ndarray,
        latitude: np.ndarray,
        heights_agl: np.ndarray,
        heights_msl: np.ndarray | None = None,
    ) -> np.ndarray:
        """Horizontally (bilinear) and vertically interpolate a 3D or multi-D WRF-Chem field onto PALM targets."""
        frame = self.get_frame(index)
        lon, lat = np.broadcast_arrays(np.asarray(longitude, dtype=np.float64), np.asarray(latitude, dtype=np.float64))
        use_msl = self.vertical_adaptation == "strict_msl" and heights_msl is not None and component != "pressure_pa"
        heights = np.asarray(heights_msl if use_msl else heights_agl, dtype=np.float64)

        x, y = self.forward.transform(lon, lat)
        column = np.clip((np.asarray(x).ravel() - self.x0) / self.dx, 0.0, self.nx - 1.0)
        row = np.clip((np.asarray(y).ravel() - self.y0) / self.dy, 0.0, self.ny - 1.0)
        i = np.clip(np.floor(column).astype(int), 0, max(0, self.nx - 2))
        j = np.clip(np.floor(row).astype(int), 0, max(0, self.ny - 2))
        fx = column - i
        fy = row - j

        arr = frame[component]
        if use_msl:
            z_grid = frame["z_w_msl" if component == "w" else "z_u_msl"]
        else:
            z_grid = frame["z_w_agl" if component == "w" else "z_u_agl"]
        extra_shape = arr.shape[3:]
        n_pts = len(column)
        n_extra = int(np.prod(extra_shape)) if extra_shape else 1
        out = np.zeros((len(heights), n_pts, n_extra) if extra_shape else (len(heights), n_pts), dtype=np.float64)

        corners = [
            (j, i, (1.0 - fx) * (1.0 - fy)),
            (j, np.minimum(i + 1, self.nx - 1), fx * (1.0 - fy)),
            (np.minimum(j + 1, self.ny - 1), i, (1.0 - fx) * fy),
            (np.minimum(j + 1, self.ny - 1), np.minimum(i + 1, self.nx - 1), fx * fy),
        ]
        for jj, ii, weight in corners:
            active = weight > 0.0
            if not active.any():
                continue
            pairs = np.stack([jj[active], ii[active]], axis=1)
            unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
            col_vals = []
            for y_idx, x_idx in unique:
                z_col = z_grid[:, y_idx, x_idx]
                v_col = arr[:, y_idx, x_idx]
                if use_msl and (heights.min() < z_col[0] or heights.max() > z_col[-1]):
                    raise ValueError(
                        f"vertical WRF extrapolation forbidden in strict_msl mode "
                        f"(target=[{heights.min():.1f}, {heights.max():.1f}] m, "
                        f"wrf=[{z_col[0]:.1f}, {z_col[-1]:.1f}] m)"
                    )
                if component == "pressure_pa":
                    t_col = frame["temperature_k"][:, y_idx, x_idx]
                    log_p = np.interp(heights, z_col, np.log(v_col))
                    below = heights < z_col[0]
                    if np.any(below):
                        log_p[below] = np.log(v_col[0]) + G_GRAV * (z_col[0] - heights[below]) / (R_DRY * t_col[0])
                    col_vals.append(np.exp(log_p))
                elif v_col.ndim == 1:
                    col_vals.append(np.interp(heights, z_col, v_col))
                else:
                    v_flat = v_col.reshape(v_col.shape[0], -1)
                    sub_cols = [np.interp(heights, z_col, v_flat[:, k]) for k in range(v_flat.shape[1])]
                    col_vals.append(np.stack(sub_cols, axis=-1))

            stacked = np.asarray(col_vals)
            if extra_shape:
                mapped = np.swapaxes(stacked[inverse], 0, 1)
                out[:, active, :] += mapped * weight[None, active, None]
            else:
                mapped = stacked[inverse].T
                out[:, active] += mapped * weight[None, active]

        if component == "mass_fracs_a":
            out = out.reshape((len(heights), n_pts) + extra_shape)
            s = np.sum(out, axis=-1, keepdims=True)
            out = np.where(s > 0.0, out / s, out)
        return out.reshape((len(heights),) + lon.shape + extra_shape)


def _met_dimensions(name: str) -> tuple[str, ...]:
    comp = name.rsplit("_", 1)[-1]
    z = "zw" if comp == "w" else "z"
    x = "xu" if comp == "u" else "x"
    y = "yv" if comp == "v" else "y"
    if name == "surface_forcing_surface_pressure":
        return ("time",)
    if name.startswith("init_atmosphere_"):
        return (z, y, x)
    side = name.split("_")[2]
    if side == "top":
        return ("time", y, x)
    return ("time", z, y if side in ("left", "right") else x)


def _extra_var_spec(name: str, palm_mode: str) -> tuple[tuple[str, ...], int, str]:
    qc_suffixes = tuple(f"{q}_aerosol_mass" for q in QC_LABELS)
    all_keys = SPECIES + AUXILIARY_GASES + ("aerosol_mass", "aerosol", "mass_fracs_a") + qc_suffixes
    key = next((k for k in sorted(all_keys, key=len, reverse=True) if name.endswith("_" + k)), None)
    if key is None:
        raise ValueError(f"Unknown chemical/aerosol field: {name}")

    if key in ("aerosol", *qc_suffixes):
        tail: tuple[str, ...] = ("Dmid",)
    elif key == "aerosol_mass":
        tail = ("Dmid", "composition_index")
    elif key == "mass_fracs_a":
        tail = ("composition_index",)
    else:
        tail = ()

    if name.startswith("init_atmosphere_"):
        dims = ("z",) + tail
        lod = 1
    else:
        side = name.split("_")[2]
        dims = (("time", "y", "x") if side == "top" else ("time", "z", "y" if side in ("left", "right") else "x")) + tail
        lod = 2 if (key in SPECIES or (palm_mode == "phstatp-salsa" and key in AUXILIARY_GASES)) else 1
        if palm_mode == "phstatp-salsa" and key in ("aerosol", "aerosol_mass", *qc_suffixes):
            lod = 1

    if key in SPECIES[:3]:
        units = "ppm"
    elif key in ("PM10", "aerosol_mass", *qc_suffixes):
        units = "kg/m3"
    elif key == "mass_fracs_a":
        units = "1"
    else:
        units = "#/m3"
    return dims, lod, units


def expected_chem_opt_val(chem_cfg: dict[str, Any]) -> int:
    return int(chem_cfg.get("wrf_chem_opt", 100))


def generate_palm_dynamic_driver(
    config_or_path: dict[str, Any] | str | Path,
    date_str: str | None = None,
    cluster_path: str | Path | None = None,
    output_path: Path | None = None,
    forecast_hours_override: float | None = None,
    palm_mode_override: str | None = None,
    wrf_dir_override: str | Path | None = None,
    static_driver_override: str | Path | None = None,
) -> dict[str, Any]:
    """Generate PALM-4U `PIDS_DYNAMIC` NetCDF4 for `meteorology`, `phstatp`, `phstatp-salsa`, or `legacy_3spec`."""
    config = (
        load_yaml_config(config_or_path, cluster_path)
        if not isinstance(config_or_path, dict)
        else config_or_path
    )
    run_cfg = dict(config.get("run", {}))
    cycle_cfg = config.get("cycle", {})
    domain_cfg = config.get("domain", {})
    chem_cfg = config.get("chemistry", {})

    palm_mode = str(palm_mode_override or chem_cfg.get("palm_mode", "phstatp-salsa")).lower()
    if palm_mode not in PALM_MODES:
        raise ValueError(f"Unsupported palm_mode '{palm_mode}'. Choose from {PALM_MODES}")

    raw_date = date_str or str(cycle_cfg.get("start_date", "2026-11-14"))
    clean_date = raw_date.replace("-", "")[:8]
    start_hour = str(cycle_cfg.get("start_hour", "00")).zfill(2)
    cycle_id = f"{clean_date}_{start_hour}"
    start_dt = datetime.strptime(f"{clean_date}{start_hour}", "%Y%m%d%H").replace(tzinfo=timezone.utc)

    env_hours = os.environ.get("OPER_FORECAST_RUN_HOURS")
    env_minutes = os.environ.get("OPER_FORECAST_RUN_MINUTES")
    if forecast_hours_override is not None:
        forecast_hours = float(forecast_hours_override)
    elif env_minutes:
        forecast_hours = max(1.0, float(env_minutes) / 60.0)
    elif env_hours:
        forecast_hours = max(1.0, float(env_hours))
    else:
        forecast_hours = float(cycle_cfg.get("forecast_hours", 24))

    static_path = (
        Path(static_driver_override)
        if static_driver_override
        else resolve_path(run_cfg.get("static_driver"), "run.static_driver", required=True)
    )
    if not static_path.exists():
        raise FileNotFoundError(f"PALM static driver not found: {static_path}")

    grid = load_target_grid_from_static(static_path, domain_cfg)
    coords, _, _ = driver_coordinates(grid)

    wrf_files, wrf_dir_used, wrf_domain, crop_bounds, wrf_dx = _discover_wrf_source(
        run_cfg, cycle_id, grid, start_dt, forecast_hours, wrf_dir_override=wrf_dir_override
    )

    vertical_adaptation = str(chem_cfg.get("vertical_adaptation", "universal"))
    dynamic_interval_s = float(cycle_cfg.get("dynamic_interval_s", 3600.0))
    reader = WRFCoupledReader(
        wrf_files,
        start_dt=start_dt,
        forecast_hours=forecast_hours,
        palm_mode=palm_mode,
        chem_cfg=chem_cfg,
        crop_bounds=crop_bounds,
        dynamic_interval_s=dynamic_interval_s,
        vertical_adaptation=vertical_adaptation,
    )
    coords["time"] = reader.lead_seconds.copy()

    if output_path is None:
        base_work = resolve_path(run_cfg.get("work_dir"), "run.work_dir", required=True)
        case_name = run_cfg.get("case_name", "DohaForecastChem")
        output_path = base_work / run_cfg.get("name", "doha_forecast_chem") / cycle_id / "INPUT" / f"{case_name}_dynamic"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    targets = list(sampling_targets(grid))
    pt_targets = [t for t in targets if t["component"] == "pt"]
    fields: dict[str, np.ndarray] = {}
    met_history: dict[str, list[np.ndarray]] = {t["name"]: [] for t in targets if t["time_scope"] == "all"}

    extra_keys: tuple[str, ...] = ()
    if palm_mode == "phstatp":
        extra_keys = SPECIES
    elif palm_mode == "phstatp-salsa":
        extra_keys = (
            SPECIES
            + AUXILIARY_GASES
            + ("aerosol", "aerosol_mass")
            + tuple(f"{q}_aerosol_mass" for q in QC_LABELS)
        )
    elif palm_mode == "legacy_3spec":
        extra_keys = SPECIES + AUXILIARY_GASES + ("aerosol", "mass_fracs_a")

    chem_history: dict[str, list[np.ndarray]] = {
        f"{t['name'][:-2]}{k}": []
        for t in pt_targets
        if t["time_scope"] == "all"
        for k in extra_keys
    }
    gas_floor = float(chem_cfg.get("salsa_gas_floor_m3", 1.0))

    def eval_met(idx: int, target: dict[str, Any]):
        x, y = target["x"], target["y"]
        lon, lat = grid.geographic(x, y)
        h_agl, h_msl = target["z_agl"], target["z_msl"]
        comp = target["component"]
        if comp in ("u", "v"):
            u = reader.sample(idx, "u_ms", lon, lat, h_agl, h_msl)
            v = reader.sample(idx, "v_ms", lon, lat, h_agl, h_msl)
            ang = np.deg2rad(grid.wind_angle_deg(x, y))
            res = u * np.cos(ang) + v * np.sin(ang) if comp == "u" else -u * np.sin(ang) + v * np.cos(ang)
        else:
            res = reader.sample(idx, comp, lon, lat, h_agl, h_msl)
        return float(res.mean()) if comp == "pressure_pa" else res

    def eval_chem(idx: int, target: dict[str, Any], key: str):
        lon, lat = grid.geographic(target["x"], target["y"])
        return reader.sample(idx, key, lon, lat, target["z_agl"], target["z_msl"])

    n_steps = len(coords["time"])
    print(f"[PALM-Prep] Mode='{palm_mode}' | {n_steps} time steps from {wrf_domain.upper()} (dx={wrf_dx:.0f}m)")
    for t_idx in range(n_steps):
        if t_idx == 0 or t_idx == n_steps - 1 or (t_idx + 1) % 6 == 0:
            print(f"[PALM-Prep]   step {t_idx + 1}/{n_steps} (+{coords['time'][t_idx] / 3600.0:.1f}h)...")
        for target in targets:
            name = target["name"]
            if target["time_scope"] == "initialization":
                if t_idx == 0:
                    fields[name] = eval_met(0, target)
            else:
                val = eval_met(t_idx, target)
                met_history[name].append(val[0] if name.startswith("ls_forcing_top_") else val)

        if extra_keys:
            for target in pt_targets:
                prefix = target["name"][:-2]
                if target["time_scope"] == "initialization":
                    if t_idx == 0:
                        for key in extra_keys:
                            if key in AUXILIARY_GASES:
                                fields[prefix + key] = np.full((len(coords["z"]),), gas_floor, dtype=np.float64)
                            else:
                                sampled = eval_chem(0, target, key)
                                fields[prefix + key] = sampled.mean(axis=(1, 2))
                else:
                    for key in extra_keys:
                        if key in AUXILIARY_GASES:
                            vals = np.full(
                                (len(target["z_agl"]),) + np.shape(target["x"]), gas_floor, dtype=np.float64
                            )
                        else:
                            vals = eval_chem(t_idx, target, key)
                        chem_history[prefix + key].append(
                            vals[0] if target["name"].startswith("ls_forcing_top_") else vals
                        )

    for name, seq in met_history.items():
        fields[name] = np.asarray(seq, dtype=np.float64)
    for name, seq in chem_history.items():
        fields[name] = np.asarray(seq, dtype=np.float64)

    edges = salsa_remap.target_edges()
    max_unres_frac = 0.0
    max_floored_kg = 0.0
    if palm_mode == "phstatp-salsa":
        prefixes = ("init_atmosphere_",) + tuple(f"ls_forcing_{s}_" for s in ("left", "right", "south", "north", "top"))
        for p in prefixes:
            n_clean, m_clean = _enforce_aerosol_compatibility(fields[p + "aerosol"], fields[p + "aerosol_mass"], edges)
            fields[p + "aerosol"] = n_clean
            fields[p + "aerosol_mass"] = m_clean
            mapped = m_clean.sum(axis=-1)
            unres = np.maximum(0.0, fields[p + "unresolved_aerosol_mass"])
            denom = np.maximum(mapped + unres, 1.0e-300)
            frac = np.where(mapped + unres > 1.0e-15, unres / denom, 0.0)
            max_unres_frac = max(max_unres_frac, float(frac.max()))
            floored = np.maximum(0.0, fields[p + "floored_source_aerosol_mass"])
            fields[p + "unresolved_aerosol_mass"] = unres
            fields[p + "antha_excluded_aerosol_mass"] = np.maximum(0.0, fields[p + "antha_excluded_aerosol_mass"])
            fields[p + "floored_source_aerosol_mass"] = floored
            max_floored_kg = max(max_floored_kg, float(floored.max()))

    pt_surface = float(fields["init_atmosphere_pt"][0].mean())
    q_surface = float(fields["init_atmosphere_qv"][0].mean())
    origin_date_time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S +00")

    metadata: dict[str, Any] = {
        **grid.origin,
        "origin_z": grid.origin_z,
        "origin_date_time": origin_date_time_str,
        "forecast_hours": forecast_hours,
        "palm_forecast_mode": palm_mode,
        "source_provider": "wrfchem",
        "source_wrf_domain": wrf_domain,
        "source_wrf_dx_m": wrf_dx,
        "source_dir": str(wrf_dir_used),
        "vertical_adaptation": vertical_adaptation,
        "pt_surface": pt_surface,
        "q_surface": q_surface,
    }
    if palm_mode != "meteorology":
        metadata["wrf_chem_opt"] = expected_chem_opt_val(chem_cfg)
        metadata["wrf_mechanism_name"] = str(chem_cfg.get("wrf_mechanism_name", "RACM_KPP_MADE_SORGAM_GOCART"))
        metadata["chem_mechanism"] = str(chem_cfg.get("mechanism", "phstatp"))
    if palm_mode == "phstatp-salsa":
        metadata.update(
            {
                "salsa_aerosol_input_schema": salsa_remap.SCHEMA,
                "salsa_antha_policy": "exclude",
                "salsa_aerosol_mass_basis": "dry_particle_mass",
                "salsa_small_bin_composition": "all_seven_dry_species",
                "salsa_auxiliary_gas_policy": "deposition_only_research_floor_1_molecule_m3",
                "salsa_aerosol_qualification": "research_only",
                "salsa_research_assumptions_acknowledged": "true",
                "salsa_unresolved_fraction_max": np.float64(max_unres_frac),
                "salsa_floored_source_imported_mass_max_kg_m3": np.float64(max_floored_kg),
                "salsa_research_qualification_receipt": json.dumps(
                    {
                        "unresolved_fraction_max": max_unres_frac,
                        "floored_source_imported_mass_max_kg_m3": max_floored_kg,
                        "antha_policy": "exclude",
                    },
                    sort_keys=True,
                ),
            }
        )

    fd, tmp_path = tempfile.mkstemp(prefix=".palm_dynamic_", suffix=".nc", dir=output_path.parent)
    os.close(fd)
    try:
        with Dataset(tmp_path, "w", format="NETCDF4") as d:
            for c_name, c_arr in coords.items():
                d.createDimension(c_name, len(c_arr))
                v = d.createVariable(c_name, "f8", (c_name,))
                v[:] = c_arr
                if c_name == "time":
                    v.units = f"seconds since {start_dt.strftime('%Y-%m-%d %H:%M:%S')}"
                    v.calendar = "standard"
                else:
                    v.units = "m"

            if palm_mode in ("phstatp-salsa", "legacy_3spec"):
                dmid = np.sqrt(edges[:-1] * edges[1:]) if palm_mode == "phstatp-salsa" else LEGACY_3SPEC_DMID_M
                comp_names = salsa_remap.COMPONENTS if palm_mode == "phstatp-salsa" else ("OC", "DU", "SS")
                d.createDimension("Dmid", len(dmid))
                d.createDimension("composition_index", len(comp_names))
                d.createDimension("max_string_length", 25)
                dv = d.createVariable("Dmid", "f8", ("Dmid",))
                dv[:] = dmid
                dv.units = "m"
                cv = d.createVariable("composition_name", "S1", ("composition_index", "max_string_length"))
                cv[:] = np.array([list(s.ljust(25)) for s in comp_names], dtype="S1")

            for f_name, f_arr in fields.items():
                if any(f_name.endswith("_" + k) for k in extra_keys):
                    dims, lod, units = _extra_var_spec(f_name, palm_mode)
                    dtype = (
                        "f8"
                        if palm_mode == "phstatp-salsa"
                        and (
                            "aerosol" in dims
                            or "Dmid" in dims
                            or "aerosol" in f_name
                            or f_name.endswith(AUXILIARY_GASES)
                            or any(f_name.endswith("_" + s) for s in SPECIES)
                        )
                        else "f4"
                    )
                    fill = np.float64(-9999.0) if dtype == "f8" else np.float32(-9999.0)
                    v = d.createVariable(f_name, dtype, dims, fill_value=fill)
                    v[:] = f_arr.astype(np.float64 if dtype == "f8" else np.float32)
                    v.units = units
                    if palm_mode == "phstatp-salsa" and any(f_name.endswith(f"_{q}_aerosol_mass") for q in QC_LABELS):
                        v.long_name = "diagnostic-only budget; not a transported SALSA component"
                    else:
                        v.lod = np.int32(lod)
                else:
                    dims = _met_dimensions(f_name)
                    v = d.createVariable(f_name, "f4", dims, fill_value=np.float32(-9999.0))
                    v[:] = f_arr.astype(np.float32)
                    comp = f_name.rsplit("_", 1)[-1]
                    v.units = (
                        "Pa"
                        if f_name == "surface_forcing_surface_pressure"
                        else "K"
                        if comp == "pt"
                        else "kg kg-1"
                        if comp == "qv"
                        else "m s-1"
                    )
                    v.lod = np.int32(2 if f_name != "surface_forcing_surface_pressure" else 1)

            for m_key, m_val in metadata.items():
                d.setncattr(m_key, int(m_val) if isinstance(m_val, bool) else m_val)

        os.replace(tmp_path, output_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    print(
        f"[PALM-Prep] Wrote PIDS_DYNAMIC ({len(fields)} fields, "
        f"{output_path.stat().st_size / (1024 * 1024):.1f} MB): {output_path}"
    )
    return {
        "dynamic_driver": str(output_path),
        "palm_mode": palm_mode,
        "field_count": len(fields),
        "wrf_domain": wrf_domain,
        "origin_date_time": origin_date_time_str,
        "end_time_s": forecast_hours * 3600.0,
        "pt_surface": pt_surface,
        "q_surface": q_surface,
        "n_timesteps": n_steps,
        "max_unresolved_fraction": max_unres_frac,
    }
