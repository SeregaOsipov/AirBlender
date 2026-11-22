"""Pure-Python ICON-ART IC/BC preprocessor from ECMWF IFS/CAMS NetCDF files.

Supports independent donor grids for:
  - `met_sfc` (`0.08°` IFS MARS, `0.1°` IFS OpenData S3, or `0.4°` CAMS SFC)
  - `met_ml` (`0.08°` IFS MARS or `0.4°` CAMS 137-ML)
  - `chem_ml` (`0.4°` CAMS 137-ML mineral dust + reactive gases)
  - `chem_ghg` (`0.1°` CAMS GHG `CO2`/`CH4` 137-ML)

Outputs:
  - `ifs2icon_R2Bxx_DOMxx.nc` (init_mode=2 meteorological + soil IC)
  - `ART_STY_iconR2Bxx-grid_TCNR.nc` (HWSD 14-class soil fractions for dust)
  - `ART_IAE_iconR2Bxx-grid_TCNR.nc` (3-mode mineral dust mass & number IC on 90 SLEVE levels)
  - `ART_ICE_iconR2Bxx-grid_TCNR.nc` (6 CAMS-derived passive gas tracer ICs on 90 SLEVE levels)
  - `<run_dir>/latbc/lbc_<YYYY><MM><DD>T<HH>0000Z.nc` (3-hourly met + dust + gas LBC)
  - `<run_dir>/emissions/{SO2,NO2}/ANT/EDGAR/R2Bxx_TCNR/ART_ANT_iconR2Bxx-grid-<YYYY-MM-DD-HH>_TCNR.nc`
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import netCDF4 as nc
import numpy as np
from scipy.ndimage import distance_transform_edt


# Exact 91 half-level heights (m) for standard 90-level ICON SLEVE setup
# (top_height=75000, htop_thcknlimit=14000, max_lay_thckn=400, stretch_fac=0.9, min_lay_thckn=20).
_VCT_A_90 = np.array(
    [
        75000.000, 72363.546, 69842.381, 67357.797, 64946.444, 62606.299,
        60335.466, 58132.167, 55976.216, 53877.930, 51824.685, 49826.951,
        47890.748, 46014.776, 44197.795, 42438.627, 40736.151, 39089.298,
        37497.048, 35958.428, 34472.507, 33038.397, 31655.249, 30322.249,
        29038.622, 27803.623, 26617.350, 25488.963, 24416.908, 23408.796,
        22460.814, 21569.375, 20731.107, 19942.837, 19201.585, 18504.545,
        17849.081, 17232.713, 16653.108, 16108.074, 15595.549, 15113.594,
        14660.386, 14234.210, 13821.524, 13421.524, 13021.524, 12621.524,
        12221.524, 11821.524, 11421.524, 11021.524, 10621.524, 10221.524,
        9821.524, 9421.524, 9021.524, 8621.524, 8221.524, 7821.524,
        7421.524, 7021.524, 6621.524, 6221.524, 5821.524, 5421.524,
        5033.731, 4659.952, 4300.121, 3954.183, 3622.092, 3303.815,
        2999.329, 2708.624, 2431.707, 2168.596, 1919.330, 1683.966,
        1462.584, 1255.291, 1062.224, 883.557, 719.514, 570.373,
        436.493, 318.336, 216.516, 131.880, 65.677, 20.000, 0.000,
    ],
    dtype=np.float64,
)

# TERRA soil parameters (sfc_terra_data.f90, 1-based soil types 1..10)
_TERRA_CFCAP = np.array([1e-10, 1e-10, 0.196, 0.260, 0.340, 0.370, 0.463, 0.763, 1e-10, 1e-10])
_TERRA_CPWP = np.array([0.0, 0.0, 0.042, 0.100, 0.110, 0.185, 0.257, 0.265, 0.0, 0.0])

# ICON-ART DUST_RAD lognormal modes (modes_dustrad.xml): d_gn [m], sigma_g, rho_p [kg/m3]
_DUST_RHO_P = 2650.0
_DUST_MODES = (
    ("dusta", "numb_dusta", "aermr04", 0.6445e-6, 1.700),
    ("dustb", "numb_dustb", "aermr05", 3.4540e-6, 1.600),
    ("dustc", "numb_dustc", "aermr06", 8.6720e-6, 1.500),
)

_GAS_SPECIES = (
    ("TRSO2", "so2", "ml", 64.066, 1.0e-11),
    ("TRNO2", "no2", "ml", 46.006, 5.0e-11),
    ("TRO3", "go3", "ml", 48.000, 6.0e-8),
    ("TRCO", "co", "ml", 28.010, 8.0e-8),
    ("TRCO2", "co2", "ghg", 44.010, 6.22e-4),
    ("TRCH4", "ch4", "ghg", 16.043, 1.05e-6),
)

_SOIL_TEXTURES = (
    "fr_hcla", "fr_silc", "fr_lcla", "fr_sicl", "fr_cloa",
    "fr_silt", "fr_silo", "fr_scla", "fr_loam", "fr_sclo",
    "fr_sloa", "fr_lsan", "fr_sand", "fr_udef",
)

_RD = 287.058
_G = 9.80665


def dust_mass_to_number(mass_kg_kg: np.ndarray, d_gn: float, sigma_g: float, rho_p: float = _DUST_RHO_P) -> np.ndarray:
    """Convert lognormal mineral dust mass mixing ratio [kg kg-1] to specific number [kg-1]."""
    ln_s = float(np.log(sigma_g))
    m_particle = (np.pi / 6.0) * rho_p * (d_gn ** 3) * np.exp(4.5 * ln_s * ln_s)
    return np.maximum(mass_kg_kg, 0.0) / m_particle


def _infill_2d_by_mask(field_2d: np.ndarray, valid_mask: np.ndarray, fallback: float) -> np.ndarray:
    """Fill invalid (`~valid_mask` or non-finite) cells from nearest valid source cell."""
    arr = np.asarray(field_2d, dtype=np.float64)
    good = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)
    if not np.any(good):
        return np.full_like(arr, fallback, dtype=np.float64)
    if np.all(good):
        return arr
    _, nearest_idx = distance_transform_edt(~good, return_distances=True, return_indices=True)
    return arr[tuple(nearest_idx)]


def soil_water_to_smi(
    swvl: np.ndarray,
    soiltyp: np.ndarray,
    lsm: np.ndarray | None = None,
) -> np.ndarray:
    """Convert volumetric soil water [m3 m-3] to TERRA Soil Moisture Index matching `smi_to_wsoil`."""
    st = np.clip(np.asarray(soiltyp, dtype=np.int32), 1, 10) - 1
    pwp = _TERRA_CPWP[st]
    fcap = _TERRA_CFCAP[st]
    denom = np.maximum(fcap - pwp, 1e-6)
    smi = np.where((st >= 2) & (st <= 7), (swvl - pwp) / denom, 0.0)
    smi = np.clip(smi, 0.0, 1.0)
    if lsm is not None:
        smi = np.where(lsm > 0.0, smi, 0.0)
    return smi.astype(np.float32)


def compute_icon_sleve_heights(
    topography_c: np.ndarray,
    flat_height: float = 16000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 91 half-level and 90 full-level SLEVE geometric heights (nlev, ncells) [m]."""
    vct_b = np.where(_VCT_A_90 >= flat_height, 0.0, (flat_height - _VCT_A_90) / flat_height)
    z_ifc = _VCT_A_90[:, None] + vct_b[:, None] * topography_c[None, :]
    z_mc = 0.5 * (z_ifc[:-1, :] + z_ifc[1:, :])
    return z_ifc, z_mc


@dataclass
class _BilinearWeights:
    lat_asc: bool
    wrap_lon: bool
    iy0: np.ndarray
    iy1: np.ndarray
    ix0: np.ndarray
    ix1: np.ndarray
    wy0: np.ndarray
    wy1: np.ndarray
    wx0: np.ndarray
    wx1: np.ndarray

    @classmethod
    def build(cls, src_lat: np.ndarray, src_lon: np.ndarray, dst_lat_deg: np.ndarray, dst_lon_deg: np.ndarray) -> "_BilinearWeights":
        lat_asc = bool(src_lat[-1] >= src_lat[0])
        lat_sorted = src_lat if lat_asc else src_lat[::-1]

        raw_lon = np.asarray(src_lon, dtype=np.float64)
        span = float(raw_lon.max() - raw_lon.min())
        wrap_lon = bool(len(raw_lon) > 10 and span > 350.0)
        if wrap_lon:
            lon_arr = raw_lon % 360.0
            dlon = float(np.median(np.diff(np.sort(lon_arr))))
            n_lon = len(lon_arr)
            target_lon = dst_lon_deg % 360.0
            fx = (target_lon - lon_arr[0]) / dlon
            ix0 = np.floor(fx).astype(np.int64) % n_lon
            ix1 = (ix0 + 1) % n_lon
            wx1 = fx - np.floor(fx)
        else:
            # Handle regional donors stored in [-180, 180) vs [0, 360) uniformly
            if float(raw_lon.min()) < 0.0:
                target_lon = ((dst_lon_deg + 180.0) % 360.0) - 180.0
                lon_arr = raw_lon
            else:
                lon_arr = raw_lon % 360.0
                target_lon = dst_lon_deg % 360.0
            target_lon = np.clip(target_lon, lon_arr[0], lon_arr[-1])
            idx = np.searchsorted(lon_arr, target_lon, side="right") - 1
            ix0 = np.clip(idx, 0, len(lon_arr) - 2)
            ix1 = ix0 + 1
            wx1 = (target_lon - lon_arr[ix0]) / np.maximum(lon_arr[ix1] - lon_arr[ix0], 1e-12)
        wx0 = 1.0 - wx1

        target_lat = np.clip(dst_lat_deg, lat_sorted[0], lat_sorted[-1])
        idy = np.searchsorted(lat_sorted, target_lat, side="right") - 1
        iy0 = np.clip(idy, 0, len(lat_sorted) - 2)
        iy1 = iy0 + 1
        wy1 = (target_lat - lat_sorted[iy0]) / np.maximum(lat_sorted[iy1] - lat_sorted[iy0], 1e-12)
        wy0 = 1.0 - wy1
        if not lat_asc:
            ny = len(lat_sorted)
            iy0 = (ny - 1) - iy0
            iy1 = (ny - 1) - iy1

        return cls(lat_asc=lat_asc, wrap_lon=wrap_lon, iy0=iy0, iy1=iy1, ix0=ix0, ix1=ix1, wy0=wy0, wy1=wy1, wx0=wx0, wx1=wx1)

    def interp(self, field: np.ndarray) -> np.ndarray:
        """Interpolate `(..., nlat, nlon)` -> `(..., ncells)`."""
        v00 = field[..., self.iy0, self.ix0]
        v01 = field[..., self.iy0, self.ix1]
        v10 = field[..., self.iy1, self.ix0]
        v11 = field[..., self.iy1, self.ix1]
        return (
            v00 * (self.wy0 * self.wx0)
            + v01 * (self.wy0 * self.wx1)
            + v10 * (self.wy1 * self.wx0)
            + v11 * (self.wy1 * self.wx1)
        )


def _find_donor_slice(search_dir: Path, prefix: str, target_dt: datetime) -> tuple[nc.Dataset, tuple[int, ...]]:
    """Open matching daily donor file (`<prefix>_*.nc`) and return `(dataset, index_tuple)` for `target_dt`."""
    candidates = sorted(search_dir.glob(f"{prefix}_*.nc"))
    if not candidates:
        raise FileNotFoundError(f"No {prefix}_*.nc files found in {search_dir}")

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    target_sec = (target_dt.astimezone(timezone.utc) - epoch).total_seconds()

    for path in candidates:
        ds = nc.Dataset(path, "r")
        vt_var = ds.variables.get("valid_time")
        if vt_var is None:
            ds.close()
            continue
        vt = np.asarray(vt_var[:])
        if vt.ndim == 2:
            vt_dims = vt_var.dimensions
            if len(vt_dims) == 2 and vt_dims[0] == "forecast_period":
                # (fp, frt)
                for frt_i in range(vt.shape[1]):
                    diffs = np.abs(vt[:, frt_i] - target_sec)
                    best = int(np.argmin(diffs))
                    if diffs[best] <= 60.0:
                        return ds, (best, frt_i)
            else:
                # Standard CDM (frt, fp) -> variable layout (fp, frt, ...)
                for frt_i in range(vt.shape[0]):
                    diffs = np.abs(vt[frt_i] - target_sec)
                    best = int(np.argmin(diffs))
                    if diffs[best] <= 60.0:
                        return ds, (best, frt_i)
        elif vt.ndim == 1:
            diffs = np.abs(vt - target_sec)
            best = int(np.argmin(diffs))
            if diffs[best] <= 60.0:
                return ds, (best,)
        ds.close()

    raise KeyError(f"Timestamp {target_dt.isoformat()} not found in {prefix}_*.nc under {search_dir}")


def _find_cams_slice(cams_dir: Path, kind: str, target_dt: datetime) -> tuple[nc.Dataset, tuple[int, ...]]:
    """Open matching daily CAMS file (`ECMWF-CAMS_<kind>_*.nc`) and return `(dataset, index_tuple)`."""
    return _find_donor_slice(cams_dir, f"ECMWF-CAMS_{kind}", target_dt)


def _read_cams_var(ds: nc.Dataset, name: str, idx: tuple[int, ...], fill: float = 0.0) -> np.ndarray:
    """Read a time-sliced NetCDF variable handling `(fp, frt, ...)` vs `(time, ...)` layouts."""
    var = ds.variables[name]
    data = var[idx + (slice(None),) * (var.ndim - len(idx))]
    if np.ma.isMaskedArray(data):
        arr = np.ma.filled(data.astype(np.float64), np.nan)
    else:
        arr = np.asarray(data, dtype=np.float64)
    if np.isnan(arr).any():
        arr = np.where(np.isnan(arr), fill, arr)
    return arr


def _read_surface_var_any(ds: nc.Dataset, names: Sequence[str], idx: tuple[int, ...], fill: float = 0.0) -> np.ndarray:
    for n in names:
        if n in ds.variables:
            return _read_cams_var(ds, n, idx, fill=fill)
    raise KeyError(f"None of {names} found in {getattr(ds, 'filepath', lambda: '<ds>')()}")


def _compute_cams_heights(
    sp: np.ndarray,
    z_sfc: np.ndarray,
    t_ml: np.ndarray,
    q_ml: np.ndarray,
    hyai: np.ndarray,
    hybi: np.ndarray,
) -> np.ndarray:
    """Compute ECMWF hybrid model-level geometric heights (nlev, ncells) [m] top-to-bottom."""
    p_half = np.maximum(hyai[:, None] + hybi[:, None] * sp[None, :], 1.0e-3)
    tv = t_ml * (1.0 + 0.608 * np.maximum(q_ml, 0.0))
    dlnp = np.log(p_half[1:, :] / p_half[:-1, :])
    dz = (_RD / _G) * tv * dlnp
    z_half_bot = z_sfc[None, :] / _G
    z_rev_cumsum = np.cumsum(dz[::-1, :], axis=0)[::-1, :]
    z_half = np.vstack([z_half_bot + z_rev_cumsum, z_half_bot])
    return 0.5 * (z_half[:-1, :] + z_half[1:, :])


def _vertical_interp_columns(z_src: np.ndarray, f_src: np.ndarray, z_dst: np.ndarray) -> np.ndarray:
    """Linearly interpolate top-to-bottom `(nsrc, ncells)` columns to `(ndst, ncells)` with clamping."""
    n_dst, n_c = z_dst.shape
    out = np.empty((n_dst, n_c), dtype=np.float32)
    for ic in range(n_c):
        out[:, ic] = np.interp(z_dst[:, ic], z_src[::-1, ic], f_src[::-1, ic])
    return out


def _read_grid_geometry(grid_path: Path, need_vertices: bool = False) -> dict[str, Any]:
    with nc.Dataset(grid_path, "r") as g:
        data: dict[str, Any] = {
            "n_cells": int(g.dimensions["cell"].size),
            "uuid": str(g.getncattr("uuidOfHGrid")),
            "clon": np.asarray(g.variables["clon"][:], dtype=np.float64),
            "clat": np.asarray(g.variables["clat"][:], dtype=np.float64),
            "clon_deg": np.rad2deg(np.asarray(g.variables["clon"][:], dtype=np.float64)),
            "clat_deg": np.rad2deg(np.asarray(g.variables["clat"][:], dtype=np.float64)),
        }
        if need_vertices:
            data["n_vertices"] = int(g.dimensions["vertex"].size)
            data["clon_vertices"] = np.asarray(g.variables["clon_vertices"][:], dtype=np.float64)
            data["clat_vertices"] = np.asarray(g.variables["clat_vertices"][:], dtype=np.float64)
        if "global_cell_index" in g.variables:
            data["global_cell_index"] = np.asarray(g.variables["global_cell_index"][:], dtype=np.int64)
        return data


def _add_unstructured_coords(
    ds: nc.Dataset,
    geom: Mapping[str, Any],
    *,
    cell_dim: str = "cell",
    with_vertices: bool = False,
) -> None:
    ds.createDimension(cell_dim, geom["n_cells"])
    clon = ds.createVariable("clon", "f8", (cell_dim,))
    clon.standard_name = "longitude"
    clon.long_name = "longitude of icon grid cell centre"
    clon.units = "radian"
    clon[:] = geom["clon"]

    clat = ds.createVariable("clat", "f8", (cell_dim,))
    clat.standard_name = "latitude"
    clat.long_name = "latitude of icon grid cell centre"
    clat.units = "radian"
    clat[:] = geom["clat"]

    if with_vertices and "clon_vertices" in geom:
        ds.createDimension("nv", 3)
        ds.createDimension("vertex", geom["n_vertices"])
        vlon = ds.createVariable("clon_vertices", "f8", (cell_dim, "nv"))
        vlon.units = "radian"
        vlon[:] = geom["clon_vertices"]
        vlat = ds.createVariable("clat_vertices", "f8", (cell_dim, "nv"))
        vlat.units = "radian"
        vlat[:] = geom["clat_vertices"]
        clon.bounds = "clon_vertices"
        clat.bounds = "clat_vertices"


def _set_cell_attrs(var: nc.Variable, long_name: str, units: str, grid_ref: int = 1) -> None:
    var.long_name = long_name
    var.units = units
    var.coordinates = "clon clat"
    var.CDI_grid_type = "unstructured"
    var.number_of_grid_in_reference = np.int32(grid_ref)


def prepare_icon_edgar_emissions(
    extpar_path: Path | str,
    grid_path: Path | str,
    run_dir: Path | str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    res_str: str = "R2B09",
    cart_io_suffix: str = "TCNR",
) -> list[Path]:
    """Generate ICON-ART ANT EDGAR emission NetCDFs for TRSO2 and TRNO2 from extpar."""
    extpar_path = Path(extpar_path)
    grid_path = Path(grid_path)
    run_dir = Path(run_dir)
    geom = _read_grid_geometry(grid_path, need_vertices=True)

    with nc.Dataset(extpar_path, "r") as ext:
        ext.set_auto_mask(False)
        emi_so2 = np.asarray(ext.variables["emi_so2"][:], dtype=np.float32)
        emi_nox = np.asarray(ext.variables["emi_nox"][:], dtype=np.float32)
    emi_so2 = np.where(np.isfinite(emi_so2) & (emi_so2 >= 0.0), emi_so2, 0.0).astype(np.float32)
    emi_nox = np.where(np.isfinite(emi_nox) & (emi_nox >= 0.0), emi_nox, 0.0).astype(np.float32)

    bracket_times = [
        datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc),
        start_dt - timedelta(hours=3),
        start_dt,
        end_dt,
        end_dt + timedelta(hours=3),
        datetime(2030, 12, 31, 0, 0, tzinfo=timezone.utc),
    ]
    specs = [
        ("SO2", "emi_so2", emi_so2, "SO2 anthropogenic emission flux (EDGAR)"),
        ("NO2", "emi_nox", emi_nox, "NO2/NOx anthropogenic emission flux (EDGAR)"),
    ]
    written: list[Path] = []
    for species, varname, data_arr, lname in specs:
        out_sub = run_dir / "emissions" / species / "ANT" / "EDGAR" / f"{res_str}_{cart_io_suffix}"
        out_sub.mkdir(parents=True, exist_ok=True)
        anchor_path: Path | None = None
        for idx_t, dt in enumerate(bracket_times):
            fname = f"ART_ANT_icon{res_str}-grid-{dt.strftime('%Y-%m-%d-%H')}_{cart_io_suffix}.nc"
            fpath = out_sub / fname
            if idx_t == 0:
                anchor_path = fpath
                if not fpath.exists():
                    with nc.Dataset(fpath, "w", format="NETCDF4") as dst:
                        dst.setncattr("Conventions", "CF-1.6")
                        dst.setncattr("uuidOfHGrid", geom["uuid"])
                        dst.setncattr("number_of_grid_used", np.int32(0))
                        _add_unstructured_coords(dst, geom, cell_dim="cell", with_vertices=True)
                        dst.createDimension("time", None)
                        tv = dst.createVariable("time", "f8", ("time",))
                        tv.standard_name = "time"
                        tv.units = f"hours since {dt.strftime('%Y-%m-%d %H:00:00')}"
                        tv.calendar = "proleptic_gregorian"
                        tv[:] = [0.0]

                        v = dst.createVariable(varname, "f4", ("time", "cell"), zlib=True, complevel=1)
                        _set_cell_attrs(v, lname, "kg m-2 s-1")
                        v[0, :] = data_arr
            elif not fpath.exists() and anchor_path is not None:
                fpath.symlink_to(anchor_path.name)
            written.append(fpath)
    return written


def _extract_hyai_hybi(ds_ml: nc.Dataset) -> tuple[np.ndarray, np.ndarray]:
    pv = None
    for vname in ("t", "u", "v", "q", "w"):
        if vname in ds_ml.variables and "GRIB_pv" in ds_ml.variables[vname].ncattrs():
            pv = np.asarray(ds_ml.variables[vname].getncattr("GRIB_pv"), dtype=np.float64)
            break
    if pv is None and "pv" in ds_ml.variables:
        pv = np.asarray(ds_ml.variables["pv"][:], dtype=np.float64)
    if pv is None:
        raise ValueError("Could not find GRIB_pv coefficients in model-level dataset")
    nhyi = len(pv) // 2
    return pv[:nhyi], pv[nhyi:]


def prepare_icon_ic(
    cams_dir: Path | str,
    grid_path: Path | str,
    extpar_path: Path | str,
    run_dir: Path | str,
    start_dt: datetime,
    *,
    res_str: str = "R2B09",
    cart_io_suffix: str = "TCNR",
    dom_id: int = 1,
    met_sfc_dir: Path | str | None = None,
    met_sfc_prefix: str = "ECMWF-CAMS_sfc",
    met_ml_dir: Path | str | None = None,
    met_ml_prefix: str = "ECMWF-CAMS_ml",
) -> dict[str, Path]:
    """
    Generate `ifs2icon`, `ART_STY`, `ART_IAE`, and `ART_ICE` initial files at `start_dt`.
    Supports independent meteorological (`met_sfc`, `met_ml`) and CAMS chemical/GHG donor grids.
    """
    cams_dir = Path(cams_dir)
    sfc_dir = Path(met_sfc_dir) if met_sfc_dir else cams_dir
    ml_dir = Path(met_ml_dir) if met_ml_dir else cams_dir
    grid_path = Path(grid_path)
    extpar_path = Path(extpar_path)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    geom = _read_grid_geometry(grid_path, need_vertices=True)
    n_c = geom["n_cells"]

    with nc.Dataset(extpar_path, "r") as ext:
        ext.set_auto_mask(False)
        fr_land = np.clip(np.asarray(ext.variables["FR_LAND"][:], dtype=np.float64), 0.0, 1.0)
        soiltyp = np.asarray(ext.variables["SOILTYP"][:], dtype=np.int32)
        t_cl = np.asarray(ext.variables["T_CL"][:], dtype=np.float64)
        topo_c = np.asarray(ext.variables["topography_c"][:], dtype=np.float64)
        tex_arrays = {k: np.asarray(ext.variables[k][:], dtype=np.float32) for k in _SOIL_TEXTURES}

    # Defense-in-depth: ensure water, coastal rock, and undefined soil classes have zero erodible fractions
    non_dust = (fr_land < 0.9) | (soiltyp <= 2) | (soiltyp >= 9)

    # 1. Write ART_STY_iconR2Bxx-grid_TCNR.nc
    sty_path = run_dir / f"ART_STY_icon{res_str}-grid_{cart_io_suffix}.nc"
    with nc.Dataset(sty_path, "w", format="NETCDF4") as dst:
        dst.setncattr("title", "HWSD soil texture fractions for ICON-ART mineral dust")
        dst.setncattr("uuidOfHGrid", geom["uuid"])
        dst.setncattr("number_of_grid_used", np.int32(0))
        _add_unstructured_coords(dst, geom, cell_dim="cell", with_vertices=True)
        for k, arr in tex_arrays.items():
            v = dst.createVariable(k, "f4", ("cell",), zlib=True, complevel=1)
            v.standard_name = f"{k}.st"
            v.units = "1"
            clean_arr = np.where(np.isfinite(arr) & (arr >= 0.0), arr, 0.0)
            if k == "fr_udef":
                v[:] = np.where(non_dust, 1.0, clean_arr).astype(np.float32)
            else:
                v[:] = np.where(non_dust, 0.0, clean_arr).astype(np.float32)

    ds_met_sfc, idx_met_sfc = _find_donor_slice(sfc_dir, met_sfc_prefix, start_dt)
    ds_met_ml, idx_met_ml = _find_donor_slice(ml_dir, met_ml_prefix, start_dt)
    same_ml_as_chem = ml_dir.resolve() == cams_dir.resolve() and met_ml_prefix == "ECMWF-CAMS_ml"
    if same_ml_as_chem:
        ds_chem_ml, idx_chem_ml = ds_met_ml, idx_met_ml
    else:
        ds_chem_ml, idx_chem_ml = _find_cams_slice(cams_dir, "ml", start_dt)

    try:
        ds_ghg, idx_ghg = _find_cams_slice(cams_dir, "ghg", start_dt)
    except (FileNotFoundError, KeyError):
        ds_ghg, idx_ghg = None, ()

    try:
        hyai_met, hybi_met = _extract_hyai_hybi(ds_met_ml)
        nhyi_met = len(hyai_met)
        n_met_ml = nhyi_met - 1
        hyam_met = 0.5 * (hyai_met[:-1] + hyai_met[1:])
        hybm_met = 0.5 * (hybi_met[:-1] + hybi_met[1:])

        w_met_sfc = _BilinearWeights.build(
            np.asarray(ds_met_sfc.variables["latitude"][:]),
            np.asarray(ds_met_sfc.variables["longitude"][:]),
            geom["clat_deg"],
            geom["clon_deg"],
        )
        w_met_ml = _BilinearWeights.build(
            np.asarray(ds_met_ml.variables["latitude"][:]),
            np.asarray(ds_met_ml.variables["longitude"][:]),
            geom["clat_deg"],
            geom["clon_deg"],
        )
        w_chem_ml = (
            w_met_ml
            if same_ml_as_chem
            else _BilinearWeights.build(
                np.asarray(ds_chem_ml.variables["latitude"][:]),
                np.asarray(ds_chem_ml.variables["longitude"][:]),
                geom["clat_deg"],
                geom["clon_deg"],
            )
        )

        sp = w_met_sfc.interp(_read_cams_var(ds_met_sfc, "sp", idx_met_sfc))
        z_sfc = (
            w_met_sfc.interp(_read_cams_var(ds_met_sfc, "z", idx_met_sfc))
            if "z" in ds_met_sfc.variables
            else np.zeros(n_c, dtype=np.float64)
        )

        # Separate land and sea pixels before interpolation to eliminate coastal contamination
        sfc_lsm_src = (
            np.clip(_read_cams_var(ds_met_sfc, "lsm", idx_met_sfc, fill=0.0), 0.0, 1.0)
            if "lsm" in ds_met_sfc.variables
            else np.ones_like(_read_cams_var(ds_met_sfc, "sp", idx_met_sfc))
        )
        land_src = sfc_lsm_src >= 0.85
        if not np.any(land_src):
            land_src = sfc_lsm_src >= 0.5
        sea_src = sfc_lsm_src <= 0.15
        if not np.any(sea_src):
            sea_src = sfc_lsm_src < 0.5

        skt_raw = (
            _read_cams_var(ds_met_sfc, "skt", idx_met_sfc, fill=np.nan)
            if "skt" in ds_met_sfc.variables
            else _read_surface_var_any(ds_met_sfc, ("t2m", "2t"), idx_met_sfc, fill=np.nan)
        )
        sst_raw = (
            _read_cams_var(ds_met_sfc, "sst", idx_met_sfc, fill=np.nan)
            if "sst" in ds_met_sfc.variables
            else np.where(sea_src, skt_raw, np.nan)
        )
        skt_land_2d = _infill_2d_by_mask(skt_raw, land_src, fallback=293.0)
        sst_sea_2d = _infill_2d_by_mask(np.where(np.isfinite(sst_raw), sst_raw, skt_raw), sea_src, fallback=298.0)
        skt_land = w_met_sfc.interp(skt_land_2d)
        sst = w_met_sfc.interp(sst_sea_2d)
        skt = fr_land * skt_land + (1.0 - fr_land) * sst

        asn_raw = _read_cams_var(ds_met_sfc, "asn", idx_met_sfc, fill=0.7) if "asn" in ds_met_sfc.variables else np.full_like(skt_raw, 0.7)
        asn = np.clip(w_met_sfc.interp(asn_raw), 0.1, 0.9)
        w_snow = (
            np.maximum(w_met_sfc.interp(_read_cams_var(ds_met_sfc, "sd", idx_met_sfc)), 0.0)
            if "sd" in ds_met_sfc.variables
            else np.zeros(n_c, dtype=np.float64)
        )
        w_i = (
            np.maximum(w_met_sfc.interp(_read_cams_var(ds_met_sfc, "src", idx_met_sfc)), 0.0)
            if "src" in ds_met_sfc.variables
            else np.zeros(n_c, dtype=np.float64)
        )
        lsm = fr_land
        if "siconc" in ds_met_sfc.variables:
            ci = np.clip(w_met_sfc.interp(_read_cams_var(ds_met_sfc, "siconc", idx_met_sfc)), 0.0, 1.0)
        elif "sithick" in ds_met_sfc.variables:
            ci = np.clip(w_met_sfc.interp((_read_cams_var(ds_met_sfc, "sithick", idx_met_sfc) > 0.01).astype(np.float64)), 0.0, 1.0)
        else:
            ci = np.zeros(n_c, dtype=np.float64)
        t_snow = (
            w_met_sfc.interp(_read_cams_var(ds_met_sfc, "tsn", idx_met_sfc, fill=273.15))
            if "tsn" in ds_met_sfc.variables
            else np.minimum(skt, 273.15)
        )
        rho_snow = (
            np.maximum(w_met_sfc.interp(_read_cams_var(ds_met_sfc, "rsn", idx_met_sfc, fill=250.0)), 50.0)
            if "rsn" in ds_met_sfc.variables
            else np.full(n_c, 250.0, dtype=np.float64)
        )

        # Soil layers from met_sfc (0.08° MARS, 0.1° OpenData, or 0.4° CAMS)
        soil_depths = (0.035, 0.175, 0.64, 1.945)
        stl_list = []
        smi_list = []
        for idx_s, d_m in enumerate(soil_depths, start=1):
            sk = f"stl{idx_s}"
            wk = f"swvl{idx_s}"
            if sk in ds_met_sfc.variables:
                stl_2d = _infill_2d_by_mask(_read_cams_var(ds_met_sfc, sk, idx_met_sfc, fill=np.nan), land_src, fallback=293.0)
                stl_list.append(w_met_sfc.interp(stl_2d))
            else:
                alpha = min(d_m / 2.0, 1.0)
                stl_list.append((1.0 - alpha) * skt_land + alpha * np.where(np.isfinite(t_cl), t_cl, skt_land))
            if wk in ds_met_sfc.variables:
                swvl_2d = _infill_2d_by_mask(_read_cams_var(ds_met_sfc, wk, idx_met_sfc, fill=np.nan), land_src, fallback=0.08)
                swvl = np.maximum(w_met_sfc.interp(swvl_2d), 0.0)
                smi_list.append(soil_water_to_smi(swvl, soiltyp, lsm=fr_land))
            else:
                smi_list.append(
                    np.where((fr_land > 0.0) & (soiltyp >= 3) & (soiltyp <= 8), 0.20, 0.0).astype(np.float32)
                )

        t_ml = w_met_ml.interp(_read_cams_var(ds_met_ml, "t", idx_met_ml, fill=250.0))
        q_ml = np.maximum(w_met_ml.interp(_read_cams_var(ds_met_ml, "q", idx_met_ml)), 0.0)
        u_ml = w_met_ml.interp(_read_cams_var(ds_met_ml, "u", idx_met_ml))
        v_ml = w_met_ml.interp(_read_cams_var(ds_met_ml, "v", idx_met_ml))
        w_omega = w_met_ml.interp(_read_cams_var(ds_met_ml, "w", idx_met_ml))
        qc_ml = (
            np.maximum(w_met_ml.interp(_read_cams_var(ds_met_ml, "clwc", idx_met_ml)), 0.0)
            if "clwc" in ds_met_ml.variables
            else np.zeros_like(q_ml)
        )
        qi_ml = (
            np.maximum(w_met_ml.interp(_read_cams_var(ds_met_ml, "ciwc", idx_met_ml)), 0.0)
            if "ciwc" in ds_met_ml.variables
            else np.zeros_like(q_ml)
        )
        qr_ml = (
            np.maximum(w_met_ml.interp(_read_cams_var(ds_met_ml, "crwc", idx_met_ml)), 0.0)
            if "crwc" in ds_met_ml.variables
            else np.zeros_like(q_ml)
        )
        qs_ml = (
            np.maximum(w_met_ml.interp(_read_cams_var(ds_met_ml, "cswc", idx_met_ml)), 0.0)
            if "cswc" in ds_met_ml.variables
            else np.zeros_like(q_ml)
        )

        # 2. Write ifs2icon_R2Bxx_DOMxx.nc (init_mode = 2)
        ifs_path = run_dir / f"ifs2icon_{res_str}_DOM{dom_id:02d}.nc"
        with nc.Dataset(ifs_path, "w", format="NETCDF4") as dst:
            dst.setncattr("Conventions", "CF-1.6")
            dst.setncattr("uuidOfHGrid", geom["uuid"])
            dst.setncattr("number_of_grid_used", np.int32(0))
            _add_unstructured_coords(dst, geom, cell_dim="cell", with_vertices=True)
            dst.createDimension("time", None)
            dst.createDimension("lev", n_met_ml)
            dst.createDimension("nhyi", nhyi_met)
            dst.createDimension("nhym", n_met_ml)
            dst.createDimension("depth", 1)

            tv = dst.createVariable("time", "f8", ("time",))
            tv.standard_name = "time"
            tv.units = f"hours since {start_dt.strftime('%Y-%m-%d %H:00:00')}"
            tv.calendar = "proleptic_gregorian"
            tv[:] = [0.0]

            lv = dst.createVariable("lev", "f8", ("lev",))
            lv.standard_name = "hybrid_sigma_pressure"
            lv.formula_terms = "ap: hyam b: hybm ps: PS"
            lv.positive = "down"
            lv.units = "1"
            lv[:] = np.arange(1, n_met_ml + 1, dtype=np.float64)

            for hname, hdim, hvals, hunit in [
                ("hyai", "nhyi", hyai_met, "Pa"),
                ("hybi", "nhyi", hybi_met, "1"),
                ("hyam", "nhym", hyam_met, "Pa"),
                ("hybm", "nhym", hybm_met, "1"),
            ]:
                hv = dst.createVariable(hname, "f8", (hdim,))
                hv.units = hunit
                hv[:] = hvals

            dv = dst.createVariable("depth", "f8", ("depth",))
            dv.units = "m"
            dv[:] = [0.0]

            for vname, arr, lname, units in [
                ("T", t_ml, "Temperature", "K"),
                ("U", u_ml, "Zonal wind", "m s-1"),
                ("V", v_ml, "Meridional wind", "m s-1"),
                ("W", w_omega, "Vertical velocity (omega)", "Pa s-1"),
                ("QV", q_ml, "Specific humidity", "kg kg-1"),
                ("QC", qc_ml, "Specific cloud liquid water content", "kg kg-1"),
                ("QI", qi_ml, "Specific cloud ice water content", "kg kg-1"),
                ("QR", qr_ml, "Specific rain water content", "kg kg-1"),
                ("QS", qs_ml, "Specific snow water content", "kg kg-1"),
            ]:
                vv = dst.createVariable(vname, "f4", ("time", "lev", "cell"), zlib=True, complevel=1)
                _set_cell_attrs(vv, lname, units)
                vv[0, :, :] = arr.astype(np.float32)

            for vname, arr, lname, units in [
                ("PS", sp, "Surface pressure", "Pa"),
                ("GEOSP", z_sfc, "Surface geopotential", "m2 s-2"),
                ("SKT", skt, "Skin temperature", "K"),
                ("SST", sst, "Sea surface temperature", "K"),
                ("T_SNOW", t_snow, "Snow temperature", "K"),
                ("ALB_SNOW", asn, "Snow albedo", "1"),
                ("W_SNOW", w_snow, "Snow depth water equivalent", "m"),
                ("RHO_SNOW", rho_snow, "Snow density", "kg m-3"),
                ("W_I", w_i, "Skin reservoir content", "m"),
                ("LSM", lsm, "Land-sea mask", "1"),
                ("CI", ci, "Sea ice cover", "1"),
            ]:
                vv = dst.createVariable(vname, "f4", ("time", "cell"), zlib=True, complevel=1)
                _set_cell_attrs(vv, lname, units)
                vv[0, :] = arr.astype(np.float32)

            for idx_s in range(4):
                vs = dst.createVariable(f"STL{idx_s + 1}", "f4", ("time", "depth", "cell"), zlib=True, complevel=1)
                _set_cell_attrs(vs, f"Soil temperature level {idx_s + 1}", "K")
                vs[0, 0, :] = stl_list[idx_s].astype(np.float32)

                vm = dst.createVariable(f"SMIL{idx_s + 1}", "f4", ("time", "depth", "cell"), zlib=True, complevel=1)
                _set_cell_attrs(vm, f"Soil moisture index level {idx_s + 1}", "1")
                vm[0, 0, :] = smi_list[idx_s].astype(np.float32)

        # Compute hydrostatically consistent CAMS chemistry column heights & ICON 90 SLEVE heights
        if same_ml_as_chem:
            hyai_chem, hybi_chem = hyai_met, hybi_met
            t_chem_col, q_chem_col = t_ml, q_ml
        else:
            hyai_chem, hybi_chem = _extract_hyai_hybi(ds_chem_ml)
            t_chem_col = w_chem_ml.interp(_read_cams_var(ds_chem_ml, "t", idx_chem_ml, fill=250.0))
            q_chem_col = np.maximum(w_chem_ml.interp(_read_cams_var(ds_chem_ml, "q", idx_chem_ml)), 0.0)

        z_chem = _compute_cams_heights(sp, z_sfc, t_chem_col, q_chem_col, hyai_chem, hybi_chem)
        _, z_icon = compute_icon_sleve_heights(topo_c)

        # 3. Write ART_IAE_iconR2Bxx-grid_TCNR.nc (mineral dust mass + number on 90 levels)
        iae_path = run_dir / f"ART_IAE_icon{res_str}-grid_{cart_io_suffix}.nc"
        with nc.Dataset(iae_path, "w", format="NETCDF4") as dst:
            dst.setncattr("Conventions", "CF-1.6")
            dst.setncattr("uuidOfHGrid", geom["uuid"])
            dst.setncattr("number_of_grid_used", np.int32(0))
            _add_unstructured_coords(dst, geom, cell_dim="cell", with_vertices=True)
            dst.createDimension("time", None)
            dst.createDimension("height", 90)
            tv = dst.createVariable("time", "f8", ("time",))
            tv.units = f"hours since {start_dt.strftime('%Y-%m-%d %H:00:00')}"
            tv.calendar = "proleptic_gregorian"
            tv[:] = [0.0]
            hv = dst.createVariable("height", "f8", ("height",))
            hv.axis = "Z"
            hv[:] = np.arange(1, 91, dtype=np.float64)

            for m_var, n_var, cams_var, d_gn, sigma_g in _DUST_MODES:
                raw_3d = np.maximum(w_chem_ml.interp(_read_cams_var(ds_chem_ml, cams_var, idx_chem_ml)), 0.0)
                m_icon = _vertical_interp_columns(z_chem, raw_3d, z_icon)
                n_icon = dust_mass_to_number(m_icon, d_gn, sigma_g)

                vm = dst.createVariable(m_var, "f4", ("time", "height", "cell"), zlib=True, complevel=1)
                _set_cell_attrs(vm, f"Mineral dust mass mixing ratio ({m_var})", "mug kg-1")
                vm[0, :, :] = (m_icon * 1.0e9).astype(np.float32)

                vn = dst.createVariable(n_var, "f4", ("time", "height", "cell"), zlib=True, complevel=1)
                _set_cell_attrs(vn, f"Mineral dust specific number concentration ({n_var})", "kg-1")
                vn[0, :, :] = n_icon

        # 4. Write ART_ICE_iconR2Bxx-grid_TCNR.nc (6 CAMS passive gas tracers on 90 levels)
        w_ghg = (
            _BilinearWeights.build(
                np.asarray(ds_ghg.variables["latitude"][:]),
                np.asarray(ds_ghg.variables["longitude"][:]),
                geom["clat_deg"],
                geom["clon_deg"],
            )
            if ds_ghg is not None
            else None
        )
        ice_path = run_dir / f"ART_ICE_icon{res_str}-grid_{cart_io_suffix}.nc"
        with nc.Dataset(ice_path, "w", format="NETCDF4") as dst:
            dst.setncattr("Conventions", "CF-1.6")
            dst.setncattr("uuidOfHGrid", geom["uuid"])
            dst.setncattr("number_of_grid_used", np.int32(0))
            _add_unstructured_coords(dst, geom, cell_dim="cell", with_vertices=True)
            dst.createDimension("time", None)
            dst.createDimension("height", 90)
            tv = dst.createVariable("time", "f8", ("time",))
            tv.units = f"hours since {start_dt.strftime('%Y-%m-%d %H:00:00')}"
            tv.calendar = "proleptic_gregorian"
            tv[:] = [0.0]
            hv = dst.createVariable("height", "f8", ("height",))
            hv.axis = "Z"
            hv[:] = np.arange(1, 91, dtype=np.float64)

            for tr_name, cams_var, src_kind, _, bg_val in _GAS_SPECIES:
                if src_kind == "ml" and cams_var in ds_chem_ml.variables:
                    raw_3d = np.maximum(w_chem_ml.interp(_read_cams_var(ds_chem_ml, cams_var, idx_chem_ml)), 0.0)
                    tr_icon = _vertical_interp_columns(z_chem, raw_3d, z_icon)
                elif src_kind == "ghg" and ds_ghg is not None and cams_var in ds_ghg.variables and w_ghg is not None:
                    raw_3d = np.maximum(w_ghg.interp(_read_cams_var(ds_ghg, cams_var, idx_ghg)), 0.0)
                    tr_icon = _vertical_interp_columns(z_chem, raw_3d, z_icon)
                else:
                    tr_icon = np.full((90, n_c), bg_val, dtype=np.float32)

                vt = dst.createVariable(tr_name, "f4", ("time", "height", "cell"), zlib=True, complevel=1)
                _set_cell_attrs(vt, f"{tr_name} mass mixing ratio", "kg kg-1")
                vt[0, :, :] = tr_icon

        return {
            "ifs2icon": ifs_path,
            "art_sty": sty_path,
            "art_iae": iae_path,
            "art_ice": ice_path,
        }
    finally:
        ds_met_sfc.close()
        if ds_met_ml is not ds_met_sfc:
            ds_met_ml.close()
        if ds_chem_ml is not ds_met_ml and ds_chem_ml is not ds_met_sfc:
            ds_chem_ml.close()
        if ds_ghg is not None:
            ds_ghg.close()


def prepare_icon_latbc(
    cams_dir: Path | str,
    latbc_grid_path: Path | str,
    run_dir: Path | str,
    times: Sequence[datetime],
    *,
    overwrite: bool = False,
    met_sfc_dir: Path | str | None = None,
    met_sfc_prefix: str = "ECMWF-CAMS_sfc",
    met_ml_dir: Path | str | None = None,
    met_ml_prefix: str = "ECMWF-CAMS_ml",
) -> list[Path]:
    """Generate 3-hourly NetCDF LatBC files on `latbc_grid_path` using `met_sfc` + `met_ml` + CAMS chem/GHG."""
    cams_dir = Path(cams_dir)
    sfc_dir = Path(met_sfc_dir) if met_sfc_dir else cams_dir
    ml_dir = Path(met_ml_dir) if met_ml_dir else cams_dir
    latbc_grid_path = Path(latbc_grid_path)
    out_dir = Path(run_dir) / "latbc"
    out_dir.mkdir(parents=True, exist_ok=True)

    geom = _read_grid_geometry(latbc_grid_path, need_vertices=True)
    w_met_sfc: _BilinearWeights | None = None
    w_met_ml: _BilinearWeights | None = None
    w_chem_ml: _BilinearWeights | None = None
    w_ghg: _BilinearWeights | None = None
    same_ml_as_chem = ml_dir.resolve() == cams_dir.resolve() and met_ml_prefix == "ECMWF-CAMS_ml"
    written: list[Path] = []

    for dt in times:
        dt_utc = dt.astimezone(timezone.utc)
        fname = f"lbc_{dt_utc.strftime('%Y%m%dT%H%M%SZ')}.nc"
        fpath = out_dir / fname
        if fpath.exists() and not overwrite:
            written.append(fpath)
            continue

        ds_met_sfc, idx_met_sfc = _find_donor_slice(sfc_dir, met_sfc_prefix, dt_utc)
        ds_met_ml, idx_met_ml = _find_donor_slice(ml_dir, met_ml_prefix, dt_utc)
        if same_ml_as_chem:
            ds_chem_ml, idx_chem_ml = ds_met_ml, idx_met_ml
        else:
            ds_chem_ml, idx_chem_ml = _find_cams_slice(cams_dir, "ml", dt_utc)
        try:
            ds_ghg, idx_ghg = _find_cams_slice(cams_dir, "ghg", dt_utc)
        except (FileNotFoundError, KeyError):
            ds_ghg, idx_ghg = None, ()

        try:
            if w_met_sfc is None:
                w_met_sfc = _BilinearWeights.build(
                    np.asarray(ds_met_sfc.variables["latitude"][:]),
                    np.asarray(ds_met_sfc.variables["longitude"][:]),
                    geom["clat_deg"],
                    geom["clon_deg"],
                )
            if w_met_ml is None:
                w_met_ml = _BilinearWeights.build(
                    np.asarray(ds_met_ml.variables["latitude"][:]),
                    np.asarray(ds_met_ml.variables["longitude"][:]),
                    geom["clat_deg"],
                    geom["clon_deg"],
                )
            if w_chem_ml is None:
                w_chem_ml = (
                    w_met_ml
                    if same_ml_as_chem
                    else _BilinearWeights.build(
                        np.asarray(ds_chem_ml.variables["latitude"][:]),
                        np.asarray(ds_chem_ml.variables["longitude"][:]),
                        geom["clat_deg"],
                        geom["clon_deg"],
                    )
                )
            if ds_ghg is not None and w_ghg is None:
                w_ghg = _BilinearWeights.build(
                    np.asarray(ds_ghg.variables["latitude"][:]),
                    np.asarray(ds_ghg.variables["longitude"][:]),
                    geom["clat_deg"],
                    geom["clon_deg"],
                )

            hyai, hybi = _extract_hyai_hybi(ds_met_ml)
            nhyi = len(hyai)
            n_ml = nhyi - 1

            sp = w_met_sfc.interp(_read_cams_var(ds_met_sfc, "sp", idx_met_sfc))
            z_sfc = (
                w_met_sfc.interp(_read_cams_var(ds_met_sfc, "z", idx_met_sfc))
                if "z" in ds_met_sfc.variables
                else np.zeros(geom["n_cells"], dtype=np.float64)
            )

            t_met = w_met_ml.interp(_read_cams_var(ds_met_ml, "t", idx_met_ml, fill=250.0))
            q_met = np.maximum(w_met_ml.interp(_read_cams_var(ds_met_ml, "q", idx_met_ml)), 0.0)

            hyai_chem, hybi_chem = (hyai, hybi) if same_ml_as_chem else _extract_hyai_hybi(ds_chem_ml)
            need_vert_chem_map = len(hyai_chem) != nhyi or not np.allclose(hyai_chem, hyai, atol=1e-2) or not np.allclose(hybi_chem, hybi, atol=1e-5)
            z_met = _compute_cams_heights(sp, z_sfc, t_met, q_met, hyai, hybi) if need_vert_chem_map else None
            if need_vert_chem_map:
                t_c = w_chem_ml.interp(_read_cams_var(ds_chem_ml, "t", idx_chem_ml, fill=250.0))
                q_c = np.maximum(w_chem_ml.interp(_read_cams_var(ds_chem_ml, "q", idx_chem_ml)), 0.0)
                z_chem = _compute_cams_heights(sp, z_sfc, t_c, q_c, hyai_chem, hybi_chem)
            else:
                z_chem = None

            def _map_chem_to_met_levels(arr_chem: np.ndarray) -> np.ndarray:
                if z_chem is not None and z_met is not None:
                    return _vertical_interp_columns(z_chem, arr_chem, z_met)
                return arr_chem.astype(np.float32)

            with nc.Dataset(fpath, "w", format="NETCDF4") as dst:
                dst.setncattr("Conventions", "CF-1.6")
                dst.setncattr("uuidOfHGrid", geom["uuid"])
                dst.setncattr("number_of_grid_used", np.int32(0))
                _add_unstructured_coords(dst, geom, cell_dim="cell", with_vertices=True)
                dst.createDimension("time", None)
                dst.createDimension("lev", n_ml)
                dst.createDimension("nhyi", nhyi)

                tv = dst.createVariable("time", "f8", ("time",))
                tv.standard_name = "time"
                tv.units = f"hours since {dt_utc.strftime('%Y-%m-%d %H:00:00')}"
                tv.calendar = "proleptic_gregorian"
                tv[:] = [0.0]

                lv = dst.createVariable("lev", "f8", ("lev",))
                lv.standard_name = "hybrid_sigma_pressure"
                lv.formula_terms = "ap: hyam b: hybm ps: PS"
                lv.positive = "down"
                lv.units = "1"
                lv[:] = np.arange(1, n_ml + 1, dtype=np.float64)

                ha = dst.createVariable("hyai", "f8", ("nhyi",))
                ha.units = "Pa"
                ha[:] = hyai
                hb = dst.createVariable("hybi", "f8", ("nhyi",))
                hb.units = "1"
                hb[:] = hybi

                for vname, arr, lname, units in [
                    ("PS", sp, "Surface pressure", "Pa"),
                    ("GEOSP", z_sfc, "Surface geopotential", "m2 s-2"),
                ]:
                    v2 = dst.createVariable(vname, "f4", ("time", "cell"), zlib=True, complevel=1)
                    _set_cell_attrs(v2, lname, units)
                    v2[0, :] = arr.astype(np.float32)

                for out_name, arr_pre, src_name, nonneg in [
                    ("T", t_met, "t", False),
                    ("U", None, "u", False),
                    ("V", None, "v", False),
                    ("W", None, "w", False),
                    ("QV", q_met, "q", True),
                    ("QC", None, "clwc", True),
                    ("QI", None, "ciwc", True),
                ]:
                    if arr_pre is not None:
                        arr = arr_pre
                    elif src_name in ds_met_ml.variables:
                        arr = w_met_ml.interp(_read_cams_var(ds_met_ml, src_name, idx_met_ml))
                    else:
                        arr = np.zeros((n_ml, geom["n_cells"]), dtype=np.float32)
                    if nonneg:
                        arr = np.maximum(arr, 0.0)
                    v3 = dst.createVariable(out_name, "f4", ("time", "lev", "cell"), zlib=True, complevel=1)
                    _set_cell_attrs(v3, out_name, "1")
                    v3[0, :, :] = arr.astype(np.float32)

                for opt_out, opt_src in [("QR", "crwc"), ("QS", "cswc")]:
                    arr = (
                        np.maximum(w_met_ml.interp(_read_cams_var(ds_met_ml, opt_src, idx_met_ml)), 0.0)
                        if opt_src in ds_met_ml.variables
                        else np.zeros((n_ml, geom["n_cells"]), dtype=np.float32)
                    )
                    v3 = dst.createVariable(opt_out, "f4", ("time", "lev", "cell"), zlib=True, complevel=1)
                    _set_cell_attrs(v3, opt_out, "kg kg-1")
                    v3[0, :, :] = arr.astype(np.float32)

                for m_var, n_var, cams_var, d_gn, sigma_g in _DUST_MODES:
                    m_raw = np.maximum(w_chem_ml.interp(_read_cams_var(ds_chem_ml, cams_var, idx_chem_ml)), 0.0)
                    m_arr = _map_chem_to_met_levels(m_raw)
                    n_arr = dust_mass_to_number(m_arr, d_gn, sigma_g).astype(np.float32)
                    vm = dst.createVariable(m_var, "f4", ("time", "lev", "cell"), zlib=True, complevel=1)
                    _set_cell_attrs(vm, m_var, "mug kg-1")
                    vm[0, :, :] = (m_arr * 1.0e9).astype(np.float32)
                    vn = dst.createVariable(n_var, "f4", ("time", "lev", "cell"), zlib=True, complevel=1)
                    _set_cell_attrs(vn, n_var, "kg-1")
                    vn[0, :, :] = n_arr

                for tr_name, cams_var, src_kind, _, bg_val in _GAS_SPECIES:
                    if src_kind == "ml" and cams_var in ds_chem_ml.variables:
                        raw = np.maximum(w_chem_ml.interp(_read_cams_var(ds_chem_ml, cams_var, idx_chem_ml)), 0.0)
                        arr = _map_chem_to_met_levels(raw)
                    elif src_kind == "ghg" and ds_ghg is not None and cams_var in ds_ghg.variables and w_ghg is not None:
                        raw = np.maximum(w_ghg.interp(_read_cams_var(ds_ghg, cams_var, idx_ghg)), 0.0)
                        arr = _map_chem_to_met_levels(raw)
                    else:
                        arr = np.full((n_ml, geom["n_cells"]), bg_val, dtype=np.float32)
                    vt = dst.createVariable(tr_name, "f4", ("time", "lev", "cell"), zlib=True, complevel=1)
                    _set_cell_attrs(vt, tr_name, "kg kg-1")
                    vt[0, :, :] = arr

            written.append(fpath)
        finally:
            ds_met_sfc.close()
            if ds_met_ml is not ds_met_sfc:
                ds_met_ml.close()
            if ds_chem_ml is not ds_met_ml and ds_chem_ml is not ds_met_sfc:
                ds_chem_ml.close()
            if ds_ghg is not None:
                ds_ghg.close()

    return written
