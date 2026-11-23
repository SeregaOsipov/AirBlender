"""ECMWF IFS/CAMS (multi-resolution surface/soil + 137 model levels) -> WPS intermediate v5 converter."""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import struct

import numpy as np
import xarray as xr

from airblender.icon_icbc import _infill_2d_by_mask


def write_wps_slab(f, nc_data: dict) -> None:
    """Write a single 2D field slab to a WPS intermediate format file (format version 5)."""
    endian_fmt = ">"
    string_encoding = "ascii"

    # Record 1: Version (integer 5)
    f.write(struct.pack(endian_fmt + "i", 4))
    f.write(struct.pack(endian_fmt + "i", nc_data["version"]))
    f.write(struct.pack(endian_fmt + "i", 4))

    # Record 2: Field metadata
    hdate_str = f"{nc_data['hdate']:<24}"[:24]
    xfcst = float(nc_data["xfcst"])
    map_src = f"{nc_data['map_source']:<32}"[:32]
    field_str = f"{nc_data['field']:<9}"[:9]
    units_str = f"{nc_data['units']:<25}"[:25]
    desc_str = f"{nc_data['desc']:<46}"[:46]
    xlvl = float(nc_data["xlvl"])
    nx = int(nc_data["nx"])
    ny = int(nc_data["ny"])
    iproj = int(nc_data["iproj"])

    rec2_fmt = endian_fmt + "24sf32s9s25s46sfiii"
    rec2_size = struct.calcsize(rec2_fmt)

    f.write(struct.pack(endian_fmt + "i", rec2_size))
    f.write(struct.pack(endian_fmt + "24s", hdate_str.encode(string_encoding)))
    f.write(struct.pack(endian_fmt + "f", xfcst))
    f.write(struct.pack(endian_fmt + "32s", map_src.encode(string_encoding)))
    f.write(struct.pack(endian_fmt + "9s", field_str.encode(string_encoding)))
    f.write(struct.pack(endian_fmt + "25s", units_str.encode(string_encoding)))
    f.write(struct.pack(endian_fmt + "46s", desc_str.encode(string_encoding)))
    f.write(struct.pack(endian_fmt + "f", xlvl))
    f.write(struct.pack(endian_fmt + "i", nx))
    f.write(struct.pack(endian_fmt + "i", ny))
    f.write(struct.pack(endian_fmt + "i", iproj))
    f.write(struct.pack(endian_fmt + "i", rec2_size))

    # Record 3: Projection parameters (Lat-Lon / Cylindrical equidistant, iproj=0)
    rec3_fmt = endian_fmt + "8sfffff"
    rec3_size = struct.calcsize(rec3_fmt)
    startloc_str = f"{nc_data['startloc']:<8}"[:8]

    f.write(struct.pack(endian_fmt + "i", rec3_size))
    f.write(struct.pack(endian_fmt + "8s", startloc_str.encode(string_encoding)))
    f.write(struct.pack(endian_fmt + "f", float(nc_data["startlat"])))
    f.write(struct.pack(endian_fmt + "f", float(nc_data["startlon"])))
    f.write(struct.pack(endian_fmt + "f", float(nc_data["deltalat"])))
    f.write(struct.pack(endian_fmt + "f", float(nc_data["deltalon"])))
    f.write(struct.pack(endian_fmt + "f", float(nc_data["earth_radius"])))
    f.write(struct.pack(endian_fmt + "i", rec3_size))

    # Record 4: Wind relative flag
    f.write(struct.pack(endian_fmt + "i", 4))
    f.write(struct.pack(endian_fmt + "i", int(nc_data.get("is_wind_grid_rel", 0))))
    f.write(struct.pack(endian_fmt + "i", 4))

    # Record 5: 2D Slab Data (nx * ny floats, big-endian float32)
    slab_bytes = np.ascontiguousarray(nc_data["slab"], dtype=">f4").tobytes()
    rec5_size = len(slab_bytes)
    f.write(struct.pack(endian_fmt + "i", rec5_size))
    f.write(slab_bytes)
    f.write(struct.pack(endian_fmt + "i", rec5_size))


def _infill_2d(field_2d: np.ndarray, valid_mask: np.ndarray, fallback: float) -> np.ndarray:
    return _infill_2d_by_mask(field_2d, valid_mask, fallback).astype(np.float32)


def _subset_ds_to_bounds(
    ds: xr.Dataset,
    g_lat_min: float,
    g_lat_max: float,
    g_lon_min: float,
    g_lon_max: float,
) -> xr.Dataset:
    raw_lats = np.asarray(ds.latitude.values, dtype=np.float64)
    raw_lons = np.asarray(ds.longitude.values, dtype=np.float64)
    lat_mask = (raw_lats >= g_lat_min) & (raw_lats <= g_lat_max)
    lon_mask = (raw_lons >= g_lon_min) & (raw_lons <= g_lon_max)
    if (
        np.sum(lat_mask) >= 4
        and np.sum(lon_mask) >= 4
        and (np.sum(lat_mask) < len(raw_lats) or np.sum(lon_mask) < len(raw_lons))
    ):
        ds = ds.isel(latitude=lat_mask, longitude=lon_mask)
    return ds


def _build_grid_meta(ds: xr.Dataset, hdate: str, map_source: str) -> tuple[dict, bool]:
    lats = np.asarray(ds.latitude.values, dtype=np.float64)
    lons = np.asarray(ds.longitude.values, dtype=np.float64)
    ny = len(lats)
    nx = len(lons)
    flip_lat = bool(ny > 1 and lats[0] > lats[1])
    startlat = float(lats[-1] if flip_lat else lats[0])
    startlon = float(lons[0])
    deltalat = float(abs(lats[-1] - lats[0]) / max(1, ny - 1))
    deltalon = float(abs(lons[-1] - lons[0]) / max(1, nx - 1))
    meta = {
        "version": 5,
        "hdate": hdate,
        "xfcst": 0.0,
        "map_source": map_source,
        "nx": nx,
        "ny": ny,
        "iproj": 0,
        "startloc": "SWCORNER",
        "startlat": startlat,
        "startlon": startlon,
        "deltalat": deltalat,
        "deltalon": deltalon,
        "earth_radius": 6371.229,
        "is_wind_grid_rel": 0,
        "desc": "",
    }
    return meta, flip_lat


def _extract_time_lookup(ds: xr.Dataset) -> tuple[str | None, list[tuple[str, int, datetime]], dict[str, int]]:
    time_dim = "forecast_period" if "forecast_period" in ds.dims else ("time" if "time" in ds.dims else None)
    if "valid_time" in ds:
        raw_times = ds["valid_time"].values.flatten()
    elif "time" in ds:
        raw_times = ds["time"].values.flatten()
    elif "forecast_reference_time" in ds:
        raw_times = [ds.forecast_reference_time.values.flatten()[0]]
    else:
        return time_dim, [], {}

    entries: list[tuple[str, int, datetime]] = []
    lookup: dict[str, int] = {}
    for idx, tval in enumerate(raw_times):
        dt_obj = np.datetime64(tval, "s").astype(datetime)
        key = dt_obj.strftime("%Y-%m-%d_%H")
        entries.append((key, idx, dt_obj))
        lookup.setdefault(key, idx)
    return time_dim, entries, lookup


def _select_2d_time(ds: xr.Dataset, time_dim: str | None, idx: int) -> xr.Dataset:
    if time_dim is None or time_dim not in ds.dims:
        return ds
    ds_t = ds.isel({time_dim: idx})
    if "forecast_reference_time" in ds_t.dims and ds_t.sizes["forecast_reference_time"] == 1:
        ds_t = ds_t.isel(forecast_reference_time=0)
    return ds_t


def _prep_2d_slab(da_2d: xr.DataArray, flip_lat: bool, fill: float = -1.0e30) -> np.ndarray:
    arr = np.asarray(da_2d.squeeze().values, dtype=np.float64)
    if flip_lat:
        arr = np.flip(arr, axis=0)
    return np.nan_to_num(arr, nan=fill).astype(np.float32)


def _resolve_static_z_da(
    ds_sfc: xr.Dataset,
    sfc_file: Path,
    sfc_time_dim: str | None,
) -> xr.DataArray | None:
    """Resolve finite 2D surface geopotential (`z`/`geopotential`), falling back to sibling 0.1° files."""
    for zk in ("z", "geopotential"):
        if zk not in ds_sfc:
            continue
        n_steps = ds_sfc.sizes[sfc_time_dim] if (sfc_time_dim and sfc_time_dim in ds_sfc.dims) else 1
        for idx in range(max(1, n_steps)):
            ds_t = _select_2d_time(ds_sfc, sfc_time_dim, idx)
            da = ds_t[zk].squeeze(drop=True)
            if np.isfinite(da.values).any():
                return da.load()

    sfc_resolved = sfc_file.resolve()
    for alt_fp in sorted(sfc_file.parent.glob("ECMWF-IFS-0p1_sfc_*.nc")):
        if alt_fp.resolve() == sfc_resolved:
            continue
        try:
            with xr.open_dataset(alt_fp) as ds_alt:
                alt_dim = "forecast_period" if "forecast_period" in ds_alt.dims else ("time" if "time" in ds_alt.dims else None)
                for zk in ("z", "geopotential"):
                    if zk not in ds_alt:
                        continue
                    da_alt = _select_2d_time(ds_alt, alt_dim, 0)[zk].squeeze(drop=True)
                    if not np.isfinite(da_alt.values).any():
                        continue
                    aligned = da_alt.sel(
                        latitude=ds_sfc.latitude.values,
                        longitude=ds_sfc.longitude.values,
                        method="nearest",
                    )
                    if np.isfinite(aligned.values).any():
                        return aligned.load()
        except Exception:
            continue
    return None


def _align_sp_to_ml_grid(ds_sfc_t: xr.Dataset, ds_ml_t: xr.Dataset, flip_lat_ml: bool) -> np.ndarray:
    sp_da = ds_sfc_t.sp.squeeze()
    ml_ny = ds_ml_t.sizes["latitude"]
    ml_nx = ds_ml_t.sizes["longitude"]
    sfc_ny = sp_da.sizes["latitude"]
    sfc_nx = sp_da.sizes["longitude"]

    if (
        sfc_ny == ml_ny
        and sfc_nx == ml_nx
        and np.allclose(ds_sfc_t.latitude.values, ds_ml_t.latitude.values, atol=1e-4)
        and np.allclose(ds_sfc_t.longitude.values, ds_ml_t.longitude.values, atol=1e-4)
    ):
        arr = np.asarray(sp_da.values, dtype=np.float64)
        if flip_lat_ml:
            arr = np.flip(arr, axis=0)
        return arr

    # Interpolate high-res/different-res surface pressure onto ds_ml grid coordinates
    sp_sorted = sp_da.sortby(["latitude", "longitude"])
    target_lats = np.asarray(ds_ml_t.latitude.values, dtype=np.float64)
    target_lons = np.asarray(ds_ml_t.longitude.values, dtype=np.float64)
    lat_clamped = np.clip(target_lats, float(sp_sorted.latitude.min()), float(sp_sorted.latitude.max()))
    lon_clamped = np.clip(target_lons, float(sp_sorted.longitude.min()), float(sp_sorted.longitude.max()))
    sp_interp = sp_sorted.interp(latitude=lat_clamped, longitude=lon_clamped, method="linear")
    arr = np.asarray(sp_interp.values, dtype=np.float64)
    if flip_lat_ml:
        arr = np.flip(arr, axis=0)
    return arr


def convert_cams_to_wps(
    cams_sfc_file: Path,
    cams_ml_file: Path,
    output_dir: Path,
    opendata_soil_file: Path | None = None,
    prefix: str = "FILE",
    overwrite: bool = False,
    target_slices: set | None = None,
    *,
    map_source: str = "ECMWF CAMS",
) -> list[Path]:
    """
    Convert ECMWF surface/soil NetCDF (`sfc`: CAMS 0.4°, OpenData 0.1°, or MARS IFS 0.08°)
    and 137 model-level NetCDF (`ml`: CAMS 0.4° or MARS IFS 0.08°) into WPS intermediate v5
    files (`FILE:YYYY-MM-DD_HH`), aligning timestamps explicitly and allowing independent grids.
    """
    cams_sfc_file = Path(cams_sfc_file)
    cams_ml_file = Path(cams_ml_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_mtime = max(cams_sfc_file.stat().st_mtime, cams_ml_file.stat().st_mtime)

    print(f"Loading SFC: {cams_sfc_file}")
    ds_sfc = xr.open_dataset(cams_sfc_file)
    print(f"Loading ML:  {cams_ml_file}")
    ds_ml = xr.open_dataset(cams_ml_file)

    ds_soil = None
    if opendata_soil_file and Path(opendata_soil_file).exists():
        print(f"Loading Soil override: {opendata_soil_file}")
        ds_soil = xr.open_dataset(opendata_soil_file)
        input_mtime = max(input_mtime, Path(opendata_soil_file).stat().st_mtime)

    geo_em_fp = output_dir / "geo_em.d01.nc"
    if geo_em_fp.exists():
        try:
            with xr.open_dataset(geo_em_fp) as ds_geo:
                lat_var = "XLAT_M" if "XLAT_M" in ds_geo else ("XLAT" if "XLAT" in ds_geo else None)
                lon_var = "XLONG_M" if "XLONG_M" in ds_geo else ("XLONG" if "XLONG" in ds_geo else None)
                if lat_var and lon_var:
                    g_lat_min = float(ds_geo[lat_var].min()) - 5.0
                    g_lat_max = float(ds_geo[lat_var].max()) + 5.0
                    g_lon_min = float(ds_geo[lon_var].min()) - 5.0
                    g_lon_max = float(ds_geo[lon_var].max()) + 5.0
                    ds_sfc = _subset_ds_to_bounds(ds_sfc, g_lat_min, g_lat_max, g_lon_min, g_lon_max)
                    ds_ml = _subset_ds_to_bounds(ds_ml, g_lat_min, g_lat_max, g_lon_min, g_lon_max)
                    if ds_soil is not None and "latitude" in ds_soil.coords and "longitude" in ds_soil.coords:
                        ds_soil = _subset_ds_to_bounds(ds_soil, g_lat_min, g_lat_max, g_lon_min, g_lon_max)
        except Exception as e:
            print(f"Notice: Could not subset donor grids to geo_em bounding box ({e}); using full donor grid.")

    pv = None
    for var_name in ["u", "v", "t", "w", "q"]:
        if var_name in ds_ml and hasattr(ds_ml[var_name], "GRIB_pv"):
            pv = ds_ml[var_name].GRIB_pv
            break
    if pv is None and "pv" in ds_ml:
        pv = ds_ml["pv"].values

    num_levels = len(ds_ml.model_level)
    if pv is not None:
        a_coeff = pv[: num_levels + 1]
        b_coeff = pv[num_levels + 1 :]
    else:
        raise ValueError("Could not find GRIB_pv coefficients in model-level dataset.")

    sfc_time_dim, sfc_entries, sfc_time_map = _extract_time_lookup(ds_sfc)
    ml_time_dim, ml_entries, ml_time_map = _extract_time_lookup(ds_ml)
    soil_time_dim, _, soil_time_map = _extract_time_lookup(ds_soil) if ds_soil is not None else (None, [], {})

    iter_entries = ml_entries if ml_entries else sfc_entries
    static_z_da = _resolve_static_z_da(ds_sfc, cams_sfc_file, sfc_time_dim)

    generated_files: list[Path] = []
    try:
        for slice_key, primary_idx, dt_obj in iter_entries:
            if target_slices is not None and slice_key not in target_slices:
                continue

            sfc_idx = sfc_time_map.get(slice_key)
            if sfc_idx is None:
                if len(sfc_entries) == 1:
                    sfc_idx = 0
                else:
                    raise KeyError(
                        f"Surface slice {slice_key} missing from {cams_sfc_file}; available={list(sfc_time_map)}"
                    )
            if ml_entries:
                ml_idx = ml_time_map.get(slice_key, primary_idx)
            else:
                ml_n = ds_ml.sizes[ml_time_dim] if ml_time_dim and ml_time_dim in ds_ml.dims else 1
                ml_idx = min(sfc_idx, max(0, ml_n - 1))

            hdate = dt_obj.strftime("%Y-%m-%d_%H:00:00")
            filename = f"{prefix}:{slice_key}"
            out_fp = output_dir / filename

            if not overwrite and out_fp.exists() and out_fp.stat().st_size > 0 and out_fp.stat().st_mtime >= input_mtime:
                print(
                    f"Skipping existing up-to-date WPS intermediate file: {out_fp} "
                    f"({out_fp.stat().st_size / (1024 * 1024):.2f} MB)"
                )
                generated_files.append(out_fp)
                continue

            ds_sfc_t = _select_2d_time(ds_sfc, sfc_time_dim, sfc_idx)
            ds_ml_t = _select_2d_time(ds_ml, ml_time_dim, ml_idx)

            sfc_meta, flip_lat_sfc = _build_grid_meta(ds_sfc_t, hdate, map_source)
            ml_meta, flip_lat_ml = _build_grid_meta(ds_ml_t, hdate, map_source)

            tmp_fp = output_dir / f".{filename}.tmp.{os.getpid()}"
            print(
                f"Writing WPS intermediate file: {out_fp} for {hdate} "
                f"(sfc={sfc_meta['nx']}x{sfc_meta['ny']}, ml={ml_meta['nx']}x{ml_meta['ny']})"
            )

            with open(tmp_fp, "wb") as f:
                def prep_sfc(da_2d, fill: float = -1.0e30):
                    return _prep_2d_slab(da_2d, flip_lat_sfc, fill=fill)

                def prep_ml(da_2d, fill: float = -1.0e30):
                    return _prep_2d_slab(da_2d, flip_lat_ml, fill=fill)

                sp_raw_sfc = prep_sfc(ds_sfc_t.sp, fill=np.nan)
                sp_sfc = np.nan_to_num(sp_raw_sfc, nan=-1.0e30)
                write_wps_slab(f, {**sfc_meta, "field": "PSFC", "units": "Pa", "xlvl": 200100.0, "slab": sp_sfc})

                if "msl" in ds_sfc_t:
                    pmsl_arr = prep_sfc(ds_sfc_t.msl)
                elif "mean_sea_level_pressure" in ds_sfc_t:
                    pmsl_arr = prep_sfc(ds_sfc_t.mean_sea_level_pressure)
                else:
                    pmsl_arr = sp_sfc
                write_wps_slab(f, {**sfc_meta, "field": "PMSL", "units": "Pa", "xlvl": 201300.0, "slab": pmsl_arr})

                z_raw = None
                for zk in ("z", "geopotential"):
                    if zk in ds_sfc_t:
                        cand = prep_sfc(ds_sfc_t[zk], fill=np.nan)
                        if np.isfinite(cand).any():
                            z_raw = cand
                            break
                if z_raw is None and static_z_da is not None:
                    z_raw = prep_sfc(static_z_da, fill=np.nan)
                if z_raw is not None:
                    soilhgt_arr = np.nan_to_num(z_raw / 9.80665, nan=0.0).astype(np.float32)
                else:
                    soilhgt_arr = np.zeros_like(sp_sfc)
                write_wps_slab(f, {**sfc_meta, "field": "SOILHGT", "units": "m", "xlvl": 200100.0, "slab": soilhgt_arr})

                # Coastal land/sea infill for clean LANDSEA / SKINTEMP / SST / soil fields
                if "lsm" in ds_sfc_t:
                    lsm_cont = np.clip(prep_sfc(ds_sfc_t.lsm, fill=0.0), 0.0, 1.0)
                else:
                    lsm_cont = np.ones_like(sp_sfc)
                land_src = lsm_cont >= 0.85
                if not np.any(land_src):
                    land_src = lsm_cont >= 0.5
                sea_src = lsm_cont <= 0.15
                if not np.any(sea_src):
                    sea_src = lsm_cont < 0.5
                lsm_arr = (lsm_cont >= 0.5).astype(np.float32)

                skt_raw = prep_sfc(ds_sfc_t.skt if "skt" in ds_sfc_t else ds_sfc_t.sst, fill=np.nan)
                skt_land_infill = _infill_2d(skt_raw, land_src, fallback=293.0)
                if "sst" in ds_sfc_t:
                    sst_raw = prep_sfc(ds_sfc_t.sst, fill=np.nan)
                    sst_in = np.where(np.isfinite(sst_raw) & (sst_raw > 100.0), sst_raw, skt_raw)
                else:
                    sst_in = skt_raw
                sst_sea_infill = _infill_2d(sst_in, sea_src, fallback=298.0)

                skintemp_clean = np.where(lsm_arr >= 0.5, skt_land_infill, sst_sea_infill).astype(np.float32)
                write_wps_slab(f, {**sfc_meta, "field": "SKINTEMP", "units": "K", "xlvl": 200100.0, "slab": skintemp_clean})
                write_wps_slab(f, {**sfc_meta, "field": "LANDSEA", "units": "0/1 Flag", "xlvl": 200100.0, "slab": lsm_arr})
                write_wps_slab(f, {**sfc_meta, "field": "SST", "units": "K", "xlvl": 200100.0, "slab": sst_sea_infill})

                if "siconc" in ds_sfc_t:
                    ice_arr = (np.nan_to_num(prep_sfc(ds_sfc_t.siconc, fill=0.0), nan=0.0) > 0.5).astype(np.float32)
                elif "sithick" in ds_sfc_t:
                    ice_arr = (np.nan_to_num(prep_sfc(ds_sfc_t.sithick, fill=0.0), nan=0.0) > 0.01).astype(np.float32)
                else:
                    ice_arr = np.zeros_like(lsm_arr)
                write_wps_slab(f, {**sfc_meta, "field": "SEAICE", "units": "0/1 Flag", "xlvl": 200100.0, "slab": ice_arr})

                if "sd" in ds_sfc_t:
                    snow_arr = np.maximum(np.nan_to_num(prep_sfc(ds_sfc_t.sd, fill=0.0), nan=0.0), 0.0) * 1000.0
                else:
                    snow_arr = np.zeros_like(lsm_arr)
                write_wps_slab(f, {**sfc_meta, "field": "SNOW", "units": "kg m-2", "xlvl": 200100.0, "slab": snow_arr.astype(np.float32)})

                if "t2m" in ds_sfc_t:
                    t_sfc = prep_sfc(ds_sfc_t.t2m)
                elif "2t" in ds_sfc_t:
                    t_sfc = prep_sfc(ds_sfc_t["2t"])
                else:
                    t_sfc = prep_ml(ds_ml_t.t.isel(model_level=-1))
                t_sfc_meta = sfc_meta if ("t2m" in ds_sfc_t or "2t" in ds_sfc_t) else ml_meta

                if "u10" in ds_sfc_t:
                    u_sfc = prep_sfc(ds_sfc_t.u10)
                    u_sfc_meta = sfc_meta
                elif "10u" in ds_sfc_t:
                    u_sfc = prep_sfc(ds_sfc_t["10u"])
                    u_sfc_meta = sfc_meta
                else:
                    u_sfc = prep_ml(ds_ml_t.u.isel(model_level=-1))
                    u_sfc_meta = ml_meta

                if "v10" in ds_sfc_t:
                    v_sfc = prep_sfc(ds_sfc_t.v10)
                    v_sfc_meta = sfc_meta
                elif "10v" in ds_sfc_t:
                    v_sfc = prep_sfc(ds_sfc_t["10v"])
                    v_sfc_meta = sfc_meta
                else:
                    v_sfc = prep_ml(ds_ml_t.v.isel(model_level=-1))
                    v_sfc_meta = ml_meta

                d2_var = "d2m" if "d2m" in ds_sfc_t else ("2d" if "2d" in ds_sfc_t else None)
                if d2_var is not None:
                    d2_arr = prep_sfc(ds_sfc_t[d2_var], fill=np.nan)
                    tc = d2_arr - 273.15
                    es_pa = 611.2 * np.exp(17.67 * tc / np.maximum(tc + 243.5, 100.0))
                    sp_clean = np.where(np.isfinite(sp_raw_sfc) & (sp_raw_sfc > 1000.0), sp_raw_sfc, 101325.0)
                    q_sfc = np.clip(0.622 * es_pa / np.maximum(sp_clean - 0.378 * es_pa, 100.0), 1.0e-6, 0.05).astype(np.float32)
                    q_sfc = np.nan_to_num(q_sfc, nan=-1.0e30)
                    q_sfc_meta = sfc_meta
                else:
                    q_sfc = prep_ml(ds_ml_t.q.isel(model_level=-1))
                    q_sfc_meta = ml_meta

                write_wps_slab(f, {**t_sfc_meta, "field": "TT", "units": "K", "xlvl": 200100.0, "slab": t_sfc})
                write_wps_slab(f, {**u_sfc_meta, "field": "UU", "units": "m s-1", "xlvl": 200100.0, "slab": u_sfc})
                write_wps_slab(f, {**v_sfc_meta, "field": "VV", "units": "m s-1", "xlvl": 200100.0, "slab": v_sfc})
                write_wps_slab(f, {**q_sfc_meta, "field": "SPECHUMD", "units": "kg kg-1", "xlvl": 200100.0, "slab": q_sfc})

                # 3D Model Levels (from ds_ml on ml_meta)
                sp_ml_arr = _align_sp_to_ml_grid(ds_sfc_t, ds_ml_t, flip_lat_ml)

                def prep_3d(da_3d):
                    arr3 = np.asarray(da_3d.squeeze().values, dtype=np.float32)
                    if flip_lat_ml:
                        arr3 = np.flip(arr3, axis=1)
                    return np.nan_to_num(arr3, nan=-1.0e30)

                t_3d = prep_3d(ds_ml_t.t)
                u_3d = prep_3d(ds_ml_t.u)
                v_3d = prep_3d(ds_ml_t.v)
                q_3d = prep_3d(ds_ml_t.q)

                for k in range(num_levels):
                    lvl_val = float(k + 1)
                    p_top = a_coeff[k] + b_coeff[k] * sp_ml_arr
                    p_bot = a_coeff[k + 1] + b_coeff[k + 1] * sp_ml_arr
                    p_rho = 0.5 * (p_top + p_bot)

                    write_wps_slab(f, {**ml_meta, "field": "PRESSURE", "units": "Pa", "xlvl": lvl_val, "slab": p_rho.astype(np.float32)})
                    write_wps_slab(f, {**ml_meta, "field": "TT", "units": "K", "xlvl": lvl_val, "slab": t_3d[k]})
                    write_wps_slab(f, {**ml_meta, "field": "UU", "units": "m s-1", "xlvl": lvl_val, "slab": u_3d[k]})
                    write_wps_slab(f, {**ml_meta, "field": "VV", "units": "m s-1", "xlvl": lvl_val, "slab": v_3d[k]})
                    write_wps_slab(f, {**ml_meta, "field": "SPECHUMD", "units": "kg kg-1", "xlvl": lvl_val, "slab": q_3d[k]})

                # Soil layers (priority: explicit ds_soil -> ds_sfc stl1..4/swvl1..4 -> fallback)
                soil_written = False
                if ds_soil is not None:
                    soil_idx = soil_time_map.get(slice_key, 0)
                    ds_soil_t = _select_2d_time(ds_soil, soil_time_dim, soil_idx)
                    soil_meta, soil_flip_lat = _build_grid_meta(ds_soil_t, hdate, "ECMWF HRES Soil")

                    def prep_soil(var_key):
                        return _prep_2d_slab(ds_soil_t[var_key], soil_flip_lat, fill=-1.0e30)

                    soil_map_legacy = [
                        ("ST000007", ("SOILTMP_soillevel0Msoillevel1", "stl1"), "K"),
                        ("ST007028", ("SOILTMP_soillevel1Msoillevel2", "stl2"), "K"),
                        ("ST028100", ("SOILTMP_soillevel2Msoillevel3", "stl3"), "K"),
                        ("ST100289", ("SOILTMP_soillevel3Msoillevel4", "stl4"), "K"),
                        ("SM000007", ("VSOILM_soillevel0Msoillevel1", "swvl1"), "m3 m-3"),
                        ("SM007028", ("VSOILM_soillevel1Msoillevel2", "swvl2"), "m3 m-3"),
                        ("SM028100", ("VSOILM_soillevel2Msoillevel3", "swvl3"), "m3 m-3"),
                        ("SM100289", ("VSOILM_soillevel3Msoillevel4", "swvl4"), "m3 m-3"),
                    ]
                    for wps_name, key_cands, s_units in soil_map_legacy:
                        hit = next((k for k in key_cands if k in ds_soil_t), None)
                        if hit is not None:
                            write_wps_slab(
                                f,
                                {**soil_meta, "field": wps_name, "units": s_units, "xlvl": 200100.0, "slab": prep_soil(hit)},
                            )
                            soil_written = True

                if not soil_written and any(f"stl{i}" in ds_sfc_t for i in (1, 2, 3, 4)):
                    st_names = ["ST000007", "ST007028", "ST028100", "ST100289"]
                    sm_names = ["SM000007", "SM007028", "SM028100", "SM100289"]
                    for lvl_i, (st_w, sm_w) in enumerate(zip(st_names, sm_names), start=1):
                        sk = f"stl{lvl_i}"
                        wk = f"swvl{lvl_i}"
                        if sk in ds_sfc_t:
                            st_arr = _infill_2d(prep_sfc(ds_sfc_t[sk], fill=np.nan), land_src, fallback=293.0)
                        else:
                            st_arr = skt_land_infill
                        write_wps_slab(f, {**sfc_meta, "field": st_w, "units": "K", "xlvl": 200100.0, "slab": st_arr})

                        if wk in ds_sfc_t:
                            sm_arr = np.clip(_infill_2d(prep_sfc(ds_sfc_t[wk], fill=np.nan), land_src, fallback=0.08), 0.0, 1.0).astype(np.float32)
                        else:
                            sm_arr = np.full_like(lsm_arr, 0.20, dtype=np.float32)
                        write_wps_slab(f, {**sfc_meta, "field": sm_w, "units": "m3 m-3", "xlvl": 200100.0, "slab": sm_arr})
                    soil_written = True

                if not soil_written:
                    for st_name in ["ST000007", "ST007028", "ST028100", "ST100289"]:
                        write_wps_slab(f, {**sfc_meta, "field": st_name, "units": "K", "xlvl": 200100.0, "slab": skintemp_clean})
                    for sm_name in ["SM000007", "SM007028", "SM028100", "SM100289"]:
                        sm_const = np.full_like(lsm_arr, 0.25, dtype=np.float32)
                        write_wps_slab(f, {**sfc_meta, "field": sm_name, "units": "m3 m-3", "xlvl": 200100.0, "slab": sm_const})

            tmp_fp.replace(out_fp)
            generated_files.append(out_fp)
            print(f"Successfully generated {out_fp} ({os.path.getsize(out_fp) / (1024 * 1024):.2f} MB)")
    finally:
        ds_sfc.close()
        ds_ml.close()
        if ds_soil is not None:
            ds_soil.close()

    return generated_files
