"""ECMWF CAMS + OpenData soil -> WPS intermediate binary format (version 5) converter."""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import struct

import numpy as np
import xarray as xr


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


def convert_cams_to_wps(
    cams_sfc_file: Path,
    cams_ml_file: Path,
    output_dir: Path,
    opendata_soil_file: Path | None = None,
    prefix: str = "FILE",
    overwrite: bool = False,
    target_slices: set | None = None,
) -> list[Path]:
    """
    Convert CAMS NetCDF `sfc` and `ml` files (and optional OpenData soil) into
    WPS intermediate binary files (`FILE:YYYY-MM-DD_HH`).
    Skips existing up-to-date slices unless `overwrite=True` and uses atomic `.tmp` renames.
    """
    cams_sfc_file = Path(cams_sfc_file)
    cams_ml_file = Path(cams_ml_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_mtime = max(cams_sfc_file.stat().st_mtime, cams_ml_file.stat().st_mtime)

    print(f"Loading CAMS SFC: {cams_sfc_file}")
    ds_sfc = xr.open_dataset(cams_sfc_file)
    print(f"Loading CAMS ML:  {cams_ml_file}")
    ds_ml = xr.open_dataset(cams_ml_file)

    ds_soil = None
    if opendata_soil_file and Path(opendata_soil_file).exists():
        print(f"Loading OpenData Soil: {opendata_soil_file}")
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
                    raw_lats = ds_sfc.latitude.values
                    raw_lons = ds_sfc.longitude.values
                    lat_mask = (raw_lats >= g_lat_min) & (raw_lats <= g_lat_max)
                    lon_mask = (raw_lons >= g_lon_min) & (raw_lons <= g_lon_max)
                    if (
                        np.sum(lat_mask) >= 10
                        and np.sum(lon_mask) >= 10
                        and (np.sum(lat_mask) < len(raw_lats) or np.sum(lon_mask) < len(raw_lons))
                    ):
                        ds_sfc = ds_sfc.isel(latitude=lat_mask, longitude=lon_mask)
                        ds_ml = ds_ml.isel(latitude=lat_mask, longitude=lon_mask)
        except Exception as e:
            print(f"Notice: Could not subset CAMS to geo_em bounding box ({e}); using full CAMS grid.")

    lats = ds_sfc.latitude.values
    lons = ds_sfc.longitude.values
    ny = len(lats)
    nx = len(lons)
    flip_lat = lats[0] > lats[1]

    startlat = float(lats[-1] if flip_lat else lats[0])
    startlon = float(lons[0])
    deltalat = float(abs(lats[-1] - lats[0]) / (ny - 1))
    deltalon = float(abs(lons[-1] - lons[0]) / (nx - 1))

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
        raise ValueError("Could not find GRIB_pv coefficients in CAMS model level dataset.")

    if "valid_time" in ds_sfc:
        times = ds_sfc["valid_time"].values.flatten()
    elif "time" in ds_sfc:
        times = ds_sfc["time"].values.flatten()
    else:
        times = [ds_sfc.forecast_reference_time.values.flatten()[0]]

    time_dim = "forecast_period" if "forecast_period" in ds_sfc.dims else ("time" if "time" in ds_sfc.dims else None)
    ml_time_dim = "forecast_period" if "forecast_period" in ds_ml.dims else ("time" if "time" in ds_ml.dims else None)

    generated_files: list[Path] = []
    try:
        for t_idx, time_val in enumerate(times):
            dt_obj = np.datetime64(time_val, "s").astype(datetime)
            slice_key = dt_obj.strftime("%Y-%m-%d_%H")
            if target_slices is not None and slice_key not in target_slices:
                continue

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

            ds_sfc_t = ds_sfc.isel({time_dim: t_idx}) if time_dim else ds_sfc
            ds_ml_t = ds_ml.isel({ml_time_dim: t_idx}) if (ml_time_dim and ml_time_dim in ds_ml.dims) else ds_ml

            tmp_fp = output_dir / f".{filename}.tmp"
            print(f"Writing WPS intermediate file: {out_fp} for {hdate}")

            with open(tmp_fp, "wb") as f:
                base_meta = {
                    "version": 5,
                    "hdate": hdate,
                    "xfcst": 0.0,
                    "map_source": "ECMWF CAMS",
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

                def prep_slab(da_2d):
                    arr = da_2d.squeeze().values
                    if flip_lat:
                        arr = np.flip(arr, axis=0)
                    return np.nan_to_num(arr, nan=-1.0e30)

                sp = ds_sfc_t.sp.squeeze()
                write_wps_slab(f, {**base_meta, "field": "PSFC", "units": "Pa", "xlvl": 200100.0, "slab": prep_slab(sp)})

                if "msl" in ds_sfc_t:
                    pmsl_arr = prep_slab(ds_sfc_t.msl)
                elif "mean_sea_level_pressure" in ds_sfc_t:
                    pmsl_arr = prep_slab(ds_sfc_t.mean_sea_level_pressure)
                else:
                    pmsl_arr = prep_slab(sp)
                write_wps_slab(f, {**base_meta, "field": "PMSL", "units": "Pa", "xlvl": 201300.0, "slab": pmsl_arr})

                if "z" in ds_sfc_t:
                    soilhgt_arr = prep_slab(ds_sfc_t.z) / 9.80665
                elif "geopotential" in ds_sfc_t:
                    soilhgt_arr = prep_slab(ds_sfc_t.geopotential) / 9.80665
                else:
                    soilhgt_arr = np.zeros_like(prep_slab(sp))
                write_wps_slab(f, {**base_meta, "field": "SOILHGT", "units": "m", "xlvl": 200100.0, "slab": soilhgt_arr})

                skt = ds_sfc_t.skt.squeeze() if "skt" in ds_sfc_t else ds_sfc_t.sst.squeeze()
                nc_skt = {**base_meta, "field": "SKINTEMP", "units": "K", "xlvl": 200100.0, "slab": prep_slab(skt)}
                write_wps_slab(f, nc_skt)

                lsm_arr = (prep_slab(ds_sfc_t.lsm) > 0.5).astype(np.float32)
                write_wps_slab(f, {**base_meta, "field": "LANDSEA", "units": "0/1 Flag", "xlvl": 200100.0, "slab": lsm_arr})

                sst_raw = prep_slab(ds_sfc_t.sst)
                sst_filled = np.where(sst_raw > 100.0, sst_raw, nc_skt["slab"])
                write_wps_slab(f, {**base_meta, "field": "SST", "units": "K", "xlvl": 200100.0, "slab": sst_filled})

                if "siconc" in ds_sfc_t:
                    ice_arr = (prep_slab(ds_sfc_t.siconc) > 0.5).astype(np.float32)
                else:
                    ice_arr = np.zeros_like(lsm_arr)
                write_wps_slab(f, {**base_meta, "field": "SEAICE", "units": "0/1 Flag", "xlvl": 200100.0, "slab": ice_arr})

                if "sd" in ds_sfc_t:
                    snow_arr = prep_slab(ds_sfc_t.sd) * 1000.0
                else:
                    snow_arr = np.zeros_like(lsm_arr)
                write_wps_slab(f, {**base_meta, "field": "SNOW", "units": "kg m-2", "xlvl": 200100.0, "slab": snow_arr})

                if "t2m" in ds_sfc_t:
                    t_sfc = prep_slab(ds_sfc_t.t2m)
                elif "2t" in ds_sfc_t:
                    t_sfc = prep_slab(ds_sfc_t["2t"])
                else:
                    t_sfc = prep_slab(ds_ml_t.t.isel(model_level=-1))

                if "u10" in ds_sfc_t:
                    u_sfc = prep_slab(ds_sfc_t.u10)
                elif "10u" in ds_sfc_t:
                    u_sfc = prep_slab(ds_sfc_t["10u"])
                else:
                    u_sfc = prep_slab(ds_ml_t.u.isel(model_level=-1))

                if "v10" in ds_sfc_t:
                    v_sfc = prep_slab(ds_sfc_t.v10)
                elif "10v" in ds_sfc_t:
                    v_sfc = prep_slab(ds_sfc_t["10v"])
                else:
                    v_sfc = prep_slab(ds_ml_t.v.isel(model_level=-1))

                q_sfc = prep_slab(ds_ml_t.q.isel(model_level=-1))

                write_wps_slab(f, {**base_meta, "field": "TT", "units": "K", "xlvl": 200100.0, "slab": t_sfc})
                write_wps_slab(f, {**base_meta, "field": "UU", "units": "m s-1", "xlvl": 200100.0, "slab": u_sfc})
                write_wps_slab(f, {**base_meta, "field": "VV", "units": "m s-1", "xlvl": 200100.0, "slab": v_sfc})
                write_wps_slab(f, {**base_meta, "field": "SPECHUMD", "units": "kg kg-1", "xlvl": 200100.0, "slab": q_sfc})

                sp_arr = sp.squeeze().values
                if flip_lat:
                    sp_arr = np.flip(sp_arr, axis=0)

                def prep_3d(da_3d):
                    arr3 = da_3d.squeeze().values
                    if flip_lat:
                        arr3 = np.flip(arr3, axis=1)
                    return np.nan_to_num(arr3, nan=-1.0e30)

                t_3d = prep_3d(ds_ml_t.t)
                u_3d = prep_3d(ds_ml_t.u)
                v_3d = prep_3d(ds_ml_t.v)
                q_3d = prep_3d(ds_ml_t.q)

                for k in range(num_levels):
                    lvl_val = float(k + 1)
                    p_top = a_coeff[k] + b_coeff[k] * sp_arr
                    p_bot = a_coeff[k + 1] + b_coeff[k + 1] * sp_arr
                    p_rho = 0.5 * (p_top + p_bot)

                    write_wps_slab(
                        f, {**base_meta, "field": "PRESSURE", "units": "Pa", "xlvl": lvl_val, "slab": p_rho.astype(np.float32)}
                    )
                    write_wps_slab(f, {**base_meta, "field": "TT", "units": "K", "xlvl": lvl_val, "slab": t_3d[k]})
                    write_wps_slab(f, {**base_meta, "field": "UU", "units": "m s-1", "xlvl": lvl_val, "slab": u_3d[k]})
                    write_wps_slab(f, {**base_meta, "field": "VV", "units": "m s-1", "xlvl": lvl_val, "slab": v_3d[k]})
                    write_wps_slab(f, {**base_meta, "field": "SPECHUMD", "units": "kg kg-1", "xlvl": lvl_val, "slab": q_3d[k]})

                if ds_soil is not None:
                    soil_lats = ds_soil.latitude.values
                    soil_lons = ds_soil.longitude.values
                    soil_ny = len(soil_lats)
                    soil_nx = len(soil_lons)
                    soil_flip_lat = soil_lats[0] > soil_lats[1]

                    soil_meta = {
                        **base_meta,
                        "map_source": "ECMWF HRES Soil",
                        "nx": soil_nx,
                        "ny": soil_ny,
                        "startlat": float(soil_lats[-1] if soil_flip_lat else soil_lats[0]),
                        "startlon": float(soil_lons[0]),
                        "deltalat": float(abs(soil_lats[-1] - soil_lats[0]) / (soil_ny - 1)),
                        "deltalon": float(abs(soil_lons[-1] - soil_lons[0]) / (soil_nx - 1)),
                    }

                    def prep_soil(var_key):
                        s_arr = ds_soil[var_key].squeeze().values
                        if soil_flip_lat:
                            s_arr = np.flip(s_arr, axis=0)
                        return np.nan_to_num(s_arr, nan=-1.0e30)

                    soil_map = [
                        ("ST000007", "SOILTMP_soillevel0Msoillevel1", "K"),
                        ("ST007028", "SOILTMP_soillevel1Msoillevel2", "K"),
                        ("ST028100", "SOILTMP_soillevel2Msoillevel3", "K"),
                        ("ST100289", "SOILTMP_soillevel3Msoillevel4", "K"),
                        ("SM000007", "VSOILM_soillevel0Msoillevel1", "m3 m-3"),
                        ("SM007028", "VSOILM_soillevel1Msoillevel2", "m3 m-3"),
                        ("SM028100", "VSOILM_soillevel2Msoillevel3", "m3 m-3"),
                        ("SM100289", "VSOILM_soillevel3Msoillevel4", "m3 m-3"),
                    ]
                    for wps_name, soil_key, s_units in soil_map:
                        if soil_key in ds_soil:
                            write_wps_slab(
                                f,
                                {**soil_meta, "field": wps_name, "units": s_units, "xlvl": 200100.0, "slab": prep_soil(soil_key)},
                            )
                else:
                    for st_name in ["ST000007", "ST007028", "ST028100", "ST100289"]:
                        write_wps_slab(f, {**base_meta, "field": st_name, "units": "K", "xlvl": 200100.0, "slab": nc_skt["slab"]})
                    for sm_name in ["SM000007", "SM007028", "SM028100", "SM100289"]:
                        sm_const = np.full_like(lsm_arr, 0.25, dtype=np.float32)
                        write_wps_slab(f, {**base_meta, "field": sm_name, "units": "m3 m-3", "xlvl": 200100.0, "slab": sm_const})

            tmp_fp.replace(out_fp)
            generated_files.append(out_fp)
            print(f"Successfully generated {out_fp} ({os.path.getsize(out_fp) / (1024 * 1024):.2f} MB)")
    finally:
        ds_sfc.close()
        ds_ml.close()
        if ds_soil is not None:
            ds_soil.close()

    return generated_files
