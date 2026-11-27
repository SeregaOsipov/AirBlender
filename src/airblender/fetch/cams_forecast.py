"""
ECMWF CAMS Global Atmospheric Composition, Meteorology & GHG Forecasts Downloader
--------------------------------------------------------------------------------
Ported from ``download_ECMWF_CAMS_AC_Forecast_to_drive_ICON.py`` with lazy
optional imports (`cdsapi`, `cfgrib`) so offline compute nodes do not require them.

Architectural features:
1. Native GRIB retrieval (``data_format='grib'``) converted locally to CDM NetCDF
   via streaming ``cfgrib`` (``read_keys=['pv']``), preserving ``GRIB_pv`` and
   standardizing dimensions ``(forecast_period, forecast_reference_time, [model_level,] latitude, longitude)``.
2. Dense request packing under the 10,000-field MARS limit.
3. Concurrent ADS queue pipelining via ``ThreadPoolExecutor`` + atomic daily slicing
   into ``ECMWF-CAMS_{sfc,ml,ghg}_{YYYYMMDD}.nc``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any, Sequence
import zipfile

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
import yaml


_PRINT_LOCK = threading.Lock()
_NETCDF_LOCK = threading.Lock()
_MERGE_LOCK = threading.Lock()
_THREAD_LOCAL = threading.local()

SURFACE_VARS = [
    "surface_pressure",
    "mean_sea_level_pressure",
    "surface_geopotential",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "sea_surface_temperature",
    "sea_ice_cover",
    "skin_temperature",
    "snow_depth",
    "snow_albedo",
    "land_sea_mask",
    "skin_reservoir_content",
]

UPPER_AIR_VARS = [
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "temperature",
    "specific_humidity",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
]

CHEM_GROUP_A = [
    "dust_aerosol_0.03-0.55um_mixing_ratio",
    "dust_aerosol_0.55-0.9um_mixing_ratio",
    "dust_aerosol_0.9-20um_mixing_ratio",
    "sea_salt_aerosol_0.03-0.5um_mixing_ratio",
    "sea_salt_aerosol_0.5-5um_mixing_ratio",
    "sea_salt_aerosol_5-20um_mixing_ratio",
    "hydrophilic_organic_matter_aerosol_mixing_ratio",
    "hydrophobic_organic_matter_aerosol_mixing_ratio",
    "hydrophilic_black_carbon_aerosol_mixing_ratio",
    "hydrophobic_black_carbon_aerosol_mixing_ratio",
    "sulphate_aerosol_mixing_ratio",
    "ammonium",
    "nitrate",
    "ozone",
    "carbon_monoxide",
    "nitrogen_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "nitric_acid",
    "hydrogen_peroxide",
    "ammonia",
    "dinitrogen_pentoxide",
    "pernitric_acid",
    "nitrate_radical",
]

CHEM_GROUP_B = [
    "hydroxyl_radical",
    "hydroperoxy_radical",
    "peroxyacetyl_nitrate",
    "peroxy_acetyl_radical",
    "organic_nitrates",
    "methane",
    "ethane",
    "propane",
    "paraffins",
    "ethene",
    "propene",
    "olefins",
    "isoprene",
    "terpenes",
    "formaldehyde",
    "aldehydes",
    "acetone",
    "ethanol",
    "methanol",
    "formic_acid",
    "methacrylic_acid",
    "methyl_glyoxal",
    "methacrolein_mvk",
    "methyl_peroxide",
]

CHEM_VARS = CHEM_GROUP_A + CHEM_GROUP_B

GHG_VARS = ["carbon_dioxide", "methane"]


def _get_chem_groups_for_date(date_dash: str) -> list[tuple[str, list[str]]]:
    """Return balanced 24-variable 3D chemistry/aerosol request groups adjusted for CAMS Cy49r1 rename."""
    group_a = list(CHEM_GROUP_A)
    if str(date_dash) < "2026-05-12":
        group_a = [
            "sea_salt_aerosol_0.03-0.55um_mixing_ratio"
            if v == "sea_salt_aerosol_0.03-0.5um_mixing_ratio"
            else v
            for v in group_a
        ]
    return [("chema", group_a), ("chemb", list(CHEM_GROUP_B))]


def log_msg(msg: str) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _PRINT_LOCK:
        print(f"[{ts}] {msg}", flush=True)


def get_thread_cds_client():
    import cdsapi

    if not hasattr(_THREAD_LOCAL, "client") or _THREAD_LOCAL.client is None:
        _THREAD_LOCAL.client = cdsapi.Client(progress=sys.stdout.isatty())
    return _THREAD_LOCAL.client


def reset_thread_cds_client():
    import cdsapi

    _THREAD_LOCAL.client = cdsapi.Client(progress=sys.stdout.isatty())
    return _THREAD_LOCAL.client


def build_arg_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="airblender fetch-cams",
            description="Download ECMWF CAMS Atmospheric Composition Forecasts/Analyses with GRIB pipelining.",
        )
    parser.add_argument("--mode", "--port", "--host", help="Ignored PyCharm debugger flags")
    parser.add_argument("--start_date", help="Start base date (YYYY-MM-DD)", required=True)
    parser.add_argument("--end_date", help="End base date (YYYY-MM-DD)", required=True)
    parser.add_argument("--data_root_folder", required=True, help="Target output directory for merged NetCDFs")
    parser.add_argument("--config", default=None, help="Optional YAML configuration file path")
    parser.add_argument(
        "--format",
        choices=["grib", "netcdf_zip"],
        default=None,
        help="Native ADS retrieval format (default: grib)",
    )
    parser.add_argument(
        "--max_fields",
        type=int,
        default=None,
        help="MARS request cost limit in fields (default: 10000)",
    )
    parser.add_argument("--sfc_lt_chunk", type=int, default=None, help="Max lead times per 2D Surface request")
    parser.add_argument("--met_lt_chunk", type=int, default=None, help="Max lead times per 3D Met request")
    parser.add_argument("--chem_lt_chunk", type=int, default=None, help="Max lead times per 3D Chem request")
    parser.add_argument("--ghg_lt_chunk", type=int, default=None, help="Max lead times per 3D GHG request")
    parser.add_argument(
        "--ghg",
        dest="ghg",
        action="store_true",
        default=None,
        help="Also download 3D CO2 & CH4 from cams-global-greenhouse-gas-forecasts (0.1 deg)",
    )
    parser.add_argument(
        "--no_ghg",
        dest="ghg",
        action="store_false",
        help="Disable GHG stream even if enabled in config",
    )
    parser.add_argument(
        "--ghg_only",
        action="store_true",
        help="Download/merge only 3D GHG fields (ECMWF-CAMS_ghg_YYYYMMDD.nc)",
    )
    parser.add_argument(
        "--level_chunks",
        type=int,
        default=None,
        help="Number of vertical chunks to split the 137 model levels into (default: 1)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of concurrent ThreadPoolExecutor workers for ADS queue pipelining (default: 5)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=None,
        help="Max retries with exponential backoff per task for transient errors (default: 3)",
    )
    parser.add_argument("--list_tasks", action="store_true", help="Print chunking table and total tasks, then exit")
    parser.add_argument(
        "--task_index",
        type=int,
        default=None,
        help="Execute a single task index (0 to N-1) for Slurm array execution",
    )
    parser.add_argument("--sfc_only", action="store_true", help="Download/update only 2D surface fields")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files or markers exist")
    parser.add_argument(
        "--type",
        choices=["analysis", "forecast"],
        default=None,
        help="Dataset type: 'analysis' or 'forecast' (default: analysis)",
    )
    parser.add_argument(
        "--time",
        default=None,
        help="Base initialization time, e.g. '00:00' or '12:00' (default: 00:00)",
    )
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        default=None,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        help="Geographic bounding box [North, West, South, East] in degrees (default: global)",
    )
    parser.add_argument(
        "--leadtimes",
        nargs="*",
        default=None,
        help="Lead times in hours (e.g. 0 3 6 ... 120). For 'forecast', defaults to 0..120 step 3.",
    )
    parser.add_argument(
        "--no_split_days",
        action="store_true",
        help="Do not split multi-day forecasts into daily files",
    )
    parser.add_argument(
        "--stream_daily",
        action="store_true",
        default=None,
        help="Order ADS requests day-by-day (0..24h first) and incrementally flush daily NetCDFs as each horizon completes",
    )
    parser.add_argument(
        "--segment_hours",
        type=int,
        default=None,
        help="Forecast horizon bucket size in hours when --stream_daily is enabled (default: 24)",
    )
    parser.add_argument(
        "--status_file",
        default=None,
        help="Optional path to atomic JSON live status file written throughout execution",
    )
    return parser


def apply_config_and_defaults(args: argparse.Namespace) -> None:
    args.data_root_folder = str(args.data_root_folder)
    if not hasattr(args, "status_file"):
        args.status_file = None
    cfg_data: dict[str, Any] = {}
    if args.config and os.path.exists(str(args.config)):
        with open(str(args.config), "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}
        cfg_data = raw_cfg.get("data", raw_cfg)

    def pick(cli_val, cfg_key, default_val):
        if cli_val is not None:
            return cli_val
        if cfg_key in cfg_data and cfg_data[cfg_key] is not None:
            return cfg_data[cfg_key]
        return default_val

    args.type = pick(args.type, "cams_type", "analysis")
    args.time = pick(args.time, "cams_time", "00:00")
    args.format = pick(args.format, "cams_format", "grib")
    args.max_fields = int(pick(args.max_fields, "cams_max_fields", 10000))
    args.sfc_lt_chunk = pick(args.sfc_lt_chunk, "cams_sfc_lt_chunk", None)
    args.met_lt_chunk = pick(args.met_lt_chunk, "cams_met_lt_chunk", None)
    args.chem_lt_chunk = pick(args.chem_lt_chunk, "cams_chem_lt_chunk", None)
    args.ghg_lt_chunk = pick(args.ghg_lt_chunk, "cams_ghg_lt_chunk", None)
    args.ghg = bool(pick(args.ghg, "cams_ghg", False))
    args.level_chunks = int(pick(args.level_chunks, "cams_level_chunks", 1))
    args.max_workers = int(pick(args.max_workers, "cams_max_workers", 5))
    args.max_retries = int(pick(args.max_retries, "cams_max_retries", 3))
    args.stream_daily = bool(pick(getattr(args, "stream_daily", None), "cams_stream_daily", False))
    args.segment_hours = max(3, int(pick(getattr(args, "segment_hours", None), "cams_segment_hours", 24)))
    if args.leadtimes is None and "cams_leadtimes" in cfg_data and cfg_data["cams_leadtimes"] is not None:
        lt_cfg = cfg_data["cams_leadtimes"]
        args.leadtimes = [str(x) for x in lt_cfg] if isinstance(lt_cfg, list) else [str(lt_cfg)]
    if args.area is None:
        raw_area = cfg_data.get("cams_area") or cfg_data.get("area")
        if raw_area is not None and isinstance(raw_area, (list, tuple)) and len(raw_area) == 4:
            args.area = [float(x) for x in raw_area]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    apply_config_and_defaults(args)
    return args


def resolve_time_and_leadtimes(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if isinstance(args.time, list):
        times = [str(t).strip() for t in args.time if str(t).strip()]
    elif "," in str(args.time):
        times = [t.strip() for t in str(args.time).split(",") if t.strip()]
    else:
        times = [str(args.time).strip()]

    if args.type == "analysis":
        leadtimes = ["0"]
    else:
        if args.leadtimes is not None:
            raw_leadtimes = []
            for item in args.leadtimes:
                for part in str(item).split(","):
                    part = part.strip()
                    if part:
                        raw_leadtimes.append(part)
            leadtimes = raw_leadtimes if raw_leadtimes else [str(h) for h in range(0, 121, 3)]
        else:
            leadtimes = [str(h) for h in range(0, 121, 3)]

    return times, leadtimes


def compute_stream_chunking(
    num_vars: int,
    levels_list: list[str] | None,
    num_times: int,
    leadtimes: list[str],
    max_fields: int = 10000,
    explicit_lt_chunk: int | None = None,
    explicit_level_chunks: int = 1,
):
    """
    Compute `(level_chunks, lt_chunks, fields_per_lt, max_lt_per_req)` so each request satisfies:
        num_vars * len(level_chunk) * len(lt_chunk) * num_times <= max_fields
    """
    total_levels = len(levels_list) if levels_list else 1
    num_lvl_chunks = max(1, int(explicit_level_chunks or 1))

    while True:
        lvl_chunk_size = math.ceil(total_levels / num_lvl_chunks)
        fields_per_lt = num_vars * lvl_chunk_size * num_times
        if fields_per_lt <= max_fields or lvl_chunk_size == 1:
            break
        num_lvl_chunks += 1

    if levels_list:
        lvl_chunk_size = math.ceil(total_levels / num_lvl_chunks)
        level_chunks = [levels_list[i : i + lvl_chunk_size] for i in range(0, total_levels, lvl_chunk_size)]
    else:
        level_chunks = [None]

    max_lt_per_req = max(1, max_fields // fields_per_lt)
    if explicit_lt_chunk is not None and int(explicit_lt_chunk) > 0:
        lt_chunk_size = min(int(explicit_lt_chunk), max_lt_per_req)
    else:
        lt_chunk_size = min(len(leadtimes), max_lt_per_req)

    lt_chunks = [leadtimes[i : i + lt_chunk_size] for i in range(0, len(leadtimes), lt_chunk_size)]
    return level_chunks, lt_chunks, fields_per_lt, max_lt_per_req


def _standardize_cfgrib_ds(ds: xr.Dataset, is_3d: bool = False) -> xr.Dataset:
    for dim in list(ds.dims):
        if dim not in ("time", "step", "hybrid", "latitude", "longitude"):
            if ds.sizes[dim] == 1:
                ds = ds.squeeze(dim, drop=True)
    for coord in list(ds.coords):
        if coord not in ("time", "step", "hybrid", "latitude", "longitude", "valid_time"):
            ds = ds.drop_vars(coord, errors="ignore")

    rename_map = {}
    if "step" in ds.dims or "step" in ds.coords:
        rename_map["step"] = "forecast_period"
    if "time" in ds.dims or "time" in ds.coords:
        rename_map["time"] = "forecast_reference_time"
    if is_3d and ("hybrid" in ds.dims or "hybrid" in ds.coords):
        rename_map["hybrid"] = "model_level"
    if rename_map:
        ds = ds.rename(rename_map)

    if "forecast_period" in ds.dims and ds.sizes["forecast_period"] > 1:
        ds = ds.sortby("forecast_period")
    if is_3d and "model_level" in ds.dims and ds.sizes["model_level"] > 1:
        ds = ds.sortby("model_level")
    return ds


def convert_grib_to_cdm_nc(grib_path: str, nc_path: str, is_3d: bool = False) -> None:
    """
    Streaming, memory-bounded (< 300 MB RAM) converter from CAMS GRIB to CDM NetCDF:
      - Reuses a single on-disk GRIB index (.idx) across all paramIds.
      - Streams one variable (paramId) and one forecast_period slice at a time.
      - Preserves GRIB_pv hybrid vertical coordinate coefficients on 3D variables.
    """
    import cfgrib
    import cfgrib.cfmessage
    import cfgrib.dataset
    import cfgrib.messages
    import cfgrib.xarray_store

    with _NETCDF_LOCK:
        idx_path = f"{grib_path}.{os.getpid()}.idx"
        tmp_nc = f"{nc_path}.tmp.{os.getpid()}.{threading.get_ident()}"
        for stale in (idx_path, tmp_nc):
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass

        try:
            stream = cfgrib.messages.FileStream(grib_path, errors="ignore")
            index = cfgrib.dataset.open_fileindex(
                stream, computed_keys=cfgrib.cfmessage.COMPUTED_KEYS, indexpath=idx_path
            )
            param_ids = sorted(index["paramId"])
            if not param_ids:
                raise RuntimeError(f"cfgrib found 0 paramIds in {grib_path}")
            lvl_vals = index.get("level:float", index.get("level", []))
            has_multi_levels = len(lvl_vals) > 1

            base_bks: dict[str, Any] = {"indexpath": idx_path, "squeeze": False, "errors": "ignore"}
            if is_3d:
                base_bks["read_keys"] = ["pv"]

            header_written = False
            nc_out = None

            try:
                for param_id in param_ids:
                    bk = base_bks.copy()
                    bk["filter_by_keys"] = {"paramId": param_id}
                    sub_datasets = cfgrib.xarray_store.raw_open_datasets(grib_path, bk)

                    for raw_ds in sub_datasets:
                        try:
                            ds = _standardize_cfgrib_ds(raw_ds, is_3d=is_3d)
                            if is_3d and has_multi_levels and ds.sizes.get("model_level", 0) <= 1:
                                continue
                            target_order = ["forecast_period", "forecast_reference_time"]
                            if is_3d and "model_level" in ds.dims:
                                target_order.append("model_level")
                            for d in ds.dims:
                                if d not in target_order:
                                    target_order.append(d)
                            target_dims = tuple(d for d in target_order if d in ds.dims)

                            if not header_written:
                                ds_coords = xr.Dataset(coords=ds.coords, attrs=ds.attrs)
                                ds_coords.to_netcdf(tmp_nc, mode="w")
                                ds_coords.close()
                                nc_out = netCDF4.Dataset(tmp_nc, mode="a")
                                header_written = True

                            for var_name, da in ds.data_vars.items():
                                if var_name in nc_out.variables:
                                    continue

                                nc_var = nc_out.createVariable(
                                    var_name, "f4", target_dims, fill_value=np.float32(np.nan)
                                )
                                for attr_k, attr_v in da.attrs.items():
                                    if attr_k in ("_FillValue", "missing_value"):
                                        continue
                                    nc_var.setncattr(attr_k, attr_v)
                                if "valid_time" in ds.coords:
                                    nc_var.setncattr("coordinates", "valid_time")

                                if "forecast_period" in target_dims:
                                    sub_dims = tuple(d for d in target_dims if d != "forecast_period")
                                    n_fp = ds.sizes["forecast_period"]
                                    for t_idx in range(n_fp):
                                        da_slice = da.isel(forecast_period=t_idx).transpose(*sub_dims)
                                        nc_var[t_idx, ...] = np.asarray(da_slice.values, dtype=np.float32)
                                else:
                                    da_T = da.transpose(*target_dims)
                                    nc_var[...] = np.asarray(da_T.values, dtype=np.float32)
                        finally:
                            try:
                                raw_ds.close()
                            except Exception:
                                pass
            finally:
                if nc_out is not None:
                    nc_out.close()

            if not header_written:
                raise RuntimeError(f"cfgrib decoded 0 variable datasets from {grib_path}")

            os.replace(tmp_nc, nc_path)
        finally:
            for cleanup_fp in (idx_path, tmp_nc):
                if os.path.exists(cleanup_fp):
                    try:
                        os.remove(cleanup_fp)
                    except OSError:
                        pass


def extract_nc_from_zip_to_target(zip_path: str, target_nc_path: str, extract_subdir: str) -> None:
    os.makedirs(extract_subdir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_subdir)
    nc_files = [os.path.join(extract_subdir, f) for f in os.listdir(extract_subdir) if f.endswith(".nc")]
    if not nc_files:
        raise RuntimeError(f"No .nc files found in {zip_path}")
    if len(nc_files) == 1:
        shutil.move(nc_files[0], target_nc_path)
    else:
        with _NETCDF_LOCK:
            ds_list = [xr.open_dataset(f) for f in nc_files]
            merged = xr.merge(ds_list, compat="override")
            tmp_nc = f"{target_nc_path}.tmp"
            merged.to_netcdf(tmp_nc)
            for ds in ds_list:
                ds.close()
            merged.close()
            os.replace(tmp_nc, target_nc_path)
    shutil.rmtree(extract_subdir, ignore_errors=True)


def _write_ds_streaming(ds_out: xr.Dataset, target_fp: str) -> None:
    """
    Stream an xarray Dataset to NetCDF one variable and one `forecast_period`
    slice at a time (< 250 MB peak RAM).
    """
    tmp_fp = f"{target_fp}.tmp.{os.getpid()}"
    if os.path.exists(tmp_fp):
        try:
            os.remove(tmp_fp)
        except OSError:
            pass
    try:
        ds_coords = xr.Dataset(coords=ds_out.coords, attrs=ds_out.attrs).compute()
        ds_coords.to_netcdf(tmp_fp, mode="w")
        ds_coords.close()

        nc_out = netCDF4.Dataset(tmp_fp, mode="a")
        try:
            for var_name, da in ds_out.data_vars.items():
                if var_name in nc_out.variables:
                    continue
                dims = da.dims
                nc_var = nc_out.createVariable(var_name, "f4", dims, fill_value=np.float32(np.nan))
                for attr_k, attr_v in da.attrs.items():
                    if attr_k in ("_FillValue", "missing_value"):
                        continue
                    nc_var.setncattr(attr_k, attr_v)
                if "valid_time" in ds_out.coords and "coordinates" not in da.attrs:
                    nc_var.setncattr("coordinates", "valid_time")

                if "forecast_period" in dims and dims[0] == "forecast_period":
                    n_fp = da.sizes["forecast_period"]
                    for t_idx in range(n_fp):
                        nc_var[t_idx, ...] = np.asarray(da.isel(forecast_period=t_idx).values, dtype=np.float32)
                else:
                    nc_var[...] = np.asarray(da.values, dtype=np.float32)
        finally:
            nc_out.close()

        os.replace(tmp_fp, target_fp)
    finally:
        if os.path.exists(tmp_fp):
            try:
                os.remove(tmp_fp)
            except OSError:
                pass


def _nc_matches_reference_date(fp: str, expected_ref_date: str | None) -> bool:
    if expected_ref_date is None:
        return True
    try:
        with netCDF4.Dataset(fp, "r") as nc_chk:
            if "forecast_reference_time" not in nc_chk.variables:
                return True
            ref_var = nc_chk.variables["forecast_reference_time"]
            units = getattr(ref_var, "units", None)
            if not units:
                return False
            decoded = netCDF4.num2date(
                ref_var[:],
                units=units,
                calendar=getattr(ref_var, "calendar", "standard"),
            )
            first_ref = np.asarray(decoded).reshape(-1)[0]
            return first_ref.strftime("%Y%m%d") == expected_ref_date
    except Exception:
        return False


def _existing_nc_steps(fp: str, time_dim: str | None, expected_ref_date: str | None = None) -> int:
    if not os.path.exists(fp):
        return 0
    if expected_ref_date is not None and not _nc_matches_reference_date(fp, expected_ref_date):
        return 0
    try:
        with netCDF4.Dataset(fp, "r") as nc_chk:
            if time_dim and time_dim in nc_chk.dimensions:
                return len(nc_chk.dimensions[time_dim])
            return 1
    except Exception:
        return 0


def save_dataset_daily_and_base(
    ds: xr.Dataset,
    stream: str,
    date_str: str,
    data_root_folder: str | Path,
    no_split_days: bool = False,
    skip_if_steps_ge: bool = False,
) -> None:
    """
    Save `ds` into `data_root_folder` as `ECMWF-CAMS_{stream}_{YYYYMMDD}.nc` daily slices
    (plus base `ECMWF-CAMS_{stream}_{date_str}.nc` if not yet written).
    """
    root = str(data_root_folder)
    os.makedirs(root, exist_ok=True)
    valid_times = None
    if "valid_time" in ds.coords:
        vt = np.asarray(ds.valid_time.values)
        valid_times = pd.to_datetime(vt.flatten() if vt.ndim > 1 else vt)
    elif "time" in ds.coords:
        valid_times = pd.to_datetime(np.asarray(ds.time.values))

    if valid_times is not None:
        unique_days = sorted(list(set(valid_times.strftime("%Y%m%d"))))
    else:
        unique_days = [date_str]

    time_dim = "forecast_period" if "forecast_period" in ds.dims else ("time" if "time" in ds.dims else None)
    expected_ref = (
        date_str
        if (
            "forecast_reference_time" in ds.coords
            or "forecast_reference_time" in ds.dims
            or "forecast_reference_time" in ds.variables
        )
        else None
    )
    if no_split_days or len(unique_days) <= 1:
        out_fp = os.path.join(root, f"ECMWF-CAMS_{stream}_{date_str}.nc")
        n_steps = ds.sizes.get(time_dim, 1) if time_dim else 1
        if skip_if_steps_ge and _existing_nc_steps(out_fp, time_dim, expected_ref_date=expected_ref) >= n_steps:
            return
        _write_ds_streaming(ds, out_fp)
        log_msg(f"[SAVED] {out_fp} ({os.path.getsize(out_fp) / (1024**2):.2f} MB)")
    else:
        for day in unique_days:
            out_fp = os.path.join(root, f"ECMWF-CAMS_{stream}_{day}.nc")
            if time_dim:
                mask = valid_times.strftime("%Y%m%d") == day
                ds_day = ds.isel({time_dim: mask})
            else:
                ds_day = ds
            n_steps = ds_day.sizes.get(time_dim, 1) if time_dim else 1
            if skip_if_steps_ge and _existing_nc_steps(out_fp, time_dim, expected_ref_date=expected_ref) >= n_steps:
                continue
            _write_ds_streaming(ds_day, out_fp)
            log_msg(f"[SAVED-DAILY] {day} ({n_steps} timesteps) -> {out_fp} ({os.path.getsize(out_fp) / (1024**2):.2f} MB)")

        base_fp = os.path.join(root, f"ECMWF-CAMS_{stream}_{date_str}.nc")
        if not os.path.exists(base_fp) or (expected_ref and not _nc_matches_reference_date(base_fp, expected_ref)):
            _write_ds_streaming(ds, base_fp)
            log_msg(f"[SAVED-BASE] {base_fp}")


def _build_horizon_buckets(leadtimes: list[str], segment_hours: int) -> list[tuple[int, list[str]]]:
    """Partition sorted forecast lead times into cumulative ``segment_hours`` buckets."""
    sorted_lts = sorted(leadtimes, key=lambda x: int(x))
    max_lt = max(int(lt) for lt in sorted_lts)
    seg_h = max(3, int(segment_hours))
    buckets: list[tuple[int, list[str]]] = []
    prev_h = -1
    cur_h = seg_h
    while True:
        bound_h = min(max_lt, cur_h)
        b_lts = [lt for lt in sorted_lts if prev_h < int(lt) <= bound_h]
        if b_lts:
            buckets.append((bound_h, b_lts))
        if bound_h >= max_lt:
            break
        prev_h = bound_h
        cur_h += seg_h
    return buckets


def _make_sfc_task(
    *,
    date_str: str,
    date_dash: str,
    base_name: str,
    lt_chunk: list[str],
    times: list[str],
    args: argparse.Namespace,
    date_spec: dict,
    temp_raw_dir: str,
    temp_extract_dir: str,
    ac_collection: str,
) -> tuple[dict, str, str]:
    ext = "grib" if args.format == "grib" else "zip"
    raw_fp = os.path.join(temp_raw_dir, f"{base_name}.{ext}")
    nc_fp = os.path.join(temp_extract_dir, f"{base_name}.nc")
    marker_fp = os.path.join(temp_raw_dir, f"{base_name}.done")
    lt_param = lt_chunk if args.type == "forecast" else "0"
    num_fields = len(SURFACE_VARS) * len(lt_chunk) * len(times)
    req: dict[str, Any] = {
        "variable": SURFACE_VARS,
        "date": f"{date_dash}/{date_dash}",
        "time": times,
        "data_format": args.format,
        "type": args.type,
        "leadtime_hour": lt_param,
    }
    if getattr(args, "area", None):
        req["area"] = [float(x) for x in args.area]
    task = {
        "name": f"{date_str}:{base_name}",
        "collection": ac_collection,
        "date_str": date_str,
        "date_spec": date_spec,
        "temp_raw_dir": temp_raw_dir,
        "temp_extract_dir": temp_extract_dir,
        "request": req,
        "output_raw": raw_fp,
        "output_nc": nc_fp,
        "marker": marker_fp,
        "format": args.format,
        "is_3d": False,
        "num_fields": num_fields,
        "lt_chunk": lt_chunk,
    }
    return task, marker_fp, nc_fp


def _make_3d_task(
    *,
    collection: str,
    variables: list[str],
    date_str: str,
    date_dash: str,
    base_name: str,
    lvl_chunk: list[str],
    lt_chunk: list[str],
    times: list[str] | None,
    include_type: bool,
    args: argparse.Namespace,
    date_spec: dict,
    temp_raw_dir: str,
    temp_extract_dir: str,
) -> tuple[dict, str, str]:
    ext = "grib" if args.format == "grib" else "zip"
    raw_fp = os.path.join(temp_raw_dir, f"{base_name}.{ext}")
    nc_fp = os.path.join(temp_extract_dir, f"{base_name}.nc")
    marker_fp = os.path.join(temp_raw_dir, f"{base_name}.done")
    lt_param = lt_chunk if args.type == "forecast" else "0"
    num_times = len(times) if times else 1
    num_fields = len(variables) * len(lvl_chunk) * len(lt_chunk) * num_times
    req: dict[str, Any] = {
        "variable": variables,
        "model_level": lvl_chunk,
        "date": f"{date_dash}/{date_dash}",
        "data_format": args.format,
        "leadtime_hour": lt_param,
    }
    if times is not None:
        req["time"] = times
    if include_type:
        req["type"] = args.type
    if getattr(args, "area", None):
        req["area"] = [float(x) for x in args.area]
    task = {
        "name": f"{date_str}:{base_name}",
        "collection": collection,
        "date_str": date_str,
        "date_spec": date_spec,
        "temp_raw_dir": temp_raw_dir,
        "temp_extract_dir": temp_extract_dir,
        "request": req,
        "output_raw": raw_fp,
        "output_nc": nc_fp,
        "marker": marker_fp,
        "format": args.format,
        "is_3d": True,
        "num_fields": num_fields,
        "lt_chunk": lt_chunk,
    }
    return task, marker_fp, nc_fp


def generate_tasks(args: argparse.Namespace) -> list[dict]:
    dates = pd.period_range(start=args.start_date, end=args.end_date, freq="D")
    times, leadtimes = resolve_time_and_leadtimes(args)
    num_times = len(times)
    stream_daily = bool(getattr(args, "stream_daily", False) and args.type == "forecast")
    segment_hours = max(3, int(getattr(args, "segment_hours", 24)))
    sorted_lts = sorted(leadtimes, key=lambda x: int(x)) if args.type == "forecast" else leadtimes

    all_levels = [str(l) for l in range(1, 138)]

    effective_ghg = bool(getattr(args, "ghg", False) or getattr(args, "ghg_only", False))
    if effective_ghg:
        if args.type == "analysis" and not getattr(args, "ghg_only", False):
            log_msg(
                "[GHG-SKIP] cams-global-greenhouse-gas-forecasts is forecast-only (00:00 UTC); "
                "skipping GHG stream for type='analysis'."
            )
            effective_ghg = False
        elif "00:00" not in times:
            raise ValueError("cams-global-greenhouse-gas-forecasts only supports 00:00 UTC base time.")

    _, sfc_lt_chunks, sfc_f_per_lt, sfc_max_lt = compute_stream_chunking(
        len(SURFACE_VARS),
        None,
        num_times,
        sorted_lts,
        max_fields=args.max_fields,
        explicit_lt_chunk=args.sfc_lt_chunk,
        explicit_level_chunks=1,
    )
    met_lvl_chunks, met_lt_chunks, met_f_per_lt, met_max_lt = compute_stream_chunking(
        len(UPPER_AIR_VARS),
        all_levels,
        num_times,
        sorted_lts,
        max_fields=args.max_fields,
        explicit_lt_chunk=None if stream_daily else args.met_lt_chunk,
        explicit_level_chunks=args.level_chunks,
    )
    chem_groups_sample = _get_chem_groups_for_date(dates[0].to_timestamp().strftime("%Y-%m-%d") if len(dates) else "2026-11-22")
    chem_group_chunks = [
        compute_stream_chunking(
            len(g_vars),
            all_levels,
            num_times,
            sorted_lts,
            max_fields=args.max_fields,
            explicit_lt_chunk=None if stream_daily else args.chem_lt_chunk,
            explicit_level_chunks=args.level_chunks,
        )
        for _, g_vars in chem_groups_sample
    ]

    horizons_def = _build_horizon_buckets(sorted_lts, segment_hours) if stream_daily else []
    chem_reqs_total = sum(len(gc[1]) * len(gc[0]) for gc in chem_group_chunks)
    chem_max_fields = max(len(gc[1][0]) * gc[2] for gc in chem_group_chunks)
    plan_lines = [
        f"Chunking Plan (format={args.format}, max_fields={args.max_fields}, total_leadtimes={len(sorted_lts)}, "
        f"stream_daily={stream_daily}"
        + (f", buckets={len(horizons_def)}x{segment_hours}h):" if stream_daily else "):"),
        f"  - Regional Area: {getattr(args, 'area', 'global')}",
        f"  - Surface (2D): {len(SURFACE_VARS)} vars x 1 lvl = {sfc_f_per_lt} fields/lt "
        f"(max {sfc_max_lt} lts/req) -> {len(sfc_lt_chunks)} request(s), "
        f"max {len(sfc_lt_chunks[0]) * sfc_f_per_lt} fields/req",
        f"  - 3D Met      : {len(UPPER_AIR_VARS)} vars x {len(met_lvl_chunks[0])} lvls = {met_f_per_lt} fields/lt "
        f"(max {met_max_lt} lts/req) -> {len(met_lt_chunks) * len(met_lvl_chunks)} request(s), "
        f"max {len(met_lt_chunks[0]) * met_f_per_lt} fields/req",
        f"  - 3D Chem (A/B): {len(chem_groups_sample)}x{len(chem_groups_sample[0][1])} vars x {len(chem_group_chunks[0][0][0])} lvls "
        f"= {chem_group_chunks[0][2]} fields/lt (max {chem_group_chunks[0][3]} lts/req) -> {chem_reqs_total} request(s), "
        f"max {chem_max_fields} fields/req",
    ]
    ghg_lvl_chunks, ghg_lt_chunks = [], []
    if effective_ghg:
        ghg_lvl_chunks, ghg_lt_chunks, ghg_f_per_lt, ghg_max_lt = compute_stream_chunking(
            len(GHG_VARS),
            all_levels,
            1,
            sorted_lts,
            max_fields=args.max_fields,
            explicit_lt_chunk=None if stream_daily else args.ghg_lt_chunk,
            explicit_level_chunks=args.level_chunks,
        )
        plan_lines.append(
            f"  - 3D GHG (0.1): {len(GHG_VARS)} vars x {len(ghg_lvl_chunks[0])} lvls = {ghg_f_per_lt} fields/lt "
            f"(max {ghg_max_lt} lts/req) -> {len(ghg_lt_chunks) * len(ghg_lvl_chunks)} request(s), "
            f"max {len(ghg_lt_chunks[0]) * ghg_f_per_lt} fields/req"
        )
    log_msg("\n".join(plan_lines))

    tasks: list[dict] = []
    date_specs: dict[str, dict] = {}
    ac_collection = "cams-global-atmospheric-composition-forecasts"
    ghg_collection = "cams-global-greenhouse-gas-forecasts"

    for raw_date in dates:
        date = raw_date.to_timestamp()
        date_str = date.strftime("%Y%m%d")
        date_dash = date.strftime("%Y-%m-%d")
        chem_groups = _get_chem_groups_for_date(date_dash)
        chem_lvls_for_date = [
            compute_stream_chunking(
                len(g_vars),
                all_levels,
                num_times,
                sorted_lts,
                max_fields=args.max_fields,
                explicit_lt_chunk=None if stream_daily else args.chem_lt_chunk,
                explicit_level_chunks=args.level_chunks,
            )
            for _, g_vars in chem_groups
        ]

        if args.type == "forecast" and not getattr(args, "no_split_days", False):
            max_lt = max(int(lt) for lt in sorted_lts)
            expected_days = [(date + pd.Timedelta(hours=h)).strftime("%Y%m%d") for h in range(0, max_lt + 1, 24)]
        else:
            expected_days = [date_str]

        def _stream_done(stream_name: str) -> bool:
            if str(args.type).lower() != "forecast":
                return all(
                    os.path.exists(os.path.join(args.data_root_folder, f"ECMWF-CAMS_{stream_name}_{d}.nc"))
                    for d in expected_days
                )
            for idx, d in enumerate(expected_days):
                fp = os.path.join(args.data_root_folder, f"ECMWF-CAMS_{stream_name}_{d}.nc")
                min_steps = 8 if (len(expected_days) > 1 and idx < len(expected_days) - 1) else 1
                if _existing_nc_steps(fp, "forecast_period", expected_ref_date=date_str) < min_steps:
                    return False
            return True

        sfc_done = _stream_done("sfc")
        ml_done = _stream_done("ml")
        ghg_done = _stream_done("ghg")

        ghg_only = bool(getattr(args, "ghg_only", False))
        need_sfc = (not ghg_only) and (args.force or not sfc_done)
        need_ml = (not args.sfc_only) and (not ghg_only) and (args.force or not ml_done)
        need_ghg = effective_ghg and (not args.sfc_only) and (args.force or not ghg_done)

        if not (need_sfc or need_ml or need_ghg):
            log_msg(f"Skipping {date_str}, final merged daily NetCDF files already exist.")
            if stream_daily:
                os.makedirs(args.data_root_folder, exist_ok=True)
                for bound_h, _ in horizons_def:
                    rmarker = os.path.join(args.data_root_folder, f".cams_ready_{date_str}_h{bound_h:03d}.done")
                    if not os.path.exists(rmarker):
                        with open(rmarker, "w", encoding="utf-8") as mf:
                            mf.write("DONE\n")
            continue

        temp_raw_dir = os.path.join(args.data_root_folder, f"tmp_gribs_{date_str}")
        temp_extract_dir = os.path.join(args.data_root_folder, f"tmp_extract_{date_str}")

        chem_groups_nc_by_lvl = [
            [[] for _ in range(len(gc[0]))] if need_ml else []
            for gc in chem_lvls_for_date
        ]
        date_spec = {
            "date_str": date_str,
            "temp_raw_dir": temp_raw_dir,
            "temp_extract_dir": temp_extract_dir,
            "all_markers": [],
            "need_sfc": need_sfc,
            "need_ml": need_ml,
            "need_ghg": need_ghg,
            "sfc_nc_files": [],
            "met_nc_by_lvl": [[] for _ in range(len(met_lvl_chunks))] if need_ml else [],
            "chem_groups_nc_by_lvl": chem_groups_nc_by_lvl,
            "ghg_nc_by_lvl": [[] for _ in range(len(ghg_lvl_chunks))] if need_ghg else [],
            "horizons": [],
        }
        date_specs[date_str] = date_spec

        if stream_daily:
            cum_markers: list[str] = []
            cum_met_by_lvl: list[list[str]] = [[] for _ in range(len(met_lvl_chunks))] if need_ml else []
            cum_chem_groups_by_lvl: list[list[list[str]]] = [
                [[] for _ in range(len(gc[0]))] if need_ml else []
                for gc in chem_lvls_for_date
            ]
            cum_ghg_by_lvl: list[list[str]] = [[] for _ in range(len(ghg_lvl_chunks))] if need_ghg else []

            for b_idx, (bound_h, b_lts) in enumerate(horizons_def):
                if b_idx == 0 and need_sfc:
                    for c_idx, lt_chunk in enumerate(sfc_lt_chunks):
                        base_name = "cams_sfc_all" if len(sfc_lt_chunks) == 1 else f"cams_sfc_chunk_{c_idx}"
                        task, marker_fp, nc_fp = _make_sfc_task(
                            date_str=date_str,
                            date_dash=date_dash,
                            base_name=base_name,
                            lt_chunk=lt_chunk,
                            times=times,
                            args=args,
                            date_spec=date_spec,
                            temp_raw_dir=temp_raw_dir,
                            temp_extract_dir=temp_extract_dir,
                            ac_collection=ac_collection,
                        )
                        task["horizon_h"] = bound_h
                        date_spec["all_markers"].append(marker_fp)
                        date_spec["sfc_nc_files"].append(nc_fp)
                        cum_markers.append(marker_fp)
                        tasks.append(task)

                if need_ml:
                    b_met_lvls, b_met_lts, _, _ = compute_stream_chunking(
                        len(UPPER_AIR_VARS),
                        all_levels,
                        num_times,
                        b_lts,
                        max_fields=args.max_fields,
                        explicit_lt_chunk=None,
                        explicit_level_chunks=args.level_chunks,
                    )
                    for l_idx, lvl_chunk in enumerate(b_met_lvls):
                        for lt_chunk in b_met_lts:
                            lvl_suffix = "" if len(b_met_lvls) == 1 else f"_part{l_idx}"
                            base_name = f"cams_met_h{int(lt_chunk[0]):03d}_{int(lt_chunk[-1]):03d}{lvl_suffix}"
                            task, marker_fp, nc_fp = _make_3d_task(
                                collection=ac_collection,
                                variables=UPPER_AIR_VARS,
                                date_str=date_str,
                                date_dash=date_dash,
                                base_name=base_name,
                                lvl_chunk=lvl_chunk,
                                lt_chunk=lt_chunk,
                                times=times,
                                include_type=True,
                                args=args,
                                date_spec=date_spec,
                                temp_raw_dir=temp_raw_dir,
                                temp_extract_dir=temp_extract_dir,
                            )
                            task["horizon_h"] = bound_h
                            date_spec["all_markers"].append(marker_fp)
                            date_spec["met_nc_by_lvl"][l_idx].append(nc_fp)
                            cum_markers.append(marker_fp)
                            cum_met_by_lvl[l_idx].append(nc_fp)
                            tasks.append(task)

                    for g_idx, (g_tag, g_vars) in enumerate(chem_groups):
                        b_chem_lvls, b_chem_lts, _, _ = compute_stream_chunking(
                            len(g_vars),
                            all_levels,
                            num_times,
                            b_lts,
                            max_fields=args.max_fields,
                            explicit_lt_chunk=None,
                            explicit_level_chunks=args.level_chunks,
                        )
                        for l_idx, lvl_chunk in enumerate(b_chem_lvls):
                            for lt_chunk in b_chem_lts:
                                lvl_suffix = "" if len(b_chem_lvls) == 1 else f"_part{l_idx}"
                                if len(lt_chunk) == 1:
                                    base_name = f"cams_{g_tag}_lt_{int(lt_chunk[0]):03d}{lvl_suffix}"
                                else:
                                    base_name = f"cams_{g_tag}_h{int(lt_chunk[0]):03d}_{int(lt_chunk[-1]):03d}{lvl_suffix}"
                                task, marker_fp, nc_fp = _make_3d_task(
                                    collection=ac_collection,
                                    variables=g_vars,
                                    date_str=date_str,
                                    date_dash=date_dash,
                                    base_name=base_name,
                                    lvl_chunk=lvl_chunk,
                                    lt_chunk=lt_chunk,
                                    times=times,
                                    include_type=True,
                                    args=args,
                                    date_spec=date_spec,
                                    temp_raw_dir=temp_raw_dir,
                                    temp_extract_dir=temp_extract_dir,
                                )
                                task["horizon_h"] = bound_h
                                date_spec["all_markers"].append(marker_fp)
                                date_spec["chem_groups_nc_by_lvl"][g_idx][l_idx].append(nc_fp)
                                cum_markers.append(marker_fp)
                                cum_chem_groups_by_lvl[g_idx][l_idx].append(nc_fp)
                                tasks.append(task)

                if need_ghg:
                    b_ghg_lvls, b_ghg_lts, _, _ = compute_stream_chunking(
                        len(GHG_VARS),
                        all_levels,
                        1,
                        b_lts,
                        max_fields=args.max_fields,
                        explicit_lt_chunk=None,
                        explicit_level_chunks=args.level_chunks,
                    )
                    for l_idx, lvl_chunk in enumerate(b_ghg_lvls):
                        for lt_chunk in b_ghg_lts:
                            lvl_suffix = "" if len(b_ghg_lvls) == 1 else f"_part{l_idx}"
                            base_name = f"cams_ghg_h{int(lt_chunk[0]):03d}_{int(lt_chunk[-1]):03d}{lvl_suffix}"
                            task, marker_fp, nc_fp = _make_3d_task(
                                collection=ghg_collection,
                                variables=GHG_VARS,
                                date_str=date_str,
                                date_dash=date_dash,
                                base_name=base_name,
                                lvl_chunk=lvl_chunk,
                                lt_chunk=lt_chunk,
                                times=None,
                                include_type=False,
                                args=args,
                                date_spec=date_spec,
                                temp_raw_dir=temp_raw_dir,
                                temp_extract_dir=temp_extract_dir,
                            )
                            task["horizon_h"] = bound_h
                            date_spec["all_markers"].append(marker_fp)
                            date_spec["ghg_nc_by_lvl"][l_idx].append(nc_fp)
                            cum_markers.append(marker_fp)
                            cum_ghg_by_lvl[l_idx].append(nc_fp)
                            tasks.append(task)

                date_spec["horizons"].append(
                    {
                        "horizon_h": bound_h,
                        "ready_marker": os.path.join(args.data_root_folder, f".cams_ready_{date_str}_h{bound_h:03d}.done"),
                        "required_markers": list(cum_markers),
                        "sfc_nc_files": list(date_spec["sfc_nc_files"]),
                        "met_nc_by_lvl": [list(g) for g in cum_met_by_lvl],
                        "chem_groups_nc_by_lvl": [[list(lv) for lv in grp] for grp in cum_chem_groups_by_lvl],
                        "ghg_nc_by_lvl": [list(g) for g in cum_ghg_by_lvl],
                    }
                )
        else:
            # 1. Surface Tasks
            if need_sfc:
                for c_idx, lt_chunk in enumerate(sfc_lt_chunks):
                    base_name = "cams_sfc_all" if len(sfc_lt_chunks) == 1 else f"cams_sfc_chunk_{c_idx}"
                    task, marker_fp, nc_fp = _make_sfc_task(
                        date_str=date_str,
                        date_dash=date_dash,
                        base_name=base_name,
                        lt_chunk=lt_chunk,
                        times=times,
                        args=args,
                        date_spec=date_spec,
                        temp_raw_dir=temp_raw_dir,
                        temp_extract_dir=temp_extract_dir,
                        ac_collection=ac_collection,
                    )
                    task["horizon_h"] = None
                    date_spec["all_markers"].append(marker_fp)
                    date_spec["sfc_nc_files"].append(nc_fp)
                    tasks.append(task)

            # 2. 3D Met & Chem Groups (A/B) Tasks
            if need_ml:
                for l_idx, lvl_chunk in enumerate(met_lvl_chunks):
                    for c_idx, lt_chunk in enumerate(met_lt_chunks):
                        lvl_suffix = "" if len(met_lvl_chunks) == 1 else f"_part{l_idx}"
                        base_name = f"cams_met_chunk_{c_idx}{lvl_suffix}"
                        task, marker_fp, nc_fp = _make_3d_task(
                            collection=ac_collection,
                            variables=UPPER_AIR_VARS,
                            date_str=date_str,
                            date_dash=date_dash,
                            base_name=base_name,
                            lvl_chunk=lvl_chunk,
                            lt_chunk=lt_chunk,
                            times=times,
                            include_type=True,
                            args=args,
                            date_spec=date_spec,
                            temp_raw_dir=temp_raw_dir,
                            temp_extract_dir=temp_extract_dir,
                        )
                        task["horizon_h"] = None
                        date_spec["all_markers"].append(marker_fp)
                        date_spec["met_nc_by_lvl"][l_idx].append(nc_fp)
                        tasks.append(task)

                for g_idx, ((g_tag, g_vars), (c_lvls, c_lts, _, _)) in enumerate(zip(chem_groups, chem_lvls_for_date)):
                    for l_idx, lvl_chunk in enumerate(c_lvls):
                        for c_idx, lt_chunk in enumerate(c_lts):
                            lvl_suffix = "" if len(c_lvls) == 1 else f"_part{l_idx}"
                            if len(lt_chunk) == 1:
                                base_name = f"cams_{g_tag}_lt_{int(lt_chunk[0]):03d}{lvl_suffix}"
                            else:
                                base_name = f"cams_{g_tag}_chunk_{c_idx}{lvl_suffix}"
                            task, marker_fp, nc_fp = _make_3d_task(
                                collection=ac_collection,
                                variables=g_vars,
                                date_str=date_str,
                                date_dash=date_dash,
                                base_name=base_name,
                                lvl_chunk=lvl_chunk,
                                lt_chunk=lt_chunk,
                                times=times,
                                include_type=True,
                                args=args,
                                date_spec=date_spec,
                                temp_raw_dir=temp_raw_dir,
                                temp_extract_dir=temp_extract_dir,
                            )
                            task["horizon_h"] = None
                            date_spec["all_markers"].append(marker_fp)
                            date_spec["chem_groups_nc_by_lvl"][g_idx][l_idx].append(nc_fp)
                            tasks.append(task)

            # 3. 3D GHG Tasks (0.1 deg CO2 & CH4; forecast-only at 00:00 UTC)
            if need_ghg:
                for l_idx, lvl_chunk in enumerate(ghg_lvl_chunks):
                    for c_idx, lt_chunk in enumerate(ghg_lt_chunks):
                        lvl_suffix = "" if len(ghg_lvl_chunks) == 1 else f"_part{l_idx}"
                        base_name = f"cams_ghg_chunk_{c_idx}{lvl_suffix}"
                        task, marker_fp, nc_fp = _make_3d_task(
                            collection=ghg_collection,
                            variables=GHG_VARS,
                            date_str=date_str,
                            date_dash=date_dash,
                            base_name=base_name,
                            lvl_chunk=lvl_chunk,
                            lt_chunk=lt_chunk,
                            times=None,
                            include_type=False,
                            args=args,
                            date_spec=date_spec,
                            temp_raw_dir=temp_raw_dir,
                            temp_extract_dir=temp_extract_dir,
                        )
                        task["horizon_h"] = None
                        date_spec["all_markers"].append(marker_fp)
                        date_spec["ghg_nc_by_lvl"][l_idx].append(nc_fp)
                        tasks.append(task)

    return tasks


def is_http_400_error(exc: Exception) -> bool:
    msg = str(exc)
    if "The job has failed" in msg:
        return False
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 400:
        return True
    if "400 Client Error" in msg or "400 Bad Request" in msg:
        return True
    return False


def classify_cams_error(exc: Exception) -> tuple[bool, str]:
    """
    Classify an ADS exception into `(is_fatal, category)`:
      - `auth_or_license`: HTTP 401/403, missing licence, disabled user, cost limit -> fail fast.
      - `bad_request`: HTTP 400 at submission (without server job failure) -> fail fast.
      - `transient_backend`: MarsRuntimeError (`Last error is -43`), 429/5xx, network errors -> retryable.
    """
    msg = str(exc)
    msg_l = msg.lower()
    resp = getattr(exc, "response", None)
    status_code = getattr(resp, "status_code", None)
    if (
        status_code in (401, 403)
        or "401 client error" in msg_l
        or "403 client error" in msg_l
        or "unauthorized" in msg_l
        or "required licences" in msg_l
        or "licences not accepted" in msg_l
        or "user has been disabled" in msg_l
        or "cost limits exceeded" in msg_l
    ):
        return True, "auth_or_license"
    if is_http_400_error(exc):
        return True, "bad_request"
    return False, "transient_backend"


class _CamsStatusTracker:
    """Thread-safe atomic JSON progress writer for CAMS chunk downloads & horizon readiness."""

    def __init__(self, tasks: list[dict], status_paths: Sequence[str | Path]) -> None:
        self._lock = threading.Lock()
        self._paths = [Path(p) for p in status_paths if p]
        self._state = "running"
        self._closed = False
        self._ready_horizons: set[int] = set()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._horizon_order: list[int] = []
        for t in tasks:
            hh = t.get("horizon_h")
            if hh is not None and int(hh) not in self._horizon_order:
                self._horizon_order.append(int(hh))
            self._tasks[t["name"]] = {
                "name": t["name"],
                "horizon_h": int(hh) if hh is not None else None,
                "state": "pending",
                "attempts": 0,
                "error": None,
            }
        self._horizon_order.sort()
        self._write_locked()

    def mark_state(
        self,
        task_name: str,
        state: str,
        *,
        error: str | None = None,
        attempt: int | None = None,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            entry = self._tasks.get(task_name)
            if entry is not None:
                entry["state"] = state
                if attempt is not None:
                    entry["attempts"] = int(attempt)
                if error is not None:
                    entry["error"] = str(error).splitlines()[0][:240]
            self._write_locked()

    def mark_horizon_ready(self, horizon_h: int) -> None:
        with self._lock:
            if self._closed:
                return
            self._ready_horizons.add(int(horizon_h))
            self._write_locked()

    def finish(self, state: str) -> None:
        with self._lock:
            if self._closed and self._state == state:
                return
            self._state = state
            self._write_locked()
            self._closed = True

    def _build_payload_locked(self) -> dict[str, Any]:
        counts = {"total": len(self._tasks), "completed": 0, "running": 0, "retrying": 0, "pending": 0, "failed": 0}
        active_tasks: list[str] = []
        failed_tasks: list[dict[str, str]] = []
        for entry in self._tasks.values():
            st = entry["state"]
            if st in counts:
                counts[st] += 1
            if st in ("running", "retrying"):
                active_tasks.append(entry["name"])
            elif st == "failed":
                failed_tasks.append({"name": entry["name"], "error": entry["error"] or "unknown error"})

        horizons_summary: list[dict[str, Any]] = []
        for hh in self._horizon_order:
            h_entries = [e for e in self._tasks.values() if e["horizon_h"] == hh]
            horizons_summary.append(
                {
                    "horizon_h": hh,
                    "done": sum(1 for e in h_entries if e["state"] == "completed"),
                    "total": len(h_entries),
                    "ready": hh in self._ready_horizons,
                    "failed": sum(1 for e in h_entries if e["state"] == "failed"),
                }
            )

        return {
            "state": self._state,
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "counts": counts,
            "horizons": horizons_summary,
            "active_tasks": active_tasks[:12],
            "failed_tasks": failed_tasks,
        }

    def _write_locked(self) -> None:
        if not self._paths:
            return
        payload = json.dumps(self._build_payload_locked(), indent=2) + "\n"
        for fp in self._paths:
            try:
                fp.parent.mkdir(parents=True, exist_ok=True)
                tmp_fp = fp.parent / f".{fp.name}.tmp.{os.getpid()}"
                tmp_fp.write_text(payload, encoding="utf-8")
                os.replace(tmp_fp, fp)
            except OSError:
                pass


def _merge_and_save_3d_groups(
    stream: str,
    nc_by_lvl_groups,
    date_str: str,
    args: argparse.Namespace,
    *,
    skip_if_steps_ge: bool = False,
) -> None:
    opened = []
    lvl_merged_datasets = []
    try:
        num_lvl_groups = max(len(group) for group in nc_by_lvl_groups)
        for l_idx in range(num_lvl_groups):
            group_merged = []
            for group in nc_by_lvl_groups:
                files = group[min(l_idx, len(group) - 1)]
                ds_list = [xr.open_dataset(f, chunks={"forecast_period": 1}) for f in files]
                opened.extend(ds_list)
                ds_cat = (
                    ds_list[0]
                    if len(ds_list) == 1
                    else xr.concat(
                        ds_list, dim="forecast_period", data_vars="minimal", coords="minimal", compat="override"
                    ).sortby("forecast_period")
                )
                group_merged.append(ds_cat)

            ds_lvl = group_merged[0] if len(group_merged) == 1 else xr.merge(group_merged, compat="override")
            lvl_merged_datasets.append(ds_lvl)

        if len(lvl_merged_datasets) == 1:
            ds_final = lvl_merged_datasets[0]
        else:
            lvl_dim = "model_level" if "model_level" in lvl_merged_datasets[0].dims else "level"
            ds_final = xr.concat(lvl_merged_datasets, dim=lvl_dim).sortby(lvl_dim)

        save_dataset_daily_and_base(
            ds_final,
            stream,
            date_str,
            args.data_root_folder,
            no_split_days=getattr(args, "no_split_days", False),
            skip_if_steps_ge=skip_if_steps_ge,
        )
    finally:
        for ds in opened:
            ds.close()


def _flush_date_spec(
    *,
    date_str: str,
    need_sfc: bool,
    need_ml: bool,
    need_ghg: bool,
    sfc_files: list[str],
    met_nc_by_lvl: list[list[str]],
    chem_nc_by_lvl: list[list[str]] | None = None,
    chem_groups_nc_by_lvl: list[list[list[str]]] | None = None,
    ghg_nc_by_lvl: list[list[str]],
    args: argparse.Namespace,
    skip_if_steps_ge: bool = False,
) -> None:
    if need_sfc and sfc_files:
        opened_sfc = [xr.open_dataset(f, chunks={"forecast_period": 1}) for f in sfc_files]
        try:
            ds_sfc = (
                opened_sfc[0]
                if len(opened_sfc) == 1
                else xr.concat(opened_sfc, dim="forecast_period").sortby("forecast_period")
            )
            save_dataset_daily_and_base(
                ds_sfc,
                "sfc",
                date_str,
                args.data_root_folder,
                no_split_days=getattr(args, "no_split_days", False),
                skip_if_steps_ge=skip_if_steps_ge,
            )
        finally:
            for ds in opened_sfc:
                ds.close()

    all_chem_groups: list[list[list[str]]] = []
    if chem_groups_nc_by_lvl:
        all_chem_groups.extend(g for g in chem_groups_nc_by_lvl if g)
    elif chem_nc_by_lvl:
        all_chem_groups.append(chem_nc_by_lvl)

    if need_ml and met_nc_by_lvl and all_chem_groups:
        _merge_and_save_3d_groups(
            "ml",
            [met_nc_by_lvl, *all_chem_groups],
            date_str,
            args,
            skip_if_steps_ge=skip_if_steps_ge,
        )

    if need_ghg and ghg_nc_by_lvl:
        _merge_and_save_3d_groups(
            "ghg",
            [ghg_nc_by_lvl],
            date_str,
            args,
            skip_if_steps_ge=skip_if_steps_ge,
        )


def check_and_flush_horizons(
    date_spec: dict,
    args: argparse.Namespace,
    status_tracker: _CamsStatusTracker | None = None,
) -> None:
    """Incrementally flush cumulative daily NetCDFs whenever a horizon bucket finishes."""
    horizons = date_spec.get("horizons")
    if not horizons:
        return
    date_str = date_spec["date_str"]
    with _MERGE_LOCK:
        for h_spec in horizons:
            ready_marker = h_spec["ready_marker"]
            if os.path.exists(ready_marker):
                if status_tracker is not None:
                    status_tracker.mark_horizon_ready(int(h_spec["horizon_h"]))
                continue
            if not all(os.path.exists(m) for m in h_spec["required_markers"]):
                break
            t_start = time.time()
            _flush_date_spec(
                date_str=date_str,
                need_sfc=bool(date_spec.get("need_sfc")),
                need_ml=bool(date_spec.get("need_ml")),
                need_ghg=bool(date_spec.get("need_ghg")),
                sfc_files=h_spec["sfc_nc_files"],
                met_nc_by_lvl=h_spec["met_nc_by_lvl"],
                chem_nc_by_lvl=h_spec.get("chem_nc_by_lvl"),
                chem_groups_nc_by_lvl=h_spec.get("chem_groups_nc_by_lvl"),
                ghg_nc_by_lvl=h_spec["ghg_nc_by_lvl"],
                args=args,
                skip_if_steps_ge=True,
            )
            tmp_marker = f"{ready_marker}.tmp.{os.getpid()}"
            with open(tmp_marker, "w", encoding="utf-8") as f:
                f.write("DONE\n")
            os.replace(tmp_marker, ready_marker)
            if status_tracker is not None:
                status_tracker.mark_horizon_ready(int(h_spec["horizon_h"]))
            log_msg(
                f"[STREAM-READY] {date_str}: horizon 0..{h_spec['horizon_h']}h ready "
                f"in {time.time() - t_start:.1f}s ({os.path.basename(ready_marker)})"
            )


def check_and_merge(date_spec: dict, args: argparse.Namespace, *, skip_if_steps_ge: bool = False) -> None:
    date_str = date_spec["date_str"]
    temp_raw_dir = date_spec["temp_raw_dir"]
    temp_extract_dir = date_spec["temp_extract_dir"]

    for marker_fp in date_spec["all_markers"]:
        if not os.path.exists(marker_fp):
            return

    lock_dir = os.path.join(temp_raw_dir, f"merge_{date_str}.lockdir")
    try:
        os.mkdir(lock_dir)
    except FileExistsError:
        return

    t_merge_start = time.time()
    try:
        log_msg(f"[MERGE-START] All {len(date_spec['all_markers'])} chunks for {date_str} are complete. Merging...")
        _flush_date_spec(
            date_str=date_str,
            need_sfc=bool(date_spec.get("need_sfc")),
            need_ml=bool(date_spec.get("need_ml")),
            need_ghg=bool(date_spec.get("need_ghg")),
            sfc_files=date_spec.get("sfc_nc_files", []),
            met_nc_by_lvl=date_spec.get("met_nc_by_lvl", []),
            chem_nc_by_lvl=date_spec.get("chem_nc_by_lvl"),
            chem_groups_nc_by_lvl=date_spec.get("chem_groups_nc_by_lvl"),
            ghg_nc_by_lvl=date_spec.get("ghg_nc_by_lvl", []),
            args=args,
            skip_if_steps_ge=skip_if_steps_ge,
        )
        log_msg(
            f"[MERGE-DONE] Full merge and daily slicing for {date_str} completed in {time.time() - t_merge_start:.1f}s."
        )

    except Exception as e:
        shutil.rmtree(lock_dir, ignore_errors=True)
        log_msg(f"[MERGE-ERROR] Error during merge for {date_str}: {e}")
        raise

    try:
        shutil.rmtree(temp_raw_dir, ignore_errors=True)
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        log_msg(f"[CLEANUP] Removed temporary directories for {date_str}.")
    except Exception as e:
        log_msg(f"[CLEANUP-WARN] Could not fully clean up temp directories: {e}")


def run_task(
    task: dict,
    args: argparse.Namespace,
    client=None,
    defer_merge: bool = False,
    *,
    abort_event: threading.Event | None = None,
    status_tracker: _CamsStatusTracker | None = None,
) -> None:
    if abort_event is not None and abort_event.is_set():
        raise RuntimeError("Task cancelled after fatal CAMS pipeline error")

    os.makedirs(task["temp_raw_dir"], exist_ok=True)
    os.makedirs(task["temp_extract_dir"], exist_ok=True)

    if getattr(args, "force", False):
        for fp in (task["marker"], task["output_raw"], task["output_nc"]):
            if os.path.exists(fp):
                os.remove(fp)

    if os.path.exists(task["marker"]) and os.path.exists(task["output_nc"]):
        nc_size_mb = os.path.getsize(task["output_nc"]) / (1024.0 * 1024.0)
        log_msg(f"[SKIPPED] {task['name']} | Already completed ({os.path.basename(task['output_nc'])}, {nc_size_mb:.2f} MB)")
        if status_tracker is not None:
            status_tracker.mark_state(task["name"], "completed")
    else:
        t_chunk_start = time.time()
        lt_desc = (
            f"{task['lt_chunk'][0]}..{task['lt_chunk'][-1]}h ({len(task['lt_chunk'])} steps)"
            if len(task["lt_chunk"]) > 1
            else f"{task['lt_chunk'][0]}h"
        )

        need_download = True
        if os.path.exists(task["output_raw"]) and os.path.getsize(task["output_raw"]) > 0:
            raw_size_mb = os.path.getsize(task["output_raw"]) / (1024.0 * 1024.0)
            log_msg(
                f"[FOUND-RAW] {task['name']} | Found existing {os.path.basename(task['output_raw'])} "
                f"({raw_size_mb:.2f} MB), attempting conversion..."
            )
            if status_tracker is not None:
                status_tracker.mark_state(task["name"], "running", attempt=0)
            try:
                t_conv_start = time.time()
                if task["format"] == "grib":
                    convert_grib_to_cdm_nc(task["output_raw"], task["output_nc"], is_3d=task["is_3d"])
                else:
                    extract_subdir = os.path.join(
                        task["temp_extract_dir"], f"unzip_{os.path.basename(task['output_raw'])}"
                    )
                    extract_nc_from_zip_to_target(task["output_raw"], task["output_nc"], extract_subdir)
                conv_elapsed = time.time() - t_conv_start
                nc_size_mb = os.path.getsize(task["output_nc"]) / (1024.0 * 1024.0)
                with open(task["marker"], "w", encoding="utf-8") as f:
                    f.write("DONE\n")
                log_msg(
                    f"[CONVERTED] {task['name']} | netcdf={os.path.basename(task['output_nc'])} ({nc_size_mb:.2f} MB) | "
                    f"conv_elapsed={conv_elapsed:.1f}s"
                )
                if status_tracker is not None:
                    status_tracker.mark_state(task["name"], "completed")
                need_download = False
            except Exception as conv_err:
                log_msg(
                    f"[RAW-INCOMPLETE] {task['name']} | Existing {os.path.basename(task['output_raw'])} "
                    f"failed conversion ({conv_err}); re-downloading..."
                )
                try:
                    os.remove(task["output_raw"])
                except OSError:
                    pass

        if need_download:
            if abort_event is not None and abort_event.is_set():
                raise RuntimeError("Task cancelled after fatal CAMS pipeline error")

            is_ghg_task = task.get("collection") == "cams-global-greenhouse-gas-forecasts"
            ghg_poll_deadline_utc = None
            if is_ghg_task:
                try:
                    ghg_init_utc = dt.datetime.strptime(task["date_str"], "%Y%m%d").replace(tzinfo=dt.timezone.utc)
                    ghg_delay_h = float(os.environ.get("CAMS_GHG_PUBLISH_DELAY_HOURS", "13.5"))
                    ghg_ready_utc = ghg_init_utc + dt.timedelta(hours=ghg_delay_h)
                    ghg_poll_deadline_utc = ghg_init_utc + dt.timedelta(hours=18.0)
                    now_utc = dt.datetime.now(dt.timezone.utc)
                    if now_utc < ghg_ready_utc:
                        wait_sec = (ghg_ready_utc - now_utc).total_seconds()
                        log_msg(
                            f"[GHG-WAIT] {task['name']} | Waiting {wait_sec / 60.0:.1f}m until expected ADS GHG "
                            f"publication window ({ghg_ready_utc:%Y-%m-%d %H:%M} UTC)..."
                        )
                        while True:
                            rem = (ghg_ready_utc - dt.datetime.now(dt.timezone.utc)).total_seconds()
                            if rem <= 0:
                                break
                            sleep_s = min(300.0, rem)
                            if abort_event is not None:
                                if abort_event.wait(sleep_s):
                                    raise RuntimeError("Task cancelled after fatal CAMS pipeline error")
                            else:
                                time.sleep(sleep_s)
                except ValueError:
                    ghg_poll_deadline_utc = None

            log_msg(
                f"[SUBMITTED] {task['name']} | format={task['format']} | fields={task['num_fields']} | "
                f"leadtimes={lt_desc} -> {os.path.basename(task['output_raw'])}"
            )

            max_retries = max(0, int(getattr(args, "max_retries", 3)))
            retry_base_s = float(getattr(args, "_retry_base_sec", 45.0))
            c = client or get_thread_cds_client()

            attempt = 0
            while True:
                if abort_event is not None and abort_event.is_set():
                    raise RuntimeError("Task cancelled after fatal CAMS pipeline error")
                if status_tracker is not None:
                    status_tracker.mark_state(task["name"], "running", attempt=attempt + 1)

                t_req_start = time.time()
                try:
                    c.retrieve(
                        task.get("collection", "cams-global-atmospheric-composition-forecasts"),
                        task["request"],
                        task["output_raw"],
                    )
                    dl_elapsed = time.time() - t_req_start
                    raw_size_mb = os.path.getsize(task["output_raw"]) / (1024.0 * 1024.0)
                    log_msg(
                        f"[COMPLETED] {task['name']} | file={os.path.basename(task['output_raw'])} | "
                        f"size={raw_size_mb:.2f} MB | download_elapsed={dl_elapsed:.1f}s ({dl_elapsed / 60.0:.2f}m)"
                    )
                    break
                except Exception as e:
                    if os.path.exists(task["output_raw"]):
                        try:
                            os.remove(task["output_raw"])
                        except OSError:
                            pass

                    is_fatal, err_cat = classify_cams_error(e)
                    if is_fatal:
                        if (
                            err_cat == "bad_request"
                            and is_ghg_task
                            and ghg_poll_deadline_utc is not None
                            and dt.datetime.now(dt.timezone.utc) < ghg_poll_deadline_utc
                        ):
                            log_msg(
                                f"[GHG-WAIT] {task['name']} | ADS GHG not yet published in catalogue "
                                f"(HTTP 400); polling again in 600s..."
                            )
                            if abort_event is not None:
                                if abort_event.wait(600.0):
                                    raise RuntimeError("Task cancelled after fatal CAMS pipeline error")
                            else:
                                time.sleep(600.0)
                            c = reset_thread_cds_client()
                            continue
                        if status_tracker is not None:
                            status_tracker.mark_state(task["name"], "failed", error=str(e), attempt=attempt + 1)
                        if abort_event is not None:
                            abort_event.set()
                        log_msg(f"[FAILED-FAST] {task['name']} | Non-retryable {err_cat}: {e}")
                        raise

                    if attempt < max_retries:
                        backoff_s = min(300.0, retry_base_s * (2**attempt))
                        if status_tracker is not None:
                            status_tracker.mark_state(task["name"], "retrying", error=str(e), attempt=attempt + 1)
                        log_msg(
                            f"[RETRY {attempt + 1}/{max_retries}] {task['name']} | "
                            f"{err_cat}: {e} | Waiting {backoff_s:g}s before retry..."
                        )
                        if abort_event is not None:
                            if abort_event.wait(backoff_s):
                                raise RuntimeError("Task cancelled after fatal CAMS pipeline error")
                        else:
                            time.sleep(backoff_s)
                        attempt += 1
                        c = reset_thread_cds_client()
                    else:
                        if status_tracker is not None:
                            status_tracker.mark_state(task["name"], "failed", error=str(e), attempt=attempt + 1)
                        if abort_event is not None:
                            abort_event.set()
                        log_msg(f"[FAILED] {task['name']} | Exhausted {max_retries} retries. Last error: {e}")
                        raise

            t_conv_start = time.time()
            if task["format"] == "grib":
                convert_grib_to_cdm_nc(task["output_raw"], task["output_nc"], is_3d=task["is_3d"])
            else:
                extract_subdir = os.path.join(
                    task["temp_extract_dir"], f"unzip_{os.path.basename(task['output_raw'])}"
                )
                extract_nc_from_zip_to_target(task["output_raw"], task["output_nc"], extract_subdir)

            conv_elapsed = time.time() - t_conv_start
            total_elapsed = time.time() - t_chunk_start
            nc_size_mb = os.path.getsize(task["output_nc"]) / (1024.0 * 1024.0)

            with open(task["marker"], "w", encoding="utf-8") as f:
                f.write("DONE\n")

            if status_tracker is not None:
                status_tracker.mark_state(task["name"], "completed")

            log_msg(
                f"[CONVERTED] {task['name']} | netcdf={os.path.basename(task['output_nc'])} ({nc_size_mb:.2f} MB) | "
                f"conv_elapsed={conv_elapsed:.1f}s | total_chunk_wall={total_elapsed:.1f}s ({total_elapsed / 60.0:.2f}m)"
            )

    if not defer_merge:
        check_and_merge(task["date_spec"], args)


def execute_cams_pipeline(args: argparse.Namespace) -> list[dict]:
    """Execute or plan the CAMS download & daily CDM NetCDF merge pipeline."""
    tasks = generate_tasks(args)

    if getattr(args, "list_tasks", False):
        print(f"{len(tasks)} tasks (0..{len(tasks) - 1})")
        return tasks

    if not tasks:
        log_msg("No pending tasks to execute.")
        return tasks

    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(tasks):
            raise ValueError(f"task_index {args.task_index} is out of bounds (0..{len(tasks) - 1})")
        task = tasks[args.task_index]
        run_task(task, args, defer_merge=False)
        return tasks

    status_paths: list[str | Path] = [
        Path(args.data_root_folder) / f".cams_status_{tasks[0]['date_str']}.json",
    ]
    if getattr(args, "status_file", None):
        status_paths.append(args.status_file)
    status_tracker = _CamsStatusTracker(tasks, status_paths)
    abort_event = threading.Event()

    max_workers = max(1, int(args.max_workers))
    log_msg(f"Starting ThreadPoolExecutor queue pipeline with max_workers={max_workers} across {len(tasks)} total tasks...")

    t_all_start = time.time()
    failures: list[tuple[str, Exception]] = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    future_to_task = {
        executor.submit(
            run_task,
            task,
            args,
            None,
            True,
            abort_event=abort_event,
            status_tracker=status_tracker,
        ): task
        for task in tasks
    }
    try:
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                future.result()
                if getattr(args, "stream_daily", False):
                    check_and_flush_horizons(task["date_spec"], args, status_tracker=status_tracker)
            except Exception as exc:
                failures.append((task["name"], exc))
                status_tracker.mark_state(task["name"], "failed", error=str(exc))
                status_tracker.finish("failed")
                log_msg(f"[TASK-ERROR] {task['name']} raised exception: {exc}")
                abort_event.set()
                for pending_fut in future_to_task:
                    pending_fut.cancel()
                break
    except Exception:
        abort_event.set()
        status_tracker.finish("failed")
        for pending_fut in future_to_task:
            pending_fut.cancel()
        raise
    finally:
        if failures:
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True, cancel_futures=True)

    if failures:
        status_tracker.finish("failed")
        log_msg(f"Pipeline failed with {len(failures)} failed task(s):")
        for name, exc in failures:
            log_msg(f"  - {name}: {exc}")
        raise RuntimeError(f"CAMS fetch pipeline failed ({len(failures)} failed task(s)).")

    seen_dates: set[str] = set()
    for task in tasks:
        d_str = task["date_str"]
        if d_str not in seen_dates:
            seen_dates.add(d_str)
            check_and_merge(
                task["date_spec"],
                args,
                skip_if_steps_ge=bool(getattr(args, "stream_daily", False)),
            )

    status_tracker.finish("completed")
    total_wall = time.time() - t_all_start
    log_msg(
        f"[ALL-COMPLETE] All {len(tasks)} tasks and daily merges finished in {total_wall:.1f}s ({total_wall / 60.0:.2f}m)."
    )
    return tasks


def download_cams_forecast(
    start_date: str,
    end_date: str,
    data_root_folder: str | Path,
    *,
    config: str | Path | None = None,
    data_format: str | None = None,
    max_fields: int | None = None,
    sfc_lt_chunk: int | None = None,
    met_lt_chunk: int | None = None,
    chem_lt_chunk: int | None = None,
    ghg_lt_chunk: int | None = None,
    ghg: bool | None = None,
    ghg_only: bool = False,
    level_chunks: int | None = None,
    max_workers: int | None = None,
    max_retries: int | None = None,
    list_tasks: bool = False,
    task_index: int | None = None,
    sfc_only: bool = False,
    force: bool = False,
    cams_type: str | None = None,
    cams_time: str | None = None,
    area: Sequence[float] | None = None,
    leadtimes: Sequence[str | int] | str | None = None,
    no_split_days: bool = False,
    stream_daily: bool | None = None,
    segment_hours: int | None = 24,
    status_file: str | Path | None = None,
) -> list[dict]:
    """Programmatic entry point matching the CAMS GRIB pipelined downloader CLI."""
    if leadtimes is not None:
        if isinstance(leadtimes, (list, tuple)):
            norm_lts = [str(x) for x in leadtimes]
        else:
            norm_lts = [str(leadtimes)]
    else:
        norm_lts = None

    ns = argparse.Namespace(
        mode=None,
        start_date=str(start_date),
        end_date=str(end_date),
        data_root_folder=str(data_root_folder),
        config=str(config) if config is not None else None,
        format=data_format,
        max_fields=max_fields,
        sfc_lt_chunk=sfc_lt_chunk,
        met_lt_chunk=met_lt_chunk,
        chem_lt_chunk=chem_lt_chunk,
        ghg_lt_chunk=ghg_lt_chunk,
        ghg=ghg,
        ghg_only=bool(ghg_only),
        level_chunks=level_chunks,
        max_workers=max_workers,
        max_retries=max_retries,
        list_tasks=bool(list_tasks),
        task_index=task_index,
        sfc_only=bool(sfc_only),
        force=bool(force),
        type=cams_type,
        time=cams_time,
        area=list(area) if area is not None else None,
        leadtimes=norm_lts,
        no_split_days=bool(no_split_days),
        stream_daily=stream_daily,
        segment_hours=segment_hours,
        status_file=str(status_file) if status_file is not None else None,
    )
    apply_config_and_defaults(ns)
    return execute_cams_pipeline(ns)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        execute_cams_pipeline(args)
    except (RuntimeError, ValueError) as exc:
        log_msg(f"Error: {exc}")
        sys.exit(1)
