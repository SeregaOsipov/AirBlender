"""ECMWF MARS operational IFS (9 km / F1280 / gridded) meteorological downloader and license probe."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import threading
from typing import Sequence
import uuid

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from airblender.fetch.cams_forecast import _write_ds_streaming, convert_grib_to_cdm_nc


WRF_SURFACE_VARS = (
    "129.128/172.128/134.128/139.128/141.128/151.128/165.128/"
    "166.128/167.128/168.128/170.128/183.128/235.128/236.128/"
    "31.128/33.128/34.128/39.128/40.128/41.128/42.128"
)
WRF_VERTICAL_LEVEL_VARS = "130.128/131.128/132.128/133.128"

IFS_FORECAST_SFC_PARAMS = (
    "129.128/172.128/134.128/151.128/235.128/34.128/31.128/"
    "167.128/168.128/165.128/166.128/141.128/33.128/32.128/"
    "238.128/198.128/139.128/170.128/183.128/236.128/"
    "39.128/40.128/41.128/42.128"
)
IFS_FORECAST_ML_PARAMS = "130.128/131.128/132.128/133.128/135.128/246.128/247.128"


def _strtobool(val: str | bool) -> bool:
    if isinstance(val, bool):
        return val
    norm = str(val).strip().lower()
    if norm in ("y", "yes", "t", "true", "on", "1"):
        return True
    if norm in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"Invalid truth value: {val!r}")


def _has_mars_credentials(rc_file: str | Path | None = None) -> bool:
    if os.environ.get("ECMWF_API_KEY") and os.environ.get("ECMWF_API_EMAIL"):
        return True
    cand = Path(rc_file).expanduser() if rc_file else Path(os.environ.get("ECMWF_API_RC_FILE", "~/.ecmwfapirc")).expanduser()
    return cand.is_file() and cand.stat().st_size > 0


def _format_mars_area(area: Sequence[float] | None) -> str | None:
    if area is None or len(area) != 4:
        return None
    n, w, s, e = [float(x) for x in area]
    return f"{n:g}/{w:g}/{s:g}/{e:g}"


def _normalize_date_str(date_str: str) -> str:
    clean = str(date_str).replace("-", "").replace("_", "")[:8]
    return f"{clean[:4]}-{clean[4:6]}-{clean[6:8]}"


def _normalize_time_str(time_str: str) -> str:
    hh = str(time_str).strip().replace(":", "")[:2].zfill(2)
    return f"{hh}:00:00"


def probe_ifs_mars_access(
    date_str: str,
    *,
    ifs_type: str = "fc",
    time_str: str = "00:00",
    max_step_h: int = 120,
    area: Sequence[float] | None = None,
    grid: str = "0.08/0.08",
    rc_file: str | Path | None = None,
    timeout_sec: float = 45.0,
) -> tuple[bool, str]:
    """
    Execute a minimal 1-field MARS probe at the furthest `valid_time` (`max_step_h`)
    to detect valid-time restrictions (`MARS_RESTRICTED_ACCESS_TO_VALID_DATA`) or
    expired/missing credentials (`401`/`403`) or queue stalls (`timeout_sec`)
    before launching heavy downloads.
    """
    if not _has_mars_credentials(rc_file):
        return False, "missing ECMWF MARS credentials (~/.ecmwfapirc)"

    from ecmwfapi import ECMWFService

    dash_date = _normalize_date_str(date_str)
    mars_time = _normalize_time_str(time_str)
    mars_area = _format_mars_area(area) or "37/41/18/61"
    probe_step = str(max(0, int(max_step_h))) if str(ifs_type).lower() == "fc" else "0"

    req = {
        "class": "od",
        "stream": "oper",
        "expver": "1",
        "type": str(ifs_type),
        "date": dash_date,
        "time": mars_time,
        "levtype": "sfc",
        "param": "167.128",
        "area": mars_area,
        "grid": str(grid),
    }
    if str(ifs_type).lower() == "fc":
        req["step"] = probe_step

    if rc_file and not os.environ.get("ECMWF_API_RC_FILE"):
        os.environ["ECMWF_API_RC_FILE"] = str(Path(rc_file).expanduser())

    tmp_fp = os.path.join(tempfile.gettempdir(), f"ifs_mars_probe_{os.getpid()}_{uuid.uuid4().hex[:8]}.grib")
    outcome: list[tuple[bool, str]] = []

    def _run_probe() -> None:
        try:
            server = ECMWFService("mars")
            server.execute(req, tmp_fp)
            if os.path.isfile(tmp_fp) and os.path.getsize(tmp_fp) > 0:
                outcome.append((True, "ok"))
            else:
                outcome.append((False, "MARS probe returned empty file"))
        except Exception as exc:
            outcome.append((False, str(exc)))
        finally:
            if os.path.exists(tmp_fp):
                try:
                    os.remove(tmp_fp)
                except OSError:
                    pass

    worker = threading.Thread(target=_run_probe, daemon=True)
    worker.start()
    worker.join(timeout=max(0.1, float(timeout_sec)))
    if worker.is_alive():
        return False, f"MARS probe timed out after {float(timeout_sec):g}s (queued or slow)"
    return outcome[0] if outcome else (False, "MARS probe failed without status")


def _split_cdm_nc_to_daily(
    src_nc: Path,
    out_dir: Path,
    stream: str,
    cycle_date_clean: str,
) -> list[Path]:
    out_files: list[Path] = []
    with xr.open_dataset(src_nc) as ds:
        if "valid_time" in ds.coords:
            vt = np.asarray(ds.valid_time.values)
            valid_times = pd.to_datetime(vt.flatten() if vt.ndim > 1 else vt)
        elif "time" in ds.coords:
            valid_times = pd.to_datetime(np.asarray(ds.time.values))
        else:
            valid_times = pd.to_datetime([datetime.strptime(cycle_date_clean, "%Y%m%d")])

        unique_days = sorted(list(set(valid_times.strftime("%Y%m%d"))))
        time_dim = "forecast_period" if "forecast_period" in ds.dims else ("time" if "time" in ds.dims else None)

        if not time_dim or len(unique_days) <= 1:
            day_tag = unique_days[0] if unique_days else cycle_date_clean
            out_fp = out_dir / f"ECMWF-IFS_{stream}_{day_tag}.nc"
            _write_ds_streaming(ds, str(out_fp))
            out_files.append(out_fp)
            return out_files

        for day in unique_days:
            out_fp = out_dir / f"ECMWF-IFS_{stream}_{day}.nc"
            mask = valid_times.strftime("%Y%m%d") == day
            ds_day = ds.isel({time_dim: mask})
            _write_ds_streaming(ds_day, str(out_fp))
            out_files.append(out_fp)
    return out_files


def _existing_cdm_steps(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    try:
        with netCDF4.Dataset(path, "r") as ds:
            if "forecast_period" in ds.dimensions:
                return len(ds.dimensions["forecast_period"])
    except Exception:
        return 0
    return 0


def download_ifs_forecast_mars(
    start_date: str,
    out_dir: str | Path,
    *,
    ifs_type: str = "fc",
    time_str: str = "00:00",
    leadtimes_hours: Sequence[int] | None = None,
    area: Sequence[float] | None = None,
    grid: str = "0.08/0.08",
    force: bool = False,
) -> dict[str, list[Path]]:
    """
    Download operational 9 km (0.08°) ECMWF IFS (`sfc` + 137 `ml`) via MARS GRIB,
    preserving `GRIB_pv`, and emit daily CDM NetCDFs `ECMWF-IFS_{sfc,ml}_{YYYYMMDD}.nc`.
    """
    from ecmwfapi import ECMWFService

    dash_date = _normalize_date_str(start_date)
    clean_date = dash_date.replace("-", "")
    mars_time = _normalize_time_str(time_str)
    hh = mars_time[:2]
    cycle_dt = datetime.strptime(f"{clean_date}{hh}", "%Y%m%d%H").replace(tzinfo=timezone.utc)
    steps = sorted({int(h) for h in (leadtimes_hours if leadtimes_hours is not None else range(0, 121, 3))})
    mars_area = _format_mars_area(area)

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = target_dir / f"tmp_ifs_mars_{clean_date}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Check if all expected daily outputs already exist
    day_groups: dict[str, list[int]] = {}
    for h in steps:
        valid_dt = cycle_dt + pd.Timedelta(hours=h)
        day_groups.setdefault(valid_dt.strftime("%Y%m%d"), []).append(h)

    existing_sfc = [target_dir / f"ECMWF-IFS_sfc_{d}.nc" for d in sorted(day_groups)]
    existing_ml = [target_dir / f"ECMWF-IFS_ml_{d}.nc" for d in sorted(day_groups)]
    if (
        not force
        and all(_existing_cdm_steps(target_dir / f"ECMWF-IFS_sfc_{d}.nc") >= len(hs) for d, hs in day_groups.items())
        and all(_existing_cdm_steps(target_dir / f"ECMWF-IFS_ml_{d}.nc") >= len(hs) for d, hs in day_groups.items())
    ):
        print(f"[SKIPPED] Daily ECMWF-IFS 9km files already complete in {target_dir}")
        return {"sfc": existing_sfc, "ml": existing_ml}

    step_str = "/".join(str(h) for h in steps)
    server = ECMWFService("mars")

    # 1. Surface + Soil GRIB -> daily CDM NetCDF
    sfc_grib = tmp_dir / f"ifs_sfc_{clean_date}.grib"
    sfc_merged_nc = tmp_dir / f"ifs_sfc_{clean_date}.nc"
    sfc_req = {
        "class": "od",
        "stream": "oper",
        "expver": "1",
        "type": str(ifs_type),
        "date": dash_date,
        "time": mars_time,
        "levtype": "sfc",
        "param": IFS_FORECAST_SFC_PARAMS,
        "grid": str(grid),
    }
    if mars_area:
        sfc_req["area"] = mars_area
    if str(ifs_type).lower() == "fc":
        sfc_req["step"] = step_str
    print(f"Downloading ECMWF MARS IFS {grid} SFC ({dash_date} {mars_time})...")
    server.execute(sfc_req, str(sfc_grib))
    convert_grib_to_cdm_nc(str(sfc_grib), str(sfc_merged_nc), is_3d=False)
    sfc_daily = _split_cdm_nc_to_daily(sfc_merged_nc, target_dir, "sfc", clean_date)

    # 2. 137 Model Levels GRIB -> daily CDM NetCDF preserving GRIB_pv
    ml_daily_map: dict[str, Path] = {}
    for day_str, day_steps in sorted(day_groups.items()):
        day_nc = target_dir / f"ECMWF-IFS_ml_{day_str}.nc"
        if not force and _existing_cdm_steps(day_nc) >= len(day_steps):
            ml_daily_map[day_str] = day_nc
            continue
        ml_grib = tmp_dir / f"ifs_ml_{day_str}.grib"
        ml_req = {
            "class": "od",
            "stream": "oper",
            "expver": "1",
            "type": str(ifs_type),
            "date": dash_date,
            "time": mars_time,
            "levtype": "ml",
            "levelist": "1/to/137",
            "param": IFS_FORECAST_ML_PARAMS,
            "grid": str(grid),
        }
        if mars_area:
            ml_req["area"] = mars_area
        if str(ifs_type).lower() == "fc":
            ml_req["step"] = "/".join(str(h) for h in day_steps)
        print(f"Downloading ECMWF MARS IFS {grid} 137-ML (day {day_str}, steps={day_steps})...")
        server.execute(ml_req, str(ml_grib))
        convert_grib_to_cdm_nc(str(ml_grib), str(day_nc), is_3d=True)
        ml_daily_map[day_str] = day_nc

    return {
        "sfc": sorted(sfc_daily),
        "ml": [ml_daily_map[d] for d in sorted(ml_daily_map)],
    }


def download_ecmwf_ifs(
    start_date: str,
    end_date: str,
    data_root_folder: str | Path,
    *,
    grid: str = "F1280",
    retrieve_ml: bool = False,
    force: bool = False,
) -> list[Path]:
    """Retrieve daily ECMWF MARS operational analysis (`ml` or `sfc`) NetCDF files."""
    from ecmwfapi import ECMWFService

    root = Path(data_root_folder)
    stream_kind = "ml" if retrieve_ml else "sfc"
    out_dir = root / "netcdf" / "global" / str(grid) / stream_kind
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.period_range(start=start_date, end=end_date, freq="D")
    pending: list[tuple[pd.Timestamp, Path]] = []
    outputs: list[Path] = []
    for raw_date in dates:
        ts = raw_date.to_timestamp()
        target = out_dir / f"ECMWF-OA_{stream_kind}_{ts.strftime('%Y%m%d')}.nc"
        if not force and target.exists() and target.stat().st_size > 0:
            print(f"[SKIPPED] {target} already exists.")
            outputs.append(target)
            continue
        pending.append((ts, target))

    if not pending:
        return outputs

    server = ECMWFService("mars")
    for ts, target in pending:
        request = {
            "class": "od",
            "date": ts.strftime("%Y-%m-%d"),
            "expver": "1",
            "levelist": "all",
            "levtype": stream_kind,
            "param": WRF_VERTICAL_LEVEL_VARS if retrieve_ml else WRF_SURFACE_VARS,
            "stream": "oper",
            "type": "an",
            "time": "00/06/12/18",
            "grid": str(grid),
            "format": "netcdf",
        }
        print(f"Downloading ECMWF IFS {stream_kind} for {ts.strftime('%Y-%m-%d')} -> {target}")
        server.execute(request, str(target))
        outputs.append(target)
    return outputs


def build_arg_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="airblender fetch-ifs",
            description="Download ECMWF MARS operational IFS NetCDFs (`ml` or `sfc`).",
        )
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--data_root_folder", required=True, help="Root output directory")
    parser.add_argument("--grid", default="F1280", help="Target MARS grid (default: F1280)")
    parser.add_argument("--retrieve_ml", type=_strtobool, default=False, help="Retrieve model levels (true/false)")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    parser.add_argument("--mode", "--port", "--host", help="Ignored PyCharm debugger flags")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    download_ecmwf_ifs(
        args.start_date,
        args.end_date,
        args.data_root_folder,
        grid=args.grid,
        retrieve_ml=args.retrieve_ml,
        force=args.force,
    )
