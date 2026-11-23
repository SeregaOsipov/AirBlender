"""ECMWF OpenData HRES 0.1 deg S3 Byte-Range fetcher and legacy soil/surface helpers."""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence
import urllib.error
import urllib.request

import netCDF4
import numpy as np
import xarray as xr


OPENDATA_MIRROR_URLS = (
    "https://storage.googleapis.com/ecmwf-open-data",
    "https://data.ecmwf.int/forecasts",
    "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
)
S3_OPENDATA_BASE_URL = OPENDATA_MIRROR_URLS[0]

OPENDATA_0P1_SFC_PARAMS = frozenset({
    "2t",
    "2d",
    "10u",
    "10v",
    "sp",
    "msl",
    "skt",
    "lsm",
    "z",
    "sd",
    "sde",
    "rsn",
    "asn",
    "sithick",
    "sot",
    "vsw",
})
OPENDATA_INVARIANT_SFC_PARAMS = ("z", "lsm")


def _default_hres_dir() -> Path:
    return Path(os.environ.get("SCRATCH", "/tmp")) / "Data" / "ECMWF" / "HRES"


def _normalize_date(date: str | None) -> tuple[str | None, str]:
    if not date:
        return None, "latest"
    clean = str(date).replace("-", "").replace("_", "")[:8]
    return f"{clean[:4]}-{clean[4:6]}-{clean[6:8]}", clean


def _parse_hh(time_str: str | int = "00:00") -> str:
    raw = str(time_str).strip().replace(":", "")
    if raw.isdigit():
        return raw[:2].zfill(2)
    return "00"


def _build_0p1_rel_paths(
    clean_date: str,
    hh: str,
    step_h: int,
    levtype_group: str = "sfc",
) -> tuple[str, str]:
    stem = f"{clean_date}/{hh}z/ifs/0p1/oper/{clean_date}{hh}0000-{int(step_h)}h-oper-fc-{levtype_group}"
    return f"{stem}.index", f"{stem}.grib2"


def _build_0p1_urls(
    clean_date: str,
    hh: str,
    step_h: int,
    s3_base_url: str = S3_OPENDATA_BASE_URL,
    levtype_group: str = "sfc",
) -> tuple[str, str]:
    base = s3_base_url.rstrip("/")
    idx_rel, grib_rel = _build_0p1_rel_paths(clean_date, hh, step_h, levtype_group=levtype_group)
    return f"{base}/{idx_rel}", f"{base}/{grib_rel}"


def _resolve_mirror_urls(s3_base_url: str) -> tuple[str, ...]:
    base = str(s3_base_url).rstrip("/")
    known = [u.rstrip("/") for u in OPENDATA_MIRROR_URLS]
    if base in known:
        return tuple([base] + [u for u in known if u != base])
    return (base,)


def _http_get_bytes(
    url: str,
    *,
    byte_range: tuple[int, int] | None = None,
    timeout: float = 25.0,
    retries: int = 3,
) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(max(1, retries)):
        headers: dict[str, str] = {}
        if byte_range is not None:
            headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            last_exc = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                raise
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _http_get_rel(
    rel_path: str,
    *,
    s3_base_url: str = S3_OPENDATA_BASE_URL,
    byte_range: tuple[int, int] | None = None,
    timeout: float = 25.0,
    retries: int = 2,
) -> bytes:
    mirrors = _resolve_mirror_urls(s3_base_url)
    rel = rel_path.lstrip("/")
    last_exc: Exception | None = None
    for attempt in range(max(1, retries)):
        for mirror in mirrors:
            url = f"{mirror}/{rel}"
            try:
                return _http_get_bytes(url, byte_range=byte_range, timeout=timeout, retries=1)
            except urllib.error.HTTPError as exc:
                if exc.code == 404 or (400 <= exc.code < 500 and exc.code != 429):
                    raise
                last_exc = exc
            except Exception as exc:
                last_exc = exc
        if attempt + 1 < retries:
            time.sleep(1.5 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _coalesce_byte_ranges(records: Sequence[dict[str, Any]]) -> list[tuple[int, int]]:
    ordered = sorted(records, key=lambda r: int(r["_offset"]))
    ranges: list[tuple[int, int]] = []
    for rec in ordered:
        off = int(rec["_offset"])
        length = int(rec["_length"])
        if length <= 0:
            continue
        end = off + length - 1
        if ranges and off == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((off, end))
    return ranges


def probe_opendata_0p1(
    date_str: str,
    time_str: str = "00:00",
    max_step_h: int = 0,
    s3_base_url: str = S3_OPENDATA_BASE_URL,
    timeout: float = 10.0,
) -> tuple[bool, str]:
    """Check whether `ifs/0p1/oper` GRIB2 `.index` sidecars exist for `(date_str, time_str)`."""
    _, clean_date = _normalize_date(date_str)
    hh = _parse_hh(time_str)
    steps_to_check = [0]
    probe_max = int(max_step_h)
    if probe_max > 144 and probe_max % 6 != 0:
        probe_max = (probe_max // 6) * 6
    if probe_max > 0:
        steps_to_check.append(probe_max)
    try:
        for step_h in steps_to_check:
            idx_rel, _ = _build_0p1_rel_paths(clean_date, hh, step_h, levtype_group="sfc")
            raw = _http_get_rel(idx_rel, s3_base_url=s3_base_url, timeout=timeout, retries=1).decode("utf-8", errors="ignore")
            lines = [line for line in raw.splitlines() if line.strip()]
            if not lines:
                return False, f"Empty OpenData 0.1° index at {idx_rel}"
            json.loads(lines[0])
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _normalize_and_subset_ds(
    ds: xr.Dataset,
    area: Sequence[float] | None,
) -> xr.Dataset:
    if "longitude" in ds.coords:
        lon_vals = np.asarray(ds.longitude.values, dtype=np.float64)
        use_360 = bool(area is not None and len(area) == 4 and float(area[1]) >= 0.0 and float(area[3]) > 180.0)
        if use_360:
            new_lon = lon_vals % 360.0
        else:
            new_lon = ((lon_vals + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords(longitude=new_lon).sortby("longitude")

    if area is not None and len(area) == 4 and "latitude" in ds.coords and "longitude" in ds.coords:
        north, west, south, east = [float(x) for x in area]
        lat_lo = min(north, south) - 0.5
        lat_hi = max(north, south) + 0.5
        lon_lo = west - 0.5
        lon_hi = east + 0.5
        lat_vals = np.asarray(ds.latitude.values, dtype=np.float64)
        lon_vals = np.asarray(ds.longitude.values, dtype=np.float64)
        lat_mask = (lat_vals >= lat_lo) & (lat_vals <= lat_hi)
        lon_mask = (lon_vals >= lon_lo) & (lon_vals <= lon_hi)
        if np.any(lat_mask) and np.any(lon_mask):
            ds = ds.isel(latitude=lat_mask, longitude=lon_mask)
    return ds


def _decode_0p1_step_grib(
    grib_bytes: bytes,
    area: Sequence[float] | None,
) -> dict[str, xr.DataArray]:
    import cfgrib

    with tempfile.NamedTemporaryFile(prefix="ifs0p1_", suffix=".grib2", delete=False) as tf:
        tf.write(grib_bytes)
        tmp_path = tf.name

    idx_path = f"{tmp_path}.idx"
    out_vars: dict[str, xr.DataArray] = {}
    try:
        dsets = cfgrib.open_datasets(
            tmp_path,
            backend_kwargs={"indexpath": idx_path, "errors": "ignore"},
        )
        for ds in dsets:
            try:
                ds_sub = _normalize_and_subset_ds(ds, area)
                if "latitude" in ds_sub.dims and len(ds_sub.latitude) > 1:
                    if float(ds_sub.latitude.values[0]) < float(ds_sub.latitude.values[-1]):
                        ds_sub = ds_sub.sortby("latitude", ascending=False)

                for vname, da in ds_sub.data_vars.items():
                    if "soilLayer" in da.dims:
                        prefix = "stl" if vname in ("sot", "stl") else ("swvl" if vname in ("vsw", "swvl") else vname)
                        for layer_val in da.coords["soilLayer"].values:
                            lvl_i = int(round(float(layer_val)))
                            sub_da = da.sel(soilLayer=layer_val)
                            for c_drop in list(sub_da.coords):
                                if c_drop not in ("latitude", "longitude"):
                                    sub_da = sub_da.drop_vars(c_drop, errors="ignore")
                            out_name = f"{prefix}{lvl_i}"
                            out_vars[out_name] = xr.DataArray(
                                np.asarray(sub_da.values, dtype=np.float32),
                                dims=("latitude", "longitude"),
                                coords={
                                    "latitude": ds_sub.latitude.values,
                                    "longitude": ds_sub.longitude.values,
                                },
                                attrs=dict(da.attrs),
                            )
                    else:
                        sub_da = da.squeeze(drop=True)
                        if sub_da.ndim != 2 or set(sub_da.dims) != {"latitude", "longitude"}:
                            continue
                        canonical = "sd" if vname == "sde" and "sd" not in out_vars else vname
                        out_vars[canonical] = xr.DataArray(
                            np.asarray(sub_da.transpose("latitude", "longitude").values, dtype=np.float32),
                            dims=("latitude", "longitude"),
                            coords={
                                "latitude": ds_sub.latitude.values,
                                "longitude": ds_sub.longitude.values,
                            },
                            attrs=dict(da.attrs),
                        )
            finally:
                try:
                    ds.close()
                except Exception:
                    pass
    finally:
        for p in (tmp_path, idx_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    # Synthesize SST (sea-only SKT) and sea-ice cover from sithick if missing
    if "sst" not in out_vars and "skt" in out_vars and "lsm" in out_vars:
        lsm_arr = out_vars["lsm"].values
        skt_arr = out_vars["skt"].values
        sst_arr = np.where(lsm_arr < 0.5, skt_arr, np.nan).astype(np.float32)
        out_vars["sst"] = xr.DataArray(
            sst_arr,
            dims=out_vars["skt"].dims,
            coords=out_vars["skt"].coords,
            attrs={"long_name": "Sea surface temperature (from skt over sea)", "units": "K"},
        )
    if "siconc" not in out_vars and "lsm" in out_vars:
        if "sithick" in out_vars:
            ci_arr = np.where(np.nan_to_num(out_vars["sithick"].values, nan=0.0) > 0.01, 1.0, 0.0).astype(np.float32)
        else:
            ci_arr = np.zeros_like(out_vars["lsm"].values, dtype=np.float32)
        out_vars["siconc"] = xr.DataArray(
            ci_arr,
            dims=out_vars["lsm"].dims,
            coords=out_vars["lsm"].coords,
            attrs={"long_name": "Sea ice area fraction", "units": "(0 - 1)"},
        )

    return out_vars


def _existing_cdm_steps(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    try:
        with netCDF4.Dataset(path, "r") as ds:
            if "forecast_period" not in ds.dimensions:
                return 0
            n_fp = len(ds.dimensions["forecast_period"])
            if n_fp <= 0:
                return 0
            for k in OPENDATA_INVARIANT_SFC_PARAMS:
                if k not in ds.variables:
                    return 0
                v = ds.variables[k]
                if not np.isfinite(np.ma.filled(v[0], np.nan)).any():
                    return 0
                if n_fp > 1 and not np.isfinite(np.ma.filled(v[-1], np.nan)).any():
                    return 0
            return n_fp
    except Exception:
        return 0


def _load_or_fetch_step0_invariants(
    clean_date: str,
    hh: str,
    target_dir: Path,
    area: Sequence[float] | None,
    s3_base_url: str,
) -> dict[str, xr.DataArray]:
    """Load step-0 static invariants (`z`, `lsm`) from Day-1 NetCDF or OpenData step 0."""
    day0_fp = target_dir / f"ECMWF-IFS-0p1_sfc_{clean_date}.nc"
    if day0_fp.is_file() and day0_fp.stat().st_size > 0:
        try:
            with xr.open_dataset(day0_fp) as ds0:
                found: dict[str, xr.DataArray] = {}
                for k in OPENDATA_INVARIANT_SFC_PARAMS:
                    if k not in ds0:
                        continue
                    da = ds0[k]
                    if "forecast_period" in da.dims:
                        da = da.isel(forecast_period=0)
                    if "forecast_reference_time" in da.dims:
                        da = da.isel(forecast_reference_time=0)
                    da2 = da.squeeze(drop=True)
                    if set(da2.dims) == {"latitude", "longitude"} and np.isfinite(da2.values).any():
                        found[k] = da2.load()
                if all(k in found for k in OPENDATA_INVARIANT_SFC_PARAMS):
                    return found
        except Exception:
            pass

    idx_rel, grib_rel = _build_0p1_rel_paths(clean_date, hh, 0, levtype_group="sfc")
    idx_raw = _http_get_rel(idx_rel, s3_base_url=s3_base_url, timeout=20.0).decode("utf-8")
    records = [json.loads(line) for line in idx_raw.splitlines() if line.strip()]
    wanted = [r for r in records if r.get("param") in OPENDATA_INVARIANT_SFC_PARAMS]
    ranges = _coalesce_byte_ranges(wanted)
    if not ranges:
        return {}
    chunks = [
        _http_get_rel(grib_rel, s3_base_url=s3_base_url, byte_range=(start_b, end_b), timeout=30.0)
        for start_b, end_b in ranges
    ]
    decoded = _decode_0p1_step_grib(b"".join(chunks), area=area)
    return {
        k: decoded[k]
        for k in OPENDATA_INVARIANT_SFC_PARAMS
        if k in decoded and np.isfinite(decoded[k].values).any()
    }


def _write_cdm_sfc_daily(
    out_path: Path,
    cycle_dt: datetime,
    step_hours_list: Sequence[int],
    step_dicts: Sequence[dict[str, xr.DataArray]],
) -> Path:
    if not step_dicts or not step_dicts[0]:
        raise RuntimeError(f"No decoded surface variables to write for {out_path}")

    ref_da = next(iter(step_dicts[0].values()))
    lats = np.asarray(ref_da.coords["latitude"].values, dtype=np.float64)
    lons = np.asarray(ref_da.coords["longitude"].values, dtype=np.float64)
    n_fp = len(step_hours_list)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fp = out_path.with_suffix(f".nc.tmp.{os.getpid()}")
    if tmp_fp.exists():
        tmp_fp.unlink()

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    frt_sec = int((cycle_dt.replace(tzinfo=timezone.utc) - epoch).total_seconds())
    vt_secs = np.array(
        [[float(frt_sec + int(h) * 3600) for h in step_hours_list]],
        dtype=np.float64,
    )

    try:
        with netCDF4.Dataset(tmp_fp, "w", format="NETCDF4") as nc_out:
            nc_out.setncattr("Conventions", "CF-1.6")
            nc_out.setncattr("source", "ECMWF OpenData IFS 0.1 deg")
            nc_out.createDimension("forecast_period", n_fp)
            nc_out.createDimension("forecast_reference_time", 1)
            nc_out.createDimension("latitude", len(lats))
            nc_out.createDimension("longitude", len(lons))

            v_fp = nc_out.createVariable("forecast_period", "f8", ("forecast_period",), fill_value=np.float64(np.nan))
            v_fp.long_name = "time since forecast_reference_time"
            v_fp.standard_name = "forecast_period"
            v_fp.units = "hours"
            v_fp[:] = np.asarray(step_hours_list, dtype=np.float64)

            v_frt = nc_out.createVariable("forecast_reference_time", "i8", ("forecast_reference_time",))
            v_frt.long_name = "initial time of forecast"
            v_frt.standard_name = "forecast_reference_time"
            v_frt.units = "seconds since 1970-01-01"
            v_frt.calendar = "proleptic_gregorian"
            v_frt[:] = [frt_sec]

            v_vt = nc_out.createVariable(
                "valid_time", "f8", ("forecast_reference_time", "forecast_period"), fill_value=np.float64(np.nan)
            )
            v_vt.standard_name = "time"
            v_vt.long_name = "time"
            v_vt.units = "seconds since 1970-01-01"
            v_vt.calendar = "proleptic_gregorian"
            v_vt[:] = vt_secs

            v_lat = nc_out.createVariable("latitude", "f8", ("latitude",), fill_value=np.float64(np.nan))
            v_lat.units = "degrees_north"
            v_lat.standard_name = "latitude"
            v_lat.long_name = "latitude"
            if len(lats) > 1 and lats[0] > lats[-1]:
                v_lat.stored_direction = "decreasing"
            v_lat[:] = lats

            v_lon = nc_out.createVariable("longitude", "f8", ("longitude",), fill_value=np.float64(np.nan))
            v_lon.units = "degrees_east"
            v_lon.standard_name = "longitude"
            v_lon.long_name = "longitude"
            v_lon[:] = lons

            var_names = sorted({k for sd in step_dicts for k in sd.keys()})
            for vname in var_names:
                sample = next((sd[vname] for sd in step_dicts if vname in sd), None)
                if sample is None:
                    continue
                nc_var = nc_out.createVariable(
                    vname,
                    "f4",
                    ("forecast_period", "forecast_reference_time", "latitude", "longitude"),
                    fill_value=np.float32(np.nan),
                    zlib=True,
                    complevel=1,
                )
                for ak, av in sample.attrs.items():
                    if ak in ("_FillValue", "missing_value"):
                        continue
                    try:
                        nc_var.setncattr(ak, av)
                    except Exception:
                        pass
                nc_var.setncattr("coordinates", "valid_time")

                fallback_2d = None
                if vname in OPENDATA_INVARIANT_SFC_PARAMS:
                    fallback_da = next(
                        (sd[vname] for sd in step_dicts if vname in sd and np.isfinite(sd[vname].values).any()),
                        None,
                    )
                    if fallback_da is not None:
                        fallback_2d = np.asarray(fallback_da.values, dtype=np.float32)

                for t_idx, sd in enumerate(step_dicts):
                    if vname in sd and (fallback_2d is None or np.isfinite(sd[vname].values).any()):
                        nc_var[t_idx, 0, :, :] = np.asarray(sd[vname].values, dtype=np.float32)
                    elif fallback_2d is not None:
                        nc_var[t_idx, 0, :, :] = fallback_2d

        os.replace(tmp_fp, out_path)
    finally:
        if tmp_fp.exists():
            try:
                tmp_fp.unlink()
            except OSError:
                pass
    return out_path


def download_opendata_0p1_sfc_daily(
    start_date: str,
    out_dir: str | Path,
    *,
    time_str: str = "00:00",
    leadtimes_hours: Sequence[int] | None = None,
    area: Sequence[float] | None = None,
    max_workers: int = 6,
    force: bool = False,
    s3_base_url: str = S3_OPENDATA_BASE_URL,
) -> list[Path]:
    """
    Download 0.1 deg ECMWF OpenData HRES surface + 4-layer soil fields via S3 `.index`
    HTTP Byte-Range subsetting and write daily CDM NetCDFs `ECMWF-IFS-0p1_sfc_{YYYYMMDD}.nc`.
    """
    dash_date, clean_date = _normalize_date(start_date)
    assert dash_date is not None
    hh = _parse_hh(time_str)
    cycle_dt = datetime.strptime(f"{clean_date}{hh}", "%Y%m%d%H").replace(tzinfo=timezone.utc)
    raw_steps = sorted({int(h) for h in (leadtimes_hours if leadtimes_hours is not None else range(0, 121, 3))})
    steps = [h for h in raw_steps if h <= 144 or h % 6 == 0]
    if len(steps) != len(raw_steps):
        dropped = [h for h in raw_steps if h not in steps]
        print(f"Notice: Dropping OpenData 0.1° unsupported >144h odd steps {dropped} (6h cadence after 144h).")

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Group lead times by UTC valid calendar day
    day_groups: dict[str, list[int]] = {}
    for h in steps:
        valid_dt = cycle_dt + timedelta(hours=h)
        day_groups.setdefault(valid_dt.strftime("%Y%m%d"), []).append(h)

    written: list[Path] = []
    invariant_cache: dict[str, xr.DataArray] = {}
    for day_str, day_steps in sorted(day_groups.items()):
        out_fp = target_dir / f"ECMWF-IFS-0p1_sfc_{day_str}.nc"
        if not force and _existing_cdm_steps(out_fp) >= len(day_steps):
            print(f"[SKIPPED] {out_fp} already has {len(day_steps)} steps.")
            written.append(out_fp)
            continue

        def _fetch_one_step(step_h: int) -> tuple[int, dict[str, xr.DataArray]]:
            idx_rel, grib_rel = _build_0p1_rel_paths(clean_date, hh, step_h, levtype_group="sfc")
            idx_raw = _http_get_rel(idx_rel, s3_base_url=s3_base_url, timeout=20.0).decode("utf-8")
            records = [json.loads(line) for line in idx_raw.splitlines() if line.strip()]
            wanted = [r for r in records if r.get("param") in OPENDATA_0P1_SFC_PARAMS]
            ranges = _coalesce_byte_ranges(wanted)
            if not ranges:
                raise RuntimeError(f"No matching 0.1 deg surface/soil records found in {idx_rel}")
            chunks = []
            for start_b, end_b in ranges:
                chunks.append(_http_get_rel(grib_rel, s3_base_url=s3_base_url, byte_range=(start_b, end_b), timeout=30.0))
            decoded = _decode_0p1_step_grib(b"".join(chunks), area=area)
            return step_h, decoded

        step_results: dict[int, dict[str, xr.DataArray]] = {}
        n_workers = max(1, min(int(max_workers), len(day_steps)))
        print(
            f"Downloading ECMWF OpenData 0.1° SFC/Soil ({clean_date} {hh}Z -> day {day_str}, "
            f"{len(day_steps)} step(s))..."
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_fetch_one_step, h) for h in day_steps]
            for fut in concurrent.futures.as_completed(futs):
                sh, sdict = fut.result()
                step_results[sh] = sdict

        if 0 in step_results:
            for k in OPENDATA_INVARIANT_SFC_PARAMS:
                if k in step_results[0] and np.isfinite(step_results[0][k].values).any():
                    invariant_cache[k] = step_results[0][k]

        if any(k not in invariant_cache for k in OPENDATA_INVARIANT_SFC_PARAMS):
            fetched_inv = _load_or_fetch_step0_invariants(clean_date, hh, target_dir, area, s3_base_url)
            for k, da in fetched_inv.items():
                invariant_cache.setdefault(k, da)

        ordered_dicts = [step_results[h] for h in day_steps]
        for sdict in ordered_dicts:
            for k, da_inv in invariant_cache.items():
                if k not in sdict or not np.isfinite(sdict[k].values).any():
                    sdict[k] = da_inv

        _write_cdm_sfc_daily(out_fp, cycle_dt, day_steps, ordered_dicts)
        print(f"[SAVED] {out_fp} ({out_fp.stat().st_size / (1024 ** 2):.2f} MB)")
        written.append(out_fp)

    return written


def download_opendata_soil(
    date: str | None = None,
    out_dir: str | Path | None = None,
    *,
    time: int = 0,
    step: int = 0,
    source: str = "ecmwf",
    force: bool = False,
) -> Path:
    """Download ECMWF OpenData 4-layer soil fields (`vsw`, `sot`) into GRIB."""
    from ecmwf.opendata import Client

    target_dir = Path(out_dir) if out_dir is not None else _default_hres_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    dash_date, tag = _normalize_date(date)
    target_fp = target_dir / f"ecmwf_hres_sol_{tag}.grib"
    if not force and target_fp.exists() and target_fp.stat().st_size > 0:
        print(f"[SKIPPED] {target_fp} already exists.")
        return target_fp

    client = Client(source=source)
    req: dict = {
        "type": "fc",
        "stream": "oper",
        "time": int(time),
        "step": int(step),
        "levtype": "sol",
        "param": ["vsw", "sot"],
        "levelist": [1, 2, 3, 4],
        "target": str(target_fp),
    }
    if dash_date is not None:
        req["date"] = dash_date

    print(f"Downloading OpenData SOL ({tag}) -> {target_fp}")
    client.retrieve(**req)
    return target_fp


def download_opendata_sfc(
    date: str | None = None,
    out_dir: str | Path | None = None,
    *,
    time: int = 0,
    step: int = 0,
    source: str = "ecmwf",
    force: bool = False,
) -> Path:
    """Download ECMWF OpenData surface meteorological and land fields into GRIB."""
    from ecmwf.opendata import Client

    target_dir = Path(out_dir) if out_dir is not None else _default_hres_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    dash_date, tag = _normalize_date(date)
    fname = f"ecmwf_sfc_hres_{tag}.grib" if dash_date is not None else "ecmwf_sfc_hres.grib"
    target_fp = target_dir / fname
    if not force and target_fp.exists() and target_fp.stat().st_size > 0:
        print(f"[SKIPPED] {target_fp} already exists.")
        return target_fp

    client = Client(source=source)
    req: dict = {
        "type": "fc",
        "stream": "oper",
        "time": int(time),
        "step": int(step),
        "param": [
            "swvl1", "swvl2", "swvl3", "swvl4",
            "stl1", "stl2", "stl3", "stl4",
            "skt", "sd", "rsn", "lsm",
        ],
        "target": str(target_fp),
    }
    if dash_date is not None:
        req["date"] = dash_date

    print(f"Downloading OpenData SFC ({tag}) -> {target_fp}")
    client.retrieve(**req)
    return target_fp


def build_arg_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="airblender fetch-opendata",
            description="Download ECMWF OpenData 0.1 deg HRES surface/soil NetCDFs or raw GRIBs.",
        )
    parser.add_argument(
        "--mode",
        choices=["sol", "sfc", "both", "s3-0p1"],
        default="sol",
        help="OpenData subset/method to fetch (`s3-0p1` writes regional daily NetCDFs without ecmwf-opendata)",
    )
    parser.add_argument("--date", default=None, help="Forecast date YYYY-MM-DD or YYYYMMDD (default: latest)")
    parser.add_argument("--outdir", default=None, help="Target directory (default: ${SCRATCH:-/tmp}/Data/ECMWF/HRES)")
    parser.add_argument("--time", type=int, default=0, help="Cycle hour (default: 0)")
    parser.add_argument("--step", type=int, default=0, help="Forecast step in hours for sol/sfc mode (default: 0)")
    parser.add_argument("--leadtimes", nargs="+", type=int, default=None, help="Lead times in hours for s3-0p1 mode")
    parser.add_argument("--area", nargs=4, type=float, default=None, help="Bounding box [N, W, S, E] for s3-0p1 mode")
    parser.add_argument("--source", default="ecmwf", help="ecmwf-opendata source (default: ecmwf)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.mode == "s3-0p1":
        if not args.date:
            parser.error("--date is required with --mode s3-0p1")
        out_dir = Path(args.outdir) if args.outdir is not None else _default_hres_dir()
        download_opendata_0p1_sfc_daily(
            args.date,
            out_dir,
            time_str=f"{int(args.time):02d}:00",
            leadtimes_hours=args.leadtimes,
            area=args.area,
            force=args.force,
        )
        return
    if args.mode in ("sol", "both"):
        download_opendata_soil(
            date=args.date,
            out_dir=args.outdir,
            time=args.time,
            step=args.step,
            source=args.source,
            force=args.force,
        )
    if args.mode in ("sfc", "both"):
        download_opendata_sfc(
            date=args.date,
            out_dir=args.outdir,
            time=args.time,
            step=args.step,
            source=args.source,
            force=args.force,
        )
