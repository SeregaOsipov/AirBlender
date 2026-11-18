"""Unified donor-to-WRF-Chem IC/BC injector (`cams`, `merra2`, `emac`)."""
from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Iterable, Sequence

import netCDF4 as nc
import numpy as np
import xarray as xr

from airblender.physics.dsl import parse_mapping_rule
from airblender.physics.mappings import get_mapping_for_source
from airblender.regrid.wrf_regrid import (
    flatten_and_line_up_boundaries,
    get_met_file_by_time,
    hor_interpolate_3d_field_on_wrf_boundary,
    hor_interpolate_3d_field_on_wrf_grid,
    update_boundaries,
    update_tendency_boundaries,
    ver_interpolate_3d_field_on_wrf_boundary,
    ver_interpolate_3d_field_on_wrf_grid,
)
from airblender.sources import get_donor_adapter
from airblender.utils.time_io import derive_wrf_eta_pressure, extract_wrf_times, parse_wrf_datetime


def _resolve_dates(
    wrf_ic_fp: Path,
    wrf_bdy_fp: Path,
    start_date,
    end_date,
    hourly_interval: float,
    do_bc: bool,
):
    """Extract processing timestamps and boundary step `delta_t` from `wrfbdy` (or `wrfinput` if IC-only)."""
    use_bdy = wrf_bdy_fp.exists()
    if do_bc and not use_bdy:
        raise FileNotFoundError(f"WRF boundary file not found: {wrf_bdy_fp}")

    if use_bdy:
        print("\nDeriving dates to process from wrf_bdy file\n")
        with nc.Dataset(wrf_bdy_fp, "r") as wrfbdy_f:
            wrf_bdy_dates = extract_wrf_times(wrfbdy_f)
        dates_all = wrf_bdy_dates
    else:
        print("\nDeriving date to process from wrfinput file (IC-only mode)\n")
        with nc.Dataset(wrf_ic_fp, "r") as wrfic_f:
            wrf_bdy_dates = extract_wrf_times(wrfic_f)
        dates_all = wrf_bdy_dates

    delta_t = (
        (dates_all[1] - dates_all[0]).total_seconds()
        if len(dates_all) >= 2
        else float(hourly_interval) * 3600.0
    )

    dates_to_process = dates_all
    if start_date is not None:
        s_dt = parse_wrf_datetime(start_date) if isinstance(start_date, str) else start_date
        dates_to_process = dates_to_process[dates_to_process >= s_dt]
    if end_date is not None:
        e_dt = parse_wrf_datetime(end_date) if isinstance(end_date, str) else end_date
        dates_to_process = dates_to_process[dates_to_process <= e_dt]
    return dates_to_process, wrf_bdy_dates, delta_t


def _write_ic_slice(
    *,
    adapter,
    mappings,
    date,
    wrf_ic_fp: Path,
    wrf_ic_ds: xr.Dataset,
    wrf_met_dir: Path,
    wrf_met_files_name_template: str,
    wrf_met_files_date_format: str,
    zero_out_first: bool,
) -> None:
    """Populate chemical and aerosol initial conditions in `wrfinput_d0*` in place."""
    print(f"INITIAL CONDITIONS: {date}")
    active_streams = list(dict.fromkeys(m.output_stream for m in mappings))
    ctx = adapter.prepare_date_context(date, sel_method=None, active_streams=active_streams)

    met_file_name = get_met_file_by_time(date.strftime(wrf_met_files_date_format), wrf_met_files_name_template)
    met_fp = os.path.join(str(wrf_met_dir), met_file_name)
    print(f"Opening metfile: {met_fp}")
    wrf_met_ds = xr.open_dataset(met_fp).set_coords(["XLAT_M", "XLONG_M"])
    derive_wrf_eta_pressure(wrf_met_ds, wrf_ic_ds)
    wrf_met_ds = wrf_met_ds.squeeze()

    print(f"Opening wrfinput: {wrf_ic_fp}")
    wrfinput_f = nc.Dataset(wrf_ic_fp, "r+")

    print("INITIAL CONDITIONS: Increments")
    zeroed_wrf_keys: set[str] = set()
    try:
        for mapping in mappings:
            print(f"Processing mapping: {mapping.mapping_rule_str}")
            for rule_vo in parse_mapping_rule(mapping.mapping_rule_str):
                donor_key = str(rule_vo["merra_key"])
                wrf_key = str(rule_vo["wrf_key"])
                print(f"\t\t - Reading {donor_key}")
                if wrf_key not in wrfinput_f.variables:
                    print(f"\t\t   [WARNING] Variable {wrf_key} not found in wrfinput, skipping.")
                    continue

                resolved_stream, donor_stream_ds, resolved_key = adapter.resolve_variable(
                    mapping.output_stream, donor_key, wrf_key, ctx
                )
                if donor_stream_ds is None or resolved_stream is None or resolved_key is None:
                    print(
                        f"\t\t   [WARNING] Variable {donor_key} not found in "
                        f"'{mapping.output_stream}' (or fallback), skipping {wrf_key}."
                    )
                    continue

                if zero_out_first and wrf_key not in zeroed_wrf_keys:
                    print(f"\t\t - Zeroing out {wrf_key} in wrfinput before increments")
                    wrfinput_f.variables[wrf_key][0, :] = 0
                    zeroed_wrf_keys.add(wrf_key)

                donor_p_rho_hi_da = adapter.get_stream_pressure(
                    resolved_stream, donor_stream_ds, wrf_ic_ds, ctx, is_boundary=False
                )
                donor_var_da = donor_stream_ds[resolved_key].load()

                print(f"\t\t - Horizontal interpolation of {resolved_key} on WRF horizontal grid")
                donor_var_hi_da = hor_interpolate_3d_field_on_wrf_grid(donor_var_da, wrf_ic_ds)

                print(f"\t\t - Vertical interpolation of {resolved_key} on WRF vertical grid")
                donor_var_hvi_da = ver_interpolate_3d_field_on_wrf_grid(
                    donor_var_hi_da, donor_p_rho_hi_da, wrf_met_ds.p_rho
                )

                print(
                    f"\t\t - Updating wrfinput field {wrf_key}[0] += {resolved_key} * "
                    f"{rule_vo['wrf_multiplier']} * {rule_vo['wrf_exponent']:.1e}"
                )
                increment_da = donor_var_hvi_da * float(rule_vo["wrf_multiplier"]) * float(rule_vo["wrf_exponent"])
                wrfinput_f.variables[wrf_key][0, :] = (
                    wrfinput_f.variables[wrf_key][0, :]
                    + increment_da.transpose("bottom_top", "south_north", "west_east")
                )
    finally:
        wrfinput_f.close()
        wrf_met_ds.close()
        adapter.close_context(ctx)

    print("DONE: INITIAL CONDITIONS")


def _write_bc_slices(
    *,
    adapter,
    mappings,
    dates_to_process,
    wrf_bdy_dates,
    delta_t: float,
    wrf_bdy_fp: Path,
    wrf_ic_ds: xr.Dataset,
    wrf_met_dir: Path,
    wrf_met_files_name_template: str,
    wrf_met_files_date_format: str,
    zero_out_first: bool,
    start_time: float,
) -> None:
    """Populate lateral boundary concentrations and tendencies in `wrfbdy_d01` in place."""
    print("\n\nSTART BOUNDARY CONDITIONS")
    print(f"Opening WRF bdy: {wrf_bdy_fp}")
    wrfbdy_f = nc.Dataset(wrf_bdy_fp, "r+")
    active_streams = list(dict.fromkeys(m.output_stream for m in mappings))

    prev_donor_time_val = None
    prev_p_rho_on_bnd = None
    prev_time_index = None
    prev_updated_wrf_keys: list[str] = []

    try:
        for date in dates_to_process:
            print(f"\n\tBCs, next date: {date}")
            sel_method = "nearest" if adapter.source_name == "cams" else None
            ctx = adapter.prepare_date_context(date, sel_method=sel_method, active_streams=active_streams)

            met_file_name = get_met_file_by_time(
                date.strftime(wrf_met_files_date_format), wrf_met_files_name_template
            )
            met_fp = os.path.join(str(wrf_met_dir), met_file_name)
            print(f"\tReading WRF Pressure from: {met_fp}")
            wrf_met_ds = xr.open_dataset(met_fp).set_coords(["XLAT_M", "XLONG_M"])
            derive_wrf_eta_pressure(wrf_met_ds, wrf_ic_ds)
            wrf_met_ds = wrf_met_ds.squeeze()

            wrf_met_ds["p_rho_on_bnd"] = flatten_and_line_up_boundaries(wrf_met_ds.p_rho)
            wrf_met_ds["boundary_lons"] = flatten_and_line_up_boundaries(wrf_met_ds.XLONG_M)
            wrf_met_ds["boundary_lats"] = flatten_and_line_up_boundaries(wrf_met_ds.XLAT_M)

            time_index_in_wrfbdy = wrf_bdy_dates.get_loc(date)
            donor_time_val = adapter.get_time_signature(ctx)
            is_same_as_prev = (
                prev_time_index is not None
                and bool(donor_time_val)
                and donor_time_val == prev_donor_time_val
                and prev_p_rho_on_bnd is not None
                and np.array_equal(wrf_met_ds.p_rho_on_bnd.values, prev_p_rho_on_bnd)
            )

            updated_wrf_keys: list[str] = []
            try:
                if is_same_as_prev:
                    print(
                        f"\tReusing boundary concentrations from previous time index {prev_time_index} "
                        "(donor data and pressure are identical)"
                    )
                    for wrf_key in prev_updated_wrf_keys:
                        if wrf_key + "_BXS" in wrfbdy_f.variables:
                            for sfx in ("_BXS", "_BXE", "_BYS", "_BYE"):
                                wrfbdy_f.variables[wrf_key + sfx][time_index_in_wrfbdy, :] = wrfbdy_f.variables[
                                    wrf_key + sfx
                                ][prev_time_index, :]
                    updated_wrf_keys = list(prev_updated_wrf_keys)
                else:
                    zeroed_wrf_keys: set[str] = set()
                    for mapping in mappings:
                        print(f"Processing mapping: {mapping.mapping_rule_str}")
                        for rule_vo in parse_mapping_rule(mapping.mapping_rule_str):
                            donor_key = str(rule_vo["merra_key"])
                            wrf_key = str(rule_vo["wrf_key"])
                            print(f"\t\t - Reading {donor_key}")

                            if wrf_key + "_BXS" not in wrfbdy_f.variables:
                                print(f"\t\t   [WARNING] Variable {wrf_key} not found in wrfbdy, skipping.")
                                continue

                            resolved_stream, donor_stream_ds, resolved_key = adapter.resolve_variable(
                                mapping.output_stream, donor_key, wrf_key, ctx
                            )
                            if donor_stream_ds is None or resolved_stream is None or resolved_key is None:
                                print(
                                    f"\t\t   [WARNING] Variable {donor_key} not found in "
                                    f"'{mapping.output_stream}' (or fallback), skipping {wrf_key}."
                                )
                                continue

                            if zero_out_first and wrf_key not in zeroed_wrf_keys:
                                for sfx in ("_BXS", "_BXE", "_BYS", "_BYE", "_BTXS", "_BTXE", "_BTYS", "_BTYE"):
                                    wrfbdy_f.variables[wrf_key + sfx][time_index_in_wrfbdy, :] = 0
                                zeroed_wrf_keys.add(wrf_key)

                            if wrf_key not in updated_wrf_keys:
                                updated_wrf_keys.append(wrf_key)

                            donor_p_rho_hi_da = adapter.get_stream_pressure(
                                resolved_stream, donor_stream_ds, wrf_met_ds, ctx, is_boundary=True
                            )
                            donor_var_da = donor_stream_ds[resolved_key].load()
                            print(f"\t\tHorizontal interpolation of {resolved_key} on WRF boundary")
                            donor_var_hi_da = hor_interpolate_3d_field_on_wrf_boundary(donor_var_da, wrf_met_ds)
                            print(f"\t\tVertical interpolation of {resolved_key} on WRF boundary")
                            donor_var_hvi_da = ver_interpolate_3d_field_on_wrf_boundary(
                                donor_var_hi_da, donor_p_rho_hi_da, wrf_met_ds.p_rho_on_bnd
                            )

                            print(
                                f"\t\t - Updating wrfbdy: {wrf_key}[{time_index_in_wrfbdy}] += {resolved_key} * "
                                f"{rule_vo['wrf_multiplier']} * {rule_vo['wrf_exponent']:.1e}"
                            )
                            increment_on_bnd_da = (
                                donor_var_hvi_da
                                * float(rule_vo["wrf_multiplier"])
                                * float(rule_vo["wrf_exponent"])
                            )
                            update_boundaries(
                                increment_on_bnd_da, wrf_met_ds, wrfbdy_f, wrf_key, time_index_in_wrfbdy
                            )

                print(f"Prescribing uniform tendency in BC for these fields: {updated_wrf_keys}")
                for wrf_key in updated_wrf_keys:
                    if wrf_key + "_BXS" in wrfbdy_f.variables:
                        update_tendency_boundaries(wrfbdy_f, wrf_key, time_index_in_wrfbdy, delta_t)

                prev_donor_time_val = donor_time_val
                prev_p_rho_on_bnd = wrf_met_ds.p_rho_on_bnd.values.copy()
                prev_time_index = time_index_in_wrfbdy
                prev_updated_wrf_keys = list(updated_wrf_keys)
                print("--- %s seconds ---" % (time.time() - start_time))
            finally:
                adapter.close_context(ctx)
                wrf_met_ds.close()
    finally:
        print(f"Closing {wrf_bdy_fp.name}")
        wrfbdy_f.close()

    print("FINISH BOUNDARY CONDITIONS")


def run_wrf_icbc(
    *,
    source: str,
    reanalysis_dir: str | Path,
    wrf_dir: str | Path,
    wrf_met_dir: str | Path,
    reanalysis_file_name_template: str | None = None,
    wrf_input: str | Sequence[str] = "wrfinput_d01",
    wrf_bdy_file: str = "wrfbdy_d01",
    wrf_met_files_name_template: str = "met_em.d01.{}.nc",
    wrf_met_files_date_format: str = "%Y-%m-%d_%H_%M_%S",
    start_date=None,
    end_date=None,
    hourly_interval: float = 3,
    do_ic: bool = False,
    do_bc: bool = False,
    zero_out_first: bool = False,
    streams: Iterable[str] | None = None,
    scheme: str = "made_sorgam",
) -> None:
    """
    Unified chemical and aerosol IC/BC injector for WRF-Chem from CAMS, MERRA-2, or EMAC.
    """
    start_time = time.time()
    wrf_dir_p = Path(wrf_dir)
    wrf_met_dir_p = Path(wrf_met_dir)

    wrf_inputs = [wrf_input] if isinstance(wrf_input, (str, Path)) else list(wrf_input)
    primary_ic_fp = wrf_dir_p / str(wrf_inputs[0])
    wrf_bdy_fp = wrf_dir_p / wrf_bdy_file
    print(f"WRF bdy fp: {wrf_bdy_fp}")
    print(f"WRF ic fp: {primary_ic_fp}")

    mappings = get_mapping_for_source(source, scheme=scheme)
    if streams:
        allowed_streams = set(streams)
        mappings = [m for m in mappings if m.output_stream in allowed_streams]

    print("\nConversion MAP:")
    for mapping in mappings:
        print(f"{mapping.mapping_rule_str}:\t")

    adapter = get_donor_adapter(source, str(reanalysis_dir), reanalysis_file_name_template)

    dates_to_process, wrf_bdy_dates, delta_t = _resolve_dates(
        primary_ic_fp, wrf_bdy_fp, start_date, end_date, hourly_interval, do_bc=do_bc
    )
    if len(dates_to_process) == 0:
        raise ValueError(
            f"No matching timestamps found between wrfbdy/wrfinput and [{start_date}, {end_date}]."
        )

    for ic_idx, ic_name in enumerate(wrf_inputs):
        wrf_ic_fp = wrf_dir_p / str(ic_name)
        if not wrf_ic_fp.exists():
            raise FileNotFoundError(f"WRF input file not found: {wrf_ic_fp}")

        with xr.open_dataset(wrf_ic_fp) as ds_tmp:
            wrf_ic_ds = ds_tmp.load()

        # Infer met_em domain template for extra domains if user passed a list of wrfinputs.
        met_tpl = wrf_met_files_name_template
        if ic_idx > 0 and "d01" in met_tpl and "d0" in str(ic_name):
            import re as _re

            m_dom = _re.search(r"(d\d{2})", str(ic_name))
            if m_dom:
                met_tpl = met_tpl.replace("d01", m_dom.group(1))

        if do_ic:
            _write_ic_slice(
                adapter=adapter,
                mappings=mappings,
                date=dates_to_process[0],
                wrf_ic_fp=wrf_ic_fp,
                wrf_ic_ds=wrf_ic_ds,
                wrf_met_dir=wrf_met_dir_p,
                wrf_met_files_name_template=met_tpl,
                wrf_met_files_date_format=wrf_met_files_date_format,
                zero_out_first=zero_out_first,
            )

        if do_bc and ic_idx == 0:
            _write_bc_slices(
                adapter=adapter,
                mappings=mappings,
                dates_to_process=dates_to_process,
                wrf_bdy_dates=wrf_bdy_dates,
                delta_t=delta_t,
                wrf_bdy_fp=wrf_bdy_fp,
                wrf_ic_ds=wrf_ic_ds,
                wrf_met_dir=wrf_met_dir_p,
                wrf_met_files_name_template=met_tpl,
                wrf_met_files_date_format=wrf_met_files_date_format,
                zero_out_first=zero_out_first,
                start_time=start_time,
            )

    print("--- %s seconds ---" % (time.time() - start_time))
