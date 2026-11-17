import argparse
import os
import time

import regridding_utils
from scipy import interpolate
from climpy.utils.time_utils import argsparse_datetime
from dateutil import rrule
import numpy as np
from datetime import datetime

from climpy.utils.wrf_utils import derive_wrf_pressure_from_met_and_input_files
from mapping_utils import get_cams2wrf_mapping
from climpy.utils.cams_utils import derive_cams_pressure_profile
from lexical_utils import parse_mapping_rule, get_unique_wrf_keys_from_mappings
import wrf as wrf
import pandas as pd
import datetime as dt
import netCDF4 as nc
import xarray as xr

from regridding_utils import flatten_and_line_up_boundaries

"""
TODO: I SHOULD BE ABLE TO MERGE ALL THREE (CAMS, MERRA2, EMAC) into one

Vertical Grid in CAMS: https://confluence.ecmwf.int/display/UDOC/L60+model+level+definitions
The $a$ and $b$ coefficients are stored in the pv (vertical coordinate parameters) array.

How to run AREAD 2022:
gogomamba
year=2022
data_dir=/project/k10048/Projects/AirQuality/Data/AREAD/IC_BC_debug/
python -u ${MERRA2BC}/merra2wrf_main.py --hourly_interval=3 --do_IC --do_BC --zero_out_first --wrf_dir=${data_dir}/debugICBC/ --wrf_met_dir=${data_dir}/met_em/ --wrf_met_files=met_em.d01.${year}-* >& log.emac2wrf

source /project/k10066/osipovs/.commonrc; gogomamba; mamba activate py311
wrf_dir=/project/k10048/Projects/AirQuality/Data/NCEC/IC_BC/MERRA2_CAMS/
wrf_dir=/scratch/osipovs/Data/AirQuality/NCEC/IC_BC/MERRA2_CAMS
wrf_met_dir=/scratch/osipovs/Models/WRF_run/run_real/
python -u ${MERRA2BC}/cams2wrf_main.py --hourly_interval=3 --do_IC --do_BC --zero_out_first --wrf_dir=${wrf_dir} --wrf_met_dir=${wrf_met_dir} --wrf_met_files=met_em.d01.* --wrf_met_files_date_format=%Y-%m-%d_%H:%M:%S >& log.cams2wrf

Submit script on Shaheen 3
#NCEC
sbatch --job-name=cams2wrf --account=k10009 --time=24:00:00 --partition=shared <<'EOT'
#!/bin/bash -l
source /project/k10066/osipovs/.commonrc; gogomamba; mamba activate py311
wrf_dir=/scratch/osipovs/Data/AirQuality/NCEC/IC_BC/MERRA2_CAMS
wrf_met_dir=/scratch/osipovs/Models/WRF_run/run_real/
#python -u ${MERRA2BC}/cams2wrf_main.py --hourly_interval=3 --do_IC --do_BC --zero_out_first --wrf_dir=${wrf_dir} --wrf_met_dir=${wrf_met_dir} --wrf_met_files=met_em.d01.* --wrf_met_files_date_format=%Y-%m-%d_%H:%M:%S >& log.cams2wrf
EOT

#THOFA
sbatch --job-name=cams2wrf --account=k10009 --time=24:00:00 --partition=shared <<'EOT'
#!/bin/bash -l
source /project/k10066/osipovs/.commonrc; gogomamba; mamba activate py311
wrf_dir=/scratch/osipovs/Data/AirQuality/THOFA/IC_BC/MERRA2CAMS
wrf_met_dir=/project/k10048/osipovs/Data/AirQuality/THOFA/IC_BC/met_em/
python -u ${MERRA2BC}/cams2wrf_main.py --hourly_interval=3 --do_IC --do_BC --zero_out_first --wrf_dir=${wrf_dir} --wrf_met_dir=${wrf_met_dir} --wrf_met_files=met_em.d01.* --wrf_met_files_date_format=%Y-%m-%d_%H:%M:%S
EOT

"""
#%%
parser = argparse.ArgumentParser()
parser.add_argument("--start_date", help="YYYY-MM-DD_HH:MM:SS format", default=None, type=argsparse_datetime)  # default='2017-08-31_21:00:00')  #
parser.add_argument("--end_date", help="YYYY-MM-DD_HH:MM:SS format", default=None, type=argsparse_datetime)  # default='2017-09-01_00:00:00')  #
parser.add_argument("--hourly_interval", help="time delta in dates to process", default=3)
parser.add_argument("--do_IC", help="process initial conditions?", action='store_true')
parser.add_argument("--do_BC", help="process boundary conditions?", action='store_true')
parser.add_argument("--zero_out_first", help="zero out fields in IC and BC first?", action='store_true')
# setup parent model
parser.add_argument("--reanalysis_dir", default='/project/k10048/Data/ECMWF/CAMS/')  # CAMS in this case
parser.add_argument("--reanalysis_file_name_template", default='ECMWF-CAMS_{stream}_{date_time}.nc')
# setup downscaling model (WRF)
parser.add_argument("--wrf_dir", help="folder containing WRF IC & BC files")   # , default=/Data/AirQuality/AQABA/IC_BC')
parser.add_argument("--wrf_input", help="use default wrfinput_d01", default='wrfinput_d01')
parser.add_argument("--wrf_bdy_file", help="use default wrfbdy_d01", default='wrfbdy_d01')
parser.add_argument("--wrf_met_dir", help="use default wrfbdy_d01")  # default=root_path + '/Data/AirQuality/EMME/IC_BC/met_em'
# parser.add_argument("--wrf_met_files", help="met_em file names template")  # , default='met_em.d01.2017-0*')
parser.add_argument("--wrf_met_files_name_template", help="met_em file names template", default='met_em.d01.{}.nc')
parser.add_argument("--wrf_met_files_date_format", help="met_em file names template", default='%Y-%m-%d_%H_%M_%S')  # or '%Y-%m-%d_%H:%M:%S'
parser.add_argument("--streams", nargs="*", default=None, help="Optional subset of donor streams to process (e.g. --streams ghg)")

parser.add_argument("--mode", "--port", "--host", help="the are only to support pycharm debugging")
args = parser.parse_args()
#%% DEBUG settings
# args.wrf_dir = '/project/k10048/Projects/AirQuality/Data/NCEC/IC_BC/MERRA2_CAMS/'
# args.wrf_met_dir = '/scratch/osipovs/Models/WRF_run/run_real/'
# args.wrf_met_files = 'met_em.d01.2021-0*'
# args.do_IC = True
# args.do_BC = True
# args.zero_out_first = True # False  # just speed up
# args.wrf_met_files_date_format='%Y-%m-%d_%H:%M:%S'
#%%
wrf_bdy_fp = args.wrf_dir + "/" + args.wrf_bdy_file
wrf_ic_fp = args.wrf_dir + "/" + args.wrf_input
print('WRF bdy fp: {}'.format(wrf_bdy_fp))
print('WRF ic fp: {}'.format(wrf_ic_fp))

wrf_ic_ds = xr.open_dataset(wrf_ic_fp).load()
wrf_ic_ds.close()

start_time = time.time()

mappings = get_cams2wrf_mapping()
if args.streams:
    allowed_streams = set(args.streams)
    mappings = [m for m in mappings if m.output_stream in allowed_streams]

print("\nConversion MAP:")
for mapping in mappings:
    print(mapping.mapping_rule_str + ":\t")
#%%
print('\nDeriving dates to process from wrf_bdy file\n')
wrfbdy_f = nc.Dataset(wrf_bdy_fp, 'r')
wrf_bdy_dates = wrf.extract_times(wrfbdy_f, wrf.ALL_TIMES)
wrfbdy_f.close()
wrf_bdy_dates = pd.to_datetime(wrf_bdy_dates)
dates_to_process = wrf_bdy_dates
delta_t = (dates_to_process[1] - dates_to_process[0]).total_seconds()

if args.start_date is not None:
    dates_to_process = dates_to_process[dates_to_process >= args.start_date]
if args.end_date is not None:
    dates_to_process = dates_to_process[dates_to_process <= args.end_date]
#%% Aux
def xr_open_cams_dataset(fp):  # standardize file opening
    print("Opening file: {}".format(fp))
    ds = xr.open_dataset(fp)

    # Standardize time coordinate: cfgrib creates 2D valid_time(forecast_reference_time, forecast_period)
    if 'forecast_reference_time' in ds.dims and ds.sizes['forecast_reference_time'] == 1:
        ds = ds.squeeze(dim='forecast_reference_time', drop=True)
    if 'forecast_period' in ds.dims:
        if 'valid_time' in ds.coords:
            times = pd.to_datetime(ds.valid_time.values.flatten() if ds.valid_time.ndim > 1 else ds.valid_time.values)
            ds = ds.rename({'forecast_period': 'time'}).assign_coords(time=times)
            if 'valid_time' in ds:
                ds = ds.drop_vars('valid_time')
        else:
            ds = ds.rename({'forecast_period': 'time'})
    elif 'valid_time' in ds.coords:
        if 'valid_time' in ds.dims:
            ds = ds.rename({'valid_time': 'time'})
        elif 'time' not in ds.dims:
            times = pd.to_datetime(ds.valid_time.values.flatten() if ds.valid_time.ndim > 1 else ds.valid_time.values)
            if ds.dims and list(ds.dims.keys())[0] not in ['level', 'model_level', 'lat', 'latitude', 'lon', 'longitude']:
                first_dim = list(ds.dims.keys())[0]
                ds = ds.rename({first_dim: 'time'}).assign_coords(time=times)
            else:
                ds = ds.rename({'valid_time': 'time'})

    rename_dict = {}
    if 'latitude' in ds.coords or 'latitude' in ds.dims:
        rename_dict['latitude'] = 'lat'
    if 'longitude' in ds.coords or 'longitude' in ds.dims:
        rename_dict['longitude'] = 'lon'
    if rename_dict:
        ds = ds.rename(rename_dict)

    if 'lat' in ds.dims:
        ds = ds.isel(lat=slice(None, None, -1))  # flip lat dimension, to make ascending, satisfying interpolation requirements

    if 'model_level' in ds.dims:
        ds = ds.rename({'model_level': 'level'})
        # ds = ds.isel(level=slice(None, None, -1))  # flip dimensions

    return ds


xr_open_dataset_impl = xr_open_cams_dataset


def extract_cams_pv(ds):
    """Return 1D hybrid sigma-pressure pv array (a_stag followed by b_stag)."""
    if 'pv' in ds:
        return np.asarray(ds['pv'].values, dtype=np.float64)
    for var_name in ds.data_vars:
        if hasattr(ds[var_name], 'GRIB_pv'):
            return np.asarray(ds[var_name].GRIB_pv, dtype=np.float64)
    raise AttributeError("No variable with GRIB_pv or 'pv' found in CAMS dataset.")


def _streams_share_horizontal_grid(stream_ds, sfc_ds):
    if 'lat' not in stream_ds.dims or 'lon' not in stream_ds.dims:
        return False
    if stream_ds.sizes['lat'] != sfc_ds.sizes['lat'] or stream_ds.sizes['lon'] != sfc_ds.sizes['lon']:
        return False
    return bool(
        np.allclose(stream_ds.lat.values, sfc_ds.lat.values)
        and np.allclose(stream_ds.lon.values, sfc_ds.lon.values)
    )


def _derive_stream_p_rho(donor_stream_ds, donor_sfc_ds, wrf_target_ds, is_boundary=False):
    """
    Derive 3D hybrid midpoint pressure for any CAMS stream directly on the target
    WRF grid or boundary points by interpolating 2D surface pressure from sfc
    (a_k + b_k * sp is linear in sp, supporting mixed 0.1/0.4 deg grids).
    """
    pv = extract_cams_pv(donor_stream_ds)
    num_levels = donor_stream_ds.sizes['level']
    a = pv[:num_levels + 1]
    b = pv[num_levels + 1:]
    a_mid = 0.5 * (a[:-1] + a[1:])
    b_mid = 0.5 * (b[:-1] + b[1:])

    sp_vals = donor_sfc_ds['sp'].transpose('lat', 'lon').values
    f_sp = interpolate.RectBivariateSpline(donor_sfc_ds.lat, donor_sfc_ds.lon, sp_vals)
    if is_boundary:
        sp_target = np.asarray(
            f_sp(wrf_target_ds.boundary_lats, wrf_target_ds.boundary_lons, grid=False),
            dtype=np.float64,
        ).reshape(-1)
        p_rho = a_mid[:, None] + b_mid[:, None] * sp_target[None, :]
        return xr.DataArray(
            p_rho,
            dims=['level', 'boundary_points'],
            coords={'level': donor_stream_ds.level},
        )

    ny = wrf_target_ds.south_north.size
    nx = wrf_target_ds.west_east.size
    sp_target = np.asarray(
        f_sp(wrf_target_ds.XLAT, wrf_target_ds.XLONG, grid=False),
        dtype=np.float64,
    ).reshape(ny, nx)
    p_rho = a_mid[:, None, None] + b_mid[:, None, None] * sp_target[None, :, :]
    return xr.DataArray(
        p_rho,
        dims=['level', 'south_north', 'west_east'],
        coords={'level': donor_stream_ds.level},
    )


def _open_stream_for_date(stream, date, sel_method, stream_cache):
    if stream in stream_cache:
        return stream_cache[stream]
    fp = os.path.join(
        args.reanalysis_dir,
        args.reanalysis_file_name_template.format(date_time=date.strftime('%Y%m%d'), stream=stream),
    )
    if not os.path.exists(fp):
        print(f"\t\t   [WARNING] Donor stream '{stream}' file not found at {fp}.")
        stream_cache[stream] = None
        return None
    ds = xr_open_dataset_impl(fp)
    ds_t = ds.sel(time=date) if sel_method is None else ds.sel(time=date, method=sel_method)
    stream_cache[stream] = ds_t
    return ds_t


def _get_stream_p_rho(stream, donor_stream_ds, donor_sfc_ds, wrf_target_ds, p_rho_cache, is_boundary=False):
    if stream in p_rho_cache:
        return p_rho_cache[stream]
    if stream == 'ml' and _streams_share_horizontal_grid(donor_stream_ds, donor_sfc_ds):
        if 'p_rho' not in donor_stream_ds:
            derive_cams_pressure_profile(donor_stream_ds, donor_sfc_ds)
        if is_boundary:
            p_rho_da = regridding_utils.hor_interpolate_3d_field_on_wrf_boundary(donor_stream_ds.p_rho, wrf_target_ds)
        else:
            p_rho_da = regridding_utils.hor_interpolate_3d_field_on_wrf_grid(donor_stream_ds.p_rho, wrf_target_ds)
    else:
        p_rho_da = _derive_stream_p_rho(donor_stream_ds, donor_sfc_ds, wrf_target_ds, is_boundary=is_boundary)
    p_rho_cache[stream] = p_rho_da
    return p_rho_da


def _resolve_donor_source(mapping_stream, merra_key, wrf_key, date, sel_method, donor_streams):
    ds = _open_stream_for_date(mapping_stream, date, sel_method, donor_streams)
    if ds is not None and merra_key in ds:
        return mapping_stream, ds, merra_key

    if mapping_stream == 'ghg':
        ml_ds = _open_stream_for_date('ml', date, sel_method, donor_streams)
        if ml_ds is not None:
            fallback_keys = [merra_key, 'ch4_c'] if wrf_key == 'ch4' else [merra_key]
            for cand in fallback_keys:
                if cand in ml_ds:
                    print(f"\t\t   [FALLBACK] Using '{cand}' from 'ml' stream for '{wrf_key}'.")
                    return 'ml', ml_ds, cand

    return None, None, None


#%% IC
if args.do_IC:
    date = dates_to_process[0]
    print("INITIAL CONDITIONS: {}".format(date))

    donor_streams = {}
    donor_p_rho_by_stream = {}
    donor_sfc_ds = _open_stream_for_date('sfc', date, None, donor_streams)
    if donor_sfc_ds is None:
        raise FileNotFoundError(f"Required CAMS 'sfc' stream file missing for IC date {date}")

    met_file_name = regridding_utils.get_met_file_by_time(date.strftime(args.wrf_met_files_date_format), args.wrf_met_files_name_template)
    met_fp = os.path.join(args.wrf_met_dir, met_file_name)
    print("Opening metfile: " + met_fp)
    wrf_met_ds = xr.open_dataset(met_fp).set_coords(['XLAT_M', 'XLONG_M'])
    derive_wrf_pressure_from_met_and_input_files(wrf_met_ds, wrf_ic_ds)
    wrf_met_ds = wrf_met_ds.squeeze()  # squeeze Time, otherwise causes size issues with pure netcdf

    print("Opening wrfintput: " + wrf_ic_fp)
    wrfinput_f = nc.Dataset(wrf_ic_fp, 'r+')

    print("INITIAL CONDITIONS: Increments")
    zeroed_wrf_keys = set()
    for mapping in mappings:
        print('Processing mapping: {}'.format(mapping.mapping_rule_str))
        pipe_to_process = parse_mapping_rule(mapping.mapping_rule_str)

        for rule_vo in pipe_to_process:
            merra_key = rule_vo['merra_key']
            wrf_key = rule_vo['wrf_key']
            print("\t\t - Reading " + merra_key)
            if wrf_key not in wrfinput_f.variables:
                print(f"\t\t   [WARNING] Variable {wrf_key} not found in wrfinput, skipping.")
                continue

            resolved_stream, donor_stream_ds, resolved_key = _resolve_donor_source(
                mapping.output_stream, merra_key, wrf_key, date, None, donor_streams
            )
            if donor_stream_ds is None:
                print(f"\t\t   [WARNING] Variable {merra_key} not found in '{mapping.output_stream}' (or fallback), skipping {wrf_key}.")
                continue

            if args.zero_out_first and wrf_key not in zeroed_wrf_keys:
                print(f"\t\t - Zeroing out {wrf_key} in wrfinput before increments")
                wrfinput_f.variables[wrf_key][0, :] = 0
                zeroed_wrf_keys.add(wrf_key)

            donor_p_rho_hi_da = _get_stream_p_rho(
                resolved_stream, donor_stream_ds, donor_sfc_ds, wrf_ic_ds, donor_p_rho_by_stream, is_boundary=False
            )
            donor_var_da = donor_stream_ds[resolved_key].load()

            print("\t\t - Horizontal interpolation of " + resolved_key + " on WRF horizontal grid")
            donor_var_hi_da = regridding_utils.hor_interpolate_3d_field_on_wrf_grid(donor_var_da, wrf_ic_ds)

            print("\t\t - Vertical interpolation of " + resolved_key + " on WRF vertical grid")
            donor_var_hvi_da = regridding_utils.ver_interpolate_3d_field_on_wrf_grid(donor_var_hi_da, donor_p_rho_hi_da, wrf_met_ds.p_rho)

            print("\t\t - Updating wrfinput field {}[0] += {} * {} * {:.1e}".format(wrf_key, resolved_key, rule_vo['wrf_multiplier'], rule_vo['wrf_exponent']))
            increment_da = donor_var_hvi_da * rule_vo['wrf_multiplier'] * rule_vo['wrf_exponent']
            wrfinput_f.variables[wrf_key][0, :] = wrfinput_f.variables[wrf_key][0, :] + increment_da.transpose('bottom_top', 'south_north', 'west_east')

    wrfinput_f.close()
    for ds in donor_streams.values():
        if ds is not None:
            ds.close()
    wrf_met_ds.close()
    print("DONE: INITIAL CONDITIONS")
#%% BC
if args.do_BC:
    print("\n\nSTART BOUNDARY CONDITIONS")

    print("Opening WRF bdy: " + wrf_bdy_fp)
    wrfbdy_f = nc.Dataset(wrf_bdy_fp, 'r+')

    active_streams = []
    for m in mappings:
        if m.output_stream not in active_streams:
            active_streams.append(m.output_stream)

    prev_donor_time_val = None
    prev_p_rho_on_bnd = None
    prev_time_index = None
    prev_updated_wrf_keys = []

    for date in dates_to_process:
        print("\n\tBCs, next date: {}".format(date))

        donor_streams = {}
        donor_p_rho_by_stream = {}
        donor_sfc_ds = _open_stream_for_date('sfc', date, 'nearest', donor_streams)
        if donor_sfc_ds is None:
            raise FileNotFoundError(f"Required CAMS 'sfc' stream file missing for BC date {date}")
        for s in active_streams:
            _open_stream_for_date(s, date, 'nearest', donor_streams)

        met_file_name = regridding_utils.get_met_file_by_time(date.strftime(args.wrf_met_files_date_format), args.wrf_met_files_name_template)
        met_fp = os.path.join(args.wrf_met_dir, met_file_name)
        print("\tReading WRF Pressure from: {}".format(met_fp))
        wrf_met_ds = xr.open_dataset(met_fp).set_coords(['XLAT_M', 'XLONG_M'])
        derive_wrf_pressure_from_met_and_input_files(wrf_met_ds, wrf_ic_ds)
        wrf_met_ds = wrf_met_ds.squeeze()  # squeeze Time, otherwise causes size issues with pure netcdf

        # prep boundary arrays for interpolation
        wrf_met_ds['p_rho_on_bnd'] = flatten_and_line_up_boundaries(wrf_met_ds.p_rho)  # concatenate east, west, nort and south boundaries
        wrf_met_ds['boundary_lons'] = flatten_and_line_up_boundaries(wrf_met_ds.XLONG_M)
        wrf_met_ds['boundary_lats'] = flatten_and_line_up_boundaries(wrf_met_ds.XLAT_M)

        time_index_in_wrfbdy = wrf_bdy_dates.get_loc(date)

        donor_time_val = tuple(
            (s, str(ds.time.values) if 'time' in ds.coords else (str(ds.valid_time.values) if 'valid_time' in ds.coords else None))
            for s, ds in sorted(donor_streams.items())
            if ds is not None
        )
        is_same_as_prev = (
            prev_time_index is not None and
            bool(donor_time_val) and
            donor_time_val == prev_donor_time_val and
            prev_p_rho_on_bnd is not None and
            np.array_equal(wrf_met_ds.p_rho_on_bnd.values, prev_p_rho_on_bnd)
        )

        updated_wrf_keys = []
        if is_same_as_prev:
            print("\tReusing boundary concentrations from previous time index {} (donor data and pressure are identical)".format(prev_time_index))
            for wrf_key in prev_updated_wrf_keys:
                if wrf_key + "_BXS" in wrfbdy_f.variables:
                    wrfbdy_f.variables[wrf_key + "_BXS"][time_index_in_wrfbdy, :] = wrfbdy_f.variables[wrf_key + "_BXS"][prev_time_index, :]
                    wrfbdy_f.variables[wrf_key + "_BXE"][time_index_in_wrfbdy, :] = wrfbdy_f.variables[wrf_key + "_BXE"][prev_time_index, :]
                    wrfbdy_f.variables[wrf_key + "_BYS"][time_index_in_wrfbdy, :] = wrfbdy_f.variables[wrf_key + "_BYS"][prev_time_index, :]
                    wrfbdy_f.variables[wrf_key + "_BYE"][time_index_in_wrfbdy, :] = wrfbdy_f.variables[wrf_key + "_BYE"][prev_time_index, :]
            updated_wrf_keys = list(prev_updated_wrf_keys)
        else:
            zeroed_wrf_keys = set()
            for mapping in mappings:
                print('Processing mapping: {}'.format(mapping.mapping_rule_str))
                pipe_to_process = parse_mapping_rule(mapping.mapping_rule_str)

                for rule_vo in pipe_to_process:
                    merra_key = rule_vo['merra_key']
                    wrf_key = rule_vo['wrf_key']
                    print("\t\t - Reading " + merra_key)

                    if wrf_key + "_BXS" not in wrfbdy_f.variables:
                        print("\t\t   [WARNING] Variable {} not found in wrfbdy, skipping.".format(wrf_key))
                        continue

                    resolved_stream, donor_stream_ds, resolved_key = _resolve_donor_source(
                        mapping.output_stream, merra_key, wrf_key, date, 'nearest', donor_streams
                    )
                    if donor_stream_ds is None:
                        print(f"\t\t   [WARNING] Variable {merra_key} not found in '{mapping.output_stream}' (or fallback), skipping {wrf_key}.")
                        continue

                    if args.zero_out_first and wrf_key not in zeroed_wrf_keys:
                        for sfx in ("_BXS", "_BXE", "_BYS", "_BYE", "_BTXS", "_BTXE", "_BTYS", "_BTYE"):
                            wrfbdy_f.variables[wrf_key + sfx][time_index_in_wrfbdy, :] = 0
                        zeroed_wrf_keys.add(wrf_key)

                    if wrf_key not in updated_wrf_keys:
                        updated_wrf_keys.append(wrf_key)

                    donor_p_rho_hi_da = _get_stream_p_rho(
                        resolved_stream, donor_stream_ds, donor_sfc_ds, wrf_met_ds, donor_p_rho_by_stream, is_boundary=True
                    )
                    donor_var_da = donor_stream_ds[resolved_key].load()
                    print("\t\tHorizontal interpolation of " + resolved_key + " on WRF boundary")
                    donor_var_hi_da = regridding_utils.hor_interpolate_3d_field_on_wrf_boundary(donor_var_da, wrf_met_ds)
                    print("\t\tVertical interpolation of " + resolved_key + " on WRF boundary")
                    donor_var_hvi_da = regridding_utils.ver_interpolate_3d_field_on_wrf_boundary(donor_var_hi_da, donor_p_rho_hi_da, wrf_met_ds.p_rho_on_bnd)

                    print("\t\t - Updating wrfbdy: {}[{}] += {} * {} * {:.1e}".format(wrf_key, time_index_in_wrfbdy, resolved_key, rule_vo['wrf_multiplier'], rule_vo['wrf_exponent']))
                    increment_on_bnd_da = donor_var_hvi_da * rule_vo['wrf_multiplier'] * rule_vo['wrf_exponent']
                    regridding_utils.update_boundaries(increment_on_bnd_da, wrf_met_ds, wrfbdy_f, wrf_key, time_index_in_wrfbdy)

        print('Prescribing uniform tendency in BC for these fields: {}'.format(updated_wrf_keys))
        for wrf_key in updated_wrf_keys:
            if wrf_key + "_BXS" in wrfbdy_f.variables:
                regridding_utils.update_tendency_boundaries(wrfbdy_f, wrf_key, time_index_in_wrfbdy, delta_t)

        prev_donor_time_val = donor_time_val
        prev_p_rho_on_bnd = wrf_met_ds.p_rho_on_bnd.values.copy()
        prev_time_index = time_index_in_wrfbdy
        prev_updated_wrf_keys = list(updated_wrf_keys)

        print("--- %s seconds ---" % (time.time() - start_time))
        for ds in donor_streams.values():
            if ds is not None:
                ds.close()
        wrf_met_ds.close()

    print("Closing " + args.wrf_bdy_file)
    wrfbdy_f.close()

    print("FINISH BOUNDARY CONDITIONS")

#%% time stats
print("--- %s seconds ---" % (time.time() - start_time))
