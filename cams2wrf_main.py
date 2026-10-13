import argparse
import time

import regridding_utils
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
#%% IC
if args.do_IC:
    date = dates_to_process[0]
    print("INITIAL CONDITIONS: {}".format(date))

    fp = args.reanalysis_dir + args.reanalysis_file_name_template.format(date_time=date.strftime('%Y%m%d'), stream='ml')
    donor_ml_ds = xr_open_dataset_impl(fp).sel(time=date)
    fp = args.reanalysis_dir + args.reanalysis_file_name_template.format(date_time=date.strftime('%Y%m%d'), stream='sfc')
    donor_sfc_ds = xr_open_dataset_impl(fp).sel(time=date)
    derive_cams_pressure_profile(donor_ml_ds, donor_sfc_ds)

    # alternative way to sort
    # merra_df = merra_df.sortby('lev', ascending=False) # set TOA as last layer
    # pressure_stag = pressure_stag.sortby('lev', ascending=False)
    # pressure_rho = pressure_rho.sortby('lev', ascending=False)

    donor_p_rho_hi_da = regridding_utils.hor_interpolate_3d_field_on_wrf_grid(donor_ml_ds.p_rho, wrf_ic_ds)

    met_file_name = regridding_utils.get_met_file_by_time(date.strftime(args.wrf_met_files_date_format), args.wrf_met_files_name_template)
    met_fp = args.wrf_met_dir + "/" + met_file_name
    print("Opening metfile: " + met_fp)
    wrf_met_ds = xr.open_dataset(met_fp).set_coords(['XLAT_M', 'XLONG_M'])
    derive_wrf_pressure_from_met_and_input_files(wrf_met_ds, wrf_ic_ds)
    wrf_met_ds = wrf_met_ds.squeeze()  # squeeze Time, otherwise causes size issues with pure netcdf

    print("Opening wrfintput: " + wrf_ic_fp)
    wrfinput_f = nc.Dataset(wrf_ic_fp, 'r+')

    if args.zero_out_first:  # zero out the fields in wrfinput before increments
        unique_wrf_keys = get_unique_wrf_keys_from_mappings(mappings)
        print("Following fields will be zeroed out FIRST in ICs: {}".format(unique_wrf_keys))
        for wrf_key in unique_wrf_keys:
            if wrf_key in wrfinput_f.variables:
                wrfinput_f.variables[wrf_key][0, :] = 0  # minuscule might be better than 0

    print("INITIAL CONDITIONS: Increments")
    donor_streams = {'ml': donor_ml_ds, 'sfc': donor_sfc_ds}
    for mapping in mappings:
        print('Processing mapping: {}'.format(mapping.mapping_rule_str))
        pipe_to_process = parse_mapping_rule(mapping.mapping_rule_str)
        if mapping.output_stream in donor_streams:
            donor_stream_ds = donor_streams[mapping.output_stream]
            close_stream = False
        else:
            fp = args.reanalysis_dir + args.reanalysis_file_name_template.format(date_time=date.strftime('%Y%m%d'), stream=mapping.output_stream)
            donor_stream_ds = xr_open_dataset_impl(fp).sel(time=date, method='nearest')
            close_stream = True

        for rule_vo in pipe_to_process:
            merra_key = rule_vo['merra_key']
            wrf_key = rule_vo['wrf_key']
            print("\t\t - Reading " + merra_key)
            if merra_key not in donor_stream_ds:
                print(f"\t\t   [WARNING] Variable {merra_key} not found in donor dataset, skipping.")
                continue
            if wrf_key not in wrfinput_f.variables:
                print(f"\t\t   [WARNING] Variable {wrf_key} not found in wrfinput, skipping.")
                continue

            donor_var_da = donor_stream_ds[merra_key].load()

            print("\t\t - Horizontal interpolation of " + merra_key + " on WRF horizontal grid")
            donor_var_hi_da = regridding_utils.hor_interpolate_3d_field_on_wrf_grid(donor_var_da, wrf_ic_ds)

            print("\t\t - Vertical interpolation of " + merra_key + " on WRF vertical grid")
            donor_var_hvi_da = regridding_utils.ver_interpolate_3d_field_on_wrf_grid(donor_var_hi_da, donor_p_rho_hi_da, wrf_met_ds.p_rho)

            print("\t\t - Updating wrfinput field {}[0] += {} * {} * {:.1e}".format(wrf_key, merra_key, rule_vo['wrf_multiplier'], rule_vo['wrf_exponent']))
            increment_da = donor_var_hvi_da * rule_vo['wrf_multiplier'] * rule_vo['wrf_exponent']
            wrfinput_f.variables[wrf_key][0, :] = wrfinput_f.variables[wrf_key][0, :] + increment_da.transpose('bottom_top', 'south_north', 'west_east')

        if close_stream:
            donor_stream_ds.close()

    wrfinput_f.close()
    donor_ml_ds.close()
    donor_sfc_ds.close()
    wrf_met_ds.close()
    print("DONE: INITIAL CONDITIONS")
#%% BC
if args.do_BC:
    print("\n\nSTART BOUNDARY CONDITIONS")

    print("Opening WRF bdy: " + wrf_bdy_fp)
    wrfbdy_f = nc.Dataset(wrf_bdy_fp, 'r+')

    unique_wrf_keys = get_unique_wrf_keys_from_mappings(mappings)
    prev_donor_time_val = None
    prev_p_rho_on_bnd = None
    prev_time_index = None

    for date in dates_to_process:
        print("\n\tBCs, next date: {}".format(date))

        fp = args.reanalysis_dir + args.reanalysis_file_name_template.format(date_time=date.strftime('%Y%m%d'), stream='ml')
        donor_ml_ds = xr_open_dataset_impl(fp).sel(time=date, method='nearest')
        fp = args.reanalysis_dir + args.reanalysis_file_name_template.format(date_time=date.strftime('%Y%m%d'), stream='sfc')
        donor_sfc_ds = xr_open_dataset_impl(fp).sel(time=date, method='nearest')
        derive_cams_pressure_profile(donor_ml_ds, donor_sfc_ds)

        met_file_name = regridding_utils.get_met_file_by_time(date.strftime(args.wrf_met_files_date_format), args.wrf_met_files_name_template)
        met_fp = args.wrf_met_dir + "/" + met_file_name
        print("\tReading WRF Pressure from: {}".format(met_fp))
        wrf_met_ds = xr.open_dataset(met_fp).set_coords(['XLAT_M', 'XLONG_M'])
        derive_wrf_pressure_from_met_and_input_files(wrf_met_ds, wrf_ic_ds)
        wrf_met_ds = wrf_met_ds.squeeze()  # squeeze Time, otherwise causes size issues with pure netcdf

        # prep boundary arrays for interpolation
        wrf_met_ds['p_rho_on_bnd'] = flatten_and_line_up_boundaries(wrf_met_ds.p_rho)  # concatenate east, west, nort and south boundaries
        wrf_met_ds['boundary_lons'] = flatten_and_line_up_boundaries(wrf_met_ds.XLONG_M)
        wrf_met_ds['boundary_lats'] = flatten_and_line_up_boundaries(wrf_met_ds.XLAT_M)

        time_index_in_wrfbdy = wrf_bdy_dates.get_loc(date)

        donor_time_val = str(donor_ml_ds.time.values) if 'time' in donor_ml_ds.coords else (str(donor_ml_ds.valid_time.values) if 'valid_time' in donor_ml_ds.coords else None)
        is_same_as_prev = (
            prev_time_index is not None and
            donor_time_val is not None and
            donor_time_val == prev_donor_time_val and
            prev_p_rho_on_bnd is not None and
            np.array_equal(wrf_met_ds.p_rho_on_bnd.values, prev_p_rho_on_bnd)
        )

        if is_same_as_prev:
            print("\tReusing boundary concentrations from previous time index {} (donor data and pressure are identical)".format(prev_time_index))
            for wrf_key in unique_wrf_keys:
                if wrf_key + "_BXS" in wrfbdy_f.variables:
                    wrfbdy_f.variables[wrf_key + "_BXS"][time_index_in_wrfbdy, :] = wrfbdy_f.variables[wrf_key + "_BXS"][prev_time_index, :]
                    wrfbdy_f.variables[wrf_key + "_BXE"][time_index_in_wrfbdy, :] = wrfbdy_f.variables[wrf_key + "_BXE"][prev_time_index, :]
                    wrfbdy_f.variables[wrf_key + "_BYS"][time_index_in_wrfbdy, :] = wrfbdy_f.variables[wrf_key + "_BYS"][prev_time_index, :]
                    wrfbdy_f.variables[wrf_key + "_BYE"][time_index_in_wrfbdy, :] = wrfbdy_f.variables[wrf_key + "_BYE"][prev_time_index, :]
        else:
            print("\tHorizontal interpolation of Pressure on WRF boundary")
            donor_p_rho_hi_da = regridding_utils.hor_interpolate_3d_field_on_wrf_boundary(donor_ml_ds.p_rho, wrf_met_ds)

            if args.zero_out_first:  # zero out the fields before increments
                print("Following fields will be zeroed out FIRST in BCs: {}".format(unique_wrf_keys))
                for wrf_key in unique_wrf_keys:
                    if wrf_key + "_BXS" in wrfbdy_f.variables:
                        wrfbdy_f.variables[wrf_key + "_BXS"][time_index_in_wrfbdy, :] = 0  # BCs
                        wrfbdy_f.variables[wrf_key + "_BXE"][time_index_in_wrfbdy, :] = 0
                        wrfbdy_f.variables[wrf_key + "_BYS"][time_index_in_wrfbdy, :] = 0
                        wrfbdy_f.variables[wrf_key + "_BYE"][time_index_in_wrfbdy, :] = 0
                        wrfbdy_f.variables[wrf_key + "_BTXS"][time_index_in_wrfbdy, :] = 0  # BCs tendencies
                        wrfbdy_f.variables[wrf_key + "_BTXE"][time_index_in_wrfbdy, :] = 0
                        wrfbdy_f.variables[wrf_key + "_BTYS"][time_index_in_wrfbdy, :] = 0
                        wrfbdy_f.variables[wrf_key + "_BTYE"][time_index_in_wrfbdy, :] = 0

            donor_streams = {'ml': donor_ml_ds, 'sfc': donor_sfc_ds}
            for mapping in mappings:
                print('Processing mapping: {}'.format(mapping.mapping_rule_str))
                pipe_to_process = parse_mapping_rule(mapping.mapping_rule_str)
                if mapping.output_stream in donor_streams:
                    donor_stream_ds = donor_streams[mapping.output_stream]
                    close_stream = False
                else:
                    fp = args.reanalysis_dir + args.reanalysis_file_name_template.format(date_time=date.strftime('%Y%m%d'), stream=mapping.output_stream)
                    donor_stream_ds = xr_open_dataset_impl(fp).sel(time=date, method='nearest')
                    close_stream = True

                for rule_vo in pipe_to_process:
                    merra_key = rule_vo['merra_key']
                    wrf_key = rule_vo['wrf_key']
                    print("\t\t - Reading " + merra_key)

                    if merra_key not in donor_stream_ds:
                        print("\t\t   [WARNING] Variable {} not found in donor dataset, skipping.".format(merra_key))
                        continue
                    if wrf_key + "_BXS" not in wrfbdy_f.variables:
                        print("\t\t   [WARNING] Variable {} not found in wrfbdy, skipping.".format(wrf_key))
                        continue

                    donor_var_da = donor_stream_ds[merra_key].load()
                    print("\t\tHorizontal interpolation of " + merra_key + " on WRF boundary")
                    donor_var_hi_da = regridding_utils.hor_interpolate_3d_field_on_wrf_boundary(donor_var_da, wrf_met_ds)
                    print("\t\tVertical interpolation of " + merra_key + " on WRF boundary")
                    donor_var_hvi_da = regridding_utils.ver_interpolate_3d_field_on_wrf_boundary(donor_var_hi_da, donor_p_rho_hi_da, wrf_met_ds.p_rho_on_bnd)

                    print("\t\t - Updating wrfbdy: {}[{}] += {} * {} * {:.1e}".format(wrf_key, time_index_in_wrfbdy, merra_key, rule_vo['wrf_multiplier'], rule_vo['wrf_exponent']))
                    increment_on_bnd_da = donor_var_hvi_da * rule_vo['wrf_multiplier'] * rule_vo['wrf_exponent']
                    regridding_utils.update_boundaries(increment_on_bnd_da, wrf_met_ds, wrfbdy_f, wrf_key, time_index_in_wrfbdy)

                if close_stream:
                    donor_stream_ds.close()

        print('Prescribing uniform tendency in BC for these fields: {}'.format(unique_wrf_keys))
        for wrf_key in unique_wrf_keys:
            if wrf_key + "_BXS" in wrfbdy_f.variables:
                regridding_utils.update_tendency_boundaries(wrfbdy_f, wrf_key, time_index_in_wrfbdy, delta_t)

        prev_donor_time_val = donor_time_val
        prev_p_rho_on_bnd = wrf_met_ds.p_rho_on_bnd.values.copy()
        prev_time_index = time_index_in_wrfbdy

        print("--- %s seconds ---" % (time.time() - start_time))
        donor_ml_ds.close()
        donor_sfc_ds.close()
        wrf_met_ds.close()

    print("Closing " + args.wrf_bdy_file)
    wrfbdy_f.close()

    print("FINISH BOUNDARY CONDITIONS")

#%% time stats
print("--- %s seconds ---" % (time.time() - start_time))
