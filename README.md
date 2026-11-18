# Merra2BC (`merra2wrf` & `cams2wrf`)

Chemical and aerosol Initial & Boundary Condition (IC/BC) pre-processor for **WRF-Chem** (`wrfinput_d0*` and `wrfbdy_d01`), driven by NASA MERRA-2 reanalysis or ECMWF CAMS operational atmospheric composition & greenhouse gas forecasts.

## CAMS Workflow (`cams2wrf_main.py`)

### Required Donor Streams
Placed under `--reanalysis_dir` using template `ECMWF-CAMS_{stream}_{date_time}.nc` (`YYYYMMDD`):
* `sfc`: Surface pressure (`sp`) and single-level parameters (`0.4°`)
* `ml`: 137 hybrid model-level meteorological, trace-gas, and aerosol fields (`0.4°`)
* `ghg` *(optional, `--streams ghg` or automatic when present)*: High-resolution **`0.1°` 3D `co2` and `ch4`** from `cams-global-greenhouse-gas-forecasts` (`ECMWF-CAMS_ghg_{YYYYMMDD}.nc`, with automatic fallback to `ch4`/`ch4_c` in `ml` for methane)

### Mixed-Grid (`0.1°` / `0.4°`) & Vertical Interpolation
* `_derive_stream_p_rho` interpolates `sp` from `sfc` onto WRF interior or boundary coordinates and reconstructs 3D hybrid midpoint pressure $p_{\rho}(k) = a_k + b_k \cdot p_s$ using `GRIB_pv` coefficients on each stream grid.
* Vertical interpolation to WRF eta pressure coordinates clamps pressure out-of-bounds cleanly to the top/bottom donor levels (`CO2` and `CH4` injected in `ppmv` for RACM/MADE-SORGAM/GOCART).

### Example Invocation
```bash
python -u "${MERRA2BC_DIR}/cams2wrf_main.py" \
    --start_date "${START_DATE}_00:00:00" \
    --end_date "${END_DATE}_00:00:00" \
    --hourly_interval 3 \
    --do_IC --do_BC --zero_out_first \
    --reanalysis_dir "${CAMS_DIR}/" \
    --reanalysis_file_name_template "ECMWF-CAMS_{stream}_{date_time}.nc" \
    --wrf_dir "${WRF_RUN_DIR}" \
    --wrf_input wrfinput_d01 \
    --wrf_bdy_file wrfbdy_d01 \
    --wrf_met_dir "${MET_EM_DIR}" \
    --wrf_met_files_name_template "met_em.d01.{}.nc" \
    --wrf_met_files_date_format "%Y-%m-%d_%H_%M_%S"
```

## Required `namelist.input` Settings (`&chem`)

```fortran
&chem
 chem_in_opt                         = 1,
 have_bcs_chem                       = .true.,
 gas_ic_opt                          = 1,
 gas_bc_opt                          = 1,
 aer_ic_opt                          = 1,
 aer_bc_opt                          = 1,
/
```