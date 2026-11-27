# AirBlender

**Unified Atmospheric Meteorology, Chemistry, and Aerosol Coupler (`v0.4.0`)**

`AirBlender` (`airblender`) couples global atmospheric composition, NWP, and reanalysis models (**ECMWF IFS**, **ECMWF CAMS**, **NASA MERRA-2**, **EMAC / MESSy**) with regional **WPS / WRF-Chem**, **ICON-ART**, and urban microscale **PALM-4U** (`PIDS_DYNAMIC`), including multi-provider dynamics resolution, gas-phase conversion (RADM2 & RACM), modal $\leftrightarrow$ sectional aerosol remapping, mixed-grid vertical pressure reconstruction, and direct Fortran unformatted WPS binary generation.

It is a standalone, `pip`-installable Python package.

---

## Key Capabilities & Coupling Pathways

| Subcommand / Entry Point | Source $\rightarrow$ Target | Scope & Highlights |
| :--- | :--- | :--- |
| **`cams2wrf`** (`airblender.wrf_icbc`) | ECMWF CAMS $\rightarrow$ WRF-Chem | Reactive trace gases covering both **RACM (`chem_opt=100`)** and **RADM2 (`chem_opt=106`)** (`c2h4` mapped to both `ete` and `ol2`, plus `macr <- 0.413*[ispd]`, `op1 <- 0.603*[ch3ooh]`, `aco3 <- 0.386*[c2o3]`), `MADE/SORGAM + GOCART` aerosols (including CAMS Cy49r1 `aermr01` `0.03–0.5 µm` with pre-`2026-05-12` `0.03–0.55 µm` fallback), and mixed-grid `0.1°` CAMS GHG (`co2`, `ch4` with `ml` `ch4_c` fallback). Supports multiple nested `wrfinput_d0*` + `wrfbdy_d01` using vectorized log-P vertical plans (`_build_or_get_vert_plan`, `_ver_interpolate_edge_clamped`, monotone non-negative profiles eliminating cubic-spline undershoots) and cached bilinear horizontal weights (`_get_cached_bilinear_weights`, `hor_interpolate_array_on_points`), delivering a **4.12x `cams2wrf` speedup** (`540.6s -> 131.3s`) with identical-step BC reuse. |
| **`merra2wrf`** (`airblender.wrf_icbc`) | NASA MERRA-2 $\rightarrow$ WRF-Chem | `PL` + `DELP` fallback hybrid pressure reconstruction, `VVV` naming support, selectable WRF schemes (`made_sorgam`, `gocart`, `gocart_dust`, `mosaic_8bin`), full `CAMS2WRF` parity (stream caching + BC array reuse). |
| **`emac2wrf`** (`airblender.wrf_icbc`) | EMAC / MESSy $\rightarrow$ WRF-Chem | Log-normal 4-mode (`ns, ks, as, cs`) $\rightarrow$ MADE/SORGAM 3-mode (`nu0/ai`, `ac0/aj`, `corn`) moment mapping, `[0, 360) -> [-180, 180)` longitude shifting, glob restart handling, and strict DSL syntax checking (including fixed `2.01*[SS_cs]` rule). |
| **`cams2wps`** (`airblender.cams2wps`) | ECMWF IFS / CAMS $\rightarrow$ WPS | Direct Fortran unformatted WPS v5 intermediate generator (`FILE:YYYY-MM-DD_HH`) supporting independent mixed-resolution `sfc` (`0.08°` MARS IFS, `0.1°` OpenData S3, or `0.4°` CAMS) + 137-level `ml` (`0.08°` or `0.4°`) NetCDFs, exact UTC slice lookup, coastal land/sea mask infill (`_infill_2d_by_mask`), Tetens 2 m `d2m -> SPECHUMD`, 4-layer soil (`stl1..4`, `swvl1..4`, or external OpenData soil NetCDF), `sp` alignment onto the `ml` grid, `GRIB_pv` `(a_k, b_k)` midpoint pressures, atomic `.tmp` writes, and up-to-date slice skipping. |
| **`cams2icon`** (`airblender.icon_icbc`) | ECMWF IFS / CAMS $\rightarrow$ ICON-ART | Pure-Python ICON-ART coupler (`prepare_icon_ic`, `prepare_icon_latbc`, `prepare_icon_edgar_emissions`, zero `cdo`/`nco`/`wgrib2` runtime dependency) across independent `met_sfc`, `met_ml`, `chem_ml`, and `chem_ghg` donor grids. Generates `ifs2icon_R2Bxx_DOMxx.nc` (with TERRA SMI conversion via `soil_water_to_smi`), HWSD 14-class `ART_STY_*`, 3-mode mineral dust `ART_IAE_*` (`dusta..c` + `numb_dusta..c` via `dust_mass_to_number`), passive tracer `ART_ICE_*` (`TRSO2, TRNO2, TRO3, TRCO, TRCO2, TRCH4`), EDGAR `emi_so2`/`emi_nox`, and 3-hourly `latbc/lbc_*.nc` on 90 SLEVE levels (`compute_icon_sleve_heights`). |
| **`wrf2palm`** (`airblender.palm_dynamic`) | WRF-Chem $\rightarrow$ PALM-4U | Generates `PIDS_DYNAMIC` (`_dynamic`) NetCDF-4 for urban microscale domains (`100 m` SLUrb / `10 m` LES) with `d02 -> d01` parent fallback, LCC projection + wind rotation, `PALM-meteo` (`universal`) barometric surface-layer pressure adaptation, and selectable `chemistry.palm_mode`: `meteorology` (`31` fields), `phstatp` (`55` fields), **`phstatp-salsa`** (`115` fields: conservative 7-species `['SO4', 'OC', 'BC', 'DU', 'SS', 'NO', 'NH']` $\times$ 10-bin sectional WRF–SALSA remapping, `antha` excluded under `absolute_mass_number_v1`), and `legacy_3spec` (`97` fields). |
| **`fetch-*`** (`airblender.fetch`) *(optional `[fetch]`)* | ADS / CDS / MARS / OpenData / GES DISC $\rightarrow$ Local NetCDF/GRIB | Donor forcing acquisition: **`fetch-dynamics`** (`resolve_and_fetch_dynamics`: ordered `ifs_mars_9km -> ifs_0p1_sfc_cams_ml -> cams_0p4_ml` waterfall with 45 s bounded MARS pre-flight probe, `DYNAMICS_PROVIDER` override, and atomic `dynamics_provenance.json`), **`fetch-cams`** (Cy49r1-aligned 14 `sfc` + 7 137-ML `met` + `2×24` `chema`/`chemb` + `0.1°` `ghg` dense GRIB packing: 41 streaming / 36 batch requests with Day-1 = 9 tasks, `classify_cams_error` fail-fast circuit-breaker, `--stream_daily` with `forecast_reference_time` validation preventing stale reuse across overlapping `< 5`-day cycles, and `--status_file` JSON tracker), **`fetch-opendata`** (keyless `--mode s3-0p1` HTTP Byte-Range `0.1°` `sfc` + 4-layer soil across GCS/ECMWF/AWS mirrors, plus `sol`/`sfc`), **`fetch-ifs`** (MARS `0.08°` forecast + `F1280`), **`fetch-eac4`**, **`fetch-era5-soil`**, and **`fetch-merra2`**. |

---

## Package Architecture

```text
AirBlender/
├── pyproject.toml
├── src/airblender/
│   ├── cli.py                        # Unified `airblender` CLI & console entry points
│   ├── wrf_icbc.py                   # Unified donor -> WRF-Chem IC/BC injector
│   ├── cams2wps.py                   # Mixed-grid IFS/CAMS NetCDF -> WPS intermediate v5 converter
│   ├── icon_icbc.py                  # Pure-Python IFS/CAMS -> ICON-ART IC, ART & LatBC coupler
│   ├── palm_dynamic.py               # WRF-Chem -> PALM-4U PIDS_DYNAMIC driver
│   ├── fetch/                        # Optional donor data fetchers (pip install airblender[fetch])
│   │   ├── dynamics.py               # Multi-provider dynamics resolver (9 km MARS -> 0.1° OpenData -> 0.4° CAMS)
│   │   ├── cams_forecast.py          # CAMS Cy49r1 GRIB pipelining, fail-fast circuit breaker & live status JSON
│   │   ├── cams_eac4.py              # ECMWF CAMS EAC4 60-level daily reanalysis
│   │   ├── ecmwf_ifs.py              # ECMWF MARS operational 0.08°/F1280 IFS downloader & pre-flight probe
│   │   ├── opendata.py               # Keyless 0.1° HTTP Byte-Range OpenData (GCS/ECMWF/AWS) & sol/sfc helpers
│   │   ├── era5_soil.py              # ECMWF ERA5 land/soil NetCDF downloader
│   │   └── merra2.py                 # NASA MERRA-2 GES DISC wget downloader (wrfchem & zittis_aot presets)
│   ├── sources/
│   │   ├── base.py                   # DonorAdapter base class & stream/p_rho caching
│   │   ├── cams.py                   # ECMWF CAMS (sfc, ml, ghg + valid_time/coord handling)
│   │   ├── merra2.py                 # NASA MERRA-2 (PL + DELP fallback, VVV collections)
│   │   └── emac.py                   # EMAC / MESSy (hybrid pressure, lon shift, glob files)
│   ├── physics/
│   │   ├── dsl.py                    # Strict mapping rule parser ("out <- a*[in];exp")
│   │   ├── sectional_to_modal.py     # Log-normal modal number/mass factor computation
│   │   ├── mappings.py               # CAMS (RACM/RADM2/MADE-SORGAM), MERRA-2 (4 presets), and EMAC tables
│   │   └── salsa_remap.py            # Conservative 7-species x 10-bin WRF -> SALSA remapper
│   ├── regrid/
│   │   └── wrf_regrid.py             # Vectorized log-P vertical & cached bilinear horizontal/BC interp (4.12x cams2wrf)
│   └── utils/
│       ├── time_io.py                # Native datetime/char Times parsing & eta pressure derivation
│       └── config_io.py              # YAML loader & ${VAR:-default} path expansion
├── cams2wrf_main.py                  # Backwards-compatible CLI wrapper
├── merra2wrf_main.py                 # Backwards-compatible CLI wrapper
├── emac2wrf_main.py                  # Backwards-compatible CLI wrapper
├── scripts/merra2/                   # Legacy MERRA-2 wget shell scripts & README
└── tests/
    └── test_airblender.py            # Synthetic & performance verification suite (Tests 1–9)
```

---

## Installation

```bash
pip install -e /path/to/AirBlender
# With optional donor downloaders (cdsapi, cfgrib, eccodes, ecmwf-opendata, ecmwf-api-client):
pip install -e "/path/to/AirBlender[fetch]"
```

Inside a pre-populated Conda/Mamba environment (skipping extra PyPI downloads):

```bash
pip install --no-deps --no-build-isolation -e /path/to/AirBlender
```

Core dependencies: `numpy`, `scipy`, `xarray`, `netCDF4`, `pandas`, `pyproj`, `pyyaml`.  
Optional `[fetch]` dependencies (`cdsapi`, `cfgrib`, `eccodes`, `ecmwf-opendata`, `ecmwf-api-client`) are lazy-imported so offline compute nodes do not require them.

---

## CLI Usage

### Unified `airblender` CLI

```bash
airblender --help

# 1. CAMS -> WRF-Chem IC & BC (d01, RACM + RADM2 + MADE/SORGAM + GOCART)
airblender cams2wrf \
    --start_date 2026-11-14_00:00:00 \
    --end_date 2026-11-19_00:00:00 \
    --hourly_interval 3 \
    --do_IC --do_BC --zero_out_first \
    --reanalysis_dir /path/to/CAMS \
    --reanalysis_file_name_template "ECMWF-CAMS_{stream}_{date_time}.nc" \
    --wrf_dir /path/to/WRF/run \
    --wrf_input wrfinput_d01 \
    --wrf_bdy_file wrfbdy_d01 \
    --wrf_met_dir /path/to/WPS \
    --wrf_met_files_name_template "met_em.d01.{}.nc" \
    --wrf_met_files_date_format "%Y-%m-%d_%H_%M_%S"

# 2. MERRA-2 -> WRF-Chem IC & BC (selectable aerosol scheme)
airblender merra2wrf --scheme mosaic_8bin ...   # presets: made_sorgam | gocart | gocart_dust | mosaic_8bin

# 3. EMAC -> WRF-Chem IC & BC
airblender emac2wrf ...

# 4. IFS / CAMS -> WPS Intermediate Binaries (FILE:YYYY-MM-DD_HH, supports mixed sfc/ml grids)
airblender cams2wps --sfc ECMWF-IFS-0p1_sfc_20261114.nc --ml ECMWF-CAMS_ml_20261114.nc --outdir /path/to/WPS

# 5. IFS / CAMS -> ICON-ART IC, ART Dust/Tracers & 3h LatBC (pure Python)
airblender cams2icon \
    --cams_dir /path/to/CAMS \
    --grid_file /path/to/DOM01.nc \
    --extpar_file /path/to/DOM01_external_parameter.nc \
    --latbc_grid_file /path/to/DOM01_latbc.grid.nc \
    --run_dir /path/to/ICON/run \
    --start_date 2026-11-14_00:00:00 \
    --end_date 2026-11-15_00:00:00 \
    --met_sfc_dir /path/to/IFS \
    --met_sfc_prefix ECMWF-IFS-0p1_sfc

# 6. WRF-Chem -> PALM-4U Dynamic Driver (PIDS_DYNAMIC)
airblender wrf2palm --config palm_doha_100m.yaml --date 2026-11-14 --mode phstatp-salsa

# 7. Optional Donor Data Acquisition (`airblender[fetch]`)
airblender fetch-dynamics --date 2026-11-14 --cams_dir /path/to/CAMS --out_dir /path/to/IFS --mode auto --primary ifs_mars_9km
airblender fetch-cams --start_date 2026-11-14 --end_date 2026-11-14 --data_root_folder /path/to/CAMS --type forecast --time 00:00 --ghg --stream_daily --segment_hours 24 --status_file /path/to/cams_status.json
airblender fetch-opendata --mode s3-0p1 --date 2026-11-14 --outdir /path/to/IFS --leadtimes 0 3 6 9 12 15 18 21 24
airblender fetch-opendata --mode sol --date 2026-11-14 --outdir /path/to/HRES
airblender fetch-eac4 --start_date 2026-03-01 --end_date 2026-03-05 --data_root_folder /path/to/EAC4
airblender fetch-ifs --start_date 2026-03-01 --end_date 2026-03-02 --data_root_folder /path/to/IFS --grid F1280 --retrieve_ml true
airblender fetch-era5-soil --date 2026-03-01 --outdir /path/to/ERA5
airblender fetch-merra2 --start_date 2026-11-14 --ndays 2 --outdir /path/to/MERRA2 --preset wrfchem   # presets: wrfchem | zittis_aot
```

Console scripts `cams2wrf`, `merra2wrf`, `emac2wrf`, `cams2wps`, `cams2icon`, and `wrf2palm`, as well as legacy repo-root entry scripts (`cams2wrf_main.py`, `merra2wrf_main.py`, `emac2wrf_main.py`), are also installed for backwards compatibility.

---

## Python Library API

```python
from datetime import datetime, timedelta, timezone
from airblender.wrf_icbc import run_wrf_icbc
from airblender.cams2wps import convert_cams_to_wps
from airblender.icon_icbc import prepare_icon_ic, prepare_icon_latbc
from airblender.palm_dynamic import generate_palm_dynamic_driver
from airblender.fetch import download_cams_forecast, download_merra2, download_opendata_soil
from airblender.fetch.dynamics import resolve_and_fetch_dynamics

download_cams_forecast(
    start_date="2026-11-14",
    end_date="2026-11-14",
    data_root_folder="/path/to/CAMS",
    cams_type="forecast",
    cams_time="00:00",
    ghg=True,
    stream_daily=True,
    segment_hours=24,
    status_file="/path/to/cams_status.json",
)
dyn = resolve_and_fetch_dynamics(
    "2026-11-14",
    cams_dir="/path/to/CAMS",
    out_dir="/path/to/IFS",
    dynamics_cfg={"mode": "auto", "primary": "ifs_mars_9km"},
    provenance_dir="/path/to/IFS",
)
download_opendata_soil(date="2026-11-14", out_dir="/path/to/HRES")
download_merra2(start_date="2026-11-14", ndays=2, out_dir="/path/to/MERRA2", preset="wrfchem")

run_wrf_icbc(
    source="cams",
    reanalysis_dir="/path/to/CAMS",
    reanalysis_file_name_template="ECMWF-CAMS_{stream}_{date_time}.nc",
    wrf_dir="/path/to/WRF/run",
    wrf_met_dir="/path/to/WPS",
    start_date="2026-11-14_00:00:00",
    end_date="2026-11-15_00:00:00",
    hourly_interval=3,
    do_ic=True,
    do_bc=True,
    zero_out_first=True,
    scheme="made_sorgam",
)

wps_files = convert_cams_to_wps(
    cams_sfc_file=f"{dyn.met_sfc_dir}/{dyn.met_sfc_prefix}_20261114.nc",
    cams_ml_file=f"{dyn.met_ml_dir}/{dyn.met_ml_prefix}_20261114.nc",
    output_dir="/path/to/WPS",
)

t0 = datetime(2026, 11, 14, tzinfo=timezone.utc)
prepare_icon_ic(
    "/path/to/CAMS",
    "/path/to/DOM01.nc",
    "/path/to/DOM01_external_parameter.nc",
    "/path/to/ICON/run",
    t0,
    met_sfc_dir=dyn.met_sfc_dir,
    met_sfc_prefix=dyn.met_sfc_prefix,
    met_ml_dir=dyn.met_ml_dir,
    met_ml_prefix=dyn.met_ml_prefix,
)
prepare_icon_latbc(
    "/path/to/CAMS",
    "/path/to/DOM01_latbc.grid.nc",
    "/path/to/ICON/run",
    [t0 + timedelta(hours=h) for h in range(0, 25, 3)],
    met_sfc_dir=dyn.met_sfc_dir,
    met_sfc_prefix=dyn.met_sfc_prefix,
    met_ml_dir=dyn.met_ml_dir,
    met_ml_prefix=dyn.met_ml_prefix,
)

palm_summary = generate_palm_dynamic_driver(
    config_or_path="palm_doha_100m.yaml",
    date_str="2026-11-14",
    palm_mode_override="phstatp-salsa",
)
```

---

## Verification

Run the synthetic verification suite (Tests 1–8 covering physics DSL, RACM/RADM2 `ete`/`macr`/`op1`/`aco3` mappings, `SS_cs` bugfix, MADE moment factors, SALSA mass/number conservation, CAMS/MERRA-2/EMAC IC/BC injection, mixed-grid `cams2wps`, `wrf2palm` `meteorology`/`phstatp`/`phstatp-salsa`, `airblender.fetch` request chunking & `41`-task `--stream_daily`, `DynamicsResolver` multi-provider fallback & OpenData S3 mock, `cams2icon` pure-Python IC/ART/LatBC generation, Cy49r1 sea-salt rename, fail-fast circuit breaker, and atomic `cams_status.json`):

```bash
python tests/test_airblender.py
```

---

## Renaming Notice (`Merra2BC` $\rightarrow$ `AirBlender`)

This repository was renamed from **`Merra2BC`** to **`AirBlender`** (with default branch updated from `master` to `main`).

* Downstream pipelines such as `operational-forecasting` now import `airblender` directly as a Python dependency instead of spawning subprocesses with `PYTHONPATH="${CLIMPY_DIR}:${MERRA2BC_DIR}"`.
* Meteorological, chemistry, and aerosol donor acquisition scripts formerly maintained under `Climate-Modelling/data_downloading/` (CAMS Forecasts, EAC4, IFS, OpenData `sol`/`sfc`, ERA5 soil, and `MERRA2/` wget scripts) were consolidated into `airblender.fetch` and [`scripts/merra2/`](scripts/merra2/README.md).
* Legacy wrapper scripts (`cams2wrf_main.py`, `merra2wrf_main.py`, `emac2wrf_main.py`) remain in the repository root for existing cron/Slurm workflows.

---

## License

See [LICENSE](LICENSE).
