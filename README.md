# AirBlender

**Unified Atmospheric Meteorology, Chemistry, and Aerosol Coupler**

`AirBlender` (`airblender`) couples global atmospheric composition and reanalysis models (**ECMWF CAMS**, **NASA MERRA-2**, **EMAC / MESSy**) with regional **WPS / WRF-Chem** and urban microscale **PALM-4U** (`PIDS_DYNAMIC`), including gas-phase conversion, modal $\leftrightarrow$ sectional aerosol remapping, mixed-grid vertical pressure reconstruction, and direct Fortran unformatted WPS binary generation.

It is a standalone, `pip`-installable Python package.

---

## Key Capabilities & Coupling Pathways

| Subcommand / Entry Point | Source $\rightarrow$ Target | Scope & Highlights |
| :--- | :--- | :--- |
| **`cams2wrf`** (`airblender.wrf_icbc`) | ECMWF CAMS $\rightarrow$ WRF-Chem | Reactive trace gases, `RACM_KPP_MADE_SORGAM_GOCART` aerosols, and mixed-grid `0.1°` CAMS GHG (`co2`, `ch4` with `ml` `ch4_c` fallback). Supports multiple nested `wrfinput_d0*` + `wrfbdy_d01` with edge-clamped vertical pressure interpolation and identical-step BC reuse. |
| **`merra2wrf`** (`airblender.wrf_icbc`) | NASA MERRA-2 $\rightarrow$ WRF-Chem | `PL` + `DELP` fallback hybrid pressure reconstruction, `VVV` naming support, selectable WRF schemes (`made_sorgam`, `gocart`, `gocart_dust`, `mosaic_8bin`), full `CAMS2WRF` parity (stream caching + BC array reuse). |
| **`emac2wrf`** (`airblender.wrf_icbc`) | EMAC / MESSy $\rightarrow$ WRF-Chem | Log-normal 4-mode (`ns, ks, as, cs`) $\rightarrow$ MADE/SORGAM 3-mode (`nu0/ai`, `ac0/aj`, `corn`) moment mapping, `[0, 360) -> [-180, 180)` longitude shifting, glob restart handling, and strict DSL syntax checking (including fixed `2.01*[SS_cs]` rule). |
| **`cams2wps`** (`airblender.cams2wps`) | ECMWF CAMS $\rightarrow$ WPS | Direct Fortran unformatted WPS intermediate generator (`FILE:YYYY-MM-DD_HH`), computing 137-level midpoint pressures from `GRIB_pv` `(a_k, b_k)`, mapping `SOILHGT`, optional ECMWF OpenData 4-layer soil (`ST*`, `SM*`), atomic `.tmp` writes, and up-to-date slice skipping. |
| **`wrf2palm`** (`airblender.palm_dynamic`) | WRF-Chem $\rightarrow$ PALM-4U | Generates `PIDS_DYNAMIC` (`_dynamic`) NetCDF-4 for urban microscale domains (`100 m` SLUrb / `10 m` LES) with `d02 -> d01` parent fallback, LCC projection + wind rotation, `PALM-meteo` (`universal`) barometric surface-layer pressure adaptation, and selectable `chemistry.palm_mode`: `meteorology` (`31` fields), `phstatp` (`55` fields), **`phstatp-salsa`** (`115` fields: conservative 7-species `['SO4', 'OC', 'BC', 'DU', 'SS', 'NO', 'NH']` $\times$ 10-bin sectional WRF–SALSA remapping, `antha` excluded under `absolute_mass_number_v1`), and `legacy_3spec` (`97` fields). |
| **`fetch-*`** (`airblender.fetch`) *(optional `[fetch]`)* | ADS / CDS / MARS / OpenData / GES DISC $\rightarrow$ Local NetCDF/GRIB | Donor forcing acquisition: **`fetch-cams`** (137-level CAMS GRIB pipelining + CDM `ECMWF-CAMS_{sfc,ml,ghg}_YYYYMMDD.nc`), **`fetch-eac4`** (EAC4 reanalysis), **`fetch-ifs`** (MARS IFS `F1280`), **`fetch-opendata`** (ECMWF HRES 4-layer soil `vsw`/`sot` & `sfc`), **`fetch-era5-soil`** (ERA5 soil NetCDFs), and **`fetch-merra2`** (concurrent NASA GES DISC `.nc4` collections). |

---

## Package Architecture

```text
AirBlender/
├── pyproject.toml
├── src/airblender/
│   ├── cli.py                        # Unified `airblender` CLI & legacy entry points
│   ├── wrf_icbc.py                   # Unified donor -> WRF-Chem IC/BC injector
│   ├── cams2wps.py                   # CAMS NetCDF -> WPS intermediate binary converter
│   ├── palm_dynamic.py               # WRF-Chem -> PALM-4U PIDS_DYNAMIC driver
│   ├── fetch/                        # Optional donor data fetchers (pip install airblender[fetch])
│   │   ├── cams_forecast.py          # ECMWF CAMS forecast/analysis GRIB pipelining & daily NetCDFs
│   │   ├── cams_eac4.py              # ECMWF CAMS EAC4 60-level daily reanalysis
│   │   ├── ecmwf_ifs.py              # ECMWF MARS operational IFS F1280 downloader
│   │   ├── opendata.py               # ECMWF OpenData HRES soil (sol) & surface (sfc) GRIBs
│   │   ├── era5_soil.py              # ECMWF ERA5 land/soil NetCDF downloader
│   │   └── merra2.py                 # NASA MERRA-2 GES DISC wget downloader
│   ├── sources/
│   │   ├── base.py                   # DonorAdapter base class & stream/p_rho caching
│   │   ├── cams.py                   # ECMWF CAMS (sfc, ml, ghg + valid_time/coord handling)
│   │   ├── merra2.py                 # NASA MERRA-2 (PL + DELP fallback, VVV collections)
│   │   └── emac.py                   # EMAC / MESSy (hybrid pressure, lon shift, glob files)
│   ├── physics/
│   │   ├── dsl.py                    # Strict mapping rule parser ("out <- a*[in];exp")
│   │   ├── sectional_to_modal.py     # Log-normal modal number/mass factor computation
│   │   ├── mappings.py               # CAMS, MERRA-2 (4 presets), and EMAC conversion tables
│   │   └── salsa_remap.py            # Conservative 7-species x 10-bin WRF -> SALSA remapper
│   ├── regrid/
│   │   └── wrf_regrid.py             # Horizontal RegularGrid + log-P vertical & BC interpolation
│   └── utils/
│       ├── time_io.py                # Native datetime/char Times parsing & eta pressure derivation
│       └── config_io.py              # YAML loader & ${VAR:-default} path expansion
├── cams2wrf_main.py                  # Backwards-compatible CLI wrapper
├── merra2wrf_main.py                 # Backwards-compatible CLI wrapper
├── emac2wrf_main.py                  # Backwards-compatible CLI wrapper
└── tests/
    └── test_airblender.py            # Synthetic verification suite (DSL, IC/BC, WPS, PALM)
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

# 1. CAMS -> WRF-Chem IC & BC (d01)
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

# 4. CAMS -> WPS Intermediate Binaries (FILE:YYYY-MM-DD_HH)
airblender cams2wps --sfc ECMWF-CAMS_sfc_20261114.nc --ml ECMWF-CAMS_ml_20261114.nc --outdir /path/to/WPS

# 5. WRF-Chem -> PALM-4U Dynamic Driver (PIDS_DYNAMIC)
airblender wrf2palm --config palm_doha_100m.yaml --date 2026-11-14 --mode phstatp-salsa

# 6. Optional Donor Data Acquisition (`airblender[fetch]`)
airblender fetch-cams --start_date 2026-11-14 --end_date 2026-11-14 --data_root_folder /path/to/CAMS --type forecast --time 00:00 --ghg
airblender fetch-eac4 --start_date 2026-03-01 --end_date 2026-03-05 --data_root_folder /path/to/EAC4
airblender fetch-ifs --start_date 2026-03-01 --end_date 2026-03-02 --data_root_folder /path/to/IFS --grid F1280 --retrieve_ml true
airblender fetch-opendata --mode sol --date 2026-11-14 --outdir /path/to/HRES
airblender fetch-era5-soil --date 2026-03-01 --outdir /path/to/ERA5
airblender fetch-merra2 --start_date 2026-11-14 --ndays 2 --outdir /path/to/MERRA2
```

Console scripts `cams2wrf`, `merra2wrf`, `emac2wrf`, `cams2wps`, and `wrf2palm`, as well as legacy repo-root entry scripts (`cams2wrf_main.py`, `merra2wrf_main.py`, `emac2wrf_main.py`), are also installed for backwards compatibility.

---

## Python Library API

```python
from airblender.wrf_icbc import run_wrf_icbc
from airblender.cams2wps import convert_cams_to_wps
from airblender.palm_dynamic import generate_palm_dynamic_driver
from airblender.fetch import download_cams_forecast, download_opendata_soil

download_cams_forecast(
    start_date="2026-11-14",
    end_date="2026-11-14",
    data_root_folder="/path/to/CAMS",
    cams_type="forecast",
    cams_time="00:00",
    ghg=True,
)
download_opendata_soil(date="2026-11-14", out_dir="/path/to/HRES")

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
    cams_sfc_file="ECMWF-CAMS_sfc_20261114.nc",
    cams_ml_file="ECMWF-CAMS_ml_20261114.nc",
    output_dir="/path/to/WPS",
)

palm_summary = generate_palm_dynamic_driver(
    config_or_path="palm_doha_100m.yaml",
    date_str="2026-11-14",
    palm_mode_override="phstatp-salsa",
)
```

---

## Verification

Run the synthetic verification suite (tests physics DSL, `SS_cs` bugfix, MADE moment factors, SALSA mass/number conservation, CAMS/MERRA-2/EMAC IC/BC injection, CAMS2WPS binary generation, and WRF2PALM `meteorology`/`phstatp`/`phstatp-salsa` drivers):

```bash
python tests/test_airblender.py
```

---

## Renaming Notice (`Merra2BC` $\rightarrow$ `AirBlender`)

This repository was renamed from **`Merra2BC`** to **`AirBlender`** (with default branch updated from `master` to `main`).

* Downstream pipelines such as `operational-forecasting` now import `airblender` directly as a Python dependency instead of spawning subprocesses with `PYTHONPATH="${CLIMPY_DIR}:${MERRA2BC_DIR}"`.
* Legacy wrapper scripts (`cams2wrf_main.py`, `merra2wrf_main.py`, `emac2wrf_main.py`) remain in the repository root for existing cron/Slurm workflows.

---

## License

See [LICENSE](LICENSE).
