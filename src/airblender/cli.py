"""Unified CLI entry points for AirBlender (`cams2wrf`, `merra2wrf`, `emac2wrf`, `cams2wps`, `wrf2palm`, `fetch-*`)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from airblender.cams2wps import convert_cams_to_wps
from airblender.fetch import cams_eac4, cams_forecast, dynamics, ecmwf_ifs, era5_soil, merra2, opendata
from airblender.icon_icbc import prepare_icon_ic, prepare_icon_latbc
from airblender.palm_dynamic import PALM_MODES, generate_palm_dynamic_driver
from airblender.utils.time_io import parse_wrf_datetime
from airblender.wrf_icbc import run_wrf_icbc


def _add_wrf_icbc_arguments(parser: argparse.ArgumentParser, source: str) -> None:
    defaults = {
        "cams": {
            "reanalysis_dir": "/project/k10048/Data/ECMWF/CAMS/",
            "template": "ECMWF-CAMS_{stream}_{date_time}.nc",
        },
        "merra2": {
            "reanalysis_dir": "/project/k10048/Data/NASA/MERRA2/",
            "template": "{stream}/MERRA2_{VVV}.{stream}.{date_time}.nc4",
        },
        "emac": {
            "reanalysis_dir": "",
            "template": "test01_________{date_time}_{stream}.nc",
        },
    }[source]

    parser.add_argument("--start_date", default=None, type=parse_wrf_datetime, help="YYYY-MM-DD_HH:MM:SS format")
    parser.add_argument("--end_date", default=None, type=parse_wrf_datetime, help="YYYY-MM-DD_HH:MM:SS format")
    parser.add_argument("--hourly_interval", type=float, default=3, help="Time step in hours")
    parser.add_argument("--do_IC", action="store_true", help="Process initial conditions")
    parser.add_argument("--do_BC", action="store_true", help="Process boundary conditions")
    parser.add_argument("--zero_out_first", action="store_true", help="Zero out target WRF fields before increments")
    parser.add_argument("--reanalysis_dir", "--emac_dir", dest="reanalysis_dir", default=defaults["reanalysis_dir"])
    parser.add_argument(
        "--reanalysis_file_name_template",
        "--emac_file_name_template",
        dest="reanalysis_file_name_template",
        default=defaults["template"],
    )
    parser.add_argument("--wrf_dir", required=True, help="Folder containing WRF IC & BC files")
    parser.add_argument("--wrf_input", default="wrfinput_d01", help="WRF initial condition filename")
    parser.add_argument("--wrf_bdy_file", default="wrfbdy_d01", help="WRF boundary condition filename")
    parser.add_argument("--wrf_met_dir", required=True, help="Folder containing met_em files")
    parser.add_argument("--wrf_met_files_name_template", default="met_em.d01.{}.nc")
    parser.add_argument("--wrf_met_files_date_format", default="%Y-%m-%d_%H_%M_%S")
    parser.add_argument("--wrf_met_files", default=None, help="Legacy informational glob flag (optional)")
    parser.add_argument("--streams", nargs="*", default=None, help="Optional subset of donor streams (e.g. ghg)")
    parser.add_argument(
        "--scheme",
        default="made_sorgam",
        help="Target WRF aerosol/chemistry scheme (e.g. made_sorgam, gocart, mosaic_8bin, messy_channels)",
    )
    parser.add_argument("--mode", "--port", "--host", help="Ignored PyCharm debugger flags")


def _dispatch_wrf_icbc(source: str, argv: Sequence[str] | None = None) -> None:
    prog_name = "merra2wrf" if source == "merra2" else f"{source}2wrf"
    parser = argparse.ArgumentParser(prog=prog_name, description=f"AirBlender {source.upper()} -> WRF IC/BC")
    _add_wrf_icbc_arguments(parser, source)
    args = parser.parse_args(argv)
    run_wrf_icbc(
        source=source,
        reanalysis_dir=args.reanalysis_dir,
        reanalysis_file_name_template=args.reanalysis_file_name_template,
        wrf_dir=args.wrf_dir,
        wrf_met_dir=args.wrf_met_dir,
        wrf_input=args.wrf_input,
        wrf_bdy_file=args.wrf_bdy_file,
        wrf_met_files_name_template=args.wrf_met_files_name_template,
        wrf_met_files_date_format=args.wrf_met_files_date_format,
        start_date=args.start_date,
        end_date=args.end_date,
        hourly_interval=args.hourly_interval,
        do_ic=args.do_IC,
        do_bc=args.do_BC,
        zero_out_first=args.zero_out_first,
        streams=args.streams,
        scheme=args.scheme,
    )


def main_cams2wrf(argv: Sequence[str] | None = None) -> None:
    _dispatch_wrf_icbc("cams", argv)


def main_merra2wrf(argv: Sequence[str] | None = None) -> None:
    _dispatch_wrf_icbc("merra2", argv)


def main_emac2wrf(argv: Sequence[str] | None = None) -> None:
    _dispatch_wrf_icbc("emac", argv)


def main_cams2wps(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cams2wps", description="Convert CAMS NetCDFs to WPS intermediate format")
    parser.add_argument("--sfc", required=True, help="Path to ECMWF-CAMS_sfc_*.nc")
    parser.add_argument("--ml", required=True, help="Path to ECMWF-CAMS_ml_*.nc")
    parser.add_argument("--soil", default=None, help="Optional path to ecmwf_hres_sol_*.nc")
    parser.add_argument("--outdir", required=True, help="Output directory for WPS FILE:* files")
    parser.add_argument("--prefix", default="FILE", help="WPS intermediate file prefix (default: FILE)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing WPS intermediate slices")
    args = parser.parse_args(argv)
    convert_cams_to_wps(
        Path(args.sfc),
        Path(args.ml),
        Path(args.outdir),
        opendata_soil_file=Path(args.soil) if args.soil else None,
        prefix=args.prefix,
        overwrite=args.overwrite,
    )


def main_wrf2palm(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="wrf2palm", description="Convert WRF-Chem output to PALM-4U PIDS_DYNAMIC")
    parser.add_argument("--config", required=True, help="Path to PALM YAML configuration")
    parser.add_argument("--cluster", default=None, help="Optional cluster YAML configuration path")
    parser.add_argument("--date", default=None, help="Cycle date (YYYY-MM-DD)")
    parser.add_argument("--output", default=None, help="Optional output PIDS_DYNAMIC path override")
    parser.add_argument("--wrf-dir", default=None, help="Optional directory containing wrfout_d02*/d01* files")
    parser.add_argument("--static-driver", default=None, help="Optional PALM _static driver override")
    parser.add_argument("--hours", type=float, default=None, help="Optional forecast duration override (hours)")
    parser.add_argument("--palm-mode", "--mode", dest="palm_mode", choices=PALM_MODES, default=None, help="Override configured palm_mode")
    args = parser.parse_args(argv)
    generate_palm_dynamic_driver(
        config_or_path=args.config,
        date_str=args.date,
        cluster_path=args.cluster,
        output_path=Path(args.output) if args.output else None,
        forecast_hours_override=args.hours,
        palm_mode_override=args.palm_mode,
        wrf_dir_override=args.wrf_dir,
        static_driver_override=args.static_driver,
    )


def _add_cams2icon_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cams_dir", required=True, help="Directory with ECMWF-CAMS_{ml,sfc,ghg}_*.nc")
    parser.add_argument("--grid_file", required=True, help="ICON domain grid NetCDF")
    parser.add_argument("--extpar_file", default=None, help="ICON external parameter NetCDF (required unless --latbc_only)")
    parser.add_argument("--latbc_grid_file", default=None, help="ICON LatBC grid NetCDF (optional)")
    parser.add_argument("--run_dir", required=True, help="Target ICON run directory")
    parser.add_argument("--start_date", required=True, help="Cycle/window start UTC (YYYY-MM-DD or YYYY-MM-DD_HH:MM:SS)")
    parser.add_argument("--end_date", default=None, help="LatBC end UTC (YYYY-MM-DD or YYYY-MM-DD_HH:MM:SS)")
    parser.add_argument("--interval_hours", type=int, default=3, help="LatBC time step in hours (default: 3)")
    parser.add_argument("--res_str", default="R2B09")
    parser.add_argument("--suffix", default="TCNR")
    parser.add_argument("--dom_id", type=int, default=1)
    parser.add_argument("--met_sfc_dir", default=None, help="Optional separate surface donor directory")
    parser.add_argument("--met_sfc_prefix", default="ECMWF-CAMS_sfc", help="Surface donor filename prefix")
    parser.add_argument("--met_ml_dir", default=None, help="Optional separate 3D model-level donor directory")
    parser.add_argument("--met_ml_prefix", default="ECMWF-CAMS_ml", help="3D model-level donor filename prefix")
    parser.add_argument("--latbc_only", action="store_true", help="Only generate LatBC files")


def _parse_iso_or_date_utc(raw: str):
    from datetime import datetime, timezone

    s = str(raw).strip().replace("_", " ").rstrip("Z")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"Unrecognized UTC timestamp format: {raw!r}")


def _dispatch_cams2icon(args: argparse.Namespace) -> None:
    from datetime import timedelta

    start_dt = _parse_iso_or_date_utc(args.start_date)
    if not args.latbc_only:
        if not args.extpar_file:
            raise ValueError("--extpar_file is required unless --latbc_only is set")
        prepare_icon_ic(
            args.cams_dir,
            args.grid_file,
            args.extpar_file,
            args.run_dir,
            start_dt,
            res_str=args.res_str,
            cart_io_suffix=args.suffix,
            dom_id=args.dom_id,
            met_sfc_dir=args.met_sfc_dir,
            met_sfc_prefix=args.met_sfc_prefix,
            met_ml_dir=args.met_ml_dir,
            met_ml_prefix=args.met_ml_prefix,
        )
    if args.latbc_grid_file:
        end_dt = _parse_iso_or_date_utc(args.end_date) if args.end_date else start_dt
        times = []
        cur = start_dt
        while cur <= end_dt:
            times.append(cur)
            cur += timedelta(hours=max(1, int(args.interval_hours)))
        prepare_icon_latbc(
            args.cams_dir,
            args.latbc_grid_file,
            args.run_dir,
            times,
            met_sfc_dir=args.met_sfc_dir,
            met_sfc_prefix=args.met_sfc_prefix,
            met_ml_dir=args.met_ml_dir,
            met_ml_prefix=args.met_ml_prefix,
        )


def main_cams2icon(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cams2icon", description="Convert ECMWF IFS/CAMS NetCDFs to ICON-ART IC & LatBC")
    _add_cams2icon_arguments(parser)
    args = parser.parse_args(argv)
    _dispatch_cams2icon(args)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="airblender",
        description="AirBlender — Unified atmospheric meteorology, chemistry, aerosol coupler & donor fetcher",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)
    icbc_subcommands = {
        "cams2wrf": "cams",
        "merra2wrf": "merra2",
        "emac2wrf": "emac",
    }
    for cmd_name, src in icbc_subcommands.items():
        p_sub = sub.add_parser(cmd_name, help=f"{src.upper()} -> WRF-Chem IC/BC injector")
        _add_wrf_icbc_arguments(p_sub, src)

    p_wps = sub.add_parser("cams2wps", help="CAMS NetCDF -> WPS intermediate binary converter")
    p_wps.add_argument("--sfc", required=True)
    p_wps.add_argument("--ml", required=True)
    p_wps.add_argument("--soil", default=None)
    p_wps.add_argument("--outdir", required=True)
    p_wps.add_argument("--prefix", default="FILE")
    p_wps.add_argument("--overwrite", action="store_true")

    p_palm = sub.add_parser("wrf2palm", help="WRF-Chem -> PALM-4U PIDS_DYNAMIC generator")
    p_palm.add_argument("--config", required=True)
    p_palm.add_argument("--cluster", default=None)
    p_palm.add_argument("--date", default=None)
    p_palm.add_argument("--output", default=None)
    p_palm.add_argument("--wrf-dir", default=None)
    p_palm.add_argument("--static-driver", default=None)
    p_palm.add_argument("--hours", type=float, default=None)
    p_palm.add_argument("--palm-mode", "--mode", dest="palm_mode", choices=PALM_MODES, default=None)

    p_icon = sub.add_parser("cams2icon", help="ECMWF IFS/CAMS NetCDFs -> ICON-ART IC & LatBC")
    _add_cams2icon_arguments(p_icon)

    cams_forecast.build_arg_parser(sub.add_parser("fetch-cams", help="Download ECMWF CAMS forecasts/analyses (GRIB pipelined)"))
    cams_eac4.build_arg_parser(sub.add_parser("fetch-eac4", help="Download daily ECMWF CAMS EAC4 reanalysis"))
    ecmwf_ifs.build_arg_parser(sub.add_parser("fetch-ifs", help="Download ECMWF MARS operational IFS NetCDFs"))
    opendata.build_arg_parser(sub.add_parser("fetch-opendata", help="Download ECMWF OpenData HRES soil/surface NetCDFs or GRIBs"))
    dynamics.build_arg_parser(sub.add_parser("fetch-dynamics", help="Resolve and download highest-available dynamics with fallback"))
    era5_soil.build_arg_parser(sub.add_parser("fetch-era5-soil", help="Download ERA5 soil/surface NetCDFs"))
    merra2.build_arg_parser(sub.add_parser("fetch-merra2", help="Download NASA MERRA-2 GES DISC collections"))

    args = parser.parse_args(argv)
    cmd = args.subcommand
    if cmd in icbc_subcommands:
        src = icbc_subcommands[cmd]
        run_wrf_icbc(
            source=src,
            reanalysis_dir=args.reanalysis_dir,
            reanalysis_file_name_template=args.reanalysis_file_name_template,
            wrf_dir=args.wrf_dir,
            wrf_met_dir=args.wrf_met_dir,
            wrf_input=args.wrf_input,
            wrf_bdy_file=args.wrf_bdy_file,
            wrf_met_files_name_template=args.wrf_met_files_name_template,
            wrf_met_files_date_format=args.wrf_met_files_date_format,
            start_date=args.start_date,
            end_date=args.end_date,
            hourly_interval=args.hourly_interval,
            do_ic=args.do_IC,
            do_bc=args.do_BC,
            zero_out_first=args.zero_out_first,
            streams=args.streams,
            scheme=args.scheme,
        )
    elif cmd == "cams2wps":
        convert_cams_to_wps(
            Path(args.sfc),
            Path(args.ml),
            Path(args.outdir),
            opendata_soil_file=Path(args.soil) if args.soil else None,
            prefix=args.prefix,
            overwrite=args.overwrite,
        )
    elif cmd == "wrf2palm":
        generate_palm_dynamic_driver(
            config_or_path=args.config,
            date_str=args.date,
            cluster_path=args.cluster,
            output_path=Path(args.output) if args.output else None,
            forecast_hours_override=args.hours,
            palm_mode_override=args.palm_mode,
            wrf_dir_override=args.wrf_dir,
            static_driver_override=args.static_driver,
        )
    elif cmd == "cams2icon":
        _dispatch_cams2icon(args)
    elif cmd == "fetch-cams":
        cams_forecast.apply_config_and_defaults(args)
        try:
            cams_forecast.execute_cams_pipeline(args)
        except (RuntimeError, ValueError) as exc:
            cams_forecast.log_msg(f"Error: {exc}")
            sys.exit(1)
    elif cmd == "fetch-eac4":
        cams_eac4.download_cams_eac4(
            args.start_date,
            args.end_date,
            args.data_root_folder,
            area=args.area,
            force=args.force,
        )
    elif cmd == "fetch-ifs":
        ecmwf_ifs.download_ecmwf_ifs(
            args.start_date,
            args.end_date,
            args.data_root_folder,
            grid=args.grid,
            retrieve_ml=args.retrieve_ml,
            force=args.force,
        )
    elif cmd == "fetch-opendata":
        if args.mode == "s3-0p1":
            if not args.date:
                parser.error("fetch-opendata --mode s3-0p1 requires --date.")
            out_dir = Path(args.outdir) if args.outdir is not None else opendata._default_hres_dir()
            opendata.download_opendata_0p1_sfc_daily(
                args.date,
                out_dir,
                time_str=f"{int(args.time):02d}:00",
                leadtimes_hours=args.leadtimes,
                area=args.area,
                force=args.force,
            )
        else:
            if args.mode in ("sol", "both"):
                opendata.download_opendata_soil(
                    date=args.date, out_dir=args.outdir, time=args.time, step=args.step, source=args.source, force=args.force
                )
            if args.mode in ("sfc", "both"):
                opendata.download_opendata_sfc(
                    date=args.date, out_dir=args.outdir, time=args.time, step=args.step, source=args.source, force=args.force
                )
    elif cmd == "fetch-dynamics":
        import json

        res = dynamics.resolve_and_fetch_dynamics(
            args.date,
            cams_dir=args.cams_dir,
            out_dir=args.out_dir,
            dynamics_cfg={"mode": args.mode, "primary": args.primary},
            leadtimes_hours=args.leadtimes,
            area=args.area,
            provenance_dir=args.out_dir or Path(args.cams_dir).parent / "IFS",
            force=args.force,
        )
        print(json.dumps(res.to_dict(), indent=2))
    elif cmd == "fetch-era5-soil":
        if not (args.date or args.start_date):
            parser.error("fetch-era5-soil requires --date or --start_date.")
        era5_soil.download_era5_soil(
            start_date=args.start_date,
            end_date=args.end_date,
            out_dir=args.outdir,
            date=args.date,
            time=args.time,
            area=args.area,
            force=args.force,
        )
    elif cmd == "fetch-merra2":
        merra2.download_merra2(
            args.start_date,
            args.end_date,
            ndays=args.ndays,
            out_dir=args.outdir,
            collections=args.collections,
            preset=args.preset,
            cookies_path=args.cookies,
            max_workers=args.max_workers,
            dry_run=args.dry_run,
            force=args.force,
        )
