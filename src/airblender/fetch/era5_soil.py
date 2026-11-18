"""ECMWF ERA5 single-level soil & land-surface NetCDF downloader."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

ERA5_SOIL_VARIABLES = [
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "temperature_of_snow_layer",
    "snow_density",
    "skin_reservoir_content",
    "land_sea_mask",
]


def _default_era5_dir() -> Path:
    return Path(os.environ.get("SCRATCH", "/tmp")) / "Data" / "ECMWF" / "ERA5"


def download_era5_soil(
    start_date: str | None = None,
    end_date: str | None = None,
    out_dir: str | Path | None = None,
    *,
    date: str | None = None,
    time: str = "00:00",
    area: Sequence[float] | None = None,
    force: bool = False,
) -> list[Path]:
    """Download daily ERA5 soil temperature/moisture NetCDF slices."""
    import cdsapi

    s_date = start_date or date
    if not s_date:
        raise ValueError("Specify `date` or `start_date` for download_era5_soil.")
    e_date = end_date or s_date

    target_dir = Path(out_dir) if out_dir is not None else _default_era5_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.period_range(start=s_date, end=e_date, freq="D")

    pending: list[tuple[pd.Timestamp, Path]] = []
    outputs: list[Path] = []
    for raw_d in dates:
        ts = raw_d.to_timestamp()
        out_fp = target_dir / f"ERA5_sfc_{ts.strftime('%Y%m%d')}.nc"
        if not force and out_fp.exists() and out_fp.stat().st_size > 0:
            print(f"[SKIPPED] {out_fp} already exists.")
            outputs.append(out_fp)
            continue
        pending.append((ts, out_fp))

    if not pending:
        return outputs

    client = cdsapi.Client()
    for ts, out_fp in pending:
        req: dict = {
            "product_type": "reanalysis",
            "variable": ERA5_SOIL_VARIABLES,
            "year": ts.strftime("%Y"),
            "month": ts.strftime("%m"),
            "day": ts.strftime("%d"),
            "time": str(time),
            "format": "netcdf",
        }
        if area and len(area) == 4:
            req["area"] = [float(x) for x in area]

        print(f"Downloading ERA5 soil ({ts.strftime('%Y-%m-%d')} {time}) -> {out_fp}")
        client.retrieve("reanalysis-era5-single-levels", req, str(out_fp))
        outputs.append(out_fp)
    return outputs


def build_arg_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="airblender fetch-era5-soil",
            description="Download ERA5 soil/surface NetCDF files via cdsapi.",
        )
    parser.add_argument("--date", default=None, help="Single date (YYYY-MM-DD)")
    parser.add_argument("--start_date", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--outdir", default=None, help="Output directory (default: ${SCRATCH:-/tmp}/Data/ECMWF/ERA5)")
    parser.add_argument("--time", default="00:00", help="Analysis hour (default: 00:00)")
    parser.add_argument("--area", nargs=4, type=float, default=None, metavar=("N", "W", "S", "E"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not (args.date or args.start_date):
        parser.error("One of --date or --start_date is required.")
    download_era5_soil(
        start_date=args.start_date,
        end_date=args.end_date,
        out_dir=args.outdir,
        date=args.date,
        time=args.time,
        area=args.area,
        force=args.force,
    )
