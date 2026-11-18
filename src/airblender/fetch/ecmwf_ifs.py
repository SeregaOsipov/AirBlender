"""ECMWF MARS operational IFS (F1280 / gridded) meteorological downloader."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

WRF_SURFACE_VARS = (
    "129.128/172.128/134.128/139.128/141.128/151.128/165.128/"
    "166.128/167.128/168.128/170.128/183.128/235.128/236.128/"
    "31.128/33.128/34.128/39.128/40.128/41.128/42.128"
)
WRF_VERTICAL_LEVEL_VARS = "130.128/131.128/132.128/133.128"


def _strtobool(val: str | bool) -> bool:
    if isinstance(val, bool):
        return val
    norm = str(val).strip().lower()
    if norm in ("y", "yes", "t", "true", "on", "1"):
        return True
    if norm in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"Invalid truth value: {val!r}")


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
            description="Download ECMWF MARS operational IFS NetCDF files (`ml` or `sfc`).",
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
