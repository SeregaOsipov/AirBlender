"""ECMWF OpenData HRES soil (`levtype=sol`) and surface GRIB downloaders."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence


def _default_hres_dir() -> Path:
    return Path(os.environ.get("SCRATCH", "/tmp")) / "Data" / "ECMWF" / "HRES"


def _normalize_date(date: str | None) -> tuple[str | None, str]:
    if not date:
        return None, "latest"
    clean = str(date).replace("-", "").replace("_", "")[:8]
    return f"{clean[:4]}-{clean[4:6]}-{clean[6:8]}", clean


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
            description="Download ECMWF OpenData HRES soil (`sol`) and/or surface (`sfc`) GRIBs.",
        )
    parser.add_argument("--mode", choices=["sol", "sfc", "both"], default="sol", help="OpenData subset to fetch")
    parser.add_argument("--date", default=None, help="Forecast date YYYY-MM-DD or YYYYMMDD (default: latest)")
    parser.add_argument("--outdir", default=None, help="Target directory (default: ${SCRATCH:-/tmp}/Data/ECMWF/HRES)")
    parser.add_argument("--time", type=int, default=0, help="Cycle hour (default: 0)")
    parser.add_argument("--step", type=int, default=0, help="Forecast step in hours (default: 0)")
    parser.add_argument("--source", default="ecmwf", help="ecmwf-opendata source (default: ecmwf)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
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
