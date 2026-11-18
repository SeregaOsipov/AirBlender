"""NASA MERRA-2 GES DISC collection downloader (`wget` + Earthdata session cookies)."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
from pathlib import Path
import subprocess
from typing import Sequence

import pandas as pd

MERRA2_COLLECTIONS: dict[str, str] = {
    "inst3_3d_asm_Nv": "https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVASM.5.12.4",
    "inst1_2d_asm_Nx": "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I1NXASM.5.12.4",
    "tavg1_2d_slv_Nx": "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXSLV.5.12.4",
    "tavg1_2d_ocn_Nx": "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXOCN.5.12.4",
    "tavg1_2d_lnd_Nx": "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXLND.5.12.4",
    "inst3_3d_aer_Nv": "https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVAER.5.12.4",
    "inst3_3d_chm_Nv": "https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVCHM.5.12.4",
}

MERRA2_EXTRA_COLLECTIONS: dict[str, str] = {
    "tavg1_2d_aer_Nx": "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXAER.5.12.4",
    "tavgM_2d_aer_Nx": "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2TMNXAER.5.12.4",
    "tavg3_3d_nav_Ne": "https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T3NENAV.5.12.4",
    "tavg1_2d_adg_Nx": "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXADG.5.12.4",
}

ALL_MERRA2_COLLECTIONS: dict[str, str] = {**MERRA2_COLLECTIONS, **MERRA2_EXTRA_COLLECTIONS}

MERRA2_PRESETS: dict[str, list[str]] = {
    "wrfchem": list(MERRA2_COLLECTIONS.keys()),
    "zittis_aot": [
        "tavg1_2d_aer_Nx",
        "tavgM_2d_aer_Nx",
        "inst3_3d_aer_Nv",
        "tavg3_3d_nav_Ne",
        "tavg1_2d_adg_Nx",
    ],
}


def merra2_stream_version(date_obj: dt.date | dt.datetime | pd.Timestamp | str) -> str:
    """Return NASA MERRA-2 file stream prefix (`100`..`401`) for a calendar date."""
    if isinstance(date_obj, str):
        ts = pd.Timestamp(date_obj)
        year, month = ts.year, ts.month
    else:
        year, month = date_obj.year, date_obj.month

    if year == 2021 and 6 <= month <= 9:
        return "401"
    if year >= 2011:
        return "400"
    if year >= 2001:
        return "300"
    if year >= 1992:
        return "200"
    return "100"


def build_merra2_url(collection: str, date_obj: dt.date | dt.datetime | pd.Timestamp | str) -> tuple[str, str]:
    """Return `(url, relative_target_path)` for one MERRA-2 collection and date."""
    if collection not in ALL_MERRA2_COLLECTIONS:
        raise KeyError(f"Unsupported MERRA-2 collection '{collection}'. Valid: {list(ALL_MERRA2_COLLECTIONS)}")

    ts = pd.Timestamp(date_obj)
    vvv = merra2_stream_version(ts)
    base_url = ALL_MERRA2_COLLECTIONS[collection].rstrip("/")
    if collection.startswith("tavgM_"):
        filename = f"MERRA2_{vvv}.{collection}.{ts.strftime('%Y%m')}.nc4"
        url = f"{base_url}/{ts.year:04d}/{filename}"
    else:
        filename = f"MERRA2_{vvv}.{collection}.{ts.strftime('%Y%m%d')}.nc4"
        url = f"{base_url}/{ts.year:04d}/{ts.month:02d}/{filename}"
    return url, f"{collection}/{filename}"


def _fetch_one_wget(url: str, target_path: Path, cookies_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "--directory-prefix",
        str(target_path.parent),
        "--no-verbose",
        "-c",
        "--load-cookies",
        str(cookies_path),
        "--save-cookies",
        str(cookies_path),
        "--keep-session-cookies",
        "--content-disposition",
        url,
    ]
    subprocess.run(cmd, check=True)
    return target_path


def download_merra2(
    start_date: str,
    end_date: str | None = None,
    *,
    ndays: int | None = None,
    out_dir: str | Path = ".",
    collections: Sequence[str] | None = None,
    preset: str = "wrfchem",
    cookies_path: str | Path = "~/.urs_cookies",
    max_workers: int = 5,
    dry_run: bool = False,
    force: bool = False,
) -> list[Path]:
    """Download MERRA-2 daily/monthly `.nc4` collections via `wget` + Earthdata session cookies."""
    start_ts = pd.Timestamp(start_date)
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
    elif ndays is not None:
        end_ts = start_ts + pd.Timedelta(days=int(ndays))
    else:
        end_ts = start_ts

    if collections:
        active_collections = list(collections)
    else:
        if preset not in MERRA2_PRESETS:
            raise KeyError(f"Unknown MERRA-2 preset '{preset}'. Valid: {list(MERRA2_PRESETS)}")
        active_collections = MERRA2_PRESETS[preset]

    root = Path(out_dir)
    cookies = Path(cookies_path).expanduser()

    dates = pd.date_range(start=start_ts, end=end_ts, freq="D")
    planned: list[tuple[str, Path]] = []
    outputs: list[Path] = []
    seen_targets: set[Path] = set()

    for col in active_collections:
        for ts in dates:
            url, rel_path = build_merra2_url(col, ts)
            target_fp = root / rel_path
            if target_fp in seen_targets:
                continue
            seen_targets.add(target_fp)

            if dry_run:
                print(f"[DRY-RUN] {url} -> {target_fp}")
                outputs.append(target_fp)
                continue
            if not force and target_fp.exists() and target_fp.stat().st_size > 0:
                print(f"[SKIPPED] {target_fp} already exists.")
                outputs.append(target_fp)
                continue
            planned.append((url, target_fp))

    if dry_run or not planned:
        return outputs

    workers = max(1, int(max_workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futs = [executor.submit(_fetch_one_wget, url, fp, cookies) for url, fp in planned]
        for fut in concurrent.futures.as_completed(futs):
            outputs.append(fut.result())

    return outputs


def build_arg_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="airblender fetch-merra2",
            description="Download NASA MERRA-2 collections from GES DISC via wget.",
        )
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--ndays", type=int, default=None, help="Number of days after start_date if end_date omitted")
    parser.add_argument("--outdir", default=".", help="Root MERRA-2 output directory (default: .)")
    parser.add_argument(
        "--preset",
        choices=list(MERRA2_PRESETS.keys()),
        default="wrfchem",
        help="Collection preset when --collections is omitted (default: wrfchem)",
    )
    parser.add_argument(
        "--collections",
        nargs="*",
        default=None,
        help=f"Explicit subset of collections ({', '.join(ALL_MERRA2_COLLECTIONS)})",
    )
    parser.add_argument("--cookies", default="~/.urs_cookies", help="Path to NASA Earthdata cookies file")
    parser.add_argument("--max_workers", type=int, default=5, help="Concurrent wget transfers (default: 5)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned URLs and output paths without downloading")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    download_merra2(
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
