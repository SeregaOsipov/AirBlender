"""ECMWF CAMS Global Reanalysis (EAC4, 60 levels) daily downloader."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Sequence

import pandas as pd


EAC4_VARIABLES = [
    "surface_pressure",
    "carbon_monoxide",
    "dust_aerosol_0.03-0.55um_mixing_ratio",
    "dust_aerosol_0.55-0.9um_mixing_ratio",
    "dust_aerosol_0.9-20um_mixing_ratio",
    "ethane",
    "formaldehyde",
    "hydrogen_peroxide",
    "hydrophilic_black_carbon_aerosol_mixing_ratio",
    "hydrophilic_organic_matter_aerosol_mixing_ratio",
    "hydrophobic_black_carbon_aerosol_mixing_ratio",
    "hydrophobic_organic_matter_aerosol_mixing_ratio",
    "hydroxyl_radical",
    "isoprene",
    "nitric_acid",
    "nitrogen_dioxide",
    "nitrogen_monoxide",
    "ozone",
    "peroxyacetyl_nitrate",
    "propane",
    "sea_salt_aerosol_0.03-0.5um_mixing_ratio",
    "sea_salt_aerosol_0.5-5um_mixing_ratio",
    "sea_salt_aerosol_5-20um_mixing_ratio",
    "specific_humidity",
    "sulphate_aerosol_mixing_ratio",
    "sulphur_dioxide",
    "temperature",
    "acetone",
    "acetone_product",
    "aldehydes",
    "amine",
    "ammonia",
    "ammonium",
    "dimethyl_sulfide",
    "dinitrogen_pentoxide",
    "ethanol",
    "ethene",
    "formic_acid",
    "hydroperoxy_radical",
    "lead",
    "methacrolein_mvk",
    "methacrylic_acid",
    "methane_chemistry",
    "methane_sulfonic_acid",
    "methanol",
    "methyl_glyoxal",
    "methyl_peroxide",
    "methylperoxy_radical",
    "nitrate",
    "nitrate_radical",
    "olefins",
    "organic_ethers",
    "organic_nitrates",
    "paraffins",
    "pernitric_acid",
    "peroxides",
    "peroxy_acetyl_radical",
    "propene",
    "radon",
    "stratospheric_ozone_tracer",
    "terpenes",
]


def download_cams_eac4(
    start_date: str,
    end_date: str,
    data_root_folder: str | Path,
    *,
    area: Sequence[float] | None = None,
    force: bool = False,
) -> list[Path]:
    """Download daily ECMWF CAMS EAC4 60-level NetCDF pairs (`ml` and `sfc`)."""
    import zipfile
    import cdsapi

    root = Path(data_root_folder)
    root.mkdir(parents=True, exist_ok=True)
    dates = pd.period_range(start=start_date, end=end_date, freq="D")

    pending: list[tuple[pd.Timestamp, str, Path, Path]] = []
    outputs: list[Path] = []
    for raw_date in dates:
        ts = raw_date.to_timestamp()
        date_str = ts.strftime("%Y%m%d")
        ml_fp = root / f"ECMWF-CAMS_ml_{date_str}.nc"
        sfc_fp = root / f"ECMWF-CAMS_sfc_{date_str}.nc"
        if not force and ml_fp.exists() and sfc_fp.exists():
            print(f"[SKIPPED] {date_str}: {ml_fp.name} and {sfc_fp.name} already exist.")
            outputs.extend([ml_fp, sfc_fp])
            continue
        pending.append((ts, date_str, ml_fp, sfc_fp))

    if not pending:
        return outputs

    client = cdsapi.Client()
    for ts, date_str, ml_fp, sfc_fp in pending:
        temp_extract_dir = root / f"tmp_eac4_{date_str}"
        temp_extract_dir.mkdir(parents=True, exist_ok=True)
        output_zip = root / f"ECMWF-CAMS_{date_str}.zip"
        request: dict = {
            "variable": EAC4_VARIABLES,
            "model_level": [str(i) for i in range(1, 61)],
            "date": f"{ts.strftime('%Y-%m-%d')}/{ts.strftime('%Y-%m-%d')}",
            "time": ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
            "data_format": "netcdf_zip",
        }
        if area and len(area) == 4:
            request["area"] = [float(x) for x in area]

        try:
            print(f"Downloading EAC4 {ts.strftime('%Y-%m-%d')} into {output_zip}")
            client.retrieve("cams-global-reanalysis-eac4", request, str(output_zip))
            with zipfile.ZipFile(output_zip, "r") as zf:
                zf.extractall(temp_extract_dir)
            shutil.move(str(temp_extract_dir / "data_mlev.nc"), str(ml_fp))
            shutil.move(str(temp_extract_dir / "data_sfc.nc"), str(sfc_fp))
            outputs.extend([ml_fp, sfc_fp])
        finally:
            if output_zip.exists():
                output_zip.unlink()
            shutil.rmtree(temp_extract_dir, ignore_errors=True)

    return outputs


def build_arg_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="airblender fetch-eac4",
            description="Download daily ECMWF CAMS Global Reanalysis (EAC4) NetCDFs.",
        )
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--data_root_folder", required=True, help="Target output directory")
    parser.add_argument("--area", nargs=4, type=float, default=None, metavar=("N", "W", "S", "E"))
    parser.add_argument("--force", action="store_true", help="Re-download existing daily files")
    parser.add_argument("--mode", "--port", "--host", help="Ignored PyCharm debugger flags")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    download_cams_eac4(
        args.start_date,
        args.end_date,
        args.data_root_folder,
        area=args.area,
        force=args.force,
    )
