"""NASA MERRA-2 donor adapter with stream caching, version auto-selection, and DELP/PL pressure."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from airblender.sources.base import DateDonorContext, DonorAdapter, ensure_ascending_lat_lon


def generate_merra2_file_name(template: str, date: Any, stream: str) -> str:
    """Resolve MERRA-2 filename with collection version (`100..401`) from `date`."""
    year = date.year
    month = date.month
    if year == 2021 and 6 <= month <= 9:
        vvv = 401
    elif year >= 2011:
        vvv = 400
    elif year >= 2001:
        vvv = 300
    elif year >= 1992:
        vvv = 200
    else:
        vvv = 100
    return template.format(date_time=date.strftime("%Y%m%d"), stream=stream, VVV=vvv)


def derive_merra2_pressure_profile(merra_ds: xr.Dataset) -> xr.DataArray:
    """Integrate 3D layer thickness `DELP` downward from 1 Pa TOA to midpoint pressure `p_rho`."""
    vdim = "level" if "level" in merra_ds.dims else "lev"
    delp = merra_ds["DELP"].sortby(vdim)
    if not np.all(merra_ds["DELP"][vdim].to_numpy() == delp[vdim].to_numpy()):
        raise ValueError("MERRA-2 vertical coordinate order is inverted; expected TOA-first levels.")

    p_stag_no_toa = 1.0 + delp.cumsum(dim=vdim)
    first_coord = delp[vdim].values[0]
    toa = p_stag_no_toa.sel({vdim: first_coord}, drop=False).copy(deep=True)
    toa[:] = 1.0
    toa[vdim] = 0
    p_stag = xr.concat([toa, p_stag_no_toa], dim=vdim)
    p_rho = p_stag.rolling({vdim: 2}).mean().dropna(vdim).rename("p_rho")
    merra_ds["p_stag"] = p_stag
    merra_ds["p_rho"] = p_rho
    return merra_ds["p_rho"]


class Merra2DonorAdapter(DonorAdapter):
    """Donor adapter for MERRA-2 3D NetCDF-4 collections (`inst3_3d_{aer,chm,asm}_Nv`)."""

    source_name = "merra2"

    def build_stream_path(self, stream: str, date: Any) -> Path:
        return self.root_dir / generate_merra2_file_name(self.file_name_template, date, stream=stream)

    def open_stream_dataset(self, path_pattern: Path | str) -> xr.Dataset:
        print(f"Opening file: {path_pattern}")
        ds = xr.open_dataset(path_pattern)
        if "lev" in ds.dims:
            ds = ds.rename({"lev": "level"})
        return ensure_ascending_lat_lon(ds)

    def prepare_date_context(
        self, date: Any, sel_method: str | None, active_streams: list[str]
    ) -> DateDonorContext:
        ctx = DateDonorContext(date=date, sel_method=sel_method)
        # Prefer inst3_3d_asm_Nv PL pressure when present; fallback to DELP on aer_Nv stream.
        self.open_stream_for_date("inst3_3d_asm_Nv", ctx)
        for s in active_streams:
            if s != "inst3_3d_asm_Nv":
                self.open_stream_for_date(s, ctx)
        return ctx

    def derive_pressure_for_stream(
        self,
        stream: str,
        donor_stream_ds: xr.Dataset,
        wrf_target_ds: xr.Dataset,
        context: DateDonorContext,
        is_boundary: bool,
    ) -> xr.DataArray:
        if "p_rho" in donor_stream_ds:
            p_da = donor_stream_ds["p_rho"]
        elif "PL" in donor_stream_ds:
            p_da = donor_stream_ds["PL"]
        else:
            asm_ds = context.streams.get("inst3_3d_asm_Nv")
            if asm_ds is not None and "PL" in asm_ds:
                p_da = asm_ds["PL"]
            elif "DELP" in donor_stream_ds:
                p_da = derive_merra2_pressure_profile(donor_stream_ds)
            else:
                aer_ds = self.open_stream_for_date("inst3_3d_aer_Nv", context)
                if aer_ds is not None and "DELP" in aer_ds:
                    p_da = derive_merra2_pressure_profile(aer_ds)
                else:
                    raise FileNotFoundError(
                        f"Cannot determine MERRA-2 3D pressure for stream '{stream}' on {context.date} "
                        "(neither inst3_3d_asm_Nv/PL nor DELP available)."
                    )
        return self.interpolate_pressure_field(p_da, wrf_target_ds, is_boundary=is_boundary)
