"""EMAC (ECHAM5/MESSy) donor adapter with [0..360] -> [-180..180] longitude shift & caching."""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import xarray as xr

from airblender.sources.base import DateDonorContext, DonorAdapter, ensure_ascending_lat_lon


class EmacDonorAdapter(DonorAdapter):
    """Donor adapter for EMAC (`WRF_bc_met` / `WRF_bc_chem` or MESSy channel files)."""

    source_name = "emac"

    def build_stream_path(self, stream: str, date: Any) -> str:
        """Resolve exact `YYYYMMDD_0000` or wildcard `YYYYMMDD_*` for split ECHAM5 restarts."""
        cand_exact = str(self.root_dir / self.file_name_template.format(date_time=date.strftime("%Y%m%d_0000"), stream=stream))
        if Path(cand_exact).exists():
            return cand_exact
        cand_glob = str(self.root_dir / self.file_name_template.format(date_time=date.strftime("%Y%m%d_*"), stream=stream))
        if glob.glob(cand_glob):
            return cand_glob
        return str(self.root_dir / self.file_name_template.format(date_time=date.strftime("%Y%m%d"), stream=stream))

    def open_stream_dataset(self, path_pattern: Path | str) -> xr.Dataset:
        print(f"Opening file: {path_pattern}")
        p_str = str(path_pattern)
        ds = xr.open_mfdataset(p_str) if ("*" in p_str or "?" in p_str) else xr.open_dataset(p_str)

        rename_dict = {}
        if "lev" in ds.dims:
            rename_dict["lev"] = "level"
        elif "mlev" in ds.dims:
            rename_dict["mlev"] = "level"
        if "press" in ds.data_vars and "p_rho" not in ds.data_vars:
            rename_dict["press"] = "p_rho"
        if rename_dict:
            ds = ds.rename(rename_dict)

        return ensure_ascending_lat_lon(ds)

    def prepare_date_context(
        self, date: Any, sel_method: str | None, active_streams: list[str]
    ) -> DateDonorContext:
        ctx = DateDonorContext(date=date, sel_method=sel_method)
        # Prefer `WRF_bc_met` for EMAC pressure if available; fallback to `ECHAM5` or embedded pressure.
        if self.open_stream_for_date("WRF_bc_met", ctx) is None:
            self.open_stream_for_date("ECHAM5", ctx)
        for s in active_streams:
            if s not in ("WRF_bc_met", "ECHAM5"):
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
        p_da = None
        if "p_rho" in donor_stream_ds:
            p_da = donor_stream_ds["p_rho"]
        else:
            for met_stream in ("WRF_bc_met", "ECHAM5"):
                met_ds = context.streams.get(met_stream)
                if met_ds is not None and "p_rho" in met_ds:
                    p_da = met_ds["p_rho"]
                    break
        if p_da is None:
            raise FileNotFoundError(
                f"Cannot find EMAC pressure field (`p_rho` / `press`) for stream '{stream}' on {context.date}"
            )
        return self.interpolate_pressure_field(p_da, wrf_target_ds, is_boundary=is_boundary)
