"""Base interfaces and shared per-date stream/pressure caching for donor models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from airblender.regrid.wrf_regrid import (
    hor_interpolate_3d_field_on_wrf_boundary,
    hor_interpolate_3d_field_on_wrf_grid,
)


@dataclass
class DateDonorContext:
    """Holds cached opened donor stream datasets and interpolated pressure fields for one cycle time."""

    date: Any
    sel_method: str | None = None
    streams: dict[str, xr.Dataset | None] = field(default_factory=dict)
    p_rho_interior: dict[str, xr.DataArray] = field(default_factory=dict)
    p_rho_boundary: dict[str, xr.DataArray] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class DonorAdapter(ABC):
    """Abstract base adapter providing uniform cached access across CAMS, MERRA-2, and EMAC."""

    source_name: str = "base"

    def __init__(self, root_dir: str | Path, file_name_template: str):
        self.root_dir = Path(root_dir)
        self.file_name_template = file_name_template

    @abstractmethod
    def open_stream_dataset(self, path_pattern: Path | str) -> xr.Dataset:
        """Open a raw donor file and standardize coordinates `(time, level, lat, lon)`."""

    @abstractmethod
    def build_stream_path(self, stream: str, date: Any) -> Path | str:
        """Construct the file path or glob pattern for `stream` and `date`."""

    def open_stream_for_date(self, stream: str, context: DateDonorContext) -> xr.Dataset | None:
        """Open and cache the time-selected dataset for `stream` inside `context`."""
        if stream in context.streams:
            return context.streams[stream]

        path_spec = self.build_stream_path(stream, context.date)
        ds = self._open_matching_path(path_spec)
        if ds is None:
            print(f"\t\t   [WARNING] Donor stream '{stream}' file not found at {path_spec}.")
            context.streams[stream] = None
            return None

        sel_kwargs = {"method": context.sel_method} if context.sel_method is not None else {}
        ds_t = ds.sel(time=context.date, **sel_kwargs)
        ds_t.attrs["_airblender_parent_ds"] = ds
        context.streams[stream] = ds_t
        return ds_t

    def _open_matching_path(self, path_spec: Path | str) -> xr.Dataset | None:
        path_str = str(path_spec)
        if "*" in path_str or "?" in path_str:
            import glob

            matches = sorted(glob.glob(path_str))
            if not matches:
                return None
            return self.open_stream_dataset(path_str)
        p = Path(path_spec)
        if not p.exists():
            return None
        return self.open_stream_dataset(p)

    def prepare_date_context(
        self, date: Any, sel_method: str | None, active_streams: list[str]
    ) -> DateDonorContext:
        """Initialize context for `date` and preload required coordinate/pressure streams."""
        ctx = DateDonorContext(date=date, sel_method=sel_method)
        for s in active_streams:
            self.open_stream_for_date(s, ctx)
        return ctx

    def resolve_variable(
        self,
        stream: str,
        donor_key: str,
        wrf_key: str,
        context: DateDonorContext,
    ) -> tuple[str | None, xr.Dataset | None, str | None]:
        """Resolve variable dataset and variable name, allowing subclass fallbacks."""
        ds = self.open_stream_for_date(stream, context)
        if ds is not None and donor_key in ds:
            return stream, ds, donor_key
        return None, None, None

    @abstractmethod
    def derive_pressure_for_stream(
        self,
        stream: str,
        donor_stream_ds: xr.Dataset,
        wrf_target_ds: xr.Dataset,
        context: DateDonorContext,
        is_boundary: bool,
    ) -> xr.DataArray:
        """Return 3D `p_rho` horizontally aligned with `wrf_target_ds` (interior or boundary)."""

    def get_stream_pressure(
        self,
        stream: str,
        donor_stream_ds: xr.Dataset,
        wrf_target_ds: xr.Dataset,
        context: DateDonorContext,
        is_boundary: bool = False,
    ) -> xr.DataArray:
        """Cache and return horizontally interpolated 3D pressure for `stream` on `wrf_target_ds`."""
        cache = context.p_rho_boundary if is_boundary else context.p_rho_interior
        if stream in cache:
            return cache[stream]
        p_rho_da = self.derive_pressure_for_stream(
            stream, donor_stream_ds, wrf_target_ds, context, is_boundary=is_boundary
        )
        cache[stream] = p_rho_da
        return p_rho_da

    def interpolate_pressure_field(
        self, p_rho: xr.DataArray, wrf_target_ds: xr.Dataset, is_boundary: bool
    ) -> xr.DataArray:
        """Utility to interpolate an explicit donor `p_rho` field onto WRF interior or boundary."""
        if is_boundary:
            return hor_interpolate_3d_field_on_wrf_boundary(p_rho, wrf_target_ds)
        return hor_interpolate_3d_field_on_wrf_grid(p_rho, wrf_target_ds)

    def get_time_signature(self, context: DateDonorContext) -> tuple[tuple[str, str | None], ...]:
        """Return immutable donor timestamp signature for reusing identical consecutive BC frames."""
        sig: list[tuple[str, str | None]] = []
        for s, ds in sorted(context.streams.items()):
            if ds is None:
                continue
            if "time" in ds.coords:
                t_val = str(ds.time.values)
            elif "valid_time" in ds.coords:
                t_val = str(ds.valid_time.values)
            else:
                t_val = None
            sig.append((s, t_val))
        return tuple(sig)

    def close_context(self, context: DateDonorContext) -> None:
        """Close underlying file handles referenced by cached stream slices."""
        for ds in context.streams.values():
            if ds is not None:
                parent = ds.attrs.pop("_airblender_parent_ds", None)
                ds.close()
                if parent is not None:
                    parent.close()
        context.streams.clear()
        context.p_rho_interior.clear()
        context.p_rho_boundary.clear()


def ensure_ascending_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    """Normalize latitude ascending [-90, 90] and shift [0, 360) longitudes to [-180, 180) ascending."""
    if "lon" in ds.coords and ds.lon.size > 0 and float(ds.lon.max()) > 180.0:
        ds = ds.assign_coords(lon=(((ds.lon + 180.0) % 360.0) - 180.0)).sortby("lon")
    elif "lon" in ds.dims and ds.lon.size > 1 and float(ds.lon.values[0]) > float(ds.lon.values[-1]):
        ds = ds.isel(lon=slice(None, None, -1))
    if "lat" in ds.dims and ds.lat.size > 1 and float(ds.lat.values[0]) > float(ds.lat.values[-1]):
        ds = ds.isel(lat=slice(None, None, -1))
    return ds
