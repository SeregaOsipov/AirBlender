"""ECMWF CAMS donor adapter supporting mixed 0.4° ml/sfc and 0.1° ghg streams."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from airblender.regrid.wrf_regrid import hor_interpolate_array_on_points
from airblender.sources.base import DateDonorContext, DonorAdapter, ensure_ascending_lat_lon


def extract_cams_pv(ds: xr.Dataset) -> np.ndarray:
    """Return 1D hybrid sigma-pressure `pv` array (`a_stag` followed by `b_stag`)."""
    if "pv" in ds:
        return np.asarray(ds["pv"].values, dtype=np.float64)
    for var_name in ds.data_vars:
        if hasattr(ds[var_name], "GRIB_pv"):
            return np.asarray(ds[var_name].GRIB_pv, dtype=np.float64)
    raise AttributeError("No variable with GRIB_pv or 'pv' found in CAMS dataset.")


def derive_cams_pressure_profile(reanalysis_ml_ds: xr.Dataset, reanalysis_sfc_ds: xr.Dataset) -> xr.DataArray:
    """Derive 3D hybrid midpoint pressure `p_rho` when `ml` and `sfc` share the horizontal grid."""
    pv = extract_cams_pv(reanalysis_ml_ds)
    num_levels = reanalysis_ml_ds.sizes["level"]
    a = pv[: num_levels + 1]
    b = pv[num_levels + 1 :]
    reanalysis_ml_ds["a"] = xr.DataArray(a, dims=("level_stag",))
    reanalysis_ml_ds["b"] = xr.DataArray(b, dims=("level_stag",))
    reanalysis_ml_ds["p_stag"] = reanalysis_ml_ds.a + reanalysis_ml_ds.b * reanalysis_sfc_ds.sp
    reanalysis_ml_ds["p_rho"] = (
        reanalysis_ml_ds.p_stag.rolling(level_stag=2)
        .mean()
        .isel(level_stag=slice(1, None))
        .rename({"level_stag": "level"})
    )
    return reanalysis_ml_ds["p_rho"]


class CamsDonorAdapter(DonorAdapter):
    """Donor adapter for ECMWF CAMS daily CDM NetCDFs (`ECMWF-CAMS_{stream}_{YYYYMMDD}.nc`)."""

    source_name = "cams"

    def build_stream_path(self, stream: str, date: Any) -> Path:
        return self.root_dir / self.file_name_template.format(date_time=date.strftime("%Y%m%d"), stream=stream)

    def open_stream_dataset(self, path_pattern: Path | str) -> xr.Dataset:
        print(f"Opening file: {path_pattern}")
        ds = xr.open_dataset(path_pattern)

        if "forecast_reference_time" in ds.dims and ds.sizes["forecast_reference_time"] == 1:
            ds = ds.squeeze(dim="forecast_reference_time", drop=True)
        if "valid_time" in ds.data_vars:
            ds = ds.set_coords("valid_time")
        if "forecast_period" in ds.dims:
            if "valid_time" in ds:
                times = pd.to_datetime(ds.valid_time.values.flatten() if ds.valid_time.ndim > 1 else ds.valid_time.values)
                ds = ds.rename({"forecast_period": "time"}).assign_coords(time=times)
                if "valid_time" in ds:
                    ds = ds.drop_vars("valid_time")
            else:
                ds = ds.rename({"forecast_period": "time"})
        elif "valid_time" in ds:
            if "valid_time" in ds.dims:
                ds = ds.rename({"valid_time": "time"})
            elif "time" not in ds.dims:
                times = pd.to_datetime(ds.valid_time.values.flatten() if ds.valid_time.ndim > 1 else ds.valid_time.values)
                spatial_coords = ("level", "model_level", "lat", "latitude", "lon", "longitude")
                if ds.dims and list(ds.dims.keys())[0] not in spatial_coords:
                    first_dim = list(ds.dims.keys())[0]
                    ds = ds.rename({first_dim: "time"}).assign_coords(time=times)
                    if "valid_time" in ds:
                        ds = ds.drop_vars("valid_time")
                else:
                    ds = ds.rename({"valid_time": "time"})

        rename_dict = {}
        if "latitude" in ds.coords or "latitude" in ds.dims:
            rename_dict["latitude"] = "lat"
        if "longitude" in ds.coords or "longitude" in ds.dims:
            rename_dict["longitude"] = "lon"
        if "model_level" in ds.dims:
            rename_dict["model_level"] = "level"
        if rename_dict:
            ds = ds.rename(rename_dict)

        return ensure_ascending_lat_lon(ds)

    def prepare_date_context(
        self, date: Any, sel_method: str | None, active_streams: list[str]
    ) -> DateDonorContext:
        ctx = DateDonorContext(date=date, sel_method=sel_method)
        sfc_ds = self.open_stream_for_date("sfc", ctx)
        if sfc_ds is None:
            raise FileNotFoundError(
                f"Required CAMS 'sfc' stream file missing for date {date} in {self.root_dir}"
            )
        for s in active_streams:
            if s != "sfc":
                self.open_stream_for_date(s, ctx)
        return ctx

    def resolve_variable(
        self,
        stream: str,
        donor_key: str,
        wrf_key: str,
        context: DateDonorContext,
    ) -> tuple[str | None, xr.Dataset | None, str | None]:
        ds = self.open_stream_for_date(stream, context)
        if ds is not None and donor_key in ds:
            return stream, ds, donor_key

        if stream == "ghg":
            ml_ds = self.open_stream_for_date("ml", context)
            if ml_ds is not None:
                candidates = [donor_key, "ch4_c"] if wrf_key == "ch4" else [donor_key]
                for cand in candidates:
                    if cand in ml_ds:
                        print(f"\t\t   [FALLBACK] Using '{cand}' from 'ml' stream for '{wrf_key}'.")
                        return "ml", ml_ds, cand

        return None, None, None

    @staticmethod
    def _streams_share_horizontal_grid(stream_ds: xr.Dataset, sfc_ds: xr.Dataset) -> bool:
        if "lat" not in stream_ds.dims or "lon" not in stream_ds.dims:
            return False
        if stream_ds.sizes["lat"] != sfc_ds.sizes["lat"] or stream_ds.sizes["lon"] != sfc_ds.sizes["lon"]:
            return False
        return bool(
            np.allclose(stream_ds.lat.values, sfc_ds.lat.values)
            and np.allclose(stream_ds.lon.values, sfc_ds.lon.values)
        )

    def _derive_mixed_grid_p_rho(
        self,
        donor_stream_ds: xr.Dataset,
        donor_sfc_ds: xr.Dataset,
        wrf_target_ds: xr.Dataset,
        is_boundary: bool,
    ) -> xr.DataArray:
        """Derive 3D hybrid midpoint pressure directly on target WRF grid using interpolated `sp`."""
        pv = extract_cams_pv(donor_stream_ds)
        num_levels = donor_stream_ds.sizes["level"]
        a = pv[: num_levels + 1]
        b = pv[num_levels + 1 :]
        a_mid = 0.5 * (a[:-1] + a[1:])
        b_mid = 0.5 * (b[:-1] + b[1:])

        sp_vals = donor_sfc_ds["sp"].transpose("lat", "lon").values
        if is_boundary:
            sp_target = hor_interpolate_array_on_points(
                sp_vals,
                donor_sfc_ds.lat.values,
                donor_sfc_ds.lon.values,
                wrf_target_ds.boundary_lats.values,
                wrf_target_ds.boundary_lons.values,
                out_shape=(wrf_target_ds.boundary_points.size,),
            )
            p_rho = a_mid[:, None] + b_mid[:, None] * sp_target[None, :]
            return xr.DataArray(
                p_rho, dims=["level", "boundary_points"], coords={"level": donor_stream_ds.level}
            )

        ny = wrf_target_ds.south_north.size
        nx = wrf_target_ds.west_east.size
        sp_target = hor_interpolate_array_on_points(
            sp_vals,
            donor_sfc_ds.lat.values,
            donor_sfc_ds.lon.values,
            wrf_target_ds.XLAT.values,
            wrf_target_ds.XLONG.values,
            out_shape=(ny, nx),
        )
        p_rho = a_mid[:, None, None] + b_mid[:, None, None] * sp_target[None, :, :]
        return xr.DataArray(
            p_rho, dims=["level", "south_north", "west_east"], coords={"level": donor_stream_ds.level}
        )

    def derive_pressure_for_stream(
        self,
        stream: str,
        donor_stream_ds: xr.Dataset,
        wrf_target_ds: xr.Dataset,
        context: DateDonorContext,
        is_boundary: bool,
    ) -> xr.DataArray:
        sfc_ds = context.streams.get("sfc")
        if sfc_ds is None:
            raise FileNotFoundError(f"Required CAMS 'sfc' stream missing for date {context.date}")
        if stream == "ml" and self._streams_share_horizontal_grid(donor_stream_ds, sfc_ds):
            if "p_rho" not in donor_stream_ds:
                derive_cams_pressure_profile(donor_stream_ds, sfc_ds)
            return self.interpolate_pressure_field(donor_stream_ds.p_rho, wrf_target_ds, is_boundary=is_boundary)
        return self._derive_mixed_grid_p_rho(donor_stream_ds, sfc_ds, wrf_target_ds, is_boundary=is_boundary)
