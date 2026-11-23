"""Horizontal spline interpolation, edge-clamped vertical pressure interpolation, and WRF boundary writers."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from scipy import interpolate
import xarray as xr

from airblender.icon_icbc import _BilinearWeights

_HORIZ_WEIGHT_CACHE: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, _BilinearWeights]] = []
_MAX_HORIZ_CACHE = 16
# Cache entries: (p_don, p_wrf, order_mode, sort_idx, idx_lo, idx_hi, alpha)
_VERT_PLAN_CACHE: list[tuple[np.ndarray, np.ndarray, str, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]] = []
_MAX_VERT_CACHE = 4


def _get_cached_bilinear_weights(
    lats: np.ndarray,
    lons: np.ndarray,
    target_lats_1d: np.ndarray,
    target_lons_1d: np.ndarray,
) -> _BilinearWeights:
    """Return cached `_BilinearWeights` for matching 1D donor coordinates and flattened target points."""
    for i, (c_lats, c_lons, c_tlat, c_tlon, weights) in enumerate(_HORIZ_WEIGHT_CACHE):
        if (
            c_lats.shape == lats.shape
            and c_lons.shape == lons.shape
            and c_tlat.shape == target_lats_1d.shape
            and c_tlon.shape == target_lons_1d.shape
            and c_lats[0] == lats[0]
            and c_lats[-1] == lats[-1]
            and c_lons[0] == lons[0]
            and c_lons[-1] == lons[-1]
            and c_tlat[0] == target_lats_1d[0]
            and c_tlat[-1] == target_lats_1d[-1]
            and c_tlon[0] == target_lons_1d[0]
            and c_tlon[-1] == target_lons_1d[-1]
            and np.array_equal(c_lats, lats)
            and np.array_equal(c_lons, lons)
            and np.array_equal(c_tlat, target_lats_1d)
            and np.array_equal(c_tlon, target_lons_1d)
        ):
            _HORIZ_WEIGHT_CACHE.append(_HORIZ_WEIGHT_CACHE.pop(i))
            return weights

    weights = _BilinearWeights.build(lats, lons, target_lats_1d, target_lons_1d)
    _HORIZ_WEIGHT_CACHE.append(
        (lats.copy(), lons.copy(), target_lats_1d.copy(), target_lons_1d.copy(), weights)
    )
    if len(_HORIZ_WEIGHT_CACHE) > _MAX_HORIZ_CACHE:
        _HORIZ_WEIGHT_CACHE.pop(0)
    return weights


def hor_interpolate_array_on_points(
    vals: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    out_shape: tuple[int, ...],
    method: str | None = None,
) -> np.ndarray:
    """Interpolate a 2D `(lat, lon)` or 3D `(level, lat, lon)` array onto target coordinates."""
    interp_method = (method or os.environ.get("AIRBLENDER_WRF_HORIZ_INTERP", "bilinear")).strip().lower()
    if interp_method not in ("bilinear", "spline"):
        raise ValueError(f"Unsupported horizontal interpolation method: {interp_method!r}")

    lats_arr = np.asarray(lats, dtype=np.float64)
    lons_arr = np.asarray(lons, dtype=np.float64)
    vals_arr = np.asarray(vals, dtype=np.float64)
    is_2d = vals_arr.ndim == 2
    if is_2d:
        vals_arr = vals_arr[None, :, :]

    if lats_arr.size > 1 and lats_arr[0] > lats_arr[-1]:
        vals_arr = vals_arr[:, ::-1, :]
        lats_arr = lats_arr[::-1]
    if lons_arr.size > 1 and lons_arr[0] > lons_arr[-1]:
        vals_arr = vals_arr[:, :, ::-1]
        lons_arr = lons_arr[::-1]

    t_lat = np.asarray(target_lats, dtype=np.float64)
    t_lon = np.asarray(target_lons, dtype=np.float64)

    if interp_method == "bilinear" and lats_arr.size >= 2 and lons_arr.size >= 2:
        weights = _get_cached_bilinear_weights(lats_arr, lons_arr, t_lat.ravel(), t_lon.ravel())
        out = weights.interp(vals_arr).reshape((vals_arr.shape[0],) + out_shape)
        return out[0] if is_2d else out

    n_lev = vals_arr.shape[0]
    data = np.zeros((n_lev,) + out_shape, dtype=np.float64)
    for k in range(n_lev):
        spline = interpolate.RectBivariateSpline(lats_arr, lons_arr, vals_arr[k])
        res = spline(t_lat, t_lon, grid=False)
        data[k] = res.reshape(out_shape) if res.shape != out_shape else res
    return data[0] if is_2d else data


def _interp_2d_spline_levels(
    donor_var_da: xr.DataArray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    out_shape: tuple[int, ...],
    out_dims: tuple[str, ...],
    method: str | None = None,
) -> xr.DataArray:
    """Shared level-by-level horizontal interpolator (cached bilinear by default, or `spline`)."""
    vals = np.asarray(donor_var_da.transpose("level", "lat", "lon").values, dtype=np.float64)
    data = hor_interpolate_array_on_points(
        vals,
        donor_var_da.lat.values,
        donor_var_da.lon.values,
        target_lats,
        target_lons,
        out_shape,
        method=method,
    )
    return xr.DataArray(data=data, dims=list(out_dims), coords={"level": donor_var_da.level})


def hor_interpolate_3d_field_on_wrf_boundary(donor_var_da: xr.DataArray, wrf_met_ds: xr.Dataset) -> xr.DataArray:
    """Horizontally interpolate a 3D donor field onto flattened WRF perimeter points."""
    return _interp_2d_spline_levels(
        donor_var_da,
        wrf_met_ds.boundary_lats,
        wrf_met_ds.boundary_lons,
        out_shape=(wrf_met_ds.boundary_points.size,),
        out_dims=("level", "boundary_points"),
    )


def hor_interpolate_3d_field_on_wrf_grid(donor_var_da: xr.DataArray, wrf_grid_ds: xr.Dataset) -> xr.DataArray:
    """Horizontally interpolate a 3D donor field onto interior WRF mass grid points."""
    return _interp_2d_spline_levels(
        donor_var_da,
        wrf_grid_ds.XLAT,
        wrf_grid_ds.XLONG,
        out_shape=(wrf_grid_ds.south_north.size, wrf_grid_ds.west_east.size),
        out_dims=("level", "south_north", "west_east"),
    )


def _build_or_get_vert_plan(
    p_don: np.ndarray,
    p_wrf: np.ndarray,
) -> tuple[str, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Compute or retrieve vectorized linear vertical-interpolation indices and weights."""
    k_src, n_col = p_don.shape
    k_dst = p_wrf.shape[0]
    p_don_00 = float(p_don[0, 0]) if p_don.size else 0.0
    p_don_11 = float(p_don[-1, -1]) if p_don.size else 0.0
    p_wrf_00 = float(p_wrf[0, 0]) if p_wrf.size else 0.0
    p_wrf_11 = float(p_wrf[-1, -1]) if p_wrf.size else 0.0

    for i, (c_pdon, c_pwrf, order_mode, sort_idx, idx_lo, idx_hi, alpha) in enumerate(_VERT_PLAN_CACHE):
        if (
            c_pdon.shape == p_don.shape
            and c_pwrf.shape == p_wrf.shape
            and (float(c_pdon[0, 0]) if c_pdon.size else 0.0) == p_don_00
            and (float(c_pdon[-1, -1]) if c_pdon.size else 0.0) == p_don_11
            and (float(c_pwrf[0, 0]) if c_pwrf.size else 0.0) == p_wrf_00
            and (float(c_pwrf[-1, -1]) if c_pwrf.size else 0.0) == p_wrf_11
            and np.array_equal(c_pdon, p_don)
            and np.array_equal(c_pwrf, p_wrf)
        ):
            _VERT_PLAN_CACHE.append(_VERT_PLAN_CACHE.pop(i))
            return order_mode, sort_idx, idx_lo, idx_hi, alpha

    sort_idx: np.ndarray | None = None
    if k_src <= 1 or np.all(np.diff(p_don, axis=0) >= 0.0):
        order_mode = "asc"
        p_don_asc = p_don
    elif p_don[0, 0] > p_don[-1, 0] and np.all(np.diff(p_don[::-1], axis=0) >= 0.0):
        order_mode = "desc"
        p_don_asc = p_don[::-1]
    else:
        order_mode = "sort"
        sort_idx = np.argsort(p_don, axis=0)
        p_don_asc = np.take_along_axis(p_don, sort_idx, axis=0)

    idx_lo = np.empty((k_dst, n_col), dtype=np.int32)
    idx_hi = np.empty((k_dst, n_col), dtype=np.int32)
    alpha = np.empty((k_dst, n_col), dtype=np.float64)

    chunk_cols = 2048
    for c0 in range(0, n_col, chunk_cols):
        c1 = min(n_col, c0 + chunk_cols)
        hi = np.sum(p_don_asc[None, :, c0:c1] <= p_wrf[:, None, c0:c1], axis=1, dtype=np.int32)
        c_hi = np.clip(hi, 0, k_src - 1)
        c_lo = np.clip(hi - 1, 0, k_src - 1)
        p0 = np.take_along_axis(p_don_asc[:, c0:c1], c_lo, axis=0)
        p1 = np.take_along_axis(p_don_asc[:, c0:c1], c_hi, axis=0)
        denom = p1 - p0
        w = np.where(denom > 0.0, (p_wrf[:, c0:c1] - p0) / np.where(denom > 0.0, denom, 1.0), 0.0)
        idx_lo[:, c0:c1] = c_lo
        idx_hi[:, c0:c1] = c_hi
        alpha[:, c0:c1] = np.clip(w, 0.0, 1.0)

    _VERT_PLAN_CACHE.append((p_don.copy(), p_wrf.copy(), order_mode, sort_idx, idx_lo, idx_hi, alpha))
    if len(_VERT_PLAN_CACHE) > _MAX_VERT_CACHE:
        _VERT_PLAN_CACHE.pop(0)
    return order_mode, sort_idx, idx_lo, idx_hi, alpha


def _ver_interpolate_edge_clamped(
    donor_var_da: xr.DataArray, donor_p_rho_da: xr.DataArray, wrf_p_rho_da: xr.DataArray
) -> xr.DataArray:
    """
    Vectorized linear interpolation of 3D or boundary donor columns onto WRF pressure
    levels with boundary clamping (matching `np.interp` edge clamp outside donor bounds).
    """
    donor_vdim = "level" if "level" in donor_var_da.dims else donor_var_da.dims[0]
    wrf_vdim = "bottom_top" if "bottom_top" in wrf_p_rho_da.dims else wrf_p_rho_da.dims[0]

    other_dims = tuple(d for d in wrf_p_rho_da.dims if d != wrf_vdim)
    wrf_p_bt = wrf_p_rho_da.transpose(wrf_vdim, *other_dims)
    donor_p_bt = donor_p_rho_da.transpose(donor_vdim, *other_dims)
    donor_v_bt = donor_var_da.transpose(donor_vdim, *other_dims)

    p_wrf = np.asarray(wrf_p_bt.values, dtype=np.float64).reshape(wrf_p_bt.shape[0], -1)
    p_don = np.asarray(donor_p_bt.values, dtype=np.float64).reshape(donor_p_bt.shape[0], -1)
    v_don = np.asarray(donor_v_bt.values, dtype=np.float64).reshape(donor_v_bt.shape[0], -1)

    order_mode, sort_idx, idx_lo, idx_hi, alpha = _build_or_get_vert_plan(p_don, p_wrf)
    if order_mode == "asc":
        v_asc = v_don
    elif order_mode == "desc":
        v_asc = v_don[::-1]
    else:
        assert sort_idx is not None
        v_asc = np.take_along_axis(v_don, sort_idx, axis=0)

    v0 = np.take_along_axis(v_asc, idx_lo, axis=0)
    v1 = np.take_along_axis(v_asc, idx_hi, axis=0)
    out = v0 + alpha * (v1 - v0)

    return (
        xr.DataArray(out.reshape(wrf_p_bt.shape), dims=wrf_p_bt.dims, coords=wrf_p_bt.coords)
        .transpose(*wrf_p_rho_da.dims)
        .astype(wrf_p_rho_da.dtype)
        .rename("diag")
    )


def ver_interpolate_3d_field_on_wrf_boundary(
    donor_var_da: xr.DataArray, donor_p_rho_da: xr.DataArray, wrf_p_rho_da: xr.DataArray
) -> xr.DataArray:
    """Vertically interpolate a boundary donor profile onto WRF eta pressure levels."""
    return _ver_interpolate_edge_clamped(donor_var_da, donor_p_rho_da, wrf_p_rho_da)


def ver_interpolate_3d_field_on_wrf_grid(
    donor_var_da: xr.DataArray, donor_p_rho_da: xr.DataArray, wrf_p_rho_da: xr.DataArray
) -> xr.DataArray:
    """Vertically interpolate a 3D donor field onto WRF interior eta pressure levels."""
    return _ver_interpolate_edge_clamped(donor_var_da, donor_p_rho_da, wrf_p_rho_da)


def update_tendency_boundaries(wrfbdy_f, name: str, index: int, dt: float) -> None:
    """Compute finite-difference tendency `(B[t] - B[t-1]) / dt` for index-1 in `wrfbdy`."""
    if index > 0:
        for sfx_b, sfx_bt in (("_BXS", "_BTXS"), ("_BXE", "_BTXE"), ("_BYS", "_BTYS"), ("_BYE", "_BTYE")):
            bvar = wrfbdy_f.variables[name + sfx_b]
            wrfbdy_f.variables[name + sfx_bt][index - 1, :] = (bvar[index, :] - bvar[index - 1, :]) / dt


def update_boundaries(increment_on_bnd_da: xr.DataArray, wrf_met_ds: xr.Dataset, wrfbdy_f, name: str, index: int) -> None:
    """Increment WEST/EAST/SOUTH/NORTH boundary arrays across `bdy_width` in `wrfbdy`."""
    bdy_width = wrfbdy_f.dimensions["bdy_width"].size
    west, north, east, south = unravel_1d_boundary(increment_on_bnd_da, wrf_met_ds)
    west = west.transpose("bottom_top", "south_north").expand_dims({"bdy_width": np.arange(bdy_width)})
    east = east.transpose("bottom_top", "south_north").expand_dims({"bdy_width": np.arange(bdy_width)})
    south = south.transpose("bottom_top", "west_east").expand_dims({"bdy_width": np.arange(bdy_width)})
    north = north.transpose("bottom_top", "west_east").expand_dims({"bdy_width": np.arange(bdy_width)})

    wrfbdy_f.variables[name + "_BXS"][index, :] = wrfbdy_f.variables[name + "_BXS"][index, :] + west
    wrfbdy_f.variables[name + "_BXE"][index, :] = wrfbdy_f.variables[name + "_BXE"][index, :] + east
    wrfbdy_f.variables[name + "_BYS"][index, :] = wrfbdy_f.variables[name + "_BYS"][index, :] + south
    wrfbdy_f.variables[name + "_BYE"][index, :] = wrfbdy_f.variables[name + "_BYE"][index, :] + north


def get_met_file_by_time(time_str: str, met_em_name_template: str) -> str:
    """Format `met_em` filename for a given formatted timestamp."""
    return met_em_name_template.format(time_str)


def is_child_domain_covered_by_parent_domain(parent_domain, child_domain) -> bool:
    """Return True if child boundary lon/lat extent lies strictly inside parent domain lon/lat bounds."""
    return bool(
        (np.min(child_domain.boundary_lons) > np.min(parent_domain.lon))
        and (np.max(child_domain.boundary_lons) < np.max(parent_domain.lon))
        and (np.min(child_domain.boundary_lats) > np.min(parent_domain.lat))
        and (np.max(child_domain.boundary_lats) < np.max(parent_domain.lat))
    )


def flatten_and_line_up_boundaries(da: xr.DataArray) -> xr.DataArray:
    """Concatenate WRF West/North/East/South perimeter slices along `boundary_points`."""
    north = da.isel(south_north=-1).rename({"west_east": "boundary_points"})
    south = da.isel(south_north=0).rename({"west_east": "boundary_points"})
    east = da.isel(west_east=-1).rename({"south_north": "boundary_points"})
    west = da.isel(west_east=0).rename({"south_north": "boundary_points"})
    return xr.concat([west, north, east, south], dim="boundary_points")


def unravel_1d_boundary(diag_on_bnd_da: xr.DataArray, wrf_grid_ds: xr.Dataset):
    """Split a `boundary_points` array back into `(west, north, east, south)` faces."""
    ny = wrf_grid_ds.sizes["south_north"]
    nx = wrf_grid_ds.sizes["west_east"]
    west = diag_on_bnd_da.isel(boundary_points=slice(0, ny)).rename({"boundary_points": "south_north"})
    north = diag_on_bnd_da.isel(boundary_points=slice(ny, ny + nx)).rename({"boundary_points": "west_east"})
    east = diag_on_bnd_da.isel(boundary_points=slice(ny + nx, 2 * ny + nx)).rename({"boundary_points": "south_north"})
    south = diag_on_bnd_da.isel(boundary_points=slice(2 * ny + nx, 2 * ny + 2 * nx)).rename(
        {"boundary_points": "west_east"}
    )
    return west, north, east, south
