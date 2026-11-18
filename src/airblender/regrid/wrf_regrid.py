"""Horizontal spline interpolation, edge-clamped vertical pressure interpolation, and WRF boundary writers."""
from __future__ import annotations

import numpy as np
from scipy import interpolate
import xarray as xr


def _interp_2d_spline_levels(
    donor_var_da: xr.DataArray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    out_shape: tuple[int, ...],
    out_dims: tuple[str, ...],
) -> xr.DataArray:
    """Shared level-by-level `RectBivariateSpline` interpolator for WRF interiors and boundaries."""
    lats = np.asarray(donor_var_da.lat.values, dtype=np.float64)
    lons = np.asarray(donor_var_da.lon.values, dtype=np.float64)
    if lats.size > 1 and lats[0] > lats[-1]:
        donor_var_da = donor_var_da.isel(lat=slice(None, None, -1))
        lats = lats[::-1]
    if lons.size > 1 and lons[0] > lons[-1]:
        donor_var_da = donor_var_da.isel(lon=slice(None, None, -1))
        lons = lons[::-1]

    n_lev = donor_var_da.level.size
    data = np.zeros((n_lev,) + out_shape, dtype=np.float64)
    t_lat = np.asarray(target_lats)
    t_lon = np.asarray(target_lons)
    vals = np.asarray(donor_var_da.transpose("level", "lat", "lon").values, dtype=np.float64)
    for k in range(n_lev):
        spline = interpolate.RectBivariateSpline(lats, lons, vals[k])
        res = spline(t_lat, t_lon, grid=False)
        data[k] = res.reshape(out_shape) if res.shape != out_shape else res

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


def _ver_interpolate_edge_clamped(
    donor_var_da: xr.DataArray, donor_p_rho_da: xr.DataArray, wrf_p_rho_da: xr.DataArray
) -> xr.DataArray:
    """
    Linearly interpolate 3D or boundary-unraveled donor columns onto WRF pressure
    levels with boundary clamping (`np.interp` edge clamp outside donor pressure bounds).
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

    out = np.empty_like(p_wrf)
    for col in range(p_wrf.shape[1]):
        xp = p_don[:, col]
        fp = v_don[:, col]
        if xp[0] > xp[-1]:
            xp = xp[::-1]
            fp = fp[::-1]
        elif not np.all(np.diff(xp) >= 0):
            order = np.argsort(xp)
            xp = xp[order]
            fp = fp[order]
        out[:, col] = np.interp(p_wrf[:, col], xp, fp)

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
