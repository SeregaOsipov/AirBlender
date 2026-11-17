import xarray as xr
import re
import numpy as np
from scipy import interpolate
import netCDF4 as nc
from types import SimpleNamespace
import itertools


def hor_interpolate_3d_field_on_wrf_boundary(donor_var_da, wrf_met_ds):
    field_bnd_da = xr.DataArray(  # wrap into xarray structure
        data=np.zeros([donor_var_da.level.size, wrf_met_ds.boundary_points.size]),
        dims=["level", "boundary_points"],
        coords=dict(level=donor_var_da.level, ),
    )

    for level_index, level in enumerate(donor_var_da.level):
        f = interpolate.RectBivariateSpline(donor_var_da.lat, donor_var_da.lon, donor_var_da.isel(level=level_index))
        field_bnd_da.isel(level=level_index).values[:] = f(wrf_met_ds.boundary_lats, wrf_met_ds.boundary_lons, grid=False)

    return field_bnd_da


def _ver_interpolate_edge_clamped(donor_var_da, donor_p_rho_da, wrf_p_rho_da):
    """
    Linearly interpolate 3D or boundary-unraveled donor columns onto WRF pressure
    levels with boundary clamping (nearest edge level outside donor pressure bounds)
    so long-lived species never drop to 0 near surface/top levels.
    """
    donor_vdim = 'level' if 'level' in donor_var_da.dims else donor_var_da.dims[0]
    wrf_vdim = 'bottom_top' if 'bottom_top' in wrf_p_rho_da.dims else wrf_p_rho_da.dims[0]

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

    out_da = xr.DataArray(
        out.reshape(wrf_p_bt.shape),
        dims=wrf_p_bt.dims,
        coords=wrf_p_bt.coords,
    ).transpose(*wrf_p_rho_da.dims).astype(wrf_p_rho_da.dtype).rename('diag')
    return out_da


def ver_interpolate_3d_field_on_wrf_boundary(donor_var_da, donor_p_rho_da, wrf_p_rho_da):
    return _ver_interpolate_edge_clamped(donor_var_da, donor_p_rho_da, wrf_p_rho_da)


def hor_interpolate_3d_field_on_wrf_grid(donor_var_da, wrf_grid_ds):  #, wrf_ny, wrf_nx, wrf_lon, wrf_lat):
    field_hor_da = xr.DataArray(  # wrap into xarray structure
        data=np.zeros([donor_var_da.level.size, wrf_grid_ds.south_north.size, wrf_grid_ds.west_east.size]),
        dims=["level", "south_north", "west_east"],
        coords=dict(level=donor_var_da.level, ),
    )

    for level_index, level in enumerate(donor_var_da.level):
        f = interpolate.RectBivariateSpline(donor_var_da.lat, donor_var_da.lon, donor_var_da.isel(level=level_index))
        field_hor_da.isel(level=level_index).values[:] = f(wrf_grid_ds.XLAT, wrf_grid_ds.XLONG, grid=False).reshape(wrf_grid_ds.south_north.size, wrf_grid_ds.west_east.size)

    return field_hor_da


def ver_interpolate_3d_field_on_wrf_grid(donor_var_da, donor_p_rho_da, wrf_p_rho_da):
    return _ver_interpolate_edge_clamped(donor_var_da, donor_p_rho_da, wrf_p_rho_da)


def update_tendency_boundaries(wrfbdy_f, name, index, dt):
    if index > 0:
        print("\t\t\tUpdating Tendency BC for " + name)
        wrfbdy_f.variables[name + "_BTXS"][index - 1, :] = (wrfbdy_f.variables[name + "_BXS"][index, :] -
                                                            wrfbdy_f.variables[name + "_BXS"][index - 1, :]) / dt
        wrfbdy_f.variables[name + "_BTXE"][index - 1, :] = (wrfbdy_f.variables[name + "_BXE"][index, :] -
                                                            wrfbdy_f.variables[name + "_BXE"][index - 1, :]) / dt
        wrfbdy_f.variables[name + "_BTYS"][index - 1, :] = (wrfbdy_f.variables[name + "_BYS"][index, :] -
                                                            wrfbdy_f.variables[name + "_BYS"][index - 1, :]) / dt
        wrfbdy_f.variables[name + "_BTYE"][index - 1, :] = (wrfbdy_f.variables[name + "_BYE"][index, :] -
                                                            wrfbdy_f.variables[name + "_BYE"][index - 1, :]) / dt


def update_boundaries(increment_on_bnd_da, wrf_met_ds, wrfbdy_f, name, index):
    '''
    wrf_met_ds is wrf_gid_ds, needs to provide grid information
    '''
    bdy_width = wrfbdy_f.dimensions['bdy_width'].size

    west, north, east, south = unravel_1d_boundary(increment_on_bnd_da, wrf_met_ds)
    west = west.transpose('bottom_top', 'south_north')
    east = east.transpose('bottom_top', 'south_north')
    south = south.transpose('bottom_top', 'west_east')
    north = north.transpose('bottom_top', 'west_east')

    west = west.expand_dims({'bdy_width': np.arange(bdy_width)})#, axis=1)  #
    east = east.expand_dims({'bdy_width': np.arange(bdy_width)})
    south = south.expand_dims({'bdy_width': np.arange(bdy_width)})
    north = north.expand_dims({'bdy_width': np.arange(bdy_width)})

    # print("\t\t\tUpdating BC for "+name)
    # make sure this is the dims order in da: bdy_width, bottom_top, south_north
    wrfbdy_f.variables[name + "_BXS"][index, :] = wrfbdy_f.variables[name + "_BXS"][index, :] + west
    wrfbdy_f.variables[name + "_BXE"][index, :] = wrfbdy_f.variables[name + "_BXE"][index, :] + east
    wrfbdy_f.variables[name + "_BYS"][index, :] = wrfbdy_f.variables[name + "_BYS"][index, :] + south
    wrfbdy_f.variables[name + "_BYE"][index, :] = wrfbdy_f.variables[name + "_BYE"][index, :] + north


def get_met_file_by_time(time, met_em_name_template):
    '''
    :param time:
    :param met_em_name_template: 'met_em.d01.{}.nc'
    :return:
    '''
    # return "met_em.d01." + time + ".nc"
    return met_em_name_template.format(time)


def is_child_domain_covered_by_parent_domain(parent_domain, child_domain):
    '''
    This check does not account for the distortions
    :param parent_domain:
    :param child_domain:
    :return:
    '''

    result = (min(child_domain.boundary_lons) > min(parent_domain.lon)) | (
            max(child_domain.boundary_lons) < max(parent_domain.lon)) | (
                     min(child_domain.boundary_lats) > min(parent_domain.lat)) | (
                     max(child_domain.boundary_lats) < max(parent_domain.lat))

    return result


def flatten_and_line_up_boundaries(da):
    '''
    Extract the four boundaries and align in 1D for interpolation
    '''
    north = da.isel(south_north=-1).rename({'west_east': 'boundary_points'})  # WRF_PRES[:, ny-1, :]
    south = da.isel(south_north=0).rename({'west_east': 'boundary_points'})  # WRF_PRES[:, 0, :]
    east = da.isel(west_east=-1).rename({'south_north': 'boundary_points'})  # WRF_PRES[:, :, nx-1]
    west = da.isel(west_east=0).rename({'south_north': 'boundary_points'})  # WRF_PRES[:, :, 0]
    diag_on_bnd_da = xr.concat([west, north, east, south], dim='boundary_points')
    return diag_on_bnd_da


def unravel_1d_boundary(diag_on_bnd_da, wrf_grid_ds):
    ny = wrf_grid_ds.sizes['south_north']
    nx = wrf_grid_ds.sizes['west_east']

    # 2. Slice the 1D array based on the original concat order:
    # Order was: [west (ny), north (nx), east (ny), south (nx)]

    west = diag_on_bnd_da.isel(boundary_points=slice(0, ny)).rename({'boundary_points': 'south_north'})
    north = diag_on_bnd_da.isel(boundary_points=slice(ny, ny + nx)).rename({'boundary_points': 'west_east'})
    east = diag_on_bnd_da.isel(boundary_points=slice(ny + nx, 2 * ny + nx)).rename({'boundary_points': 'south_north'})
    south = diag_on_bnd_da.isel(boundary_points=slice(2 * ny + nx, 2 * ny + 2 * nx)).rename({'boundary_points': 'west_east'})

    return west, north, east, south

