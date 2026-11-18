"""Comprehensive self-contained verification suite for AirBlender."""
from __future__ import annotations

from datetime import datetime, timedelta
import math
from pathlib import Path
import tempfile

from netCDF4 import Dataset
import numpy as np
from pyproj import CRS, Transformer
import xarray as xr
import yaml

from airblender.cams2wps import convert_cams_to_wps
from airblender.palm_dynamic import generate_palm_dynamic_driver
from airblender.physics import (
    MADE_MODES_DGS_M,
    MADE_MODES_SIGMAS,
    compute_made_number_factor,
    get_mapping_for_source,
    get_wrf_made_modpar,
    parse_mapping_rule,
    salsa_remap,
)
from airblender.regrid import is_child_domain_covered_by_parent_domain
from airblender.wrf_icbc import run_wrf_icbc


def _write_char_times(ds: Dataset, timestamps: list[str]) -> None:
    ds.createDimension("DateStrLen", 19)
    t_var = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
    for idx, ts in enumerate(timestamps):
        t_var[idx, :] = np.array(list(ts.ljust(19)[:19]), dtype="S1")


def _make_wrf_grid_and_met(run_dir: Path, met_dir: Path, timestamps: list[str], dom: str = "d01", nx: int = 8, ny: int = 7, nz: int = 6):
    lat2d, lon2d = np.meshgrid(np.linspace(24.0, 27.0, ny), np.linspace(50.0, 53.0, nx), indexing="ij")
    znu = np.linspace(0.98, 0.05, nz, dtype=np.float32)

    ic_fp = run_dir / f"wrfinput_{dom}"
    with Dataset(ic_fp, "w") as d:
        d.createDimension("Time", 1)
        d.createDimension("bottom_top", nz)
        d.createDimension("south_north", ny)
        d.createDimension("west_east", nx)
        _write_char_times(d, [timestamps[0]])
        p_top = d.createVariable("P_TOP", "f4", ("Time",))
        p_top[:] = 5000.0
        znu_v = d.createVariable("ZNU", "f4", ("Time", "bottom_top"))
        znu_v[0, :] = znu
        xlat = d.createVariable("XLAT", "f4", ("Time", "south_north", "west_east"))
        xlat[0, :, :] = lat2d
        xlon = d.createVariable("XLONG", "f4", ("Time", "south_north", "west_east"))
        xlon[0, :, :] = lon2d
        for vname in ("co2", "ch4", "o3", "so2", "co", "no2", "so4aj", "ac0", "soila", "seas", "corn", "ecj", "orgpaj", "nu0", "no3aj", "nh4aj"):
            v = d.createVariable(vname, "f4", ("Time", "bottom_top", "south_north", "west_east"))
            v[0, :, :, :] = 123.0

    bdy_fp = None
    if dom == "d01":
        bdy_fp = run_dir / "wrfbdy_d01"
        with Dataset(bdy_fp, "w") as d:
            d.createDimension("Time", len(timestamps))
            d.createDimension("bdy_width", 5)
            d.createDimension("bottom_top", nz)
            d.createDimension("south_north", ny)
            d.createDimension("west_east", nx)
            _write_char_times(d, timestamps)
            for vname in ("co2", "ch4", "o3", "so2", "co", "no2", "so4aj", "ac0", "soila", "seas", "corn", "ecj", "orgpaj", "nu0", "no3aj", "nh4aj"):
                for sfx, xdim in (("_BXS", "south_north"), ("_BXE", "south_north"), ("_BYS", "west_east"), ("_BYE", "west_east")):
                    vb = d.createVariable(vname + sfx, "f4", ("Time", "bdy_width", "bottom_top", xdim))
                    vb[:, :, :, :] = 99.0
                    vt = d.createVariable(vname + sfx.replace("_B", "_BT"), "f4", ("Time", "bdy_width", "bottom_top", xdim))
                    vt[:, :, :, :] = 99.0

    for ts in timestamps:
        fn = ts.replace(":", "_")
        met_fp = met_dir / f"met_em.{dom}.{fn}.nc"
        with Dataset(met_fp, "w") as d:
            d.createDimension("Time", 1)
            d.createDimension("south_north", ny)
            d.createDimension("west_east", nx)
            _write_char_times(d, [ts])
            psfc = d.createVariable("PSFC", "f4", ("Time", "south_north", "west_east"))
            psfc[0, :, :] = 101325.0
            xlat = d.createVariable("XLAT_M", "f4", ("Time", "south_north", "west_east"))
            xlat[0, :, :] = lat2d
            xlon = d.createVariable("XLONG_M", "f4", ("Time", "south_north", "west_east"))
            xlon[0, :, :] = lon2d
    return ic_fp, bdy_fp


def test_physics_and_regrid_fixes() -> None:
    print("[Test 1] Physics DSL, EMAC SS_cs bugfix, MADE moment factors, SALSA budgets...")
    emac_rules = get_mapping_for_source("emac")
    seas_corn = [r for r in emac_rules if r.mapping_rule_str.startswith("corn -> 2.01*[SS_ks]")]
    assert len(seas_corn) == 1
    parsed = parse_mapping_rule(seas_corn[0].mapping_rule_str)
    assert [p["merra_key"] for p in parsed] == ["SS_ks", "SS_as", "SS_cs"], f"Unexpected parsed terms: {parsed}"

    try:
        parse_mapping_rule("corn -> 2.01[SS_cs];1.e7")
        raise AssertionError("Expected malformed rule without '*' to raise ValueError")
    except ValueError:
        pass

    for scheme in ("made_sorgam", "gocart", "gocart_dust", "mosaic_8bin"):
        assert len(get_mapping_for_source("merra2", scheme=scheme)) >= 5

    k_so4 = compute_made_number_factor(1800.0, MADE_MODES_DGS_M[1], MADE_MODES_SIGMAS[1])
    assert abs(k_so4 - 3.5602e11) / 3.5602e11 < 1e-3
    so4 = np.array([5.0])
    m3 = so4 * (1e-6 * 6.0 / math.pi / 1800.0)
    ac0 = k_so4 * so4
    dgs = get_wrf_made_modpar(MADE_MODES_SIGMAS[1:2], (ac0,), (m3,))
    assert np.allclose(dgs[0], MADE_MODES_DGS_M[1], rtol=1e-12)

    parent = xr.Dataset(coords={"lon": [49.0, 54.0], "lat": [23.0, 28.0]})
    child_ok = xr.Dataset({"boundary_lons": ("bp", [50.0, 52.0]), "boundary_lats": ("bp", [24.0, 26.0])})
    child_bad = xr.Dataset({"boundary_lons": ("bp", [48.0, 52.0]), "boundary_lats": ("bp", [24.0, 26.0])})
    assert is_child_domain_covered_by_parent_domain(parent, child_ok)
    assert not is_child_domain_covered_by_parent_domain(parent, child_bad)

    comp = {k: np.full((2, 2), 0.1) for k in salsa_remap.SOURCE_COMPONENTS}
    comp["soila"] = np.full((2, 2), 50.0)
    comp["seas"] = np.full((2, 2), 10.0)
    comp["antha"] = np.full((2, 2), 5.0)
    nums = {"nu0": np.full((2, 2), 1e9), "ac0": np.full((2, 2), 1e8), "corn": np.full((2, 2), 1e6)}
    res = salsa_remap.remap_modes(comp, nums, np.full((2, 2), 1.15))
    assert res.number.shape == (2, 2, 10)
    assert res.mass.shape == (2, 2, 10, 7)
    assert np.all(res.excluded_mass[..., salsa_remap.SOURCE_COMPONENTS.index("antha")] > 0.0)
    print("  -> Test 1 PASSED.")


def test_unified_cams_merra2_emac_icbc() -> None:
    print("[Test 2] Unified CAMS, MERRA-2, and EMAC -> WRF IC/BC...")
    with tempfile.TemporaryDirectory(prefix="airblender_icbc_") as tmp:
        root = Path(tmp)
        timestamps = ["2026-11-14_00:00:00", "2026-11-14_03:00:00", "2026-11-14_06:00:00"]
        lats = np.linspace(28.5, 22.5, 9)  # Descending (real CAMS order)
        lons = np.linspace(48.5, 54.5, 9)
        levs = np.arange(1, 6, dtype=np.float32)
        pv = np.array([2000.0, 15000.0, 35000.0, 60000.0, 85000.0, 0.0, 0.0, 0.05, 0.25, 0.55, 0.85, 1.0], dtype=np.float32)

        # --- A. CAMS (with mixed 0.1 deg ghg + fallback + multi-domain d02) ---
        cams_dir = root / "cams"
        met_dir = root / "met"
        wrf_cams = root / "wrf_cams"
        cams_dir.mkdir()
        met_dir.mkdir()
        wrf_cams.mkdir()
        _make_wrf_grid_and_met(wrf_cams, met_dir, timestamps, dom="d01")
        _make_wrf_grid_and_met(wrf_cams, met_dir, timestamps, dom="d02", nx=5, ny=5)

        times_np = np.array(["2026-11-14T00:00:00", "2026-11-14T03:00:00", "2026-11-14T06:00:00"], dtype="datetime64[s]")
        with Dataset(cams_dir / "ECMWF-CAMS_sfc_20261114.nc", "w") as d:
            d.createDimension("forecast_period", 3)
            d.createDimension("latitude", len(lats))
            d.createDimension("longitude", len(lons))
            vt = d.createVariable("valid_time", "i8", ("forecast_period",))
            vt.units = "seconds since 1970-01-01 00:00:00"
            vt[:] = (times_np - np.datetime64("1970-01-01T00:00:00")).astype("int64")
            la = d.createVariable("latitude", "f4", ("latitude",))
            la[:] = lats
            lo = d.createVariable("longitude", "f4", ("longitude",))
            lo[:] = lons
            sp = d.createVariable("sp", "f4", ("forecast_period", "latitude", "longitude"))
            sp[:, :, :] = 101325.0

        with Dataset(cams_dir / "ECMWF-CAMS_ml_20261114.nc", "w") as d:
            d.createDimension("forecast_period", 3)
            d.createDimension("model_level", len(levs))
            d.createDimension("latitude", len(lats))
            d.createDimension("longitude", len(lons))
            vt = d.createVariable("valid_time", "i8", ("forecast_period",))
            vt.units = "seconds since 1970-01-01 00:00:00"
            vt[:] = (times_np - np.datetime64("1970-01-01T00:00:00")).astype("int64")
            ml = d.createVariable("model_level", "f4", ("model_level",))
            ml[:] = levs
            la = d.createVariable("latitude", "f4", ("latitude",))
            la[:] = lats
            lo = d.createVariable("longitude", "f4", ("longitude",))
            lo[:] = lons
            for name, base in [("go3", 1.0e-7), ("aermr04", 2.0e-8), ("ch4_c", 1.1e-6)]:
                v = d.createVariable(name, "f4", ("forecast_period", "model_level", "latitude", "longitude"))
                v.GRIB_pv = pv
                for ti in range(3):
                    v[ti, :, :, :] = base * (ti + 1)

        with Dataset(cams_dir / "ECMWF-CAMS_ghg_20261114.nc", "w") as d:
            ghg_lats = np.linspace(28.5, 22.5, 19)
            ghg_lons = np.linspace(48.5, 54.5, 19)
            d.createDimension("forecast_period", 3)
            d.createDimension("model_level", len(levs))
            d.createDimension("latitude", len(ghg_lats))
            d.createDimension("longitude", len(ghg_lons))
            vt = d.createVariable("valid_time", "i8", ("forecast_period",))
            vt.units = "seconds since 1970-01-01 00:00:00"
            vt[:] = (times_np - np.datetime64("1970-01-01T00:00:00")).astype("int64")
            ml = d.createVariable("model_level", "f4", ("model_level",))
            ml[:] = levs
            la = d.createVariable("latitude", "f4", ("latitude",))
            la[:] = ghg_lats
            lo = d.createVariable("longitude", "f4", ("longitude",))
            lo[:] = ghg_lons
            # Omit ch4 intentionally to exercise ml ch4_c fallback, keep co2 at 0.1 deg resolution!
            v = d.createVariable("co2", "f4", ("forecast_period", "model_level", "latitude", "longitude"))
            v.GRIB_pv = pv
            v[:, :, :, :] = 6.5e-4

        run_wrf_icbc(
            source="cams",
            reanalysis_dir=cams_dir,
            wrf_dir=wrf_cams,
            wrf_met_dir=met_dir,
            wrf_input=["wrfinput_d01", "wrfinput_d02"],
            do_ic=True,
            do_bc=True,
            zero_out_first=True,
        )
        with Dataset(wrf_cams / "wrfinput_d01", "r") as d1, Dataset(wrf_cams / "wrfinput_d02", "r") as d2, Dataset(wrf_cams / "wrfbdy_d01", "r") as db:
            assert np.allclose(d1.variables["co2"][0], 0.6582 * 6.5e-4 * 1e6, rtol=1e-4)
            assert np.allclose(d2.variables["co2"][0], 0.6582 * 6.5e-4 * 1e6, rtol=1e-4)
            assert np.allclose(d1.variables["ch4"][0], 1.8058 * 1.1e-6 * 1e6, rtol=1e-4)
            assert np.allclose(db.variables["o3_BXS"][0], 0.604 * 1e-7 * 1e6, rtol=1e-4)
            assert np.allclose(db.variables["o3_BXS"][1], 0.604 * 2e-7 * 1e6, rtol=1e-4)
            assert db.variables["o3_BTXS"][0].mean() > 0.0

        # --- B. MERRA-2 (with DELP fallback & caching) ---
        m_dir = root / "merra2" / "inst3_3d_aer_Nv"
        m_chm = root / "merra2" / "inst3_3d_chm_Nv"
        wrf_m2 = root / "wrf_m2"
        m_dir.mkdir(parents=True)
        m_chm.mkdir(parents=True)
        wrf_m2.mkdir()
        _make_wrf_grid_and_met(wrf_m2, met_dir, timestamps, dom="d01")

        m_lats = np.linspace(22.5, 28.5, 9)
        m_levs = np.arange(1, 6, dtype=np.float32)
        with Dataset(m_dir / "MERRA2_400.inst3_3d_aer_Nv.20261114.nc4", "w") as da, Dataset(
            m_chm / "MERRA2_400.inst3_3d_chm_Nv.20261114.nc4", "w"
        ) as dc:
            for d, is_aer in ((da, True), (dc, False)):
                d.createDimension("time", 3)
                d.createDimension("lev", len(m_levs))
                d.createDimension("lat", len(m_lats))
                d.createDimension("lon", len(lons))
                tv = d.createVariable("time", "i8", ("time",))
                tv.units = "seconds since 1970-01-01 00:00:00"
                tv[:] = (times_np - np.datetime64("1970-01-01T00:00:00")).astype("int64")
                lv = d.createVariable("lev", "f4", ("lev",))
                lv[:] = m_levs
                la = d.createVariable("lat", "f4", ("lat",))
                la[:] = m_lats
                lo = d.createVariable("lon", "f4", ("lon",))
                lo[:] = lons
                if is_aer:
                    delp = d.createVariable("DELP", "f4", ("time", "lev", "lat", "lon"))
                    delp[:, :, :, :] = 20000.0
                    so4 = d.createVariable("SO4", "f4", ("time", "lev", "lat", "lon"))
                    so4[:, :, :, :] = 3.0e-9
                    du1 = d.createVariable("DU001", "f4", ("time", "lev", "lat", "lon"))
                    du1[:, :, :, :] = 1.0e-8
                else:
                    o3 = d.createVariable("O3", "f4", ("time", "lev", "lat", "lon"))
                    o3[:, :, :, :] = 5.0e-8

        run_wrf_icbc(
            source="merra2",
            reanalysis_dir=root / "merra2",
            wrf_dir=wrf_m2,
            wrf_met_dir=met_dir,
            do_ic=True,
            do_bc=True,
            zero_out_first=True,
        )
        with Dataset(wrf_m2 / "wrfinput_d01", "r") as dm:
            assert np.allclose(dm.variables["so4aj"][0], 3.0, rtol=1e-4)
            assert np.allclose(dm.variables["soila"][0], 10.0, rtol=1e-4)
            assert np.allclose(dm.variables["o3"][0], 0.604 * 5.0e-8 * 1e6, rtol=1e-4)

        # --- C. EMAC (0..360 lon shift + pressure rename) ---
        emac_dir = root / "emac"
        wrf_emac = root / "wrf_emac"
        emac_dir.mkdir()
        wrf_emac.mkdir()
        _make_wrf_grid_and_met(wrf_emac, met_dir, timestamps, dom="d01")

        lons_360 = np.array([350.0, 355.0, 0.0, 48.0, 50.0, 52.0, 54.0, 56.0, 180.0])
        with Dataset(emac_dir / "test01_________20261114_0000_WRF_bc_met.nc", "w") as dm, Dataset(
            emac_dir / "test01_________20261114_0000_WRF_bc_chem.nc", "w"
        ) as dc:
            for d, is_met in ((dm, True), (dc, False)):
                d.createDimension("time", 3)
                d.createDimension("lev", 5)
                d.createDimension("lat", len(m_lats))
                d.createDimension("lon", len(lons_360))
                tv = d.createVariable("time", "i8", ("time",))
                tv.units = "seconds since 1970-01-01 00:00:00"
                tv[:] = (times_np - np.datetime64("1970-01-01T00:00:00")).astype("int64")
                lv = d.createVariable("lev", "f4", ("lev",))
                lv[:] = np.arange(1, 6, dtype=np.float32)
                la = d.createVariable("lat", "f4", ("lat",))
                la[:] = m_lats
                lo = d.createVariable("lon", "f4", ("lon",))
                lo[:] = lons_360
                if is_met:
                    pr = d.createVariable("press", "f4", ("time", "lev", "lat", "lon"))
                    pr[:, :, :, :] = np.linspace(10000.0, 100000.0, 5)[None, :, None, None]
                else:
                    ss = d.createVariable("SS_cs", "f4", ("time", "lev", "lat", "lon"))
                    ss[:, :, :, :] = 1.0e-8

        run_wrf_icbc(
            source="emac",
            reanalysis_dir=emac_dir,
            wrf_dir=wrf_emac,
            wrf_met_dir=met_dir,
            do_ic=True,
            do_bc=True,
            zero_out_first=True,
        )
        with Dataset(wrf_emac / "wrfinput_d01", "r") as de:
            assert np.allclose(de.variables["seas"][0], 2.017 * 1e-8 * 1e9, rtol=1e-4)
            assert np.allclose(de.variables["corn"][0], 2.01 * 1e-8 * 1e7, rtol=1e-4)
    print("  -> Test 2 PASSED.")


def test_cams2wps_and_wrf2palm() -> None:
    print("[Test 3] Synthetic CAMS2WPS & WRF2PALM (`meteorology`, `phstatp`, `phstatp-salsa`)...")
    with tempfile.TemporaryDirectory(prefix="airblender_palm_") as tmp:
        root = Path(tmp)
        lats = np.linspace(27.0, 23.0, 5)
        lons = np.linspace(49.0, 54.0, 6)
        pv = np.array([2000.0, 50000.0, 0.0, 0.0, 0.4, 1.0], dtype=np.float32)
        sfc_fp = root / "ECMWF-CAMS_sfc_20261114.nc"
        ml_fp = root / "ECMWF-CAMS_ml_20261114.nc"
        wps_dir = root / "wps"
        t0 = np.datetime64("2026-11-14T00:00:00")
        with Dataset(sfc_fp, "w") as d:
            d.createDimension("forecast_period", 1)
            d.createDimension("latitude", len(lats))
            d.createDimension("longitude", len(lons))
            vt = d.createVariable("valid_time", "i8", ("forecast_period",))
            vt.units = "seconds since 1970-01-01 00:00:00"
            vt[:] = (t0 - np.datetime64("1970-01-01T00:00:00")).astype("int64")
            d.createVariable("latitude", "f4", ("latitude",))[:] = lats
            d.createVariable("longitude", "f4", ("longitude",))[:] = lons
            for name, val in [("sp", 101325.0), ("skt", 300.0), ("lsm", 1.0), ("sst", 301.0), ("t2m", 299.0), ("u10", 3.0), ("v10", -1.5)]:
                d.createVariable(name, "f4", ("forecast_period", "latitude", "longitude"))[:] = val
        with Dataset(ml_fp, "w") as d:
            d.createDimension("forecast_period", 1)
            d.createDimension("model_level", 2)
            d.createDimension("latitude", len(lats))
            d.createDimension("longitude", len(lons))
            d.createVariable("model_level", "f4", ("model_level",))[:] = [1, 2]
            d.createVariable("latitude", "f4", ("latitude",))[:] = lats
            d.createVariable("longitude", "f4", ("longitude",))[:] = lons
            for name, val in [("t", 295.0), ("u", 4.0), ("v", -2.0), ("q", 0.01)]:
                v = d.createVariable(name, "f4", ("forecast_period", "model_level", "latitude", "longitude"))
                v.GRIB_pv = pv
                v[:] = val

        wps_files = convert_cams_to_wps(sfc_fp, ml_fp, wps_dir)
        assert len(wps_files) == 1 and wps_files[0].exists() and wps_files[0].stat().st_size > 100

        # Build synthetic PALM static + WRF d02 + d01 forcing
        static_fp = root / "DohaTest_static"
        with Dataset(static_fp, "w") as d:
            crs_obj = CRS.from_epsg(32639)
            fwd = Transformer.from_crs(4326, crs_obj, always_xy=True)
            ox, oy = fwd.transform(51.45, 25.25)
            d.origin_x = float(ox)
            d.origin_y = float(oy)
            d.origin_z = 5.0
            d.origin_lat = 25.25
            d.origin_lon = 51.45
            d.rotation_angle = 0.0
            d.createDimension("x", 10)
            d.createDimension("y", 10)
            cv = d.createVariable("crs", "i4", ())
            cv.crs_wkt = crs_obj.to_wkt()

        wrf_out_dir = root / "wrf_out"
        wrf_out_dir.mkdir()
        lcc_str = "+proj=lcc +lat_1=27.962 +lat_2=27.962 +lat_0=27.6706 +lon_0=42.022 +a=6370000 +b=6370000 +units=m"
        lcc_fwd = Transformer.from_crs(4326, CRS.from_proj4(lcc_str), always_xy=True)
        cx, cy = lcc_fwd.transform(51.46, 25.26)
        wrf_fp = wrf_out_dir / "wrfout_d02_2026-11-14_00_00_00"
        with Dataset(wrf_fp, "w") as d:
            d.TRUELAT1 = 27.962
            d.TRUELAT2 = 27.962
            d.MOAD_CEN_LAT = 27.6706
            d.STAND_LON = 42.022
            d.CEN_LAT = 25.26
            d.CEN_LON = 51.46
            d.DX = 2000.0
            d.DY = 2000.0
            d.CHEM_OPT = np.int32(100)
            d.createDimension("Time", 2)
            d.createDimension("bottom_top", 4)
            d.createDimension("bottom_top_stag", 5)
            d.createDimension("south_north", 12)
            d.createDimension("south_north_stag", 13)
            d.createDimension("west_east", 12)
            d.createDimension("west_east_stag", 13)
            _write_char_times(d, ["2026-11-14_00:00:00", "2026-11-14_01:00:00"])
            d.createVariable("T", "f4", ("Time", "bottom_top", "south_north", "west_east"))[:] = 0.0
            d.createVariable("P", "f4", ("Time", "bottom_top", "south_north", "west_east"))[:] = 0.0
            pb = np.linspace(100000.0, 85000.0, 4, dtype=np.float32)[None, :, None, None]
            d.createVariable("PB", "f4", ("Time", "bottom_top", "south_north", "west_east"))[:] = pb
            d.createVariable("QVAPOR", "f4", ("Time", "bottom_top", "south_north", "west_east"))[:] = 0.012
            ph = np.linspace(100.0, 15000.0, 5, dtype=np.float32)[None, :, None, None]
            d.createVariable("PH", "f4", ("Time", "bottom_top_stag", "south_north", "west_east"))[:] = ph
            d.createVariable("PHB", "f4", ("Time", "bottom_top_stag", "south_north", "west_east"))[:] = 0.0
            d.createVariable("HGT", "f4", ("Time", "south_north", "west_east"))[:] = 8.0
            d.createVariable("U", "f4", ("Time", "bottom_top", "south_north", "west_east_stag"))[:] = 3.0
            d.createVariable("V", "f4", ("Time", "bottom_top", "south_north_stag", "west_east"))[:] = -1.0
            d.createVariable("W", "f4", ("Time", "bottom_top_stag", "south_north", "west_east"))[:] = 0.0
            for vname, val in [("no", 0.002), ("no2", 0.01), ("o3", 0.04), ("PM10", 80.0), ("nu0", 1e8), ("ac0", 1e8), ("corn", 1e6), ("soila", 40.0), ("seas", 10.0), ("so4aj", 3.0)]:
                d.createVariable(vname, "f4", ("Time", "bottom_top", "south_north", "west_east"))[:] = val
            for cname in salsa_remap.SOURCE_COMPONENTS:
                if cname not in ("soila", "seas", "so4aj"):
                    d.createVariable(cname, "f4", ("Time", "bottom_top", "south_north", "west_east"))[:] = 0.1

        cfg_fp = root / "palm.yaml"
        cfg = {
            "model": "palm",
            "run": {
                "name": "test_palm",
                "case_name": "TestPalm",
                "work_dir": str(root / "palm_work"),
                "static_driver": str(static_fp),
                "wrf_source_dir": str(wrf_out_dir),
            },
            "cycle": {"start_date": "2026-11-14", "start_hour": "00", "forecast_hours": 1},
            "domain": {"nx": 9, "ny": 9, "nz": 6, "dx": 100.0, "dy": 100.0, "dz": 25.0, "dz_stretch_level": 10000.0, "dz_stretch_factor": 1.0, "dz_max": 25.0},
            "chemistry": {"palm_mode": "phstatp-salsa", "wrf_chem_opt": 100},
        }
        cfg_fp.write_text(yaml.safe_dump(cfg))

        r_met = generate_palm_dynamic_driver(cfg_fp, "2026-11-14", output_path=root / "dyn_met.nc", palm_mode_override="meteorology")
        r_ph = generate_palm_dynamic_driver(cfg_fp, "2026-11-14", output_path=root / "dyn_ph.nc", palm_mode_override="phstatp")
        r_sa = generate_palm_dynamic_driver(cfg_fp, "2026-11-14", output_path=root / "dyn_sa.nc", palm_mode_override="phstatp-salsa")
        assert r_met["field_count"] == 31
        assert r_ph["field_count"] == 55
        assert r_sa["field_count"] == 115
        assert r_sa["wrf_domain"] == "d02"
    print("  -> Test 3 PASSED.")


def test_fetch_subpackage() -> None:
    print("[Test 4] Optional airblender.fetch chunking, CDM NetCDF contract, and MERRA-2 URLs...")
    import argparse
    import xarray as xr
    from airblender.fetch.cams_forecast import (
        apply_config_and_defaults,
        generate_tasks,
        save_dataset_daily_and_base,
    )
    from airblender.fetch.merra2 import download_merra2, merra2_stream_version
    from airblender.sources.cams import CamsDonorAdapter, extract_cams_pv
    from airblender.sources.base import DateDonorContext

    with tempfile.TemporaryDirectory(prefix="airblender_fetch_") as td:
        root = Path(td)
        ns = argparse.Namespace(
            mode=None,
            start_date="2026-11-14",
            end_date="2026-11-14",
            data_root_folder=str(root / "cams_dl"),
            config=None,
            format="grib",
            max_fields=10000,
            sfc_lt_chunk=None,
            met_lt_chunk=None,
            chem_lt_chunk=None,
            ghg_lt_chunk=24,
            ghg=True,
            ghg_only=False,
            level_chunks=1,
            max_workers=5,
            max_retries=3,
            list_tasks=True,
            task_index=None,
            sfc_only=False,
            force=False,
            type="forecast",
            time="00:00",
            area=None,
            leadtimes=None,
            no_split_days=False,
        )
        apply_config_and_defaults(ns)
        tasks = generate_tasks(ns)
        assert len(tasks) == 48, f"Expected 48 tasks (1 sfc + 4 met + 41 chem + 2 ghg), got {len(tasks)}"
        assert all(t["num_fields"] <= 10000 for t in tasks), "Found task exceeding 10,000-field MARS limit"

        # Verify producer -> consumer daily NetCDF contract with CamsDonorAdapter
        times = np.array(["2026-11-14T00:00:00", "2026-11-15T00:00:00"], dtype="datetime64[ns]")
        pv = np.array([2000.0, 50000.0, 0.0, 0.0, 0.4, 1.0], dtype=np.float32)
        ds_syn = xr.Dataset(
            data_vars={
                "go3": (
                    ("forecast_period", "model_level", "latitude", "longitude"),
                    np.full((2, 2, 3, 3), 5.0e-8, dtype=np.float32),
                    {"GRIB_pv": pv, "units": "kg kg-1"},
                )
            },
            coords={
                "forecast_period": [0, 24],
                "valid_time": ("forecast_period", times),
                "model_level": [1, 2],
                "latitude": [26.0, 25.0, 24.0],
                "longitude": [50.0, 51.0, 52.0],
            },
        )
        cams_out = root / "cams_daily"
        save_dataset_daily_and_base(ds_syn, "ml", "20261114", cams_out)
        ml_14 = cams_out / "ECMWF-CAMS_ml_20261114.nc"
        ml_15 = cams_out / "ECMWF-CAMS_ml_20261115.nc"
        assert ml_14.exists() and ml_15.exists()

        import pandas as pd
        adapter = CamsDonorAdapter(cams_out, "ECMWF-CAMS_{stream}_{date_time}.nc")
        for day_dt in (pd.Timestamp("2026-11-14T00:00:00"), pd.Timestamp("2026-11-15T00:00:00")):
            ctx = DateDonorContext(date=day_dt)
            ds_loaded = adapter.open_stream_for_date("ml", ctx)
            assert ds_loaded is not None and "go3" in ds_loaded
            assert float(ds_loaded.lat.values[0]) < float(ds_loaded.lat.values[-1])
            assert len(extract_cams_pv(ds_loaded)) == 6
            adapter.close_context(ctx)

        # Verify MERRA-2 stream versions and dry-run paths
        assert merra2_stream_version("2021-07-15") == "401"
        assert merra2_stream_version("2023-06-01") == "400"
        assert merra2_stream_version("2005-01-01") == "300"
        assert merra2_stream_version("1995-01-01") == "200"
        assert merra2_stream_version("1985-01-01") == "100"

        m2_paths = download_merra2("2026-11-14", out_dir=root / "m2", dry_run=True)
        assert len(m2_paths) == 7
        assert all(p.name.startswith("MERRA2_400.") and p.name.endswith(".20261114.nc4") for p in m2_paths)
    print("  -> Test 4 PASSED.")


def main() -> None:
    print("=" * 68)
    print(" Running AirBlender Verification Suite")
    print("=" * 68)
    test_physics_and_regrid_fixes()
    test_unified_cams_merra2_emac_icbc()
    test_cams2wps_and_wrf2palm()
    test_fetch_subpackage()
    print("=" * 68)
    print(" ALL AIRBLENDER CHECKS PASSED (100% OK)")
    print("=" * 68)


if __name__ == "__main__":
    main()
