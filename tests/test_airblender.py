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
        for vname in ("co2", "ch4", "o3", "so2", "co", "no2", "ete", "macr", "so4aj", "ac0", "soila", "seas", "corn", "ecj", "orgpaj", "nu0", "no3aj", "nh4aj"):
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
            for vname in ("co2", "ch4", "o3", "so2", "co", "no2", "ete", "macr", "so4aj", "ac0", "soila", "seas", "corn", "ecj", "orgpaj", "nu0", "no3aj", "nh4aj"):
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
            for name, base in [("go3", 1.0e-7), ("aermr04", 2.0e-8), ("ch4_c", 1.1e-6), ("c2h4", 3.0e-8), ("ispd", 4.0e-8)]:
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
            assert np.allclose(d1.variables["ete"][0], 1.032 * 3.0e-8 * 1e6, rtol=1e-4)
            assert np.allclose(d1.variables["macr"][0], 0.413 * 4.0e-8 * 1e6, rtol=1e-4)
            assert "ol2" not in d1.variables
            assert np.allclose(db.variables["o3_BXS"][0], 0.604 * 1e-7 * 1e6, rtol=1e-4)
            assert np.allclose(db.variables["o3_BXS"][1], 0.604 * 2e-7 * 1e6, rtol=1e-4)
            assert np.allclose(db.variables["ete_BXS"][0], 1.032 * 3.0e-8 * 1e6, rtol=1e-4)
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
        assert len(tasks) == 36, f"Expected 36 tasks (1 sfc + 5 met + 14 chema + 14 chemb + 2 ghg), got {len(tasks)}"
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

        m2_zittis = download_merra2("2026-11-14", "2026-11-15", out_dir=root / "m2z", preset="zittis_aot", dry_run=True)
        # 4 daily collections x 2 days + 1 monthly collection deduplicated to 1 file = 9
        assert len(m2_zittis) == 9
        assert any(p.name == "MERRA2_400.tavgM_2d_aer_Nx.202611.nc4" for p in m2_zittis)
    print("  -> Test 4 PASSED.")


def test_cams_stream_daily() -> None:
    print("[Test 5] airblender.fetch CAMS daily-horizon streaming and incremental flushing...")
    import argparse
    import time as _time
    import pandas as pd
    import xarray as xr
    from airblender.fetch.cams_forecast import (
        apply_config_and_defaults,
        check_and_flush_horizons,
        generate_tasks,
    )

    with tempfile.TemporaryDirectory(prefix="airblender_stream_") as td:
        stream_dir = Path(td) / "cams_stream"
        ns = argparse.Namespace(
            mode=None,
            start_date="2026-11-14",
            end_date="2026-11-14",
            data_root_folder=str(stream_dir),
            config=None,
            format="grib",
            max_fields=10000,
            sfc_lt_chunk=None,
            met_lt_chunk=None,
            chem_lt_chunk=None,
            ghg_lt_chunk=None,
            ghg=True,
            ghg_only=False,
            level_chunks=None,
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
            stream_daily=True,
            segment_hours=24,
        )
        apply_config_and_defaults(ns)
        tasks = generate_tasks(ns)
        assert len(tasks) == 41, f"Expected 41 tasks across 5 daily buckets, got {len(tasks)}"
        date_spec = tasks[0]["date_spec"]
        assert [h["horizon_h"] for h in date_spec["horizons"]] == [24, 48, 72, 96, 120]
        # Bucket 1 (tasks[:9]) must strictly stay within 0..24h for 3D met/chema/chemb/ghg (and 0..120h sfc)
        assert all(max(int(lt) for lt in t["lt_chunk"]) <= 24 for t in tasks[1:9])
        assert max(int(lt) for lt in tasks[0]["lt_chunk"]) == 120  # single sfc_all request covers 0..120h

        pv = np.array([2000.0, 50000.0, 0.0, 0.0, 0.4, 1.0], dtype=np.float32)
        base_dt = pd.Timestamp("2026-11-14T00:00:00")

        def _populate_task_chunks(task_slice: list[dict]) -> None:
            for t in task_slice:
                nc_fp = Path(t["output_nc"])
                marker_fp = Path(t["marker"])
                nc_fp.parent.mkdir(parents=True, exist_ok=True)
                marker_fp.parent.mkdir(parents=True, exist_ok=True)
                lts = [int(x) for x in t["lt_chunk"]]
                vtimes = np.array([base_dt + pd.Timedelta(hours=h) for h in lts], dtype="datetime64[ns]")
                name = t["name"]
                if ":cams_sfc" in name:
                    ds = xr.Dataset(
                        data_vars={
                            "t2m": (
                                ("forecast_period", "latitude", "longitude"),
                                np.full((len(lts), 2, 2), 300.0, dtype=np.float32),
                            )
                        },
                        coords={
                            "forecast_period": lts,
                            "valid_time": ("forecast_period", vtimes),
                            "latitude": [26.0, 25.0],
                            "longitude": [50.0, 51.0],
                        },
                    )
                else:
                    if ":cams_met" in name:
                        vname = "u"
                    elif ":cams_chemb" in name:
                        vname = "c2h4"
                    elif ":cams_chema" in name or ":cams_chem_" in name:
                        vname = "go3"
                    else:
                        vname = "co2"
                    ds = xr.Dataset(
                        data_vars={
                            vname: (
                                ("forecast_period", "model_level", "latitude", "longitude"),
                                np.full((len(lts), 2, 2, 2), 1.0e-6, dtype=np.float32),
                                {"GRIB_pv": pv},
                            )
                        },
                        coords={
                            "forecast_period": lts,
                            "valid_time": ("forecast_period", vtimes),
                            "model_level": [1, 2],
                            "latitude": [26.0, 25.0],
                            "longitude": [50.0, 51.0],
                        },
                    )
                ds.to_netcdf(nc_fp, encoding={"valid_time": {"units": "seconds since 1970-01-01", "dtype": "int64"}})
                marker_fp.write_text("DONE\n", encoding="utf-8")

        # Populate ONLY Bucket 1 (0..24h = 9 tasks)
        _populate_task_chunks(tasks[:9])
        check_and_flush_horizons(date_spec, ns)
        assert (stream_dir / ".cams_ready_20261114_h024.done").is_file()
        assert not (stream_dir / ".cams_ready_20261114_h048.done").exists()

        for stream in ("sfc", "ml", "ghg"):
            with xr.open_dataset(stream_dir / f"ECMWF-CAMS_{stream}_20261114.nc") as ds_d0:
                assert ds_d0.sizes["forecast_period"] == 8, f"{stream} day0 steps={ds_d0.sizes['forecast_period']}"
                if stream == "ml":
                    assert "go3" in ds_d0 and "c2h4" in ds_d0, f"Missing merged chema/chemb vars: {list(ds_d0.data_vars)}"
        for stream in ("ml", "ghg"):
            with xr.open_dataset(stream_dir / f"ECMWF-CAMS_{stream}_20261115.nc") as ds_d1:
                assert ds_d1.sizes["forecast_period"] == 1, f"{stream} day1 steps={ds_d1.sizes['forecast_period']}"

        # Verify completed Day 0 ml file is NOT rewritten when Bucket 2 flushes
        ml_day0 = stream_dir / "ECMWF-CAMS_ml_20261114.nc"
        mtime_day0 = ml_day0.stat().st_mtime_ns
        _time.sleep(0.02)

        # Populate Bucket 2 (27..48h = 8 tasks)
        _populate_task_chunks(tasks[9:17])
        check_and_flush_horizons(date_spec, ns)
        assert (stream_dir / ".cams_ready_20261114_h048.done").is_file()
        assert ml_day0.stat().st_mtime_ns == mtime_day0, "Completed Day 0 file should not be rewritten on Bucket 2 flush"

        with xr.open_dataset(stream_dir / "ECMWF-CAMS_ml_20261115.nc") as ds_d1:
            assert ds_d1.sizes["forecast_period"] == 8
        with xr.open_dataset(stream_dir / "ECMWF-CAMS_ml_20261116.nc") as ds_d2:
            assert ds_d2.sizes["forecast_period"] == 1
    print("  -> Test 5 PASSED.")


def test_dynamics_resolver_and_fallback() -> None:
    print("[Test 6] DynamicsResolver auto-fallback, pinned error, env override, and provenance...")
    import json
    import os
    from unittest.mock import patch
    from airblender.fetch.dynamics import resolve_and_fetch_dynamics

    with tempfile.TemporaryDirectory(prefix="airblender_dyn_") as td:
        root = Path(td)
        cams_dir = root / "cams"
        ifs_dir = root / "ifs"
        prov_dir = root / "prov"
        cams_dir.mkdir()

        # 6a. MARS blocked (RESTRICTED_ACCESS_TO_VALID_DATA) -> falls back to S3 0.1°
        with (
            patch("airblender.fetch.dynamics.probe_ifs_mars_access", return_value=(False, "MARS_RESTRICTED_ACCESS_TO_VALID_DATA")),
            patch("airblender.fetch.dynamics.probe_opendata_0p1", return_value=(True, "ok")),
            patch("airblender.fetch.dynamics.download_opendata_0p1_sfc_daily", return_value=[]) as mock_od_dl,
        ):
            res = resolve_and_fetch_dynamics(
                "2026-11-22",
                cams_dir=cams_dir,
                out_dir=ifs_dir,
                dynamics_cfg={"mode": "auto", "primary": "ifs_mars_9km", "fallback_chain": ["ifs_0p1_sfc_cams_ml", "cams_0p4_ml"]},
                leadtimes_hours=[0, 3],
                provenance_dir=prov_dir,
            )
            assert res.provider == "ifs_0p1_sfc_cams_ml"
            assert res.sfc_res_deg == 0.1 and res.ml_res_deg == 0.4
            assert "ifs_mars_9km" in res.fallback_reasons
            mock_od_dl.assert_called_once()
            prov = json.loads((prov_dir / "dynamics_provenance.json").read_text(encoding="utf-8"))
            assert prov["provider"] == "ifs_0p1_sfc_cams_ml"

        # 6b. Both MARS and S3 fail -> clean fallback to cams_0p4_ml
        with (
            patch("airblender.fetch.dynamics.probe_ifs_mars_access", return_value=(False, "401 Unauthorized")),
            patch("airblender.fetch.dynamics.probe_opendata_0p1", return_value=(False, "TimeoutError")),
        ):
            res2 = resolve_and_fetch_dynamics(
                "2026-11-22",
                cams_dir=cams_dir,
                out_dir=ifs_dir,
                dynamics_cfg={"mode": "auto", "primary": "ifs_mars_9km"},
                leadtimes_hours=[0, 3],
            )
            assert res2.provider == "cams_0p4_ml"
            assert res2.met_sfc_dir == cams_dir and res2.met_sfc_prefix == "ECMWF-CAMS_sfc"

        # 6c. Pinned mode raises when primary unavailable
        with patch("airblender.fetch.dynamics.probe_ifs_mars_access", return_value=(False, "403 Expired")):
            try:
                resolve_and_fetch_dynamics(
                    "2026-11-22",
                    cams_dir=cams_dir,
                    out_dir=ifs_dir,
                    dynamics_cfg={"mode": "pinned", "primary": "ifs_mars_9km"},
                    leadtimes_hours=[0, 3],
                )
                raise AssertionError("Expected pinned mode to raise RuntimeError")
            except RuntimeError:
                pass

        # 6d. Archive date succeeds on MARS 9km
        with (
            patch("airblender.fetch.dynamics.probe_ifs_mars_access", return_value=(True, "ok")),
            patch("airblender.fetch.dynamics.download_ifs_forecast_mars", return_value={"sfc": [], "ml": []}) as mock_mars_dl,
        ):
            res3 = resolve_and_fetch_dynamics(
                "2026-11-19",
                cams_dir=cams_dir,
                out_dir=ifs_dir,
                dynamics_cfg={"mode": "auto", "primary": "ifs_mars_9km"},
                leadtimes_hours=[0, 3],
            )
            assert res3.provider == "ifs_mars_9km"
            assert res3.met_sfc_prefix == "ECMWF-IFS_sfc" and res3.met_ml_prefix == "ECMWF-IFS_ml"
            assert res3.ml_res_deg == 0.08
            mock_mars_dl.assert_called_once()

        # 6e. DYNAMICS_PROVIDER env override skips probes
        saved = os.environ.get("DYNAMICS_PROVIDER")
        try:
            os.environ["DYNAMICS_PROVIDER"] = "cams_0p4_ml"
            res4 = resolve_and_fetch_dynamics(
                "2026-11-22",
                cams_dir=cams_dir,
                dynamics_cfg={"mode": "auto", "primary": "ifs_mars_9km"},
            )
            assert res4.provider == "cams_0p4_ml"
        finally:
            if saved is None:
                os.environ.pop("DYNAMICS_PROVIDER", None)
            else:
                os.environ["DYNAMICS_PROVIDER"] = saved

        # 6f. Bounded MARS probe timeout + OpenData multi-mirror 503 failover
        import io
        import sys
        import time as _t
        import types
        import urllib.error
        from unittest.mock import MagicMock
        from airblender.fetch.ecmwf_ifs import probe_ifs_mars_access
        from airblender.fetch.opendata import S3_OPENDATA_BASE_URL, probe_opendata_0p1

        fake_ecmwfapi = types.ModuleType("ecmwfapi")
        class _SlowMars:
            def __init__(self, *_args, **_kwargs):
                pass
            def execute(self, _req, _target):
                _t.sleep(0.5)
        fake_ecmwfapi.ECMWFService = _SlowMars
        old_ecmwfapi = sys.modules.get("ecmwfapi")
        sys.modules["ecmwfapi"] = fake_ecmwfapi
        try:
            with patch.dict(os.environ, {"ECMWF_API_KEY": "k", "ECMWF_API_EMAIL": "e@example.com"}):
                t0 = _t.monotonic()
                ok_to, reason_to = probe_ifs_mars_access("2026-11-19", max_step_h=0, timeout_sec=0.1)
                elapsed = _t.monotonic() - t0
                assert not ok_to, f"Expected timeout failure, got {(ok_to, reason_to)}"
                assert "timed out" in reason_to.lower(), reason_to
                assert elapsed < 0.4, f"Timeout took too long: {elapsed:.2f}s"
        finally:
            if old_ecmwfapi is None:
                sys.modules.pop("ecmwfapi", None)
            else:
                sys.modules["ecmwfapi"] = old_ecmwfapi

        calls_urls: list[str] = []
        def _fake_urlopen(req, timeout=10.0):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            calls_urls.append(url)
            if len(calls_urls) == 1:
                raise urllib.error.HTTPError(url, 503, "Service Unavailable", hdrs=None, fp=io.BytesIO(b""))
            resp = MagicMock()
            resp.read.return_value = b'{"param": "2t", "_offset": 0, "_length": 100}\n'
            resp.__enter__.return_value = resp
            resp.__exit__.return_value = False
            return resp

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            ok_mir, reason_mir = probe_opendata_0p1("2026-11-22", max_step_h=0, s3_base_url=S3_OPENDATA_BASE_URL)
            assert ok_mir, f"Expected mirror failover success, got: {reason_mir}"
            assert len(calls_urls) == 2
            assert calls_urls[0].split("/20261122")[0] != calls_urls[1].split("/20261122")[0]
    print("  -> Test 6 PASSED.")


def test_mixed_grid_wps_and_cams2icon() -> None:
    print("[Test 7] Multi-grid convert_cams_to_wps (0.1° SFC + 0.4° ML) & airblender.cli cams2icon...")
    from datetime import timezone
    from airblender.cli import main as airblender_main
    from airblender.fetch.opendata import _write_cdm_sfc_daily
    from airblender.icon_icbc import prepare_icon_edgar_emissions, prepare_icon_ic, prepare_icon_latbc

    with tempfile.TemporaryDirectory(prefix="airblender_mix_") as td:
        root = Path(td)
        cams_dir = root / "cams"
        ifs_dir = root / "ifs"
        wps_dir = root / "wps"
        icon_run = root / "icon_run"
        cams_dir.mkdir()
        ifs_dir.mkdir()

        # Build 0.1° SFC/Soil dataset in ifs_dir
        lats_01 = np.linspace(27.0, 24.0, 31)
        lons_01 = np.linspace(49.0, 53.0, 41)
        lat2d_01, lon2d_01 = np.meshgrid(lats_01, lons_01, indexing="ij")
        lsm_01 = np.where((lon2d_01 >= 50.8) & (lon2d_01 <= 51.6) & (lat2d_01 >= 24.5) & (lat2d_01 <= 26.2), 1.0, 0.0).astype(np.float32)

        def _make_step_dict(temp_k: float) -> dict[str, xr.DataArray]:
            coords = {"latitude": lats_01, "longitude": lons_01}
            base = np.ones((len(lats_01), len(lons_01)), dtype=np.float32)
            out = {
                "sp": xr.DataArray(base * 101200.0, dims=("latitude", "longitude"), coords=coords),
                "msl": xr.DataArray(base * 101300.0, dims=("latitude", "longitude"), coords=coords),
                "z": xr.DataArray(lsm_01 * 150.0, dims=("latitude", "longitude"), coords=coords),
                "lsm": xr.DataArray(lsm_01, dims=("latitude", "longitude"), coords=coords),
                "skt": xr.DataArray(np.where(lsm_01 > 0.5, temp_k - 3.0, temp_k + 2.0).astype(np.float32), dims=("latitude", "longitude"), coords=coords),
                "sst": xr.DataArray(np.where(lsm_01 < 0.5, temp_k + 2.0, np.nan).astype(np.float32), dims=("latitude", "longitude"), coords=coords),
                "t2m": xr.DataArray(base * temp_k, dims=("latitude", "longitude"), coords=coords),
                "d2m": xr.DataArray(base * (temp_k - 8.0), dims=("latitude", "longitude"), coords=coords),
                "u10": xr.DataArray(base * 3.5, dims=("latitude", "longitude"), coords=coords),
                "v10": xr.DataArray(base * -2.0, dims=("latitude", "longitude"), coords=coords),
                "sd": xr.DataArray(base * 0.0, dims=("latitude", "longitude"), coords=coords),
                "siconc": xr.DataArray(base * 0.0, dims=("latitude", "longitude"), coords=coords),
            }
            for i in range(1, 5):
                out[f"stl{i}"] = xr.DataArray((base * (temp_k - i)).astype(np.float32), dims=("latitude", "longitude"), coords=coords)
                out[f"swvl{i}"] = xr.DataArray(np.where(lsm_01 > 0.5, 0.14, 0.0).astype(np.float32), dims=("latitude", "longitude"), coords=coords)
            return out

        cycle_dt = datetime(2026, 11, 22, 0, 0, tzinfo=timezone.utc)
        sfc_01_fp = ifs_dir / "ECMWF-IFS-0p1_sfc_20261122.nc"
        _write_cdm_sfc_daily(sfc_01_fp, cycle_dt, [0, 3], [_make_step_dict(298.0), _make_step_dict(300.0)])

        # Build 0.4° CAMS ML dataset (3 levels) + 0.1° GHG dataset
        lats_04 = np.linspace(27.6, 23.6, 11)
        lons_04 = np.linspace(48.8, 53.2, 12)
        pv = np.array([1000.0, 30000.0, 50000.0, 0.0, 0.0, 0.25, 0.65, 1.0], dtype=np.float64)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        frt_sec = int((cycle_dt - epoch).total_seconds())
        vt_secs = np.array([[frt_sec, frt_sec + 10800]], dtype=np.float64)

        cams_ml_fp = cams_dir / "ECMWF-CAMS_ml_20261122.nc"
        with Dataset(cams_ml_fp, "w") as d:
            d.createDimension("forecast_period", 2)
            d.createDimension("forecast_reference_time", 1)
            d.createDimension("model_level", 3)
            d.createDimension("latitude", len(lats_04))
            d.createDimension("longitude", len(lons_04))
            d.createVariable("forecast_period", "f8", ("forecast_period",))[:] = [0.0, 3.0]
            d.createVariable("forecast_reference_time", "i8", ("forecast_reference_time",))[:] = [frt_sec]
            vt = d.createVariable("valid_time", "f8", ("forecast_reference_time", "forecast_period"))
            vt.units = "seconds since 1970-01-01"
            vt[:] = vt_secs
            d.createVariable("model_level", "f8", ("model_level",))[:] = [1, 2, 3]
            d.createVariable("latitude", "f8", ("latitude",))[:] = lats_04
            d.createVariable("longitude", "f8", ("longitude",))[:] = lons_04
            for vname, val in [
                ("t", 285.0), ("u", 5.0), ("v", -3.0), ("w", 0.01), ("q", 0.008),
                ("clwc", 0.0), ("ciwc", 0.0), ("aermr04", 1e-8), ("aermr05", 2e-8), ("aermr06", 1e-8),
                ("so2", 1e-9), ("no2", 2e-9), ("go3", 5e-8), ("co", 1e-7),
            ]:
                vv = d.createVariable(vname, "f4", ("forecast_period", "forecast_reference_time", "model_level", "latitude", "longitude"))
                vv.GRIB_pv = pv
                vv[:] = val

        # 7a. Test convert_cams_to_wps across mixed 0.1° SFC + 0.4° ML
        wps_files = convert_cams_to_wps(sfc_01_fp, cams_ml_fp, wps_dir, map_source="ECMWF IFS 0.1+CAMS")
        assert len(wps_files) == 2
        assert (wps_dir / "FILE:2026-11-22_00").is_file() and (wps_dir / "FILE:2026-11-22_03").is_file()

        # 7b. Test ICON-ART IC + LatBC + CLI `cams2icon` on mixed 0.1° SFC + 0.4° ML
        grid_fp = root / "domain.nc"
        ext_fp = root / "extpar.nc"
        lbc_grid_fp = root / "latbc_grid.nc"
        clon_deg = np.array([50.2, 51.2, 51.3, 52.4])  # sea, land, land, sea
        clat_deg = np.array([25.3, 25.3, 25.4, 25.5])
        for gpath in (grid_fp, lbc_grid_fp):
            with Dataset(gpath, "w") as g:
                g.uuidOfHGrid = "test-uuid-qatar-001"
                g.createDimension("cell", len(clon_deg))
                g.createDimension("vertex", 6)
                g.createDimension("nv", 3)
                g.createVariable("clon", "f8", ("cell",))[:] = np.deg2rad(clon_deg)
                g.createVariable("clat", "f8", ("cell",))[:] = np.deg2rad(clat_deg)
                g.createVariable("clon_vertices", "f8", ("cell", "nv"))[:] = np.deg2rad(clon_deg)[:, None]
                g.createVariable("clat_vertices", "f8", ("cell", "nv"))[:] = np.deg2rad(clat_deg)[:, None]
        with Dataset(ext_fp, "w") as e:
            e.createDimension("cell", len(clon_deg))
            e.createVariable("FR_LAND", "f8", ("cell",))[:] = [0.0, 1.0, 1.0, 0.0]
            e.createVariable("SOILTYP", "i4", ("cell",))[:] = [9, 3, 5, 9]
            e.createVariable("T_CL", "f8", ("cell",))[:] = [299.0, 297.0, 297.0, 299.0]
            e.createVariable("topography_c", "f8", ("cell",))[:] = [0.0, 20.0, 15.0, 0.0]
            e.createVariable("emi_so2", "f4", ("cell",))[:] = [0.0, 1e-10, 1e-10, 0.0]
            e.createVariable("emi_nox", "f4", ("cell",))[:] = [0.0, 2e-10, 2e-10, 0.0]
            for k in ("fr_hcla", "fr_silc", "fr_lcla", "fr_sicl", "fr_cloa", "fr_silt", "fr_silo", "fr_scla", "fr_loam", "fr_sclo", "fr_sloa", "fr_lsan", "fr_sand", "fr_udef"):
                e.createVariable(k, "f4", ("cell",))[:] = [0.0, 0.5, 0.5, 0.0] if k in ("fr_sand", "fr_udef") else [0.0, 0.0, 0.0, 0.0]

        airblender_main([
            "cams2icon",
            "--cams_dir", str(cams_dir),
            "--grid_file", str(grid_fp),
            "--extpar_file", str(ext_fp),
            "--latbc_grid_file", str(lbc_grid_fp),
            "--run_dir", str(icon_run),
            "--start_date", "2026-11-22_00:00:00",
            "--end_date", "2026-11-22_03:00:00",
            "--met_sfc_dir", str(ifs_dir),
            "--met_sfc_prefix", "ECMWF-IFS-0p1_sfc",
        ])
        with Dataset(icon_run / "ifs2icon_R2B09_DOM01.nc") as di:
            smil1 = np.asarray(di.variables["SMIL1"][0, 0, :])
            skt = np.asarray(di.variables["SKT"][0, :])
            assert np.all(np.isfinite(smil1)) and np.all(np.isfinite(skt))
            assert smil1[0] == 0.0 and smil1[3] == 0.0, f"Expected 0 SMIL over sea, got {smil1}"
            assert 0.0 < smil1[1] < 1.0 and 0.0 < smil1[2] < 1.0, f"Expected land SMIL in (0,1), got {smil1}"
            assert skt[0] > skt[1] + 2.0, f"Expected land/sea SKT contrast preserved, got {skt}"
        with Dataset(icon_run / "ART_STY_iconR2B09-grid_TCNR.nc") as dst:
            fr_sand = np.asarray(dst.variables["fr_sand"][:])
            assert fr_sand[0] == 0.0 and fr_sand[3] == 0.0 and fr_sand[1] > 0.0
        assert (icon_run / "ART_IAE_iconR2B09-grid_TCNR.nc").is_file()
        assert (icon_run / "ART_ICE_iconR2B09-grid_TCNR.nc").is_file()
        assert (icon_run / "latbc" / "lbc_20261122T000000Z.nc").is_file()
        assert (icon_run / "latbc" / "lbc_20261122T030000Z.nc").is_file()
    print("  -> Test 7 PASSED.")


def test_cams_error_classification_and_circuit_breaker() -> None:
    print("[Test 8] CAMS Cy49r1 sea-salt rename, error classification, fail-fast circuit breaker, & status JSON...")
    import json
    import sys
    import time as _t
    import types
    from airblender.fetch.cams_forecast import (
        _get_chem_groups_for_date,
        classify_cams_error,
        download_cams_forecast,
    )

    # 8a. Cy49r1 date-aware sea salt bin-1 name
    grp_post = dict(_get_chem_groups_for_date("2026-11-22"))["chema"]
    grp_pre = dict(_get_chem_groups_for_date("2026-04-01"))["chema"]
    assert "sea_salt_aerosol_0.03-0.5um_mixing_ratio" in grp_post
    assert "sea_salt_aerosol_0.03-0.55um_mixing_ratio" in grp_pre

    # 8b. Error classification
    assert classify_cams_error(RuntimeError("403 Client Error: Forbidden: required licences not accepted")) == (True, "auth_or_license")
    assert classify_cams_error(RuntimeError("401 Client Error: Unauthorized")) == (True, "auth_or_license")
    assert classify_cams_error(RuntimeError("400 Client Error: Bad Request")) == (True, "bad_request")
    assert classify_cams_error(RuntimeError("400 Client Error: Bad Request\nThe job has failed\nMarsRuntimeError: Last error is -43")) == (False, "transient_backend")

    # 8c. Fail-fast circuit breaker on 403 + atomic cams_status.json
    calls = {"n": 0}
    fake_cdsapi = types.ModuleType("cdsapi")
    class _ForbiddenClient:
        def __init__(self, **_kwargs):
            pass
        def retrieve(self, _col, _req, _target):
            calls["n"] += 1
            raise RuntimeError("403 Client Error: Forbidden: required licences not accepted")
    fake_cdsapi.Client = _ForbiddenClient
    old_cds = sys.modules.get("cdsapi")
    sys.modules["cdsapi"] = fake_cdsapi

    with tempfile.TemporaryDirectory(prefix="airblender_cb_") as td:
        root = Path(td)
        cams_dir = root / "cams"
        status_fp = root / "cams_status.json"
        t0 = _t.monotonic()
        try:
            try:
                download_cams_forecast(
                    "2026-11-22",
                    "2026-11-22",
                    cams_dir,
                    cams_type="forecast",
                    cams_time="00:00",
                    ghg=False,
                    stream_daily=True,
                    max_workers=2,
                    max_retries=5,
                    status_file=status_fp,
                )
                raise AssertionError("Expected 403 licence error to abort CAMS pipeline")
            except RuntimeError as exc:
                assert "failed" in str(exc).lower()
        finally:
            if old_cds is None:
                sys.modules.pop("cdsapi", None)
            else:
                sys.modules["cdsapi"] = old_cds

        elapsed = _t.monotonic() - t0
        assert elapsed < 2.0, f"Circuit breaker took too long: {elapsed:.2f}s"
        assert calls["n"] <= 2, f"Expected fail-fast with 0 retries across max_workers=2, got {calls['n']} calls"
        st_ext = json.loads(status_fp.read_text(encoding="utf-8"))
        st_int = json.loads((cams_dir / ".cams_status_20261122.json").read_text(encoding="utf-8"))
        assert st_ext["state"] == "failed" and st_int["state"] == "failed"
        assert st_ext["counts"]["failed"] >= 1 and len(st_ext["failed_tasks"]) >= 1
    print("  -> Test 8 PASSED.")


def main() -> None:
    print("=" * 68)
    print(" Running AirBlender Verification Suite")
    print("=" * 68)
    test_physics_and_regrid_fixes()
    test_unified_cams_merra2_emac_icbc()
    test_cams2wps_and_wrf2palm()
    test_fetch_subpackage()
    test_cams_stream_daily()
    test_dynamics_resolver_and_fallback()
    test_mixed_grid_wps_and_cams2icon()
    test_cams_error_classification_and_circuit_breaker()
    print("=" * 68)
    print(" ALL AIRBLENDER CHECKS PASSED (100% OK)")
    print("=" * 68)


if __name__ == "__main__":
    main()
