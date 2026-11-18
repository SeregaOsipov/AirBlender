"""Standalone datetime, WRF Times reader, and eta-pressure helpers (no wrf-python or ClimPy dependency)."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def parse_wrf_datetime(value: str) -> datetime:
    """Parse common WRF/operational CLI timestamp formats into naive UTC datetime."""
    text = str(value).strip()
    for fmt in (
        "%Y-%m-%d_%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d_%H_%M_%S",
        "%Y-%m-%d",
        "%Y%m%d%H",
        "%Y%m%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return pd.to_datetime(text).to_pydatetime().replace(tzinfo=None)


def extract_wrf_times(nc_dataset: Any) -> pd.DatetimeIndex:
    """Read WRF character `Times(Time, DateStrLen)` variable cleanly without `wrf-python`."""
    raw = nc_dataset.variables["Times"][:]
    rows: list[str] = []
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    for row in raw:
        if isinstance(row, (bytes, np.bytes_)):
            s = row.decode("ascii", errors="ignore")
        else:
            s = b"".join(bytes(c) if isinstance(c, (bytes, np.bytes_)) else str(c).encode("ascii") for c in row).decode(
                "ascii", errors="ignore"
            )
        rows.append(s.strip().replace("_", " "))
    return pd.to_datetime(rows)


def derive_wrf_eta_pressure(wrf_met_ds: xr.Dataset, wrf_ic_ds: xr.Dataset) -> xr.DataArray:
    """Compute WRF full-level midpoint pressure `p_rho` from `P_TOP`, `PSFC`, and `ZNU`."""
    with xr.set_options(keep_attrs=True):
        wrf_met_ds["p_rho"] = wrf_ic_ds.P_TOP + (wrf_met_ds.PSFC - wrf_ic_ds.P_TOP) * wrf_ic_ds.ZNU
    wrf_met_ds.p_rho.attrs["description"] = "Pressure at eta values on half (mass) levels"
    return wrf_met_ds["p_rho"]
