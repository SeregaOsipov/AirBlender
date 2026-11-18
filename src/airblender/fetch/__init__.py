"""Optional data acquisition adapters for donor atmospheric/soil datasets (`airblender[fetch]`)."""

from airblender.fetch.cams_eac4 import download_cams_eac4
from airblender.fetch.cams_forecast import download_cams_forecast
from airblender.fetch.ecmwf_ifs import download_ecmwf_ifs
from airblender.fetch.era5_soil import download_era5_soil
from airblender.fetch.merra2 import (
    ALL_MERRA2_COLLECTIONS,
    MERRA2_COLLECTIONS,
    MERRA2_EXTRA_COLLECTIONS,
    MERRA2_PRESETS,
    build_merra2_url,
    download_merra2,
    merra2_stream_version,
)
from airblender.fetch.opendata import download_opendata_sfc, download_opendata_soil

__all__ = [
    "download_cams_forecast",
    "download_cams_eac4",
    "download_ecmwf_ifs",
    "download_opendata_soil",
    "download_opendata_sfc",
    "download_era5_soil",
    "download_merra2",
    "merra2_stream_version",
    "build_merra2_url",
    "MERRA2_COLLECTIONS",
    "MERRA2_EXTRA_COLLECTIONS",
    "ALL_MERRA2_COLLECTIONS",
    "MERRA2_PRESETS",
]
