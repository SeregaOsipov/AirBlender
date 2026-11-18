"""Donor model adapters (CAMS, MERRA-2, EMAC) for AirBlender."""
from airblender.sources.base import DateDonorContext, DonorAdapter
from airblender.sources.cams import CamsDonorAdapter
from airblender.sources.emac import EmacDonorAdapter
from airblender.sources.merra2 import Merra2DonorAdapter


def get_donor_adapter(source: str, root_dir: str, file_name_template: str | None = None) -> DonorAdapter:
    """Factory returning the normalized `DonorAdapter` for `cams`, `merra2`, or `emac`."""
    src = str(source).lower()
    if src == "cams":
        return CamsDonorAdapter(
            root_dir, file_name_template or "ECMWF-CAMS_{stream}_{date_time}.nc"
        )
    if src == "merra2":
        return Merra2DonorAdapter(
            root_dir, file_name_template or "{stream}/MERRA2_{VVV}.{stream}.{date_time}.nc4"
        )
    if src == "emac":
        return EmacDonorAdapter(
            root_dir, file_name_template or "test01_________{date_time}_{stream}.nc"
        )
    raise ValueError(f"Unsupported donor source '{source}'. Choose from ('cams', 'merra2', 'emac').")


__all__ = [
    "CamsDonorAdapter",
    "DateDonorContext",
    "DonorAdapter",
    "EmacDonorAdapter",
    "Merra2DonorAdapter",
    "get_donor_adapter",
]
