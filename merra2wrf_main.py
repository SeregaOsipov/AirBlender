#!/usr/bin/env python
"""Backwards-compatible entrypoint delegating to `airblender.cli:main_merra2wrf`."""
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from airblender.cli import main_merra2wrf

if __name__ == "__main__":
    main_merra2wrf()
