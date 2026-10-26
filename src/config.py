"""Project-wide paths and dataset constants.

Paths can be overridden with the RETAILPULSE_DATA_DIR environment variable so tests
and CI never touch the real data directory.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("RETAILPULSE_DATA_DIR", PROJECT_ROOT / "data"))

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Official UCI download. The archive contains a single workbook with one sheet per year.
UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
UCI_WORKBOOK_NAME = "online_retail_II.xlsx"
# SHA-256 of online_retail_II.xlsx as published by UCI (archive dated 2023-05-22).
# Pinned so a silently changed or corrupted download is detected rather than ingested.
UCI_WORKBOOK_SHA256 = "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980"
UCI_SHEETS = ("Year 2009-2010", "Year 2010-2011")

RAW_PARQUET = INTERIM_DIR / "online_retail_ii_raw.parquet"
CLEAN_PARQUET = PROCESSED_DIR / "transactions_clean.parquet"
