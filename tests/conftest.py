"""Shared fixtures. These frames are tiny hand-written examples shaped like the real file
(same columns and types). They are test inputs, not samples of the real dataset."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.uci_loader import standardise

SOURCE_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


def source_frame(rows: list[tuple]) -> pd.DataFrame:
    """Build a frame with the workbook's original column names and dtypes."""
    df = pd.DataFrame(rows, columns=SOURCE_COLUMNS)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Customer ID"] = df["Customer ID"].astype("float64")
    return df


def raw_frame(rows: list[tuple], sheet: str = "Year 2010-2011") -> pd.DataFrame:
    """Build a standardised raw frame, as produced by the loader."""
    return standardise(source_frame(rows), sheet)


@pytest.fixture
def make_raw():
    return raw_frame


@pytest.fixture
def make_source():
    return source_frame
