from __future__ import annotations

import hashlib
import zipfile

import pandas as pd
import pytest

from src import config
from src.ingestion.uci_loader import (
    ChecksumError,
    SchemaError,
    SheetOverlapError,
    download_dataset,
    resolve_sheet_overlap,
    standardise,
)


def test_standardise_renames_and_types_columns(make_source):
    src = make_source(
        [("489434", "85048", 12345, 12, "2009-12-01 07:45", 6.95, 13085.0, "United Kingdom")]
    )
    out = standardise(src, "Year 2009-2010")

    assert list(out.columns[:8]) == [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country",
    ]
    row = out.iloc[0]
    # Customer ID arrives as a float and must become a clean string id.
    assert row["customer_id"] == "13085"
    # Numeric descriptions exist in the real file and must survive as text.
    assert row["description"] == "12345"
    assert row["quantity"] == 12
    assert row["source_sheet"] == "Year 2009-2010"
    assert row["source_row"] == 2  # Excel row number, header is row 1


def test_standardise_keeps_missing_customer_as_na(make_source):
    src = make_source([("489434", "85048", "X", 1, "2009-12-01", 1.0, None, "UK")])
    assert pd.isna(standardise(src, "s").iloc[0]["customer_id"])


def test_standardise_rejects_missing_columns(make_source):
    src = make_source([("489434", "85048", "X", 1, "2009-12-01", 1.0, 1.0, "UK")])
    with pytest.raises(SchemaError, match="Price"):
        standardise(src.drop(columns=["Price"]), "s")


def _sheets(make_raw):
    first = make_raw(
        [
            ("1", "10001", "A", 1, "2010-11-30 10:00", 1.0, 1.0, "UK"),
            ("2", "10002", "B", 2, "2010-12-01 09:00", 1.0, 1.0, "UK"),
            ("3", "10003", "C", 3, "2010-12-02 09:00", 1.0, 1.0, "UK"),
        ],
        "first",
    )
    second = make_raw(
        [
            ("2", "10002", "B", 2, "2010-12-01 09:00", 1.0, 1.0, "UK"),
            ("3", "10003", "C", 3, "2010-12-02 09:00", 1.0, 1.0, "UK"),
            ("4", "10004", "D", 4, "2010-12-03 09:00", 1.0, 1.0, "UK"),
        ],
        "second",
    )
    return first, second


def test_overlap_window_is_removed_once(make_raw):
    first, second = _sheets(make_raw)
    combined, report = resolve_sheet_overlap(first, second)

    assert report.rows_dropped == 2
    assert combined["invoice"].tolist() == ["1", "2", "3", "4"]
    # The overlapping rows are kept from the later sheet.
    assert combined["source_sheet"].tolist() == ["first", "second", "second", "second"]


def test_overlap_mismatch_raises(make_raw):
    first, second = _sheets(make_raw)
    second.loc[0, "quantity"] = 99
    with pytest.raises(SheetOverlapError):
        resolve_sheet_overlap(first, second)


def test_overlap_keeps_repeated_lines_inside_a_sheet(make_raw):
    # A line repeated inside the second sheet must not be collapsed by overlap handling.
    first = make_raw([("1", "10001", "A", 1, "2010-11-30 10:00", 1.0, 1.0, "UK")], "first")
    second = make_raw(
        [
            ("2", "10002", "B", 2, "2010-12-03 09:00", 1.0, 1.0, "UK"),
            ("2", "10002", "B", 2, "2010-12-03 09:00", 1.0, 1.0, "UK"),
        ],
        "second",
    )
    combined, report = resolve_sheet_overlap(first, second)
    assert report.rows_dropped == 0
    assert len(combined) == 3


def _zip_with_workbook(tmp_path, payload: bytes, member: str = config.UCI_WORKBOOK_NAME):
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(member, payload)
    return archive


def test_download_extracts_and_verifies_checksum(tmp_path):
    payload = b"fake workbook bytes"
    archive = _zip_with_workbook(tmp_path, payload)
    raw_dir = tmp_path / "raw"

    path = download_dataset(
        raw_dir=raw_dir,
        url=archive.as_uri(),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert path == raw_dir / config.UCI_WORKBOOK_NAME
    assert path.read_bytes() == payload


def test_download_rejects_checksum_mismatch(tmp_path):
    archive = _zip_with_workbook(tmp_path, b"tampered")
    raw_dir = tmp_path / "raw"
    with pytest.raises(ChecksumError):
        download_dataset(raw_dir=raw_dir, url=archive.as_uri(), expected_sha256="0" * 64)
    assert not (raw_dir / config.UCI_WORKBOOK_NAME).exists()


def test_download_rejects_archive_without_workbook(tmp_path):
    archive = _zip_with_workbook(tmp_path, b"x", member="something_else.xlsx")
    with pytest.raises(SchemaError):
        download_dataset(raw_dir=tmp_path / "raw", url=archive.as_uri(), expected_sha256=None)
