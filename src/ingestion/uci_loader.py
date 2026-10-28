"""Download, verify and load the UCI Online Retail II workbook.

The official archive holds one Excel workbook with two sheets, "Year 2009-2010" and
"Year 2010-2011". Inspection of the real file showed:

* both sheets have the same eight columns: Invoice, StockCode, Description, Quantity,
  InvoiceDate, Price, Customer ID, Country;
* the sheets overlap: 2010-12-01 to 2010-12-09 appears, row for row, in both;
* some Description cells are stored as numbers, and Customer ID is stored as a float.

This module only makes the raw data faithful and typed. It renames columns, fixes
types and removes the sheet overlap. Business rules (cancellations, duplicates,
adjustments) live in ``src.processing.clean_transactions``.

Usage::

    python -m src.ingestion.uci_loader            # download if needed, build raw parquet
    python -m src.ingestion.uci_loader --refresh  # rebuild parquet from the workbook
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# Source column name -> pipeline column name.
COLUMN_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}
BUSINESS_COLUMNS = list(COLUMN_MAP.values())


class SchemaError(ValueError):
    """The workbook does not have the columns this pipeline was written against."""


class ChecksumError(ValueError):
    """The downloaded workbook does not match the pinned SHA-256."""


class SheetOverlapError(ValueError):
    """The overlapping window between sheets is not an exact duplicate."""


@dataclass(frozen=True)
class OverlapReport:
    window_start: str | None
    window_end: str | None
    rows_in_first_sheet: int
    rows_in_second_sheet: int
    rows_dropped: int


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(
    raw_dir: Path = config.RAW_DIR,
    url: str = config.UCI_DATASET_URL,
    expected_sha256: str | None = config.UCI_WORKBOOK_SHA256,
    force: bool = False,
) -> Path:
    """Download the official archive and extract the workbook, verifying its checksum.

    Returns the path of the extracted workbook. An existing, verified workbook is reused
    unless ``force`` is set.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    workbook = raw_dir / config.UCI_WORKBOOK_NAME

    if workbook.exists() and not force:
        _verify(workbook, expected_sha256)
        logger.info("Using existing workbook %s", workbook)
        return workbook

    logger.info("Downloading %s", url)
    with tempfile.TemporaryDirectory(dir=raw_dir) as tmp:
        archive = Path(tmp) / "online_retail_ii.zip"
        with urllib.request.urlopen(url, timeout=120) as resp, archive.open("wb") as out:
            shutil.copyfileobj(resp, out)
        with zipfile.ZipFile(archive) as zf:
            if config.UCI_WORKBOOK_NAME not in zf.namelist():
                raise SchemaError(
                    f"{config.UCI_WORKBOOK_NAME} not found in archive; members: {zf.namelist()}"
                )
            # Extract only the expected member, by name, so archive paths are never trusted.
            with (
                zf.open(config.UCI_WORKBOOK_NAME) as src,
                (Path(tmp) / "wb.xlsx").open("wb") as dst,
            ):
                shutil.copyfileobj(src, dst)
        _verify(Path(tmp) / "wb.xlsx", expected_sha256)
        (Path(tmp) / "wb.xlsx").replace(workbook)

    logger.info("Saved verified workbook to %s", workbook)
    return workbook


def _verify(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    actual = sha256_of(path)
    if actual != expected_sha256:
        raise ChecksumError(
            f"{path} has SHA-256 {actual}, expected {expected_sha256}. "
            "The upstream file may have changed; inspect it before updating the pin."
        )


def read_workbook(path: Path, sheets: tuple[str, ...] = config.UCI_SHEETS) -> list[pd.DataFrame]:
    """Read each sheet as text-safe columns and standardise it. Slow (~80 s for the real file)."""
    logger.info("Reading %s (this takes a minute or two)", path)
    xl = pd.ExcelFile(path, engine="openpyxl")
    missing = set(sheets) - set(xl.sheet_names)
    if missing:
        raise SchemaError(f"Workbook is missing sheets {sorted(missing)}; found {xl.sheet_names}")
    frames = []
    for sheet in sheets:
        # Codes and descriptions are read as text: some descriptions are stored as numbers.
        df = pd.read_excel(
            xl,
            sheet_name=sheet,
            dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str},
        )
        frames.append(standardise(df, sheet))
        logger.info("Sheet %r: %d rows", sheet, len(df))
    return frames


def standardise(df: pd.DataFrame, sheet: str) -> pd.DataFrame:
    """Validate source columns, rename to snake_case and fix types. Keeps every row."""
    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise SchemaError(f"Sheet {sheet!r} is missing columns {sorted(missing)}")

    out = df[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    for col in ("invoice", "stock_code", "description", "country"):
        out[col] = out[col].astype("string")
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce").astype("Int64")
    out["unit_price"] = pd.to_numeric(out["unit_price"], errors="coerce").astype("float64")
    out["invoice_date"] = pd.to_datetime(out["invoice_date"], errors="coerce")
    # Customer ID arrives as a float (13085.0); store it as a nullable string id ("13085").
    cust = pd.to_numeric(out["customer_id"], errors="coerce").astype("Int64")
    out["customer_id"] = cust.astype("string")

    out["source_sheet"] = pd.Series(sheet, index=out.index, dtype="string")
    # Excel row number (header is row 1) for lineage back to the original file.
    out["source_row"] = (df.index + 2).astype("int64")
    return out.reset_index(drop=True)


def resolve_sheet_overlap(
    first: pd.DataFrame, second: pd.DataFrame
) -> tuple[pd.DataFrame, OverlapReport]:
    """Combine consecutive sheets, removing the window that appears in both.

    The overlap is resolved by time window rather than by row-level de-duplication. That
    way, legitimately repeated lines inside a sheet are left for the cleaning stage to judge
    and are not silently collapsed here. The window must be an exact multiset match.
    Otherwise this raises, because the files have changed in a way the pipeline does not
    understand.
    """
    start = second["invoice_date"].min()
    end = first["invoice_date"].max()
    if pd.isna(start) or pd.isna(end) or start > end:
        combined = pd.concat([first, second], ignore_index=True)
        return combined, OverlapReport(None, None, 0, 0, 0)

    in_first = first["invoice_date"] >= start
    in_second = second["invoice_date"] <= end
    a = _multiset(first.loc[in_first])
    b = _multiset(second.loc[in_second])
    if not a.equals(b):
        raise SheetOverlapError(
            f"Sheets overlap between {start} and {end} but the rows differ "
            f"({int(in_first.sum())} vs {int(in_second.sum())} rows)."
        )

    combined = pd.concat([first.loc[~in_first], second], ignore_index=True)
    report = OverlapReport(
        window_start=str(start),
        window_end=str(end),
        rows_in_first_sheet=int(in_first.sum()),
        rows_in_second_sheet=int(in_second.sum()),
        rows_dropped=int(in_first.sum()),
    )
    return combined, report


def _multiset(df: pd.DataFrame) -> pd.Series:
    keys = df[BUSINESS_COLUMNS].astype("string").fillna("<NA>")
    return keys.value_counts().sort_index()


def load_raw(
    refresh: bool = False,
    force_download: bool = False,
    parquet_path: Path = config.RAW_PARQUET,
) -> pd.DataFrame:
    """Return the combined raw transactions, building the parquet cache if needed."""
    if parquet_path.exists() and not (refresh or force_download):
        return pd.read_parquet(parquet_path)

    workbook = download_dataset(force=force_download)
    first, second = read_workbook(workbook)
    combined, report = resolve_sheet_overlap(first, second)
    logger.info("Sheet overlap: %s", asdict(report))

    combined = combined.sort_values(["invoice_date", "source_sheet", "source_row"], kind="stable")
    combined = combined.reset_index(drop=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, index=False)
    logger.info("Wrote %d raw rows to %s", len(combined), parquet_path)
    return combined


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="rebuild parquet from workbook")
    parser.add_argument("--force-download", action="store_true", help="re-download the archive")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = load_raw(refresh=args.refresh, force_download=args.force_download)
    print(f"rows: {len(df):,}")
    print(f"columns: {list(df.columns)}")
    print(f"invoice_date range: {df['invoice_date'].min()} -> {df['invoice_date'].max()}")
    print(df.groupby("source_sheet", observed=True).size().to_string())


if __name__ == "__main__":
    main()
