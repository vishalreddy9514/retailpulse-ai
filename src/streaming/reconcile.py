"""Check that the streaming pipeline reproduces the batch daily demand exactly.

The stream and batch paths share the classification rules and the event ids, but they
compute aggregates differently: incremental SQL updates versus a pandas groupby. This
compares every product-day both paths produced over the replayed date range.

The batch table only starts a product's rows at its first sale (see aggregate_sales).
Stream rows outside that window, i.e. returns or reversals before a product's first sale,
are counted separately rather than reported as mismatches.

Usage::

    python -m src.streaming.reconcile [--db data/stream/stream_state.sqlite]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from src.processing.aggregate_sales import PRODUCT_DAILY_PARQUET
from src.streaming.kafka_consumer import DEFAULT_DB

INT_COLUMNS = [
    "units_sold",
    "sale_lines",
    "invoices",
    "customers",
    "units_returned",
    "return_lines",
    "units_reversed",
]
MONEY_COLUMNS = ["gross_revenue", "return_value"]
COLUMNS = INT_COLUMNS + MONEY_COLUMNS
MONEY_TOLERANCE = 0.011  # float summation order differs; values are rounded to pence


def load_stream(db_path: Path | str) -> pd.DataFrame:
    with closing(sqlite3.connect(db_path)) as conn:
        df = pd.read_sql_query(
            f"SELECT stock_code, date, {', '.join(COLUMNS)} FROM product_daily_view", conn
        )
        days = conn.execute("SELECT MIN(invoice_day), MAX(invoice_day) FROM events").fetchone()
    df["date"] = pd.to_datetime(df["date"])
    df.attrs["day_range"] = tuple(pd.Timestamp(d) for d in days)
    return df


def reconcile(stream: pd.DataFrame, batch: pd.DataFrame) -> dict:
    first_day, last_day = stream.attrs["day_range"]
    # The batch table starts each product at its first sale (over the full history).
    first_sale = batch.groupby("stock_code")["date"].min()
    batch = batch[(batch["date"] >= first_day) & (batch["date"] <= last_day)]
    active = lambda df: df[(df[COLUMNS] != 0).any(axis=1)]  # noqa: E731
    batch, stream = active(batch[["stock_code", "date", *COLUMNS]]), active(stream)

    in_window = stream["date"] >= stream["stock_code"].map(first_sale)
    outside = stream[~in_window.fillna(False)]
    stream = stream[in_window.fillna(False)]

    merged = batch.merge(
        stream,
        on=["stock_code", "date"],
        how="outer",
        suffixes=("_batch", "_stream"),
        indicator=True,
    )
    both = merged[merged["_merge"] == "both"]
    mismatches = {}
    for col in INT_COLUMNS:
        n = int((both[f"{col}_batch"] != both[f"{col}_stream"]).sum())
        if n:
            mismatches[col] = n
    for col in MONEY_COLUMNS:
        n = int((np.abs(both[f"{col}_batch"] - both[f"{col}_stream"]) > MONEY_TOLERANCE).sum())
        if n:
            mismatches[col] = n
    totals = {
        col: {"batch": float(batch[col].sum()), "stream": float(stream[col].sum())}
        for col in COLUMNS
    }
    report = {
        "day_range": [str(first_day.date()), str(last_day.date())],
        "product_days_compared": int(len(both)),
        "only_in_batch": int((merged["_merge"] == "left_only").sum()),
        "only_in_stream": int((merged["_merge"] == "right_only").sum()),
        "mismatched_values": mismatches,
        "totals": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in totals.items()},
        "stream_rows_before_first_sale": {
            "rows": int(len(outside)),
            "units_returned": int(outside["units_returned"].sum()),
            "units_reversed": int(outside["units_reversed"].sum()),
        },
    }
    report["exact_match"] = (
        report["only_in_batch"] == 0 and report["only_in_stream"] == 0 and not mismatches
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconcile streaming vs batch daily demand")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args(argv)
    report = reconcile(load_stream(args.db), pd.read_parquet(PRODUCT_DAILY_PARQUET))
    print(json.dumps(report, indent=2))
    return 0 if report["exact_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
