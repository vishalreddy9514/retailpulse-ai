"""Profile the raw and cleaned data and write a Markdown report.

Every number in ``reports/data_profile.md`` is computed here from the real file, so the
README can quote measured figures instead of assumptions.

Usage::

    python -m src.ingestion.explore
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src import config
from src.ingestion.uci_loader import load_raw
from src.processing.clean_transactions import (
    CANCELLATION_RETURN,
    ORDER_REVERSAL_SALE,
    SALE,
    clean_transactions,
)

logger = logging.getLogger(__name__)


def _table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [
            f"{v:,.2f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)
            for v in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def profile_raw(raw: pd.DataFrame) -> dict[str, pd.DataFrame | str]:
    cols = [c for c in raw.columns if c not in ("source_sheet", "source_row")]
    schema = pd.DataFrame(
        {
            "column": cols,
            "dtype": [str(raw[c].dtype) for c in cols],
            "nulls": [int(raw[c].isna().sum()) for c in cols],
            "null_pct": [f"{raw[c].isna().mean():.2%}" for c in cols],
            "distinct": [int(raw[c].nunique()) for c in cols],
        }
    )
    per_sheet = (
        raw.groupby("source_sheet", observed=True)
        .agg(
            rows=("invoice", "size"),
            first=("invoice_date", "min"),
            last=("invoice_date", "max"),
        )
        .reset_index()
    )
    per_sheet["rows"] = per_sheet["rows"].astype(int)
    inv = raw["invoice"].str.strip().str.upper().fillna("")
    anomalies = pd.DataFrame(
        {
            "check": [
                "invoices starting with C (cancellations/returns)",
                "invoices starting with A (bad-debt adjustments)",
                "quantity <= 0",
                "quantity < 0 on a non-C invoice",
                "unit_price == 0",
                "unit_price < 0",
                "missing customer_id",
                "missing description",
            ],
            "rows": [
                int(inv.str.startswith("C").sum()),
                int(inv.str.startswith("A").sum()),
                int((raw["quantity"] <= 0).sum()),
                int(((raw["quantity"] < 0) & ~inv.str.startswith("C")).sum()),
                int((raw["unit_price"] == 0).sum()),
                int((raw["unit_price"] < 0).sum()),
                int(raw["customer_id"].isna().sum()),
                int(raw["description"].isna().sum()),
            ],
        }
    )
    return {"schema": schema, "per_sheet": per_sheet, "anomalies": anomalies}


def profile_clean(clean: pd.DataFrame) -> dict[str, pd.DataFrame]:
    types = clean.groupby("line_type", observed=True).agg(
        lines=("event_id", "size"),
        units=("quantity", "sum"),
        value_gbp=("line_revenue", "sum"),
    )
    types = types.sort_values("lines", ascending=False).reset_index()
    types["lines"] = types["lines"].astype(int)
    types["units"] = types["units"].astype(int)
    types["value_gbp"] = types["value_gbp"].round(2)

    sales = clean[clean["line_type"] == SALE]
    daily = sales.groupby(["stock_code", "invoice_day"], observed=True)["quantity"].sum()
    days_per_product = daily.groupby(level=0).size()
    span = (sales["invoice_day"].max() - sales["invoice_day"].min()).days + 1
    trading_days = sales["invoice_day"].nunique()
    sparsity = pd.DataFrame(
        {
            "metric": [
                "products with at least one sale",
                "calendar days in sales span",
                "days with at least one sale (trading days)",
                "median selling days per product",
                "products selling on >= 100 days",
                "products selling on >= 300 days",
                "products selling on < 10 days",
                "share of product-trading-days with zero sales",
            ],
            "value": [
                f"{sales['stock_code'].nunique():,}",
                f"{span:,}",
                f"{trading_days:,}",
                f"{days_per_product.median():.0f}",
                f"{int((days_per_product >= 100).sum()):,}",
                f"{int((days_per_product >= 300).sum()):,}",
                f"{int((days_per_product < 10).sum()):,}",
                f"{1 - len(daily) / (sales['stock_code'].nunique() * trading_days):.1%}",
            ],
        }
    )
    top = (
        sales.groupby(["stock_code", "product_description"], observed=True)["quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"quantity": "units_sold"})
    )
    top["units_sold"] = top["units_sold"].astype(int)

    by_weekday = (
        sales.assign(weekday=sales["invoice_day"].dt.day_name())
        .groupby("weekday", observed=True)
        .agg(trading_days=("invoice_day", "nunique"), units=("quantity", "sum"))
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    biggest = (
        clean[clean["line_type"].isin([ORDER_REVERSAL_SALE, CANCELLATION_RETURN, SALE])]
        .loc[lambda d: d["quantity"].abs() >= 10_000]
        .sort_values("invoice_date")[
            ["invoice", "invoice_date", "stock_code", "quantity", "unit_price", "line_type"]
        ]
    )
    biggest["quantity"] = biggest["quantity"].astype(int)
    return {
        "line_types": types,
        "sparsity": sparsity,
        "top_products": top,
        "weekday": by_weekday,
        "largest_lines": biggest,
    }


def render(raw_profile: dict, clean_profile: dict, report) -> str:
    checks = pd.DataFrame(
        [
            {"check": c["name"], "result": "PASS" if c["passed"] else "FAIL"}
            for c in report.quality_checks
        ]
    )
    return f"""# Data profile: UCI Online Retail II

Generated by `python -m src.ingestion.explore` from the official UCI workbook
(SHA-256 `{config.UCI_WORKBOOK_SHA256[:16]}…`). All figures are measured, not assumed.

## Raw data (after removing the sheet overlap)

{_table(raw_profile["per_sheet"])}

Rows: **{report.rows_in:,}**

### Schema

{_table(raw_profile["schema"])}

### Raw anomalies

{_table(raw_profile["anomalies"])}

## Cleaning outcome

Exact duplicate lines dropped: **{report.exact_duplicates_dropped:,}**.
Rows after cleaning: **{report.rows_out:,}**.
Same-day order reversals matched: **{report.order_reversal_pairs:,}** pairs
({report.order_reversal_units:,} units removed from demand).
Sale lines without a customer id (kept, since the sale still happened):
**{report.sale_lines_missing_customer:,}**.

{_table(clean_profile["line_types"])}

`units` and `value_gbp` are signed: returns and write-offs are negative.

### Quality checks

{_table(checks)}

## Demand sparsity (sale lines only)

{_table(clean_profile["sparsity"])}

### Sales by weekday

{_table(clean_profile["weekday"])}

### Top 10 products by units sold

{_table(clean_profile["top_products"])}

### Lines with |quantity| >= 10,000

{_table(clean_profile["largest_lines"])}
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Profile UCI Online Retail II")
    parser.add_argument(
        "--out", type=Path, default=config.REPORTS_DIR / "data_profile.md", help="output path"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw = load_raw()
    clean, report = clean_transactions(raw)
    text = render(profile_raw(raw), profile_clean(clean), report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
