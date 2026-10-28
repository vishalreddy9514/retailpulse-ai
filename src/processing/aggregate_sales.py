"""Aggregate cleaned transaction lines into product-level daily demand.

Outputs (``data/processed/``):

``product_daily.parquet``
    One row per product per trading day, from the product's first sale to the end of the
    data. Days without sales are explicit zeros. Demand (``units_sold``) counts only
    ``sale`` lines. Returns and same-day order reversals are kept in separate columns, as
    set out in the README's return-handling policy.
``daily_totals.parquet``
    Store-wide totals per trading day.
``product_summary.parquet``
    Per-product history and sparsity statistics over the *full* history. This is a
    descriptive summary for exploration and the dashboard. Forecasting must not use it
    as a feature, because it contains future information. Phase 4 recomputes what it
    needs at each forecast origin.

History uses *observed* trading days (days with at least one sale). Every return
falls on such a day. Future dates use the rule in ``trading_calendar``.

Usage::

    python -m src.processing.aggregate_sales
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.processing.clean_transactions import CANCELLATION_RETURN, ORDER_REVERSAL_SALE, SALE
from src.processing.trading_calendar import compare_with_observed

logger = logging.getLogger(__name__)

PRODUCT_DAILY_PARQUET = config.PROCESSED_DIR / "product_daily.parquet"
DAILY_TOTALS_PARQUET = config.PROCESSED_DIR / "daily_totals.parquet"
PRODUCT_SUMMARY_PARQUET = config.PROCESSED_DIR / "product_summary.parquet"

# Sparsity thresholds. A product needs at least MIN_SELLING_DAYS days with sales, over a
# window of at least MIN_WINDOW_DAYS trading days, before it gets a model forecast.
MIN_SELLING_DAYS = 10
MIN_WINDOW_DAYS = 28
RECENT_DAYS = 28
# Syntetos-Boylan cut-offs for classifying demand patterns.
ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49
# The last day is "partial" if the data stops before this share of a normal day's activity.
PARTIAL_DAY_QUANTILE = 0.10


@dataclass
class AggregationReport:
    products: int
    trading_days: int
    first_day: str
    last_day: str
    product_day_rows: int
    zero_share: float
    partial_days: list[str]
    calendar_check: dict
    returns_before_first_sale: dict
    returns_for_never_sold_products: dict
    reversals_outside_product_window: dict
    demand_classes: dict[str, int]
    history_status: dict[str, int]
    reconciliation: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.reconciliation.values())


def _daily(lines: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Per product-day counts for one family of lines. Quantities and values are made positive."""
    g = lines.groupby(["stock_code", "invoice_day"], observed=True)
    out = pd.DataFrame(
        {
            f"{prefix}_units": g["quantity"].sum().abs(),
            f"{prefix}_lines": g.size(),
            f"{prefix}_invoices": g["invoice"].nunique(),
            f"{prefix}_customers": g["customer_id"].nunique(),
            f"{prefix}_value": g["line_revenue"].sum().abs(),
        }
    )
    return out.rename_axis(["stock_code", "date"])


def detect_partial_days(clean: pd.DataFrame, trading_days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Flag the first/last day if the data starts unusually late or stops unusually early."""
    activity = clean.loc[clean["line_type"] == SALE].groupby("invoice_day")["invoice_date"]
    first_ts, last_ts = activity.min(), activity.max()
    minutes = lambda s: s.dt.hour * 60 + s.dt.minute  # noqa: E731
    partial = []
    last_day, first_day = trading_days[-1], trading_days[0]
    if minutes(last_ts).loc[last_day] < minutes(last_ts).quantile(PARTIAL_DAY_QUANTILE):
        partial.append(last_day)
    if minutes(first_ts).loc[first_day] > minutes(first_ts).quantile(1 - PARTIAL_DAY_QUANTILE):
        partial.append(first_day)
    return sorted(partial)


def classify_demand(adi: pd.Series, cv2: pd.Series) -> pd.Series:
    """Syntetos-Boylan demand classes. Products without enough non-zero sizes get <NA>."""
    cls = np.select(
        [
            (adi < ADI_CUTOFF) & (cv2 < CV2_CUTOFF),
            (adi < ADI_CUTOFF) & (cv2 >= CV2_CUTOFF),
            (adi >= ADI_CUTOFF) & (cv2 < CV2_CUTOFF),
            (adi >= ADI_CUTOFF) & (cv2 >= CV2_CUTOFF),
        ],
        ["smooth", "erratic", "intermittent", "lumpy"],
        default="",
    )
    return pd.Series(cls, index=adi.index, dtype="string").replace("", pd.NA)


def aggregate(clean: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], AggregationReport]:
    lt = clean["line_type"]
    sales = clean.loc[lt == SALE]
    returns = clean.loc[lt == CANCELLATION_RETURN]
    reversals = clean.loc[lt == ORDER_REVERSAL_SALE]

    trading_days = pd.DatetimeIndex(sorted(sales["invoice_day"].unique()))
    partial_days = detect_partial_days(clean, trading_days)
    first_sale = sales.groupby("stock_code", observed=True)["invoice_day"].min()

    # Dense grid: every product on every trading day from its first sale to the end.
    grid = pd.DataFrame(
        {
            "stock_code": np.repeat(first_sale.index.to_numpy(), len(trading_days)),
            "date": np.tile(trading_days.to_numpy(), len(first_sale)),
        }
    )
    grid = grid[grid["date"] >= grid["stock_code"].map(first_sale)]
    grid = grid.set_index(["stock_code", "date"])

    daily = grid.join(_daily(sales, "sale")).join(_daily(returns, "return"))
    daily = daily.join(_daily(reversals, "reversal")[["reversal_units"]])
    daily = daily.fillna(0)

    out = pd.DataFrame(index=daily.index)
    out["units_sold"] = daily["sale_units"].astype("int64")
    out["sale_lines"] = daily["sale_lines"].astype("int64")
    out["invoices"] = daily["sale_invoices"].astype("int64")
    out["customers"] = daily["sale_customers"].astype("int64")
    out["gross_revenue"] = daily["sale_value"].round(2)
    out["units_returned"] = daily["return_units"].astype("int64")
    out["return_lines"] = daily["return_lines"].astype("int64")
    out["return_value"] = daily["return_value"].round(2)
    out["units_reversed"] = daily["reversal_units"].astype("int64")
    out["net_units"] = out["units_sold"] - out["units_returned"]
    out["net_revenue"] = (out["gross_revenue"] - out["return_value"]).round(2)
    out["avg_unit_price"] = (out["gross_revenue"] / out["units_sold"].replace(0, np.nan)).round(4)
    out = out.reset_index()
    out["is_partial_day"] = out["date"].isin(partial_days)

    totals = (
        out.groupby("date")
        .agg(
            units_sold=("units_sold", "sum"),
            gross_revenue=("gross_revenue", "sum"),
            units_returned=("units_returned", "sum"),
            return_value=("return_value", "sum"),
            net_revenue=("net_revenue", "sum"),
            products_sold=("units_sold", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    totals["invoices"] = (
        sales.groupby("invoice_day")["invoice"].nunique().reindex(totals["date"]).to_numpy()
    )
    totals["is_partial_day"] = totals["date"].isin(partial_days)

    summary = summarise_products(out, clean, trading_days)

    # Returns and reversals that cannot be placed on a product's row (before its first sale,
    # or for a product that never had a real sale) are reported, not silently lost.
    before = returns["invoice_day"] < returns["stock_code"].map(first_sale)
    never = ~returns["stock_code"].isin(first_sale.index)
    rev_first = reversals["stock_code"].map(first_sale)
    rev_outside = rev_first.isna() | (reversals["invoice_day"] < rev_first)

    report = AggregationReport(
        products=len(first_sale),
        trading_days=len(trading_days),
        first_day=str(trading_days[0].date()),
        last_day=str(trading_days[-1].date()),
        product_day_rows=len(out),
        zero_share=round(float((out["units_sold"] == 0).mean()), 4),
        partial_days=[str(d.date()) for d in partial_days],
        calendar_check=compare_with_observed(pd.Series(trading_days)),
        returns_before_first_sale={
            "lines": int(before.sum()),
            "units": int(returns.loc[before, "quantity"].abs().sum()),
        },
        returns_for_never_sold_products={
            "lines": int(never.sum()),
            "units": int(returns.loc[never, "quantity"].abs().sum()),
        },
        reversals_outside_product_window={
            "lines": int(rev_outside.sum()),
            "units": int(reversals.loc[rev_outside, "quantity"].sum()),
        },
        demand_classes={
            str(k): int(v) for k, v in summary["demand_class"].value_counts(dropna=False).items()
        },
        history_status={
            str(k): int(v) for k, v in summary["history_status"].value_counts().items()
        },
    )
    placed_returns = returns.loc[~before & ~never]
    report.reconciliation = {
        "units_sold_match_sale_lines": int(out["units_sold"].sum()) == int(sales["quantity"].sum()),
        "revenue_matches_sale_lines": bool(
            np.isclose(out["gross_revenue"].sum(), sales["line_revenue"].sum(), atol=1.0)
        ),
        "returns_reconcile": int(out["units_returned"].sum())
        == int(placed_returns["quantity"].abs().sum()),
        "reversals_reconcile": int(out["units_reversed"].sum())
        == int(reversals.loc[~rev_outside, "quantity"].sum()),
        "one_row_per_product_day": not out.duplicated(["stock_code", "date"]).any(),
        "no_negative_demand": bool((out["units_sold"] >= 0).all()),
        "totals_match_products": int(totals["units_sold"].sum()) == int(out["units_sold"].sum()),
    }
    return {"product_daily": out, "daily_totals": totals, "product_summary": summary}, report


def summarise_products(
    daily: pd.DataFrame, clean: pd.DataFrame, trading_days: pd.DatetimeIndex
) -> pd.DataFrame:
    g = daily.groupby("stock_code")
    selling = daily[daily["units_sold"] > 0].groupby("stock_code")["units_sold"]
    summary = pd.DataFrame(
        {
            "first_sale_date": g["date"].min(),
            "last_sale_date": daily[daily["units_sold"] > 0].groupby("stock_code")["date"].max(),
            "window_days": g.size(),
            "selling_days": selling.size(),
            "total_units": g["units_sold"].sum(),
            "total_gross_revenue": g["gross_revenue"].sum().round(2),
            "total_units_returned": g["units_returned"].sum(),
            "mean_daily_units": g["units_sold"].mean().round(3),
            "mean_nonzero_units": selling.mean().round(3),
            "cv2_nonzero": (selling.std(ddof=0) / selling.mean()).pow(2).round(4),
        }
    )
    summary["zero_share"] = (1 - summary["selling_days"] / summary["window_days"]).round(4)
    summary["adi"] = (summary["window_days"] / summary["selling_days"]).round(3)
    # CV² needs at least two non-zero demand sizes to mean anything.
    cv2 = summary["cv2_nonzero"].where(summary["selling_days"] >= 2)
    summary["demand_class"] = classify_demand(summary["adi"], cv2)
    summary.loc[cv2.isna(), "demand_class"] = pd.NA

    sufficient = (summary["selling_days"] >= MIN_SELLING_DAYS) & (
        summary["window_days"] >= MIN_WINDOW_DAYS
    )
    summary["history_status"] = np.where(sufficient, "sufficient", "insufficient")
    recent_cutoff = trading_days[max(len(trading_days) - RECENT_DAYS, 0)]
    summary["recently_active"] = summary["last_sale_date"] >= recent_cutoff

    catalog = (
        clean.loc[clean["line_type"] == SALE, ["stock_code", "product_description"]]
        .drop_duplicates("stock_code")
        .set_index("stock_code")
    )
    summary = summary.join(catalog).reset_index()
    cols = ["stock_code", "product_description"] + [
        c for c in summary.columns if c not in ("stock_code", "product_description")
    ]
    return summary[cols]


def write_outputs(
    tables: dict[str, pd.DataFrame],
    report: AggregationReport,
    out_dir: Path = config.PROCESSED_DIR,
    report_path: Path = config.REPORTS_DIR / "aggregation_report.json",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n")
    logger.info("Wrote %s and %s", ", ".join(tables), report_path)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Aggregate cleaned sales to product-day").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    clean = pd.read_parquet(config.CLEAN_PARQUET)
    tables, report = aggregate(clean)
    write_outputs(tables, report)

    print(
        f"products: {report.products:,}  trading days: {report.trading_days} "
        f"({report.first_day} -> {report.last_day})"
    )
    print(
        f"product-day rows: {report.product_day_rows:,}  zero-demand share: {report.zero_share:.1%}"
    )
    print(f"partial days: {report.partial_days}")
    print(f"calendar rule agreement: {report.calendar_check['agreement_rate']:.2%}")
    print(f"history status: {report.history_status}")
    print(f"demand classes: {report.demand_classes}")
    for name, ok in report.reconciliation.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
