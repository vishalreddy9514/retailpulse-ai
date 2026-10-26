"""Clean raw Online Retail II lines and label what each line actually is.

Not every row is a sale. Rather than silently dropping rows, every surviving line gets a
``line_type`` and downstream stages pick the types they need. The rules come from
inspecting the real file. See the README section "Cleaning decisions" for the
reasoning and measured counts.

Precedence, first match wins:

=========================  ==========================================================
``invalid``                fails schema validation (missing key fields, bad types)
``bad_debt_adjustment``    invoice starts with "A" (accounting write-offs, code "B")
``non_product``            stock code is postage, fees, manual, discount, test, voucher
``cancellation_return``    invoice starts with "C" with negative quantity
``invalid``                invoice starts with "C" with non-negative quantity
``stock_adjustment``       non-"C" line with quantity <= 0 (damages, lost, found, ...)
``zero_price``             positive quantity at unit price <= 0 (free-of-charge lines)
``sale``                   positive quantity at a positive price
=========================  ==========================================================

After classification, a ``cancellation_return`` that exactly reverses a ``sale`` on the
same calendar day (same customer, product and absolute quantity, cancelled after the
sale) relabels the pair as ``order_reversal_sale`` / ``order_reversal_return``. The
cancellation is known by the end of that day, so excluding the pair from demand uses no
future information when forecasting from the next day onward.

Usage::

    python -m src.processing.clean_transactions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src import config
from src.ingestion.uci_loader import BUSINESS_COLUMNS, load_raw

logger = logging.getLogger(__name__)

SALE = "sale"
CANCELLATION_RETURN = "cancellation_return"
ORDER_REVERSAL_SALE = "order_reversal_sale"
ORDER_REVERSAL_RETURN = "order_reversal_return"
STOCK_ADJUSTMENT = "stock_adjustment"
ZERO_PRICE = "zero_price"
NON_PRODUCT = "non_product"
BAD_DEBT = "bad_debt_adjustment"
INVALID = "invalid"

LINE_TYPES = (
    SALE,
    CANCELLATION_RETURN,
    ORDER_REVERSAL_SALE,
    ORDER_REVERSAL_RETURN,
    STOCK_ADJUSTMENT,
    ZERO_PRICE,
    NON_PRODUCT,
    BAD_DEBT,
    INVALID,
)

# Merchandise stock codes: five digits with an optional letter suffix (85123A, 15056BL),
# dotcom gift sets (DCGS0058) and the SP-prefixed specials (SP1002). Everything else
# observed in the file is postage, carriage, fees, manual entries, discounts, samples,
# tests, gift vouchers or adjustments, which are not product demand.
PRODUCT_CODE_PATTERN = r"^(?:\d{5}[A-Z]{0,3}|DCGS[0-9A-Z]+|SP\d{4})$"

EVENT_ID_COLUMNS = BUSINESS_COLUMNS


@dataclass
class CleaningReport:
    rows_in: int
    rows_out: int
    exact_duplicates_dropped: int
    line_type_counts: dict[str, int]
    invalid_reasons: dict[str, int]
    sale_lines_missing_customer: int
    descriptions_filled_from_catalog: int
    order_reversal_pairs: int
    sale_units: int
    order_reversal_units: int
    quality_checks: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.quality_checks)


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Trim and canonicalise text fields. Does not drop or reorder rows."""
    df = raw.copy()
    df["invoice"] = df["invoice"].str.strip().str.upper()
    # Codes appear with trailing spaces ("47503J ") and in lower case ("m", "gift_0001_20").
    df["stock_code"] = df["stock_code"].str.strip().str.upper()
    df["description"] = df["description"].str.strip().str.replace(r"\s+", " ", regex=True)
    df.loc[df["description"] == "", "description"] = pd.NA
    df["country"] = df["country"].str.strip()
    return df


def invalid_reason(df: pd.DataFrame) -> pd.Series:
    """Return the first schema violation per row, or <NA> when the row is well-formed."""
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    checks = [
        ("missing_invoice", df["invoice"].isna() | (df["invoice"] == "")),
        ("missing_stock_code", df["stock_code"].isna() | (df["stock_code"] == "")),
        ("missing_quantity", df["quantity"].isna()),
        ("missing_invoice_date", df["invoice_date"].isna()),
        ("missing_unit_price", df["unit_price"].isna()),
        ("missing_country", df["country"].isna() | (df["country"] == "")),
        (
            "malformed_invoice",
            df["invoice"].notna() & ~df["invoice"].fillna("").str.match(r"^[A-Z]?\d{6}$"),
        ),
    ]
    # Apply in reverse so the first listed violation wins.
    for name, mask in reversed(checks):
        reason = reason.mask(mask.fillna(True), name)
    return reason


def make_event_id(df: pd.DataFrame) -> pd.Series:
    """Deterministic id from the normalised business fields.

    The same transaction line always gets the same id, which makes replay idempotent:
    the streaming consumer can de-duplicate on it across restarts.
    """
    parts = [df[c].astype("string").fillna("") for c in EVENT_ID_COLUMNS]
    joined = parts[0].str.cat(parts[1:], sep="\x1f")
    return joined.map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]).astype("string")


def classify(df: pd.DataFrame) -> pd.Series:
    """Assign a line_type using the precedence documented in the module docstring."""
    inv = df["invoice"].fillna("")
    qty = df["quantity"]
    price = df["unit_price"]
    is_product = df["stock_code"].fillna("").str.match(PRODUCT_CODE_PATTERN)
    is_c = inv.str.startswith("C")

    # Built in reverse precedence: later assignments override earlier ones.
    line_type = pd.Series(SALE, index=df.index, dtype="string")
    line_type = line_type.mask(~is_c & (price <= 0).fillna(False), ZERO_PRICE)
    line_type = line_type.mask(~is_c & (qty <= 0).fillna(False), STOCK_ADJUSTMENT)
    line_type = line_type.mask(is_c & (qty >= 0).fillna(False), INVALID)
    line_type = line_type.mask(is_c & (qty < 0).fillna(False), CANCELLATION_RETURN)
    line_type = line_type.mask(~is_product, NON_PRODUCT)
    line_type = line_type.mask(inv.str.startswith("A"), BAD_DEBT)
    line_type = line_type.mask(df["invalid_reason"].notna(), INVALID)
    return line_type


def match_order_reversals(df: pd.DataFrame) -> pd.DataFrame:
    """Relabel same-day exact cancellations of a sale, one-to-one.

    A cancellation is paired with the latest unmatched sale on the same day with the same
    customer, product and absolute quantity, timestamped at or before the cancellation.
    Lines without a customer id are never matched: there is nothing to tie them together.
    """
    keys = ["customer_id", "stock_code", "abs_qty", "day"]
    work = df.assign(abs_qty=df["quantity"].abs(), day=df["invoice_date"].dt.normalize())
    sales = work.loc[(work["line_type"] == SALE) & work["customer_id"].notna(), keys]
    rets = work.loc[(work["line_type"] == CANCELLATION_RETURN) & work["customer_id"].notna(), keys]
    sales = sales.assign(sale_idx=sales.index, sale_ts=work.loc[sales.index, "invoice_date"])
    rets = rets.assign(ret_idx=rets.index, ret_ts=work.loc[rets.index, "invoice_date"])

    cand = rets.merge(sales, on=keys, how="inner")
    cand = cand[cand["sale_ts"] <= cand["ret_ts"]]
    cand = cand.sort_values(
        ["ret_ts", "ret_idx", "sale_ts", "sale_idx"], ascending=[True, True, False, False]
    )

    used_sales: set[int] = set()
    matched_rets: set[int] = set()
    for ret_idx, sale_idx in zip(cand["ret_idx"], cand["sale_idx"], strict=True):
        if ret_idx in matched_rets or sale_idx in used_sales:
            continue
        matched_rets.add(ret_idx)
        used_sales.add(sale_idx)

    out = df.copy()
    out.loc[list(used_sales), "line_type"] = ORDER_REVERSAL_SALE
    out.loc[list(matched_rets), "line_type"] = ORDER_REVERSAL_RETURN
    return out


def build_product_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """One canonical description per product code: the most frequent one on sale lines.

    Ties go to the alphabetically first description so the result is reproducible.
    """
    named = df.loc[df["line_type"].isin([SALE, ORDER_REVERSAL_SALE]) & df["description"].notna()]
    counts = named.groupby(["stock_code", "description"], observed=True).size()
    counts = counts.rename("n").reset_index()
    counts = counts.sort_values(["stock_code", "n", "description"], ascending=[True, False, True])
    catalog = counts.drop_duplicates("stock_code")[["stock_code", "description"]]
    return catalog.rename(columns={"description": "product_description"}).reset_index(drop=True)


def clean_transactions(
    raw: pd.DataFrame, drop_exact_duplicates: bool = True
) -> tuple[pd.DataFrame, CleaningReport]:
    """Normalise, de-duplicate, validate and label raw transaction lines."""
    rows_in = len(raw)
    df = normalise(raw)

    # Exact duplicate lines: same invoice, product, description, quantity, timestamp, price,
    # customer and country. The source has no line number, so a repeated line cannot be
    # told apart from a double-scan. We drop repeats by default and report the count.
    dup_mask = df.duplicated(subset=EVENT_ID_COLUMNS, keep="first")
    duplicates = int(dup_mask.sum())
    if drop_exact_duplicates:
        df = df.loc[~dup_mask].copy()

    df["event_id"] = make_event_id(df)
    if not drop_exact_duplicates:
        occurrence = df.groupby("event_id").cumcount().astype("string")
        df["event_id"] = df["event_id"].str.cat(occurrence, sep="-")

    df["invalid_reason"] = invalid_reason(df)
    df["line_type"] = classify(df)
    df = match_order_reversals(df)

    catalog = build_product_catalog(df)
    df = df.merge(catalog, on="stock_code", how="left", validate="many_to_one")
    filled = int((df["description"].isna() & df["product_description"].notna()).sum())
    df["product_description"] = df["product_description"].fillna(df["description"])

    df["line_revenue"] = (df["quantity"].astype("float64") * df["unit_price"]).round(2)
    df["invoice_day"] = df["invoice_date"].dt.normalize()
    df["has_customer"] = df["customer_id"].notna()
    df = df.sort_values(["invoice_date", "source_sheet", "source_row"], kind="stable")
    df = df.reset_index(drop=True)

    sales = df["line_type"] == SALE
    reversal_sales = df["line_type"] == ORDER_REVERSAL_SALE
    report = CleaningReport(
        rows_in=rows_in,
        rows_out=len(df),
        exact_duplicates_dropped=duplicates if drop_exact_duplicates else 0,
        line_type_counts={t: int((df["line_type"] == t).sum()) for t in LINE_TYPES},
        invalid_reasons={str(k): int(v) for k, v in df["invalid_reason"].value_counts().items()},
        sale_lines_missing_customer=int((sales & ~df["has_customer"]).sum()),
        descriptions_filled_from_catalog=filled,
        order_reversal_pairs=int(reversal_sales.sum()),
        sale_units=int(df.loc[sales, "quantity"].sum()),
        order_reversal_units=int(df.loc[reversal_sales, "quantity"].sum()),
    )
    report.quality_checks = run_quality_checks(df, report)
    return df, report


def run_quality_checks(df: pd.DataFrame, report: CleaningReport) -> list[dict]:
    """Invariants the cleaned table must satisfy. Each is reported, not just asserted."""
    lt = df["line_type"]
    sales = df.loc[lt.isin([SALE, ORDER_REVERSAL_SALE])]
    returns = df.loc[lt.isin([CANCELLATION_RETURN, ORDER_REVERSAL_RETURN])]
    checks = {
        "event_id_unique": df["event_id"].is_unique,
        "row_count_conserved": report.rows_in == report.rows_out + report.exact_duplicates_dropped,
        "every_line_typed": bool(lt.isin(LINE_TYPES).all()),
        "sales_positive_qty_and_price": bool(
            ((sales["quantity"] > 0) & (sales["unit_price"] > 0)).all()
        ),
        "returns_negative_qty": bool((returns["quantity"] < 0).all()),
        "returns_are_c_invoices": bool(returns["invoice"].str.startswith("C").all()),
        "sales_have_product_code": bool(sales["stock_code"].str.match(PRODUCT_CODE_PATTERN).all()),
        "reversal_pairs_balanced": int((lt == ORDER_REVERSAL_SALE).sum())
        == int((lt == ORDER_REVERSAL_RETURN).sum()),
        "valid_lines_have_timestamp": bool(df.loc[lt != INVALID, "invoice_date"].notna().all()),
    }
    return [{"name": k, "passed": bool(v)} for k, v in checks.items()]


def write_outputs(
    df: pd.DataFrame,
    report: CleaningReport,
    parquet_path: Path = config.CLEAN_PARQUET,
    report_path: Path = config.REPORTS_DIR / "cleaning_report.json",
) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n")
    logger.info("Wrote %s and %s", parquet_path, report_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean UCI Online Retail II transactions")
    parser.add_argument(
        "--keep-exact-duplicates",
        action="store_true",
        help="keep repeated identical lines instead of dropping them",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df, report = clean_transactions(
        load_raw(), drop_exact_duplicates=not args.keep_exact_duplicates
    )
    write_outputs(df, report)

    print(
        f"rows in: {report.rows_in:,}  rows out: {report.rows_out:,}  "
        f"exact duplicates dropped: {report.exact_duplicates_dropped:,}"
    )
    for name, n in report.line_type_counts.items():
        print(f"  {name:<24} {n:>9,}")
    print(f"order reversal pairs: {report.order_reversal_pairs:,}")
    for check in report.quality_checks:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
