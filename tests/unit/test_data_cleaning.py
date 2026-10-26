from __future__ import annotations

import pandas as pd

from src.processing.clean_transactions import (
    BAD_DEBT,
    CANCELLATION_RETURN,
    INVALID,
    NON_PRODUCT,
    ORDER_REVERSAL_RETURN,
    ORDER_REVERSAL_SALE,
    SALE,
    STOCK_ADJUSTMENT,
    ZERO_PRICE,
    clean_transactions,
)

T = "2011-03-01 10:00"


def _types(df: pd.DataFrame) -> dict[str, str]:
    return dict(zip(df["invoice"], df["line_type"], strict=True))


def test_line_types_follow_documented_rules(make_raw):
    raw = make_raw(
        [
            ("500001", "85123A", "HEART HOLDER", 6, T, 2.55, 1.0, "United Kingdom"),
            ("C500002", "85123A", "HEART HOLDER", -2, T, 2.55, 2.0, "United Kingdom"),
            ("A500003", "B", "Adjust bad debt", 1, T, -100.0, None, "United Kingdom"),
            ("500004", "POST", "POSTAGE", 1, T, 18.0, 1.0, "France"),
            ("500005", "21733", "damages", -10, T, 0.0, None, "United Kingdom"),
            ("500006", "21734", "found", 5, T, 0.0, None, "United Kingdom"),
            ("C500007", "M", "Manual", 1, T, 373.57, None, "United Kingdom"),
            ("C500008", "21735", "X", 3, T, 1.0, 3.0, "United Kingdom"),
            ("500009", "gift_0001_20", "Gift Voucher", 1, T, 20.0, 1.0, "United Kingdom"),
        ]
    )
    clean, _ = clean_transactions(raw)
    assert _types(clean) == {
        "500001": SALE,
        "C500002": CANCELLATION_RETURN,
        "A500003": BAD_DEBT,
        "500004": NON_PRODUCT,
        "500005": STOCK_ADJUSTMENT,
        "500006": ZERO_PRICE,
        "C500007": NON_PRODUCT,  # non-product precedes the cancellation rules
        "C500008": INVALID,  # a cancellation with positive quantity is malformed
        "500009": NON_PRODUCT,
    }


def test_returns_are_never_counted_as_sales(make_raw):
    raw = make_raw(
        [
            ("500001", "22222", "MUG", 10, "2011-03-01 10:00", 1.0, 1.0, "UK"),
            ("C500002", "22222", "MUG", -4, "2011-03-05 10:00", 1.0, 1.0, "UK"),
        ]
    )
    clean, report = clean_transactions(raw)
    # Returned on a later day: an ordinary return, not an order reversal.
    assert _types(clean) == {"500001": SALE, "C500002": CANCELLATION_RETURN}
    assert report.sale_units == 10
    assert clean.loc[clean["invoice"] == "C500002", "line_revenue"].item() == -4.0


def test_exact_duplicates_dropped_and_counted(make_raw):
    row = ("500001", "22222", "MUG", 10, T, 1.0, 1.0, "UK")
    raw = make_raw([row, row, ("500001", "22223", "BOWL", 1, T, 1.0, 1.0, "UK")])
    clean, report = clean_transactions(raw)
    assert len(clean) == 2
    assert report.exact_duplicates_dropped == 1
    assert report.passed


def test_duplicates_can_be_kept_with_unique_event_ids(make_raw):
    row = ("500001", "22222", "MUG", 10, T, 1.0, 1.0, "UK")
    clean, report = clean_transactions(make_raw([row, row]), drop_exact_duplicates=False)
    assert len(clean) == 2
    assert clean["event_id"].is_unique
    assert report.passed


def test_event_id_is_deterministic_and_ignores_formatting_noise(make_raw):
    a = make_raw([("500001", "47503J", "TOOLS", 1, T, 1.0, 1.0, "UK")])
    b = make_raw([("500001", "47503j ", " TOOLS ", 1, T, 1.0, 1.0, "UK")], "other sheet")
    id_a = clean_transactions(a)[0]["event_id"].item()
    id_b = clean_transactions(b)[0]["event_id"].item()
    assert id_a == id_b
    assert len(id_a) == 32


def test_same_day_exact_cancellation_is_an_order_reversal(make_raw):
    raw = make_raw(
        [
            ("581483", "23843", "PAPER CRAFT", 80995, "2011-12-09 09:15", 2.08, 16446.0, "UK"),
            ("C581484", "23843", "PAPER CRAFT", -80995, "2011-12-09 09:27", 2.08, 16446.0, "UK"),
            ("581485", "23843", "PAPER CRAFT", 5, "2011-12-09 10:00", 2.08, 99.0, "UK"),
        ]
    )
    clean, report = clean_transactions(raw)
    assert _types(clean) == {
        "581483": ORDER_REVERSAL_SALE,
        "C581484": ORDER_REVERSAL_RETURN,
        "581485": SALE,
    }
    assert report.order_reversal_pairs == 1
    assert report.sale_units == 5


def test_reversal_matching_is_one_to_one_and_requires_order(make_raw):
    raw = make_raw(
        [
            # Two identical sales, one cancellation: only one sale is reversed (the latest).
            ("500001", "22222", "MUG", 3, "2011-03-01 09:00", 1.0, 7.0, "UK"),
            ("500002", "22222", "MUG", 3, "2011-03-01 10:00", 1.0, 7.0, "UK"),
            ("C500003", "22222", "MUG", -3, "2011-03-01 11:00", 1.0, 7.0, "UK"),
            # Cancellation before the sale cannot reverse it.
            ("C500004", "33333", "BOWL", -2, "2011-03-01 08:00", 1.0, 7.0, "UK"),
            ("500005", "33333", "BOWL", 2, "2011-03-01 12:00", 1.0, 7.0, "UK"),
            # No customer id: never matched.
            ("500006", "44444", "JUG", 1, "2011-03-01 09:00", 1.0, None, "UK"),
            ("C500007", "44444", "JUG", -1, "2011-03-01 09:30", 1.0, None, "UK"),
        ]
    )
    clean, _ = clean_transactions(raw)
    assert _types(clean) == {
        "500001": SALE,
        "500002": ORDER_REVERSAL_SALE,
        "C500003": ORDER_REVERSAL_RETURN,
        "C500004": CANCELLATION_RETURN,
        "500005": SALE,
        "500006": SALE,
        "C500007": CANCELLATION_RETURN,
    }


def test_missing_description_filled_from_catalog(make_raw):
    raw = make_raw(
        [
            ("500001", "22222", "WHITE MUG", 1, T, 1.0, 1.0, "UK"),
            ("500002", "22222", "WHITE MUG", 1, T, 1.0, 2.0, "UK"),
            ("500003", "22222", "white mug?", 1, T, 1.0, 3.0, "UK"),
            ("500004", "22222", None, -5, T, 0.0, None, "UK"),
        ]
    )
    clean, report = clean_transactions(raw)
    assert set(clean["product_description"]) == {"WHITE MUG"}
    assert report.descriptions_filled_from_catalog == 1
    # The original text is preserved alongside the canonical name.
    assert pd.isna(clean.loc[clean["invoice"] == "500004", "description"].item())


def test_missing_customer_sales_are_kept(make_raw):
    raw = make_raw([("500001", "22222", "MUG", 2, T, 1.0, None, "UK")])
    clean, report = clean_transactions(raw)
    assert clean["line_type"].item() == SALE
    assert not clean["has_customer"].item()
    assert report.sale_lines_missing_customer == 1


def test_malformed_rows_are_labelled_invalid(make_raw):
    raw = make_raw(
        [
            ("500001", "22222", "MUG", 2, None, 1.0, 1.0, "UK"),
            ("not-an-invoice", "22222", "MUG", 2, T, 1.0, 1.0, "UK"),
            ("500003", None, "MUG", 2, T, 1.0, 1.0, "UK"),
        ]
    )
    clean, report = clean_transactions(raw)
    assert set(clean["line_type"]) == {INVALID}
    assert report.invalid_reasons == {
        "missing_invoice_date": 1,
        "malformed_invoice": 1,
        "missing_stock_code": 1,
    }
    assert report.passed


def test_quality_checks_all_pass_on_mixed_input(make_raw):
    raw = make_raw(
        [
            ("500001", "85123A", "HEART", 6, T, 2.55, 1.0, "UK"),
            ("C500002", "85123A", "HEART", -6, T, 2.55, 1.0, "UK"),
            ("500003", "POST", "POSTAGE", 1, T, 18.0, 1.0, "UK"),
        ]
    )
    _, report = clean_transactions(raw)
    assert report.rows_in == report.rows_out + report.exact_duplicates_dropped
    assert all(c["passed"] for c in report.quality_checks), report.quality_checks


def test_scalar_classifier_agrees_with_vectorised_rules(make_raw):
    """The stream consumer's classify_line must give the batch result on every combination."""
    import itertools

    from src.processing.clean_transactions import classify_line

    invoices = ["500001", "C500001", "A500001"]
    codes = ["85123A", "22222", "DCGS0058", "SP1002", "POST", "M", "GIFT_0001_20", "B"]
    quantities = [-5, 0, 3]
    prices = [-1.0, 0.0, 2.5]
    rows = [
        (inv, code, "X", q, T, p, None, "UK")
        for inv, code, q, p in itertools.product(invoices, codes, quantities, prices)
    ]
    raw = make_raw(rows)
    from src.processing.clean_transactions import classify, invalid_reason, normalise

    df = normalise(raw)
    df["invalid_reason"] = invalid_reason(df)
    expected = classify(df).tolist()
    got = [classify_line(i, c, q, p) for i, c, _, q, _, p, _, _ in rows]
    assert got == expected
