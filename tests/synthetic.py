"""Small synthetic artifacts (3 products, Sep-Dec 2011) for copilot, API and dashboard tests.

Test inputs only: hand-made demand shapes, forecasts and a risk table built with the real
risk engine. Not samples of the real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.copilot.knowledge import KnowledgeBase
from src.copilot.services import Artifacts
from src.inventory.risk_engine import ErrorModel, assess
from src.inventory.synthetic_inventory import load_assumptions
from src.processing.trading_calendar import expected_trading_days

ORIGIN = pd.Timestamp("2011-12-08")
PRODUCTS = {
    "22197": ("SMALL POPCORN HOLDER", lambda i: 40 + (300 if i >= 53 else 0)),
    "85099B": ("JUMBO BAG RED RETROSPOT", lambda i: 30 + i % 5),
    "22410": ("MONEY BOX HOUSEKEEPING DESIGN", lambda i: 10 if i % 2 else 0),
}


def build_synthetic_artifacts() -> Artifacts:
    days = [pd.Timestamp("2011-09-01")] + expected_trading_days("2011-09-01", 400)
    days = [d for d in days if d <= ORIGIN]
    rows = []
    for code, (_, f) in PRODUCTS.items():
        for i, d in enumerate(days):
            u = f(i)
            rows.append(
                (
                    code,
                    d,
                    u,
                    max(u // 10, 1 if u else 0),
                    1 if u else 0,
                    round(u * 1.5, 2),
                    0,
                    False,
                )
            )
    daily = pd.DataFrame(
        rows,
        columns=["stock_code", "date", "units_sold", "invoices", "customers", "gross_revenue",
                 "units_returned", "is_partial_day"],
    )  # fmt: skip
    summary = pd.DataFrame(
        {
            "stock_code": list(PRODUCTS),
            "product_description": [v[0] for v in PRODUCTS.values()],
            "first_sale_date": days[0],
            "last_sale_date": ORIGIN,
            "total_units": daily.groupby("stock_code")["units_sold"]
            .sum()
            .reindex(list(PRODUCTS))
            .to_numpy(),
            "total_gross_revenue": 1000.5,
            "selling_days": 60,
            "demand_class": "erratic",
        }
    )
    targets = expected_trading_days(ORIGIN, 7)
    fc = pd.DataFrame(
        [
            (code, ORIGIN, t, h, rate, "lightgbm", "test-model", 20.0, 10.0)
            for code, rate in {"22197": 300.5, "85099B": 31.25, "22410": 6.0}.items()
            for h, t in enumerate(targets, 1)
        ],
        columns=["stock_code", "forecast_origin", "target_date", "horizon_trading_days",
                 "forecast_units",
                 "method", "model_version", "recent_mean_28", "units_last_trading_day"],
    )  # fmt: skip
    inv = pd.DataFrame(
        [("22197", 50, 5, 30, 1530), ("85099B", 5000, 3, 10, 104), ("22410", 2, 7, 8, 50)],
        columns=["stock_code", "current_stock", "lead_time_days", "safety_stock", "reorder_point"],
    )
    cum = pd.DataFrame(
        [(f"P{i}", ORIGIN, h, a * h / 7, 70.0 * h / 7)
         for i, a in enumerate(np.linspace(35, 105, 100))
         for h in range(1, 8)],
        columns=["stock_code", "origin_date", "h", "cum_actual", "cum_forecast"],
    )  # fmt: skip
    assumptions = load_assumptions()
    risk = assess(fc, inv, ErrorModel.fit(cum), assumptions)
    forecast_eval = {
        "test": {"segments": {"model": {"methods": {"lightgbm": {"wape": 0.9887}}}}},
        "test_cumulative_7day": {"model": {"lightgbm": {"wape": 0.5926}}},
        "splits": {"test": {"targets": ["2011-10-01", "2011-12-08"]}},
    }
    return Artifacts(
        product_daily=daily,
        product_summary=summary,
        forecasts=fc,
        risk=risk,
        forecast_eval=forecast_eval,
        risk_eval={"mean_predicted_probability": 0.139, "observed_stockout_rate": 0.165},
        assumptions=assumptions,
        knowledge=KnowledgeBase.build(summary),
    )
