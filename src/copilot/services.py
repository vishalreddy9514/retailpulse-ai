"""Backend services behind the copilot's tools (and, in phase 9, the API).

Every number the copilot reports comes from one of these functions, never from the
language model. Each result carries its provenance:

* ``value_types``: which fields are ``observed`` (UCI sales history), which are
  ``model_prediction`` (the forecasting model) and which are ``synthetic`` (the
  generated inventory, which is not real stock);
* ``data_as_of`` / ``forecast_origin``: the end-of-day timestamp the figures refer to;
* ``units``, ``model_version``, ``assumptions_version`` and ``limitations``.

All dates are in the historical replay (2009-12-01 to 2011-12-09). "Tomorrow" means the
next expected trading day after the forecast origin.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.copilot.knowledge import KnowledgeBase
from src.inventory.approvals import DB_PATH as APPROVALS_DB
from src.inventory.approvals import ApprovalError, ApprovalStore
from src.inventory.synthetic_inventory import INVENTORY_DIR, SYNTHETIC_LABEL, load_assumptions
from src.processing.aggregate_sales import PRODUCT_DAILY_PARQUET, PRODUCT_SUMMARY_PARQUET

FORECAST_EVAL_JSON = config.REPORTS_DIR / "forecast_evaluation.json"
RISK_EVAL_JSON = config.REPORTS_DIR / "inventory_risk_evaluation.json"
STOCK_CODE_RE = re.compile(r"^(?:\d{5}[A-Z]{0,3}|DCGS[0-9A-Z]+|SP\d{4})$")

SIMULATION_NOTE = (
    "All data is historical UCI Online Retail II (2009-12-01 to 2011-12-09) replayed as a "
    "simulated real-time stream; it is not live retailer data."
)


class ServiceError(ValueError):
    """A request the services cannot answer (unknown product, bad parameter)."""


@dataclass
class Artifacts:
    product_daily: pd.DataFrame
    product_summary: pd.DataFrame
    forecasts: pd.DataFrame
    risk: pd.DataFrame
    forecast_eval: dict
    risk_eval: dict | None
    assumptions: dict
    knowledge: KnowledgeBase
    approvals_path: Path | None = None

    @classmethod
    def load(cls, origin: str | None = None) -> Artifacts:
        daily = pd.read_parquet(PRODUCT_DAILY_PARQUET)
        origin_ts = (
            pd.Timestamp(origin) if origin else daily.loc[~daily["is_partial_day"], "date"].max()
        )
        forecasts_path = config.PROCESSED_DIR / f"forecasts_{origin_ts:%Y-%m-%d}.parquet"
        risk_path = INVENTORY_DIR / f"stockout_risk_{origin_ts:%Y-%m-%d}.parquet"
        missing = [str(p) for p in (forecasts_path, risk_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing {missing}. Run `python -m src.forecasting.predict --origin "
                f"{origin_ts.date()}` and `python -m src.inventory.risk_engine --origin "
                f"{origin_ts.date()}` first."
            )
        summary = pd.read_parquet(PRODUCT_SUMMARY_PARQUET)
        return cls(
            product_daily=daily,
            product_summary=summary,
            forecasts=pd.read_parquet(forecasts_path),
            risk=pd.read_parquet(risk_path),
            forecast_eval=json.loads(FORECAST_EVAL_JSON.read_text()),
            risk_eval=json.loads(RISK_EVAL_JSON.read_text()) if RISK_EVAL_JSON.exists() else None,
            assumptions=load_assumptions(),
            knowledge=KnowledgeBase.build(summary),
            approvals_path=APPROVALS_DB if APPROVALS_DB.exists() else None,
        )


def _round(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), nd)


def _date(x) -> str | None:
    return None if pd.isna(x) else str(pd.Timestamp(x).date())


class RetailPulseServices:
    def __init__(self, artifacts: Artifacts):
        self.a = artifacts
        fc = artifacts.forecasts
        self.origin = pd.Timestamp(fc["forecast_origin"].iloc[0])
        self.model_version = str(fc["model_version"].iloc[0])
        self.names = artifacts.product_summary.set_index("stock_code")["product_description"]
        daily = artifacts.product_daily
        self.daily = daily.loc[~daily["is_partial_day"] & (daily["date"] <= self.origin)]

    # -- shared provenance -----------------------------------------------------------------
    def _envelope(
        self, tool: str, results, value_types: dict, units: dict, limitations: list[str], **extra
    ):
        return {
            "tool": tool,
            "forecast_origin": f"{self.origin.date()} (end of trading day)",
            "data_as_of": f"{self.origin.date()} (end of trading day)",
            "results": results,
            "value_types": value_types,
            "units": units,
            "limitations": [*limitations, SIMULATION_NOTE],
            **extra,
        }

    @cached_property
    def _forecast_limits(self) -> list[str]:
        ev = self.a.forecast_eval
        daily_wape = ev["test"]["segments"]["model"]["methods"]["lightgbm"]["wape"]
        week_wape = ev["test_cumulative_7day"]["model"]["lightgbm"]["wape"]
        t0, t1 = ev["splits"]["test"]["targets"]
        return [
            f"Forecasts are model predictions, not observed sales. Held-out test ({t0} to {t1}): "
            f"WAPE {daily_wape} for single product-days and {week_wape} for 7-day totals; "
            "daily product demand is lumpy (wholesale orders) and hard to predict.",
            "Products with too little history get a 28-day-mean fallback instead of the model "
            "(method field).",
        ]

    def _require_product(self, stock_code: str) -> str:
        code = stock_code.strip().upper()
        if not STOCK_CODE_RE.match(code):
            raise ServiceError(f"'{stock_code}' is not a valid product stock code")
        if code not in self.names.index:
            raise ServiceError(f"unknown product {code}")
        return code

    # -- 1. forecasts ---------------------------------------------------------------------
    def top_forecast_demand(self, horizon_trading_days: int = 1, limit: int = 10) -> dict:
        if not 1 <= horizon_trading_days <= 7:
            raise ServiceError("horizon_trading_days must be 1-7")
        fc = self.a.forecasts[self.a.forecasts["horizon_trading_days"] == horizon_trading_days]
        top = fc.nlargest(limit, "forecast_units")
        results = [
            {
                "stock_code": r.stock_code,
                "product_description": self.names.get(r.stock_code),
                "target_date": _date(r.target_date),
                "forecast_units": _round(r.forecast_units, 1),
                "recent_mean_28_units_per_day": _round(r.recent_mean_28, 1),
                "units_last_trading_day": _round(r.units_last_trading_day, 0),
                "method": r.method,
            }
            for r in top.itertuples()
        ]
        limits = list(self._forecast_limits)
        if self.origin == pd.Timestamp("2011-12-08"):
            limits.append(
                "The next trading day (2011-12-09) is only partly present in the source data "
                "(it stops at 12:50), so it cannot be used to check this forecast."
            )
        return self._envelope(
            "top_forecast_demand",
            results,
            {
                "forecast_units": "model_prediction",
                "recent_mean_28_units_per_day": "observed",
                "units_last_trading_day": "observed",
            },  # fmt: skip
            {"forecast_units": "units on target_date", "recent_mean_28_units_per_day": "units/day"},
            limits,
            model_version=self.model_version,
            horizon_trading_days=horizon_trading_days,
        )

    # -- 2. unusual sales -------------------------------------------------------------------
    def unusual_sales(self, recent_days: int = 5, limit: int = 10) -> dict:
        """Products whose recent sales deviate most from their own recent baseline (observed)."""
        if not 1 <= recent_days <= 20:
            raise ServiceError("recent_days must be 1-20")
        baseline_days = 56
        days = np.sort(self.daily["date"].unique())
        recent_from = days[-recent_days]
        base_from = days[-(recent_days + baseline_days)]
        d = self.daily[self.daily["date"] >= base_from]
        recent = d[d["date"] >= recent_from].groupby("stock_code")["units_sold"].sum()
        base = d[d["date"] < recent_from].groupby("stock_code")["units_sold"]
        stats = pd.DataFrame(
            {
                "base_mean": base.mean(),
                "base_std": base.std(ddof=0),
                "base_days": base.size(),
                "base_selling_days": base.apply(lambda s: int((s > 0).sum())),
            }  # fmt: skip
        ).join(recent.rename("recent_units"), how="inner")
        stats = stats[(stats["base_days"] >= 28) & (stats["base_selling_days"] >= 10)]
        expected = stats["base_mean"] * recent_days
        scale = np.maximum(stats["base_std"], 1.0) * np.sqrt(recent_days)
        stats["expected_units"] = expected
        stats["z_score"] = (stats["recent_units"] - expected) / scale
        stats["ratio_to_expected"] = stats["recent_units"] / expected.replace(0, np.nan)
        top = stats.reindex(stats["z_score"].abs().sort_values(ascending=False).index).head(limit)
        results = [
            {
                "stock_code": code,
                "product_description": self.names.get(code),
                "direction": "above baseline" if r.z_score > 0 else "below baseline",
                "recent_units": int(r.recent_units),
                "expected_units_from_baseline": _round(r.expected_units, 1),
                "ratio_to_expected": _round(r.ratio_to_expected, 2),
                "z_score": _round(r.z_score, 1),
            }
            for code, r in top.iterrows()
        ]
        return self._envelope(
            "unusual_sales",
            results,
            {
                "recent_units": "observed",
                "expected_units_from_baseline": "observed (baseline)",
                "z_score": "statistic on observed data",
            },  # fmt: skip
            {"recent_units": f"units over the last {recent_days} trading days"},
            [
                f"Recent window: {pd.Timestamp(recent_from).date()} to {self.origin.date()} "
                f"({recent_days} trading days); baseline: the {baseline_days} trading days before.",
                "z_score = (recent - expected) / (baseline daily std x sqrt(days)), std floored at "
                "1 unit; only products with >= 10 selling days in the baseline are scored. Large "
                "single wholesale orders can produce high scores.",
            ],
            method="robust deviation from each product's own 56-trading-day baseline",
        )

    # -- 3. stockout risk -------------------------------------------------------------------
    def stockout_risk(self, min_risk_level: str = "high", limit: int = 10) -> dict:
        order = {"low": 0, "medium": 1, "high": 2}
        if min_risk_level not in order:
            raise ServiceError("min_risk_level must be low, medium or high")
        r = self.a.risk[self.a.risk["risk_level"].map(order) >= order[min_risk_level]]
        r = r.sort_values(["stockout_probability_7d", "forecast_units_7d"], ascending=False)
        counts = self.a.risk["risk_level"].value_counts().to_dict()
        results = [self._risk_row(row) for row in r.head(limit).itertuples()]
        live = self._approval_statuses([x["suggestion_id"] for x in results if x["suggestion_id"]])
        for x in results:
            x["suggestion_status"] = live.get(x["suggestion_id"], x["suggestion_status"])
        return self._envelope(
            "stockout_risk",
            results,
            self._risk_value_types(),
            {
                "current_stock": "units (synthetic)",
                "forecast_units_7d": "units over next 7 trading days",
                "stockout_probability_7d": "probability 0-1",
            },  # fmt: skip
            self._risk_limits(),
            model_version=self.model_version,
            assumptions_version=str(self.a.risk["assumptions_version"].iloc[0]),
            products_by_risk_level={k: int(v) for k, v in counts.items()},
            products_matching=int(len(r)),
        )

    def _risk_row(self, row) -> dict:
        return {
            "stock_code": row.stock_code,
            "product_description": self.names.get(row.stock_code),
            "current_stock": int(row.current_stock),
            "forecast_units_7d": _round(row.forecast_units_7d, 1),
            "projected_stock_7d": _round(row.projected_stock_7d, 1),
            "stockout_probability_7d": _round(row.stockout_probability_7d, 3),
            "risk_level": row.risk_level,
            "expected_stockout_date": _date(row.expected_stockout_date),
            "lead_time_days": int(row.lead_time_days),
            "earliest_delivery_date": _date(row.earliest_delivery_date),
            "suggested_order_qty": int(row.suggested_order_qty),
            "suggestion_status": row.suggestion_status,
            "suggestion_id": row.suggestion_id or None,
        }

    def _approval_statuses(self, suggestion_ids: list[str]) -> dict[str, str]:
        """Live approval status from the approvals store (the risk file only knows 'pending')."""
        if not suggestion_ids or not self.a.approvals_path:
            return {}
        store = ApprovalStore(self.a.approvals_path)
        try:
            marks = ",".join("?" * len(suggestion_ids))
            rows = store.conn.execute(
                f"SELECT suggestion_id, status FROM suggestions WHERE suggestion_id IN ({marks})",
                suggestion_ids,
            )
            return dict(rows.fetchall())
        finally:
            store.close()

    @staticmethod
    def _risk_value_types() -> dict:
        return {
            "current_stock": "synthetic",
            "lead_time_days": "synthetic",
            "forecast_units_7d": "model_prediction",
            "projected_stock_7d": "model_prediction x synthetic stock",
            "stockout_probability_7d": "model_prediction x synthetic stock",
            "suggested_order_qty": "derived from model_prediction and synthetic assumptions",
        }

    def _risk_limits(self) -> list[str]:
        out = [
            f"Inventory is {SYNTHETIC_LABEL}. Stock levels and lead times are invented; risk "
            "levels illustrate the method, not the retailer's real position.",
            "Reorder suggestions require human approval; nothing is ordered automatically.",
        ]
        if self.a.risk_eval:
            ev = self.a.risk_eval
            out.append(
                f"Held-out calibration: mean predicted {ev['mean_predicted_probability']} vs "
                f"observed {ev['observed_stockout_rate']} stockout rate; probabilities run "
                "slightly low, so treat them as lower bounds."
            )
        return out

    # -- 4. demand change -------------------------------------------------------------------
    def explain_demand_change(
        self, stock_code: str, recent_days: int = 7, baseline_days: int = 28
    ) -> dict:
        code = self._require_product(stock_code)
        if not (1 <= recent_days <= 28 and 7 <= baseline_days <= 112):
            raise ServiceError("recent_days must be 1-28 and baseline_days 7-112")
        p = self.daily[self.daily["stock_code"] == code].sort_values("date")
        if len(p) < recent_days + baseline_days:
            raise ServiceError(
                f"{code} has {len(p)} trading days of history; need {recent_days + baseline_days}"
            )
        recent, base = p.tail(recent_days), p.iloc[-(recent_days + baseline_days) : -recent_days]

        def window(w: pd.DataFrame) -> dict:
            units, invoices = w["units_sold"].sum(), w["invoices"].sum()
            return {
                "from": _date(w["date"].min()),
                "to": _date(w["date"].max()),
                "trading_days": int(len(w)),
                "units_per_day": _round(units / len(w), 2),
                "orders_per_day": _round(invoices / len(w), 2),
                "units_per_order": _round(units / invoices, 2) if invoices else None,
                "customers_per_day": _round(w["customers"].sum() / len(w), 2),
                "avg_unit_price_gbp": _round(w["gross_revenue"].sum() / units, 2)
                if units
                else None,
                "units_returned": int(w["units_returned"].sum()),
                "selling_days": int((w["units_sold"] > 0).sum()),
                "largest_day_units": int(w["units_sold"].max()),
            }

        r, b = window(recent), window(base)

        def pct(new, old):
            return _round((new / old - 1) * 100, 1) if new is not None and old else None

        changes = {
            "units_per_day_pct": pct(r["units_per_day"], b["units_per_day"]),
            "orders_per_day_pct": pct(r["orders_per_day"], b["orders_per_day"]),
            "units_per_order_pct": pct(r["units_per_order"], b["units_per_order"]),
            "avg_unit_price_pct": pct(r["avg_unit_price_gbp"], b["avg_unit_price_gbp"]),
        }
        drivers = []
        if changes["orders_per_day_pct"] is not None and changes["units_per_order_pct"] is not None:
            main = (
                "number of orders"
                if abs(changes["orders_per_day_pct"]) >= abs(changes["units_per_order_pct"])
                else "size of orders"
            )
            drivers.append(f"larger relative change in {main}")
        if r["largest_day_units"] and r["units_per_day"]:
            share = r["largest_day_units"] / (r["units_per_day"] * r["trading_days"])
            if share >= 0.5:
                drivers.append(
                    f"one day holds {_round(share * 100, 0)}% of recent units "
                    "(likely a single large order)"
                )
        fc = self.a.forecasts[self.a.forecasts["stock_code"] == code].sort_values(
            "horizon_trading_days"
        )
        forecast = {
            "next_7_trading_days_units": _round(fc["forecast_units"].sum(), 1),
            "next_trading_day_units": _round(fc["forecast_units"].iloc[0], 1) if len(fc) else None,
            "method": fc["method"].iloc[0] if len(fc) else None,
        }
        return self._envelope(
            "explain_demand_change",
            {
                "stock_code": code,
                "product_description": self.names.get(code),
                "recent": r,
                "baseline": b,
                "change": changes,
                "drivers": drivers,
                "forecast": forecast,
            },  # fmt: skip
            {
                "recent": "observed",
                "baseline": "observed",
                "change": "computed from observed",
                "forecast": "model_prediction",
            },  # fmt: skip
            {
                "units_per_day": "units/trading day",
                "avg_unit_price_gbp": "GBP/unit",
                "change": "percent",
            },
            [
                "Drivers are descriptive (order count vs order size vs price), not causal; "
                "the data has no promotions, stock-outs or marketing information.",
                *self._forecast_limits[:1],
            ],
            model_version=self.model_version,
        )

    # -- 5. reorder suggestion --------------------------------------------------------------
    def reorder_suggestion(self, stock_code: str) -> dict:
        code = self._require_product(stock_code)
        rows = self.a.risk[self.a.risk["stock_code"] == code]
        if rows.empty:
            raise ServiceError(f"{code} has no risk assessment at {self.origin.date()}")
        row = rows.iloc[0]
        fc = self.a.forecasts[self.a.forecasts["stock_code"] == code].sort_values(
            "horizon_trading_days"
        )
        review = int(self.a.assumptions["policy"]["review_period_days"])
        lead = int(row["lead_time_days"])
        daily = fc["forecast_units"].to_numpy()
        cover = lead + review
        within = min(cover, len(daily))
        demand_cover = daily[:within].sum() + (daily.mean() * max(cover - len(daily), 0))
        status = row["suggestion_status"]
        decision = None
        if row["reorder_suggested"] and self.a.approvals_path:
            store = ApprovalStore(self.a.approvals_path)
            try:
                rec = store.get(row["suggestion_id"])
                status, decision = rec["status"], rec["history"][-1]
            except ApprovalError:
                pass
            finally:
                store.close()
        result = {
            **self._risk_row(rows.itertuples().__next__()),
            "reorder_suggested": bool(row["reorder_suggested"]),
            "reorder_reason": row["reorder_reason"] or "stock above reorder point and safety stock",
            "suggestion_id": row["suggestion_id"] or None,
            "approval_status": status,
            "last_decision": decision,
            "calculation": {
                "policy": "order-up-to: demand over (lead time + review period) "
                "+ safety stock - stock on hand",
                "lead_time_days": lead,
                "review_period_days": review,
                "forecast_demand_over_cover_units": _round(demand_cover, 1),
                "safety_stock_units": int(row["safety_stock"]),
                "reorder_point_units": int(row["reorder_point"]),
                "stock_on_hand_units": int(row["current_stock"]),
                "beyond_7_days": "extrapolated at the mean 7-day forecast rate"
                if cover > 7
                else "not needed",
            },
        }
        return self._envelope(
            "reorder_suggestion",
            result,
            {
                **self._risk_value_types(),
                "safety_stock_units": "synthetic",
                "reorder_point_units": "synthetic",
            },
            {"suggested_order_qty": "units", "lead_time_days": "trading days"},
            self._risk_limits(),
            model_version=self.model_version,
            assumptions_version=str(row["assumptions_version"]),
            service_level=self.a.assumptions["policy"]["service_level"],
        )

    # -- 6/7. retrieval ---------------------------------------------------------------------
    def product_info(self, query: str, limit: int = 5) -> dict:
        hits = self.a.knowledge.search_products(query, limit)
        summ = self.a.product_summary.set_index("stock_code")
        results = []
        for code, score in hits:
            s = summ.loc[code]
            results.append(
                {
                    "stock_code": code,
                    "product_description": s["product_description"],
                    "match_score": "exact stock code" if np.isinf(score) else _round(score, 2),
                    "first_sale_date": _date(s["first_sale_date"]),
                    "last_sale_date": _date(s["last_sale_date"]),
                    "total_units_sold": int(s["total_units"]),
                    "total_gross_revenue_gbp": _round(s["total_gross_revenue"], 2),
                    "selling_days": int(s["selling_days"]),
                    "demand_class": None if pd.isna(s["demand_class"]) else s["demand_class"],
                }
            )
        return self._envelope(
            "product_info",
            results,
            {"total_units_sold": "observed", "total_gross_revenue_gbp": "observed"},
            {
                "total_units_sold": "units, full history",
                "total_gross_revenue_gbp": "GBP, full history",
            },
            ["Product search is keyword retrieval over the catalogue descriptions (BM25)."],
        )

    def search_docs(self, query: str, limit: int = 4) -> dict:
        hits = self.a.knowledge.search_docs(query, limit)
        return self._envelope(
            "search_docs",
            [
                {
                    "source": d.source,
                    "section": d.section,
                    "text": d.text,
                    "match_score": _round(s, 2),
                }
                for d, s in hits
            ],
            {"text": "project documentation"},
            {},
            ["Retrieved from this project's documentation; cite the source and section."],
        )

    # -- dashboard views --------------------------------------------------------------------
    def product_overview(self, stock_code: str, history_days: int = 56) -> dict:
        """Observed daily history up to the origin plus the 7-day forecast, for one product."""
        code = self._require_product(stock_code)
        if not 7 <= history_days <= 365:
            raise ServiceError("history_days must be 7-365")
        hist = self.daily[self.daily["stock_code"] == code].sort_values("date").tail(history_days)
        fc = self.a.forecasts[self.a.forecasts["stock_code"] == code].sort_values(
            "horizon_trading_days"
        )
        return self._envelope(
            "product_overview",
            {
                "stock_code": code,
                "product_description": self.names.get(code),
                "history": [
                    {
                        "date": _date(r.date),
                        "units_sold": int(r.units_sold),
                        "units_returned": int(r.units_returned),
                    }  # fmt: skip
                    for r in hist.itertuples()
                ],
                "forecast": [
                    {
                        "date": _date(r.target_date),
                        "horizon_trading_days": int(r.horizon_trading_days),
                        "forecast_units": _round(r.forecast_units, 1),
                        "method": r.method,
                    }  # fmt: skip
                    for r in fc.itertuples()
                ],
            },
            {"history": "observed", "forecast": "model_prediction"},
            {"units_sold": "units/day", "forecast_units": "units/day"},
            self._forecast_limits,
            model_version=self.model_version,
        )

    @cached_property
    def _totals(self) -> pd.DataFrame:
        return (
            self.daily.groupby("date")
            .agg(
                units_sold=("units_sold", "sum"),
                gross_revenue=("gross_revenue", "sum"),
                units_returned=("units_returned", "sum"),
            )  # fmt: skip
            .reset_index()
        )

    def daily_totals(self, days: int = 90) -> dict:
        """Store-wide observed daily totals for the last ``days`` trading days."""
        if not 7 <= days <= 700:
            raise ServiceError("days must be 7-700")
        t = self._totals.tail(days)
        return self._envelope(
            "daily_totals",
            [
                {
                    "date": _date(r.date),
                    "units_sold": int(r.units_sold),
                    "gross_revenue_gbp": _round(r.gross_revenue, 2),
                    "units_returned": int(r.units_returned),
                }  # fmt: skip
                for r in t.itertuples()
            ],
            {"units_sold": "observed", "gross_revenue_gbp": "observed"},
            {"units_sold": "units/day", "gross_revenue_gbp": "GBP/day"},
            [
                "Sale lines only; returns are shown separately "
                "(see the README's return-handling policy)."
            ],
        )

    def status(self) -> dict:
        return {
            "forecast_origin": str(self.origin.date()),
            "model_version": self.model_version,
            "assumptions_version": str(self.a.risk["assumptions_version"].iloc[0]),
            "products": int(self.a.forecasts["stock_code"].nunique()),
            "data_note": SIMULATION_NOTE,
            "inventory_note": SYNTHETIC_LABEL,
        }
