"""Projected inventory, stockout risk and reorder suggestions.

Three kinds of numbers meet here, and every output column says which kind it is:

* **observed**: sales history from UCI Online Retail II;
* **model_prediction**: demand forecasts from ``src.forecasting.predict``;
* **synthetic**: stock, lead time, safety stock and reorder point from
  ``synthetic_inventory`` (invented; not real retailer inventory).

Method, for each product at the forecast origin (end of a trading day):

1. Cumulative forecast demand F_h for the next h = 1..7 trading days.
2. Projected stock = current stock - F_h. No deliveries are assumed inside the horizon,
   because the synthetic data has no open purchase orders.
3. P(stockout by day h) = P(actual cumulative demand > current stock). The demand
   distribution is the forecast plus the **empirical** additive error of the model's
   held-out backtest (phase 4 test window), binned by horizon and forecast size. No
   normality assumption is made, which matters for lumpy wholesale demand.
4. Reorder suggestion (order-up-to). It applies when stock <= the synthetic reorder point,
   or when expected stock on the delivery day would fall below safety stock. Quantity =
   expected demand over lead time + review period + safety stock - current stock.
   Demand beyond the 7 forecast days is extrapolated at the mean forecast rate.
5. Every suggestion is ``pending_approval``. Nothing is ordered. Approval is a separate,
   human, audited step (``src.inventory.approvals``).

Usage::

    python -m src.inventory.risk_engine                    # origin = last full trading day
    python -m src.inventory.risk_engine --evaluate         # calibration check on held-out data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.forecasting import features as F
from src.forecasting.train import BACKTEST_PARQUET
from src.inventory.synthetic_inventory import (
    INVENTORY_DIR,
    SYNTHETIC_LABEL,
    assumptions_fingerprint,
    generate,
    load_assumptions,
    snapshot,
)
from src.processing.aggregate_sales import PRODUCT_DAILY_PARQUET, PRODUCT_SUMMARY_PARQUET
from src.processing.trading_calendar import expected_trading_days

HORIZON = 7
FORECAST_BINS = np.array([0.0, 0.5, 2, 5, 10, 25, 50, 100, 250, np.inf])
ZERO_FORECAST = 1e-6
MIN_BIN_CASES = 30
REPORT_MD = config.REPORTS_DIR / "inventory_risk_evaluation.md"
REPORT_JSON = config.REPORTS_DIR / "inventory_risk_evaluation.json"


# -- uncertainty -------------------------------------------------------------------------


def cumulative_backtest(
    backtest: pd.DataFrame, forecast_col: str = "forecast_pipeline"
) -> pd.DataFrame:
    """Per (product, origin, h): cumulative actual and forecast over horizons 1..h.

    Only origins with all 7 horizons are used, and products launched after the origin are
    excluded (they have no forecast at the origin).
    """
    bt = backtest[backtest["segment"] != "launched_after_origin"]
    full = bt.groupby(["stock_code", "origin_date"])["horizon"].transform("size") == HORIZON
    bt = bt[full].sort_values(["stock_code", "origin_date", "horizon"])
    g = bt.groupby(["stock_code", "origin_date"])
    return pd.DataFrame(
        {
            "stock_code": bt["stock_code"].to_numpy(),
            "origin_date": bt["origin_date"].to_numpy(),
            "h": bt["horizon"].to_numpy(),
            "cum_actual": g["actual_units"].cumsum().to_numpy(),
            "cum_forecast": g[forecast_col].cumsum().to_numpy(),
        }
    )


@dataclass
class ErrorModel:
    """Empirical distribution of actual demand relative to the forecast.

    Per (horizon, forecast-size bin) it keeps the sorted ratios r = actual / forecast, so
    demand = forecast x r. Errors on counts grow with the level, and the ratio keeps the
    real share of zero-demand outcomes, which an additive error mixed across a bin
    distorts near zero. For forecasts of (almost) exactly 0 it keeps actual demand itself.
    """

    ratios: dict[tuple[int, int], np.ndarray]
    zero_forecast_actuals: dict[int, np.ndarray]
    counts: dict[tuple[int, int], int]
    fitted_on: list[str]

    @classmethod
    def fit(cls, cum: pd.DataFrame) -> ErrorModel:
        f = cum["cum_forecast"].to_numpy()
        a = cum["cum_actual"].to_numpy()
        h_all = cum["h"].to_numpy()
        b = np.digitize(f, FORECAST_BINS[1:-1])
        positive = f > ZERO_FORECAST
        ratios, zeros, counts = {}, {}, {}
        for h in range(1, HORIZON + 1):
            at_h = h_all == h
            zeros[h] = np.sort(a[at_h & ~positive])
            for k in range(len(FORECAST_BINS) - 1):
                mask = at_h & positive & (b == k)
                if mask.sum() >= MIN_BIN_CASES:
                    ratios[(h, k)] = np.sort(a[mask] / f[mask])
                    counts[(h, k)] = int(mask.sum())
        days = pd.to_datetime(cum["origin_date"])
        return cls(ratios, zeros, counts, [str(days.min().date()), str(days.max().date())])

    def prob_exceeds(self, h: int, forecast: np.ndarray, stock: np.ndarray) -> np.ndarray:
        """P(actual cumulative demand over h days > stock), per product."""
        forecast = np.asarray(forecast, float)
        stock = np.asarray(stock, float)
        out = np.empty(len(forecast))
        zero = forecast <= ZERO_FORECAST
        if zero.any():
            sample = self.zero_forecast_actuals[h]
            out[zero] = _survival(sample, stock[zero]) if len(sample) else 0.0
        bins = np.digitize(forecast, FORECAST_BINS[1:-1])
        for k in np.unique(bins[~zero]):
            mask = ~zero & (bins == k)
            sample = self.ratios[self._nearest((h, int(k)))]
            out[mask] = _survival(sample, stock[mask] / forecast[mask])
        return out

    def _nearest(self, key: tuple[int, int]) -> tuple[int, int]:
        if key in self.ratios:
            return key
        h, k = key
        same_h = [kk for (hh, kk) in self.ratios if hh == h]
        return (h, min(same_h, key=lambda kk: abs(kk - k)))


def _survival(sorted_sample: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Empirical P(X > x): exact, including the point mass at zero demand."""
    return 1.0 - np.searchsorted(sorted_sample, x, side="right") / len(sorted_sample)


# -- risk --------------------------------------------------------------------------------


def assess(
    forecasts: pd.DataFrame, inventory: pd.DataFrame, errors: ErrorModel, assumptions: dict
) -> pd.DataFrame:
    """Risk and reorder suggestion per product. ``forecasts`` is predict.forecast() output."""
    origin = pd.Timestamp(forecasts["forecast_origin"].iloc[0])
    wide = forecasts.pivot(
        index="stock_code", columns="horizon_trading_days", values="forecast_units"
    )
    wide = wide.reindex(columns=range(1, HORIZON + 1)).fillna(0.0)
    inv = inventory.set_index("stock_code").reindex(wide.index)
    missing = inv["current_stock"].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} forecast products have no synthetic inventory row")

    stock = inv["current_stock"].to_numpy(float)
    cum = wide.cumsum(axis=1).to_numpy()
    probs = np.column_stack(
        [errors.prob_exceeds(h, cum[:, h - 1], stock) for h in range(1, HORIZON + 1)]
    )
    probs = np.maximum.accumulate(probs, axis=1)  # running out by day h implies by day h+1
    target_days = expected_trading_days(origin, HORIZON)

    out = pd.DataFrame(index=wide.index)
    out["forecast_origin"] = origin
    out["current_stock"] = inv["current_stock"].astype(int)
    out["lead_time_days"] = inv["lead_time_days"].astype(int)
    out["safety_stock"] = inv["safety_stock"].astype(int)
    out["reorder_point"] = inv["reorder_point"].astype(int)
    out["forecast_units_next_day"] = np.round(cum[:, 0], 2)
    out["forecast_units_7d"] = np.round(cum[:, -1], 2)
    out["projected_stock_7d"] = np.round(stock - cum[:, -1], 2)
    first_short = np.argmax(cum > stock[:, None], axis=1)
    has_short = (cum > stock[:, None]).any(axis=1)
    out["expected_stockout_date"] = [
        target_days[i] if ok else pd.NaT for i, ok in zip(first_short, has_short, strict=True)
    ]
    out["stockout_probability_next_day"] = np.round(probs[:, 0], 3)
    out["stockout_probability_7d"] = np.round(probs[:, -1], 3)
    risk = assumptions["risk"]
    out["risk_level"] = np.select(
        [
            out["stockout_probability_7d"] >= risk["high"],
            out["stockout_probability_7d"] >= risk["medium"],
        ],
        ["high", "medium"],
        default="low",
    )

    # Reorder suggestion: order-up-to over lead time + review period.
    lead = out["lead_time_days"].to_numpy()
    review = assumptions["policy"]["review_period_days"]
    daily_rate = cum[:, -1] / HORIZON
    cover = lead + review
    within = np.minimum(cover, HORIZON)
    demand_cover = cum[np.arange(len(cum)), within - 1] + daily_rate * np.maximum(
        cover - HORIZON, 0
    )
    demand_lead = np.where(
        lead <= HORIZON, cum[np.arange(len(cum)), np.minimum(lead, HORIZON) - 1], daily_rate * lead
    )
    stock_at_delivery = stock - demand_lead
    needs = (stock <= out["reorder_point"].to_numpy()) | (
        stock_at_delivery < out["safety_stock"].to_numpy()
    )
    qty = np.ceil(demand_cover + out["safety_stock"].to_numpy() - stock)
    out["reorder_suggested"] = needs & (qty > 0)
    out["suggested_order_qty"] = np.where(out["reorder_suggested"], qty, 0).astype(int)
    arrival = {lt: expected_trading_days(origin, int(lt))[-1] for lt in np.unique(lead)}
    out["earliest_delivery_date"] = [arrival[lt] for lt in lead]
    out["stockout_before_delivery_likely"] = out["expected_stockout_date"].notna() & (
        out["expected_stockout_date"] < out["earliest_delivery_date"]
    )
    out["reorder_reason"] = np.select(
        [~out["reorder_suggested"], stock <= out["reorder_point"].to_numpy()],
        ["", "stock at or below reorder point"],
        default="expected stock at delivery below safety stock",
    )
    out["suggestion_status"] = np.where(out["reorder_suggested"], "pending_approval", "none")
    out["suggestion_id"] = [
        _suggestion_id(origin, code, assumptions) if s else ""
        for code, s in zip(out.index, out["reorder_suggested"], strict=True)
    ]

    meta = forecasts.drop_duplicates("stock_code").set_index("stock_code")
    out["forecast_method"] = meta["method"].reindex(out.index)
    out["model_version"] = meta["model_version"].reindex(out.index)
    out["assumptions_version"] = assumptions_fingerprint(assumptions)
    out["inventory_data_label"] = SYNTHETIC_LABEL
    out["value_types"] = (
        "current_stock/lead_time/safety_stock/reorder_point: synthetic; "
        "forecast_*: model_prediction; stockout_probability_*: model_prediction x synthetic stock"
    )
    return out.reset_index().sort_values(
        ["stockout_probability_7d", "forecast_units_7d"], ascending=False, ignore_index=True
    )


def _suggestion_id(origin: pd.Timestamp, stock_code: str, assumptions: dict) -> str:
    key = f"{origin.date()}|{stock_code}|{assumptions_fingerprint(assumptions)}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# -- calibration check -------------------------------------------------------------------


def evaluate(product_daily: pd.DataFrame, backtest: pd.DataFrame, assumptions: dict) -> dict:
    """Are the stockout probabilities calibrated? Time-split, on held-out forecasts.

    The error model is fitted on the first half of the backtest origins and evaluated on the
    second half. At each evaluation origin a synthetic stock is generated for every product,
    as in production, and the 7-day stockout probability is compared with whether actual
    7-day demand exceeded that stock.
    """
    cum = cumulative_backtest(backtest)
    origins = np.sort(cum["origin_date"].unique())
    split = origins[len(origins) // 2]
    errors = ErrorModel.fit(cum[cum["origin_date"] < split])
    test = cum[(cum["origin_date"] >= split) & (cum["h"] == HORIZON)]

    panel = F.build_panel(product_daily)
    window = assumptions["policy"]["demand_window_days"]
    mean = panel.units.rolling(window, min_periods=1).mean()
    std = panel.units.rolling(window, min_periods=1).std(ddof=0)
    rows = []
    for origin, grp in test.groupby("origin_date"):
        codes = grp["stock_code"].to_numpy()
        t = panel.index_of(origin)
        snap = generate(
            codes,
            mean.iloc[t].reindex(codes).to_numpy(),
            std.iloc[t].reindex(codes).to_numpy(),
            pd.Timestamp(origin),
            assumptions,
        )
        stock = snap["current_stock"].to_numpy(float)
        p = errors.prob_exceeds(HORIZON, grp["cum_forecast"].to_numpy(), stock)
        rows.append(pd.DataFrame({"p": p, "outcome": grp["cum_actual"].to_numpy() > stock}))
    ev = pd.concat(rows, ignore_index=True)
    base_rate = ev["outcome"].mean()
    brier = float(((ev["p"] - ev["outcome"]) ** 2).mean())
    brier_base = float(((base_rate - ev["outcome"]) ** 2).mean())
    ev["bin"] = pd.cut(
        ev["p"], [0, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0], include_lowest=True
    )
    reliability = (
        ev.groupby("bin", observed=True)
        .agg(cases=("p", "size"), mean_predicted=("p", "mean"), observed_rate=("outcome", "mean"))
        .reset_index()
    )
    reliability["bin"] = reliability["bin"].astype(str)
    return {
        "error_model_fitted_on_origins": errors.fitted_on,
        "evaluated_origins": [
            str(pd.Timestamp(split).date()),
            str(pd.Timestamp(origins[-1]).date()),
        ],
        "cases": int(len(ev)),
        "observed_stockout_rate": round(float(base_rate), 4),
        "mean_predicted_probability": round(float(ev["p"].mean()), 4),
        "brier_score": round(brier, 4),
        "brier_score_base_rate": round(brier_base, 4),
        "brier_skill_score": round(1 - brier / brier_base, 4),
        "reliability": reliability.round(4).to_dict(orient="records"),
    }


def render_evaluation(ev: dict) -> str:
    fit, held = ev["error_model_fitted_on_origins"], ev["evaluated_origins"]
    lines = [
        "# Stockout-probability calibration\n",
        "Generated by `python -m src.inventory.risk_engine --evaluate`.",
        "Stock levels are **synthetic** (`config/inventory_assumptions.toml`). Demand is the",
        "**actual** observed demand. Forecasts are the phase 4 held-out backtest.\n",
        f"* Error model fitted on backtest origins {fit[0]} to {fit[1]}; evaluated on origins "
        f"from {held[0]} to {held[1]} ({ev['cases']:,} product-origins, 7-day horizon).",
        f"* Observed stockout rate (actual 7-day demand > synthetic stock): "
        f"{ev['observed_stockout_rate']:.1%}; mean predicted probability: "
        f"{ev['mean_predicted_probability']:.1%}.",
        f"* Brier score {ev['brier_score']:.4f} vs {ev['brier_score_base_rate']:.4f} for always "
        f"predicting the base rate (skill score {ev['brier_skill_score']:.3f}; 0 = no better than "
        "the base rate, 1 = perfect).\n",
        "| predicted probability | cases | mean predicted | observed rate |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['bin']} | {r['cases']:,} | {r['mean_predicted']:.3f} | {r['observed_rate']:.3f} |"
        for r in ev["reliability"]
    ]
    return "\n".join(lines) + "\n"


# -- CLI ---------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stockout risk from forecasts and SYNTHETIC stock")
    parser.add_argument("--origin", help="forecast origin (default: last full trading day)")
    parser.add_argument("--evaluate", action="store_true", help="run the calibration check")
    args = parser.parse_args(argv)
    assumptions = load_assumptions()
    daily = pd.read_parquet(PRODUCT_DAILY_PARQUET)
    backtest = pd.read_parquet(BACKTEST_PARQUET)

    if args.evaluate:
        ev = evaluate(daily, backtest, assumptions)
        REPORT_JSON.write_text(json.dumps(ev, indent=2) + "\n")
        REPORT_MD.write_text(render_evaluation(ev))
        print(render_evaluation(ev))
        return 0

    from src.forecasting.predict import forecast, load_model

    booster, meta = load_model()
    fc = forecast(daily, booster, meta, args.origin)
    origin = fc["forecast_origin"].iloc[0]
    inventory = snapshot(daily, origin, assumptions)
    risk = assess(fc, inventory, ErrorModel.fit(cumulative_backtest(backtest)), assumptions)
    names = pd.read_parquet(PRODUCT_SUMMARY_PARQUET, columns=["stock_code", "product_description"])
    risk = risk.merge(names, on="stock_code", how="left")

    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    out = INVENTORY_DIR / f"stockout_risk_{origin:%Y-%m-%d}.parquet"
    risk.to_parquet(out, index=False)
    print(SYNTHETIC_LABEL)
    print(
        f"origin {origin.date()} | model {meta['model_version']} | "
        f"assumptions {assumptions_fingerprint(assumptions)}"
    )
    print(risk["risk_level"].value_counts().to_string())
    print(f"reorder suggestions pending human approval: {int(risk['reorder_suggested'].sum()):,}")
    cols = ["stock_code", "product_description", "current_stock", "forecast_units_7d",
            "stockout_probability_7d", "expected_stockout_date", "suggested_order_qty"]  # fmt: skip
    print(risk[cols].head(10).to_string(index=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
