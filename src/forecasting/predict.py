"""Reproducible demand forecasts from the persisted model.

Given a forecast origin (end of a trading day), this produces forecasts for the next
1-7 *expected trading days* from the rule-based calendar, for every product on sale at
the origin. Only data up to the origin is used: the input is truncated before features
are built.

Each row states how it was produced (``method``) and carries the evidence the copilot
needs: recent baseline, history status and model version. These are model predictions,
not observed sales.

Usage::

    python -m src.forecasting.predict                     # origin = last full trading day
    python -m src.forecasting.predict --origin 2011-11-15 --top 15
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.forecasting import features as F
from src.forecasting.train import META_PATH, MODEL_PATH
from src.processing.aggregate_sales import PRODUCT_DAILY_PARQUET, PRODUCT_SUMMARY_PARQUET
from src.processing.trading_calendar import expected_trading_days

logger = logging.getLogger(__name__)

MODEL_METHOD = "lightgbm"
FALLBACK_METHOD = "fallback_mean_28_excl_launch"


class ForecastError(ValueError):
    pass


def load_model(model_path: Path = MODEL_PATH, meta_path: Path = META_PATH):
    if not model_path.exists():
        raise ForecastError(f"No model at {model_path}; run `python -m src.forecasting.train`")
    booster = lgb.Booster(model_file=str(model_path))
    meta = json.loads(meta_path.read_text())
    if booster.feature_name() != meta["features"]:
        raise ForecastError("Model features do not match its metadata; retrain the model")
    return booster, meta


def forecast(
    product_daily: pd.DataFrame,
    booster: lgb.Booster,
    meta: dict,
    origin: str | pd.Timestamp | None = None,
    horizons: int = 7,
) -> pd.DataFrame:
    """Forecast ``horizons`` trading days ahead of ``origin`` for every product on sale."""
    if not 1 <= horizons <= max(F.HORIZONS):
        raise ForecastError(f"horizons must be between 1 and {max(F.HORIZONS)}")
    full = product_daily.loc[~product_daily["is_partial_day"]]
    origin = pd.Timestamp(origin) if origin is not None else full["date"].max()
    if origin not in set(full["date"]):
        raise ForecastError(f"{origin.date()} is not a full trading day in the data")

    # Truncate first: nothing after the origin can reach the features.
    panel = F.build_panel(product_daily.loc[product_daily["date"] <= origin])
    t = panel.index_of(origin)
    on_sale = np.nonzero(panel.units.notna().to_numpy()[t])[0]
    targets = expected_trading_days(origin, horizons)

    n = len(on_sale)
    t_idx = np.full(n * horizons, t)
    p_idx = np.tile(on_sale, horizons)
    h = np.repeat(np.arange(1, horizons + 1), n)
    target_dates = pd.DatetimeIndex(np.repeat(np.array(targets, dtype="datetime64[ns]"), n))

    eligible = panel.eligible[t_idx, p_idx]
    fallback = F.fallback_forecast(panel, t_idx, p_idx)
    yhat = fallback.copy()
    if eligible.any():
        x = F.feature_frame(
            panel, t_idx[eligible], p_idx[eligible], h[eligible], target_dates[eligible]
        )
        yhat[eligible] = booster.predict(x)

    origin_feat = panel.origin
    out = pd.DataFrame(
        {
            "stock_code": panel.codes[p_idx],
            "forecast_origin": origin,
            "target_date": target_dates,
            "horizon_trading_days": h,
            "forecast_units": np.round(np.clip(yhat, 0, None), 3),
            "method": np.where(eligible, MODEL_METHOD, FALLBACK_METHOD),
            "history_status": np.where(eligible, "sufficient", "insufficient"),
            "recent_mean_28": np.round(origin_feat["mean_28"][t_idx, p_idx].astype(float), 3),
            "units_last_trading_day": origin_feat["lag_1"][t_idx, p_idx].astype(float),
            "trading_days_since_last_sale": origin_feat["days_since_last_sale"][
                t_idx, p_idx
            ].astype(float),
            "selling_days_to_date": origin_feat["selling_days_to_date"][t_idx, p_idx].astype(int),
            "model_version": meta["model_version"],
        }
    )
    out["unit"] = "units/day"
    out["value_type"] = "model_prediction"
    return out.sort_values(
        ["horizon_trading_days", "forecast_units"], ascending=[True, False]
    ).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forecast product demand")
    parser.add_argument("--origin", help="forecast origin date (default: last full trading day)")
    parser.add_argument("--horizons", type=int, default=7)
    parser.add_argument("--top", type=int, default=10, help="rows to print for the next day")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    booster, meta = load_model()
    daily = pd.read_parquet(PRODUCT_DAILY_PARQUET)
    fc = forecast(daily, booster, meta, args.origin, args.horizons)

    origin = fc["forecast_origin"].iloc[0]
    out = config.PROCESSED_DIR / f"forecasts_{origin:%Y-%m-%d}.parquet"
    fc.to_parquet(out, index=False)

    names = pd.read_parquet(PRODUCT_SUMMARY_PARQUET, columns=["stock_code", "product_description"])
    next_day = fc[fc["horizon_trading_days"] == 1].merge(names, on="stock_code", how="left")
    print(
        f"Model {meta['model_version']} | origin {origin.date()} (end of day) | "
        f"next trading day {next_day['target_date'].iloc[0].date()}"
    )
    print(
        f"products forecast: {fc['stock_code'].nunique():,} "
        f"({(next_day['method'] == MODEL_METHOD).sum():,} model, "
        f"{(next_day['method'] == FALLBACK_METHOD).sum():,} fallback)"
    )
    cols = ["stock_code", "product_description", "forecast_units", "recent_mean_28", "method"]
    print(next_day[cols].head(args.top).to_string(index=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
