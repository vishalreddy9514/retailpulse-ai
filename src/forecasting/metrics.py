"""Forecast error metrics. Units are product units per trading day."""

from __future__ import annotations

import numpy as np


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|y - yhat| / sum|y|.

    Unlike MAPE it stays defined on zero-demand days. NaN when there is no demand at all.
    """
    y = np.asarray(y, float)
    denom = np.abs(y).sum()
    return float(np.abs(y - np.asarray(yhat, float)).sum() / denom) if denom > 0 else float("nan")


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    """Signed total error relative to total demand. Positive means over-forecasting."""
    y = np.asarray(y, float)
    denom = np.abs(y).sum()
    return float((np.asarray(yhat, float) - y).sum() / denom) if denom > 0 else float("nan")


def all_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    return {
        "rows": int(len(y)),
        "mae": round(mae(y, yhat), 4),
        "rmse": round(rmse(y, yhat), 4),
        "wape": round(wape(y, yhat), 4),
        "bias": round(bias(y, yhat), 4),
    }
