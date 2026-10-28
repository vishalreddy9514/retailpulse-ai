# Monitoring report

Forecast origin 2011-11-15 (held-out backtest (test model)); Historical UCI Online Retail II replay; inventory is synthetic.

| check | status |
|---|---|
| forecast_accuracy | ok |
| input_drift | degraded |
| freshness | ok |

## Forecast accuracy against later actuals (4,023 model-forecast products, 7/7 target days with actuals)

| window | WAPE now | held-out test WAPE | MAE | bias |
|---|---:|---:|---:|---:|
| daily | 0.858 | 0.989 | 6.08 | -0.202 |
| 7d | 0.515 | 0.593 | 25.51 | -0.202 |

## Input drift (recent 2011-10-14 to 2011-11-15 vs reference 2011-06-05 to 2011-10-13)

| feature | PSI | reference mean | recent mean |
|---|---:|---:|---:|
| store_daily_units | 3.227 | 17910.0 | 26408.1 |
| zero_demand_share | 3.119 | 0.804 | 0.7463 |

PSI above 0.25 is flagged. Seasonal peaks also raise PSI, so a flag means 'review', not 'broken'.
