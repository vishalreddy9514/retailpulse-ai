# RetailPulse AI — Retail Demand Forecasting Copilot

RetailPulse AI forecasts product-level daily demand for an online retailer. It flags stockout risk against an inventory model and answers questions through a GenAI copilot whose numbers come from backend services, not from the LLM.

> **Important: what is real and what is simulated**
>
> * **Sales history is real.** It is the public [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset (Dec 2009 – Dec 2011).
> * **"Real-time" is simulated.** A Kafka producer replays those *historical* transactions in timestamp order. The application does **not** receive live transactions from the retailer.
> * **Inventory is synthetic.** The dataset has no stock levels or supplier lead times. The inventory figures are generated, clearly labelled assumptions, not the retailer's real stock.

## Status

Built phase by phase. Each phase lands on its own branch and pull request.

| # | Phase | Status |
|---|---|---|
| 1 | Download and inspect UCI Online Retail II | done |
| 2 | Data cleaning and quality checks | done |
| 3 | Daily product-level demand aggregation | done |
| 4 | Forecasting models and evaluation | done |
| 5 | Kafka historical event replay (producer) | done |
| 6 | Streaming processing pipeline (consumer) | done |
| 7 | Synthetic inventory and stockout risk | done |
| 8 | AI copilot (LangGraph tools and RAG) | done |
| 9 | FastAPI backend and Streamlit dashboard | planned |
| 10 | Docker, CI/CD, logging, monitoring, docs | planned |

## Quick start

Requires Python 3.11+. Reading the Excel workbook takes about 80 seconds, and the result is cached as Parquet.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m src.ingestion.uci_loader          # download, verify SHA-256, merge sheets -> data/interim/
python -m src.processing.clean_transactions # label every line, run quality checks -> data/processed/
python -m src.processing.aggregate_sales    # product x trading-day demand, reconciled -> data/processed/
python -m src.forecasting.train             # ~15 min on 4 cores: select, evaluate, save -> models/, reports/
python -m src.forecasting.predict           # next 1-7 trading days for every product (~20 s)
python -m src.ingestion.explore             # regenerate reports/data_profile.md

pytest                                      # unit tests (no download, no broker needed)
ruff check src tests
```

Expected output of `python -m src.processing.clean_transactions` (exit code 0):

```
rows in: 1,044,848  rows out: 1,033,036  exact duplicates dropped: 11,812
  sale                     1,002,104
  cancellation_return         16,678
  order_reversal_sale          1,236
  order_reversal_return        1,236
  stock_adjustment             3,391
  zero_price                   2,571
  non_product                  5,814
  bad_debt_adjustment              6
  invalid                          0
order reversal pairs: 1,236
  [PASS] event_id_unique
  ... (9 checks, all PASS)
```

The command exits non-zero if any quality check fails.

Expected output of `python -m src.processing.aggregate_sales` (exit code 0):

```
products: 4,719  trading days: 604 (2009-12-01 -> 2011-12-09)
product-day rows: 2,397,712  zero-demand share: 77.9%
partial days: ['2011-12-09']
calendar rule agreement: 99.86%
history status: {'sufficient': 4087, 'insufficient': 632}
demand classes: {'lumpy': 3731, 'intermittent': 552, 'erratic': 312, '<NA>': 121, 'smooth': 3}
  [PASS] units_sold_match_sale_lines
  ... (7 reconciliation checks, all PASS)
```

## Project layout

```
src/
  config.py                         paths, official URL, pinned checksum
  ingestion/uci_loader.py           download, verify, read sheets, resolve sheet overlap
  ingestion/explore.py              data profile -> reports/data_profile.md
  processing/clean_transactions.py  normalisation, de-duplication, line labelling, checks
  processing/trading_calendar.py    rule-based trading calendar (known in advance)
  processing/aggregate_sales.py     product x trading-day demand, totals, product summary
  forecasting/features.py           leakage-safe origin features, baselines, fallback
  forecasting/metrics.py            MAE, RMSE, WAPE, bias
  forecasting/train.py              chronological splits, config selection, evaluation, persistence
  forecasting/predict.py            reproducible inference from the saved model
  streaming/schema.py               event contract (pydantic), dead-letter payload
  streaming/replay.py               broker-independent replay engine: order, pacing, resume
  streaming/kafka_producer.py       transactional Kafka sink + CLI
  streaming/stream_processor.py     idempotent stateful processing into SQLite (events, daily demand, offsets)
  streaming/kafka_consumer.py       read_committed consumer loop + CLI
  streaming/reconcile.py            stream vs batch daily-demand comparison
  inventory/synthetic_inventory.py  SYNTHETIC stock, lead time, safety stock, reorder point
  inventory/risk_engine.py          projected stock, stockout probability, reorder suggestions
  inventory/approvals.py            human approval workflow with append-only audit log
  copilot/services.py               backend services: every number the copilot reports, with provenance
  copilot/tools.py                  strict tool schemas + pydantic validation
  copilot/knowledge.py              BM25 retrieval over the product catalogue and project docs (RAG)
  copilot/graph.py                  LangGraph loop: Claude <-> tools -> number verification
  copilot/verify.py                 checks every number in an answer against tool results
  copilot/offline.py                deterministic no-LLM mode (tests, CI, no API key)
  copilot/cli.py, evaluate.py       command line and evaluation harness
config/inventory_assumptions.toml   every synthetic-inventory assumption, in one documented file
schemas/transaction_event.v1.json   JSON Schema of the event contract (checked by a test)
docker-compose.yml                  single-node Kafka (KRaft) for local development
tests/integration/                  tests against a real broker (opt-in)
tests/unit/                         pytest suites on small hand-written fixtures
reports/                            generated, committed: data profile, cleaning, aggregation, forecast evaluation
models/                             trained model + metadata (git-ignored, rebuilt by train.py)
data/                               downloaded and derived data (git-ignored)
```

## Dataset

| | |
|---|---|
| Name | Online Retail II |
| Source | UCI Machine Learning Repository, <https://archive.ics.uci.edu/dataset/502/online+retail+ii> |
| Citation | Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5CG6D> |
| Licence | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |
| File | `online_retail_II.xlsx` inside the official zip, SHA-256 `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980` |
| Coverage | 2009-12-01 07:45 to 2011-12-09 12:50, a UK-based online gift retailer, many customers are wholesalers |

The raw data is downloaded by the pipeline and is never committed to this repository. The loader pins the workbook's checksum. If UCI ever publishes a different file, ingestion stops instead of silently ingesting it.

### Source schema

The workbook has two sheets, `Year 2009-2010` (525,461 rows) and `Year 2010-2011` (541,910 rows), with identical columns:

| Source column | Pipeline column | Type | Notes |
|---|---|---|---|
| `Invoice` | `invoice` | string | 6 digits. Prefix `C` = cancellation/return, prefix `A` = bad-debt adjustment |
| `StockCode` | `stock_code` | string | 5 digits plus optional letters for products. Also service codes (`POST`, `M`, `D`, ...) |
| `Description` | `description` | string | Free text. 0.41% missing, some stored as numbers, staff notes on adjustments |
| `Quantity` | `quantity` | Int64 | Negative for cancellations, returns and stock write-offs |
| `InvoiceDate` | `invoice_date` | datetime | Minute resolution, local time, no timezone in source |
| `Price` | `unit_price` | float64 | GBP per unit |
| `Customer ID` | `customer_id` | string | Stored as a float in the source. 22.5% missing |
| `Country` | `country` | string | 43 countries, 94% of lines are United Kingdom |

The loader also adds `source_sheet` and `source_row` (the Excel row number) for lineage.

## Cleaning decisions

All counts below are measured on the real file. The full report is in [`reports/data_profile.md`](reports/data_profile.md).

1. **Sheet overlap.** 2010-12-01 08:26 to 2010-12-09 20:01 appears in *both* sheets: 22,523 rows each, verified as an exact multiset match. The window is dropped from the first sheet once. It is resolved by time window, not by row de-duplication, so repeated lines inside a sheet are not collapsed by accident. If the window ever stops matching exactly, ingestion raises an error. Result: **1,044,848** raw rows.
2. **Text normalisation.** Codes are trimmed and upper-cased (`"47503J "` → `47503J`, `gift_0001_20` → `GIFT_0001_20`), and descriptions have whitespace trimmed and collapsed.
3. **Exact duplicates.** **11,812** lines are identical in every field (invoice, product, description, quantity, timestamp, price, customer, country). The source has no line number, so a double-scan can't be told apart from a genuine repeat. They are dropped by default and counted (`--keep-exact-duplicates` keeps them).
4. **Every line is labelled, not silently dropped.** Precedence, first match wins:

   | `line_type` | Rule | Lines | In demand? |
   |---|---|---:|---|
   | `invalid` | missing key field or malformed invoice number | 0 | no |
   | `bad_debt_adjustment` | invoice starts with `A` | 6 | no |
   | `non_product` | stock code not a product: postage, carriage, manual, discount, samples, fees, tests, gift vouchers | 5,814 | no |
   | `cancellation_return` | `C` invoice with negative quantity | 16,678 | no, tracked as returns |
   | `invalid` | `C` invoice with non-negative quantity | 0 | no |
   | `stock_adjustment` | non-`C` line with quantity ≤ 0 (staff notes like "damages", "lost", "check") | 3,391 | no |
   | `zero_price` | positive quantity at price ≤ 0 (free-of-charge, "found") | 2,571 | no |
   | `sale` | positive quantity at positive price on a product code | 1,002,104 | **yes** |
   | `order_reversal_sale` / `_return` | same-day exact cancellation of a sale (see below) | 1,236 pairs | no |

   Product codes match `^(\d{5}[A-Z]{0,3}|DCGS[0-9A-Z]+|SP\d{4})$`. The `DCGS` codes are dotcom gift sets and are real merchandise.
5. **Missing customer id.** 226,761 sale lines (22.6%) have no customer id. They are **kept**, because the goods were still sold. They are excluded only from order-reversal matching, which needs a customer.
6. **Missing descriptions.** Each product gets a canonical `product_description`: its most frequent description on sale lines, with ties broken alphabetically. 3,871 missing descriptions are filled this way. The original `description` is kept unchanged.
7. **Event id.** `event_id` is a SHA-256 hash (first 32 hex characters) of the normalised business fields. The same line always gets the same id. The streaming phase uses it for idempotent, restart-safe processing.

### Return-handling policy

The dataset doesn't link a cancellation to the invoice it cancels, and doesn't distinguish "order cancelled" from "goods returned". The policy is:

* **Demand target = units on `sale` lines.** Returns are not netted off demand. A customer returning goods later doesn't mean the demand didn't occur, and netting would push some product-days negative.
* **Returns are tracked separately** on the day the `C` invoice was raised, as returned units and refunded value. Net revenue = sales revenue + (negative) return value.
* **Same-day order reversals are removed from demand.** A `C` line is paired one-to-one with the latest earlier sale on the *same calendar day* that has the same customer, product and absolute quantity. Both lines are excluded. This catches orders keyed in error and immediately cancelled, including the two largest lines in the file: 80,995 × `23843` (2011-12-09) and 74,215 × `23166` (2011-01-18), each cancelled within 20 minutes. 1,236 pairs and 210,522 units are excluded. Because the cancellation is known by the end of that day, this uses no future information when forecasting from the next day onward. Cancellations on later days are ordinary returns.
* **Stock write-offs, bad debt and non-product lines** never count as demand or as returns.

### Observations that shape forecasting

* **No Saturday trading.** There is only one Saturday with any sales in two years. Forecast horizons are therefore measured in trading days, and calendar features must reflect this.
* **Sparse demand.** 4,719 products sold at least once, with a median of 68 selling days each. 618 products sold on fewer than 10 days, and 81.4% of product × trading-day cells are zero. Sparse products need explicit handling.
* **Large legitimate orders.** Wholesale orders of 10,000+ units at £0.10–£0.25 do occur (for example 19,152 mugs to a Danish customer). They are real sales, not errors, so outlier treatment belongs in modelling rather than cleaning.

## Daily demand aggregation

`python -m src.processing.aggregate_sales` turns cleaned lines into three tables in `data/processed/`.

**`product_daily.parquet`**: one row per product per trading day, from the product's first sale to the end of the data. Days without sales are explicit zeros.

| Column | Meaning |
|---|---|
| `units_sold` | **Forecast target.** Units on `sale` lines only |
| `sale_lines`, `invoices`, `customers` | Transaction counts: sale lines, distinct invoices, distinct known customers |
| `gross_revenue` | GBP value of `sale` lines |
| `units_returned`, `return_lines`, `return_value` | `cancellation_return` lines on that day, as positive numbers |
| `units_reversed` | Units in same-day order reversals (excluded from demand, kept for transparency) |
| `net_units`, `net_revenue` | Sales minus returns. Can be negative on days with returns but no sales |
| `avg_unit_price` | `gross_revenue / units_sold`. Empty on zero-sale days |
| `is_partial_day` | The day is cut off by the dataset boundary |

**`daily_totals.parquet`**: store-wide totals per trading day. **`product_summary.parquet`**: per-product history, sparsity and demand class.

### Trading calendar

The shop does not trade on Saturdays, England & Wales bank holidays (including one-offs such as 2011-04-29), Easter Sunday, or 24 December – 3 January. Forecasts need a calendar that is known *in advance*, so `trading_calendar.py` encodes these rules, with bank holidays from the [`holidays`](https://pypi.org/project/holidays/) package. Measured against history, the rule agrees on **738 of 739 days (99.86%)**. The one miss is 2009-12-05, the only Saturday with sales. Historical tables use the observed trading days (604). Future forecast dates use the rule. Forecast horizons are therefore *trading* days: the 7 trading days after 2011-12-09 run to 2011-12-18.

### Decisions and measured results

* **The final day is partial.** The data stops at 12:50 on 2011-12-09, which has 12,943 units against a typical 30–40k. It is flagged `is_partial_day` and must not be used as a full day for training or evaluation. A day is flagged when the data starts later, or stops earlier, than on 90% of trading days.
* **Product life starts at first sale.** Zero rows before a product's first sale would be false zeros. The dense table has 2,397,712 rows, and 77.9% have zero demand.
* **Unplaceable lines are reported, not lost.** These are lines dated before a product's first sale, or for products that never had a real sale: 68 return lines (449 units) and 14 order-reversal lines (155,741 units, including the 80,995-unit `23843` order, which is that product's only appearance).
* **Every run reconciles.** Product-day totals are checked against the cleaned lines for units, revenue, returns and reversals. The command exits non-zero on any mismatch.
* **Sparsity classes** follow Syntetos–Boylan (average demand interval cut-off 1.32, CV² cut-off 0.49), computed on non-zero demand sizes:

  | demand class | sufficient history | insufficient history |
  |---|---:|---:|
  | lumpy | 3,518 | 213 |
  | intermittent | 266 | 286 |
  | erratic | 300 | 12 |
  | smooth | 3 | 0 |
  | n/a (fewer than 2 selling days) | 0 | 121 |

  "Sufficient" means at least 10 selling days over at least 28 trading days. 2,775 products are both sufficient and sold in the last 28 trading days. Demand here is overwhelmingly lumpy, because many customers are wholesalers. The forecasting phase is designed with that in mind rather than assuming smooth series.
* **`product_summary` covers the full history and must not be used as a model feature.** It would leak future information. Forecasting recomputes history features at each forecast origin.

## Demand forecasting

Forecasts are **model predictions** of `units_sold` (units per trading day) for each product, made at the end of a trading day (the *origin*) for the next 1–7 trading days.

### Method

* **Information available at the origin only.** Features come from a trading-day × product matrix in which every rolling window, lag and expanding statistic ends at the origin. Two tests prove this by multiplying all future data and asserting that no feature changes. A deliberately leaky change (a centred rolling window) makes both tests fail. `predict.py` also truncates its input at the origin before building features.
* **Features** (35): lags 1–7; rolling means over 7/14/28/56/112 days; rolling std and max; selling-day share; mean non-zero order size; days since last sale; product age; selling days to date; invoice and customer counts; average price and price ratio; return rate; store-wide momentum. Target-day features: horizon, weekday, ISO week, month, day of month, and demand in the same week last year (364 days back, so weekdays align).
* **Chronological splits by target date:** train up to 2011-06-30, validation 2011-07-01 to 09-30, test 2011-10-01 to 12-08 (the last full day; the partial 2011-12-09 is excluded). Validation chooses the configuration and the number of boosting rounds. The test set is scored once, by a model refit on train + validation. Evaluation is rolling-origin: every trading day in the window is an origin, and every horizon whose target falls inside the window is scored.
* **Models.** The baselines are a zero forecast, seasonal naive (same weekday, most recent at or before the origin) and a 28-trading-day moving average. The stronger model is a global LightGBM regressor with horizon as a feature, and a small search over Tweedie(1.2), Tweedie(1.5) and Poisson objectives. Training uses one seeded random horizon per (product, origin), giving 1.24M rows instead of 12.5M near-duplicates.
* **Sparse products, handled explicitly.** A product gets a model forecast only with at least 10 selling days over at least 28 trading days *at the origin*. Otherwise it gets the fallback: a 28-day mean that excludes the launch day. A product enters the data on the day of its first sale, so a plain short-window mean over-forecasts. On validation, the fallback's bias is +11.0% with the launch day and −0.07% without it. A "pad to 28 days" variant had lower error but a −43% bias, which would understate demand for new products, so it was rejected. Products launched after the origin have no history and get a forecast of 0. They are reported as their own segment, not hidden.
* **Reproducibility.** Fixed seeds, LightGBM `deterministic=True` and a fixed thread count. Two full training runs produced identical numbers. `models/demand_lgbm.json` records the configuration, iteration count, feature list, data fingerprint, training window and library versions. A test checks that a saved and reloaded model gives identical forecasts.

### Measured results (test set, 2011-10-01 to 2011-12-08)

Full tables, including per-horizon WAPE and the validation results, are in [`reports/forecast_evaluation.md`](reports/forecast_evaluation.md).

**Daily demand, model-eligible products** (1,641,260 forecasts, 4,086 products):

| method | MAE | RMSE | WAPE | bias |
|---|---:|---:|---:|---:|
| zero | 6.370 | 36.03 | 1.000 | −1.000 |
| seasonal naive | 7.704 | 42.03 | 1.210 | −0.029 |
| moving average 28 | 6.391 | **31.62** | 1.003 | −0.055 |
| **LightGBM** | **6.298** | 31.72 | **0.989** | −0.050 |

**Total demand over the next 7 trading days, model-eligible products** (210,628 product-origins):

| method | MAE | RMSE | WAPE | bias |
|---|---:|---:|---:|---:|
| zero | 44.13 | 153.12 | 1.000 | −1.000 |
| seasonal naive | 31.52 | 121.72 | 0.714 | −0.013 |
| moving average 28 | 26.86 | **97.05** | 0.609 | −0.045 |
| **LightGBM** | **26.15** | 100.02 | **0.593** | −0.036 |

**All products, with the fallback and launches included**, the full pipeline scores a 7-day WAPE of 0.603, against 0.624 for the moving average. For fallback products on the 7-day test, the pipeline's WAPE is 0.816 (bias +13%), while seasonal naive scores 0.793.

### What these results do and do not show

* **Daily product-level demand here is barely predictable.** 70% of test product-days are zero, and the largest 1% of product-days carry 36.5% of all units, because many customers are wholesalers placing occasional large orders. A WAPE close to 1 at daily level reflects that, and the zero forecast is a hard baseline on MAE and WAPE. That is why the zero row is reported.
* **LightGBM is the best method on MAE and WAPE, but by a small margin**: about 1.5% on daily MAE and 2.6% on 7-day MAE over the moving average. On RMSE the moving average is slightly better: the Tweedie model reacts more to large orders, which costs it on squared error. Neither should be described as "accurate". The 7-day totals (WAPE ≈ 0.59) are the more useful signal for replenishment.
* **The test window is the Q4 peak**, so these numbers may not represent other seasons. The validation window (Q3) results are in the report.
* **Single-split evaluation.** Results come from one chronological test window, not from repeated backtests over a full year.
* **The model doesn't know about planned launches, promotions or customer orders placed in advance.** None of these are in the data.

### Inference

```bash
python -m src.forecasting.predict                       # origin = last full day, 2011-12-08
python -m src.forecasting.predict --origin 2011-11-15   # any historical full trading day
```

Output: `data/processed/forecasts_<origin>.parquet`, with one row per product × horizon:

| column | meaning |
|---|---|
| `forecast_origin`, `target_date`, `horizon_trading_days` | made at the end of `forecast_origin`; target dates come from the trading-calendar rule |
| `forecast_units` | predicted units sold (units/day, ≥ 0) |
| `method` | `lightgbm` or `fallback_mean_28_excl_launch` |
| `history_status`, `selling_days_to_date` | why that method was used |
| `recent_mean_28`, `units_last_trading_day`, `trading_days_since_last_sale` | observed evidence at the origin |
| `model_version`, `value_type` = `model_prediction`, `unit` | provenance, for the copilot and dashboard |

At the default origin, the next trading day is 2011-12-09. That day is partly observed in the source (until 12:50), so it can't be used to judge the forecast as a full day.

## Simulated real-time streaming (Kafka replay)

> **This is simulated real-time streaming using historical data.** `kafka_producer.py` replays the 2009–2011 UCI transactions in timestamp order. The application does **not** receive live transactions from the retailer. Every event carries `"data_origin": "historical_replay"`.

```bash
docker compose up -d --wait kafka                          # single-node Kafka 3.9.1 on localhost:9092

python -m src.streaming.kafka_producer --dry-run           # validate every event, no broker
python -m src.streaming.kafka_producer --mode fast --start 2010-12-01 --end 2010-12-03
python -m src.streaming.kafka_producer --mode accelerated --speed 3600 --max-gap 2   # 1 h of history per second
python -m src.streaming.kafka_producer --mode fast --rate 500                        # fast, capped at 500 events/s

RETAILPULSE_KAFKA_BOOTSTRAP=localhost:9092 pytest -m integration                     # real-broker tests
```

### Event contract

Topic `retail.transactions.v1` has 6 partitions and is keyed by `stock_code`, so each product's events stay in order within a partition. The payload is JSON, validated by `TransactionEvent` ([`schemas/transaction_event.v1.json`](schemas/transaction_event.v1.json)):

| field | meaning |
|---|---|
| `event_time` | the **original** invoice timestamp, unchanged (UK local time as recorded; the source has no timezone) |
| `ingestion_time` | UTC wall-clock time the event was published. The consumer uses it to measure processing latency |
| `event_id` | deterministic id from the normalised line, the **same id the batch pipeline gives the cleaned row** |
| `sequence` | position in the canonical replay order (event time, sheet, row) across the whole dataset |
| `invoice`, `stock_code`, `description`, `quantity`, `unit_price`, `customer_id`, `country` | the raw line, including cancellations and adjustments. Classification happens in the consumer, with the batch rules |
| `source_sheet`, `source_row`, `replay_run_id`, `schema_version`, `data_origin` | lineage and provenance |

Records that fail validation (bad invoice number, missing quantity or price, non-finite price, missing timestamp, and so on) go to `retail.transactions.dlq.v1` with the raw values and the list of errors. They are never dropped silently. On the real file, **all 1,044,848 lines pass validation**, consistent with the cleaning stage's 0 invalid rows. Unit tests inject malformed records to exercise the dead-letter path.

### Restart without duplicates

* **Transactional, atomic batches.** Each batch (default 1,000 events), including its dead-letter records, is committed in one Kafka transaction. A crash or Ctrl-C aborts the open batch, so consumers reading with `isolation.level=read_committed` never see half a batch.
* **The checkpoint lives in Kafka.** On start, the producer first calls `init_transactions`, which fences any older producer with the same `transactional.id` and aborts its open transaction. It then reads the highest committed `sequence` on both topics and resumes after it. There is no local checkpoint file to lose or fall out of sync.
* **Re-replaying on purpose** (`--from-start`) publishes events again with the **same `event_id`s**. The phase 6 consumer de-duplicates on `event_id`, which also removes the raw file's exact-duplicate lines, as the batch cleaning does.

Verified on the real broker:

* Four runs over 2010-12-01 to 12-14: a fast run; an identical rerun (published 0); an accelerated run interrupted with Ctrl-C mid-batch (aborted, committed up to sequence 522,593); and a resume (started at 522,594).
* A `read_committed` consumer then found **31,102 messages, exactly the 31,102 expected, with 0 duplicate sequences and 0 gaps**. Every payload validated against the schema, the key matched the stock code, and 100% of `event_id`s matched the batch-cleaned table. The 439 repeated `event_id`s are exactly the raw exact-duplicate lines in that window.
* The integration test (`tests/integration/`) checks the same crash → resume → exactly-once behaviour automatically, and CI runs it against a Kafka container.

### Pacing

| mode | behaviour |
|---|---|
| `fast` | as fast as possible: about **25,000 events/s** measured locally with transactions (99,602 events in 3.9 s), so the full history replays in under a minute. `--rate N` caps it at N events/s |
| `accelerated` | the wait between events is the event-time gap ÷ `--speed`, capped at `--max-gap` seconds, so nights, weekends and Christmas don't stall a demo. `--speed 1` replays at the original pace |

Before any wait of 0.25 s or longer, the open batch is committed, so consumers see events promptly in slow replays.

## Streaming processing (Kafka consumer)

```bash
python -m src.streaming.kafka_consumer                    # run continuously (Ctrl-C to stop)
python -m src.streaming.kafka_consumer --idle-exit 15     # catch up, finish the window, stop
python -m src.streaming.kafka_consumer --status           # counters, latency, offsets
python -m src.streaming.reconcile                         # stream vs batch, exits 1 on any mismatch
```

The consumer turns the replayed events into the same daily product demand the batch pipeline builds. Its state is a local SQLite database (`data/stream/stream_state.sqlite`):

| table | contents |
|---|---|
| `events` | de-duplicated events, with their line type and per-event latency |
| `product_daily` (+ `product_daily_view`) | incrementally maintained product × day counters: the batch `product_daily` columns, including distinct invoices and customers |
| `offsets` | next Kafka offset per partition |
| `dead_letters` | producer dead-letter records and any poison messages |

**Exactly once.** Each poll batch updates events, aggregates and offsets in **one SQLite transaction**. On start, the consumer assigns partitions explicitly and seeks to its stored offsets. A crash at any point either keeps the whole batch, including its offsets, or none of it. Consumer-group offsets are committed to Kafka too, but only so standard lag tooling works. The consumer reads with `isolation.level=read_committed`, so aborted producer batches are never seen. Events are also de-duplicated on `event_id`, so a deliberate `--from-start` replay changes nothing, and the raw file's exact-duplicate lines are dropped as in batch.

**Same rules as batch.** Lines are classified by `classify_line`, a single-line version of the batch rules; a test checks that both versions agree on all 216 combinations of invoice type, code, quantity and price. Same-day order reversals need state, and one detail matters. In 45 of the 1,236 reversal pairs, the sale arrives *after* its cancellation (same minute, later row). The consumer therefore records a cancellation as a return straight away, and resolves its match only once the partition's event time has moved past that minute. Products are partitioned by stock code, so by then every candidate sale has arrived. The match uses the batch's greedy rule and moves the units into `units_reversed`. A test confirms that dropping this deferral breaks batch parity.

### Measured on the full replay

* **Exact match with batch.** Replaying all 1,044,848 events and consuming them gave 1,033,036 stored events and 11,812 duplicates ignored, matching batch cleaning exactly, and every line-type count was identical. `reconcile` compared **535,156 product-days on 9 metrics** (units, lines, invoices, customers, revenue, returns, return value, reversed units) with **0 mismatches**. The 78 stream rows before a product's first sale hold exactly the 449 returned and 155,741 reversed units that the batch aggregation reports as unplaceable.
* **Crash recovery.** A consumer killed with `SIGKILL` after 469,930 events and then restarted ended with identical totals: still 11,812 duplicates ignored, meaning no message was applied twice. Reconciliation was again an exact match.
* **Throughput.** Catching up from a full backlog ran at about 6,700 events/s (1.04M in about 2.5 min), on a 4-core container with SQLite.
* **Latency, live.** Measured from the producer's `ingestion_time` to processing, with the consumer running during an accelerated replay of one trading day (600×, 3,409 events): **p50 167 ms, p95 527 ms, p99 750 ms, max 788 ms**. The floor comes from transactional visibility: the producer commits at least every 0.5 s (`max_batch_age_seconds`, added after a first run showed a 1.3 s tail when batches waited to fill), and the consumer polls with a 0.2 s timeout. Latency measured while catching up a backlog only reflects how old the backlog is, so it isn't reported as processing latency.

Tests: unit tests feed real replay output through simulated partitions and require **exact equality with the batch aggregates** at batch sizes 1, 3 and 100. They also cover redelivery, re-replay, rollback on a crash mid-batch, poison messages, and a cancellation that arrives before its sale. The integration test runs producer → Kafka → consumer (stopped and resumed) → reconcile, then a full re-replay, against a real broker.

## Inventory and stockout risk (synthetic inventory)

> **The inventory data is synthetic.** UCI Online Retail II has no stock levels, purchase orders or supplier lead times. Every stock figure in RetailPulse is generated from [`config/inventory_assumptions.toml`](config/inventory_assumptions.toml). None of it is the retailer's real inventory. Every row carries `is_synthetic = True` and a `SYNTHETIC` label, and every output records the assumptions version it was produced with.

```bash
python -m src.inventory.synthetic_inventory --as-of 2011-12-08   # the synthetic snapshot (CSV)
python -m src.inventory.risk_engine                              # risk + reorder suggestions
python -m src.inventory.risk_engine --evaluate                   # calibration check
python -m src.inventory.approvals register data/inventory/stockout_risk_2011-12-08.parquet
python -m src.inventory.approvals list
python -m src.inventory.approvals approve <suggestion_id> --by "Your Name" --note "why"
```

### Synthetic inventory assumptions

| field | how it is generated |
|---|---|
| `lead_time_days` | supplier lead time in trading days, drawn from {3, 5, 7, 10, 14} with weights {20, 35, 25, 15, 5}% |
| `mean_daily_demand`, `std_daily_demand` | **observed** units/day over the last 56 trading days up to the snapshot date |
| `safety_stock` | ⌈z × std × √lead time⌉, z for a 95% service level (1.645) |
| `reorder_point` | ⌈mean × lead time + safety stock⌉ |
| `current_stock` | uniform(0, 30) trading days of mean demand; uniform{0..20} units if there was no recent demand |

Random draws are seeded per (seed, product, snapshot date), so a snapshot is reproducible and a product's values don't depend on which other products exist. There are no open purchase orders, so projections assume no deliveries within the 7-day horizon.

### Risk method

* **Projected stock** = synthetic current stock − cumulative **forecast** demand for each of the next 7 trading days. `expected_stockout_date` is the first day the expected demand exceeds stock.
* **Stockout probability** = P(actual demand > stock) by day h. Demand = forecast × r, where r comes from the **empirical distribution of actual/forecast ratios in the model's held-out backtest**, by horizon and forecast size. For a forecast of exactly 0, actual demand is used directly. No normal distribution is assumed. A first version used additive errors. It was clearly wrong near zero demand: it predicted P(any demand) = 0.51 where 29% was observed, and 1.0 for some next-day cases. The ratio model fixed both.
* **Risk level:** `high` ≥ 50%, `medium` ≥ 20% (configurable).
* **Reorder suggestion** (order-up-to). It applies when stock ≤ the reorder point, or when expected stock on the delivery day falls below safety stock. Quantity = forecast demand over lead time + 7-day review period + safety stock − stock on hand. Demand beyond the 7 forecast days is extrapolated at the mean forecast rate. `stockout_before_delivery_likely` flags products expected to run out before an order placed now could arrive.
* Every output column states whether it is synthetic, a model prediction, or derived from both (`value_types`).

### Human approval

Suggestions are only ever `pending_approval`. **RetailPulse never places an order.** `approvals.py` stores each suggestion with its numbers, model version and assumptions version. A named person must approve or reject it, a decision is final, and every action goes into an audit log that SQLite triggers make append-only. "Approved" means accepted for manual ordering outside this system. Because the stock is synthetic, approvals here demonstrate the workflow; they are not real purchase decisions.

### Measured results

**Calibration** ([`reports/inventory_risk_evaluation.md`](reports/inventory_risk_evaluation.md)). The check uses synthetic stock against **actual** demand. The error model is fitted on backtest origins 2011-09-30 to 10-30 and evaluated on 2011-10-31 to 11-30 (126,902 product-origins, 7-day horizon):

| predicted probability | cases | mean predicted | observed rate |
|---|---:|---:|---:|
| 0–5% | 67,628 | 1.2% | 2.2% |
| 5–20% | 28,364 | 11.3% | 14.8% |
| 20–35% | 14,649 | 26.5% | 31.6% |
| 35–50% | 6,078 | 42.0% | 49.5% |
| 50–65% | 4,378 | 56.0% | 61.0% |
| 65–80% | 2,951 | 71.7% | 78.3% |
| 80–95% | 2,221 | 87.4% | 90.7% |
| 95–100% | 633 | 97.1% | 97.2% |

The Brier score is 0.090, against 0.138 for always predicting the base rate (skill 0.35). The probabilities rank risk well, but they are **consistently a few points too low**: 13.9% predicted on average against 16.5% observed. The likely cause is that demand rose from October into the November peak and the forecasts under-predicted it (test bias −5%). Under-stated risk is the unsafe direction, so treat these probabilities as lower bounds. The calibration was not tuned on this window.

**At the last full day (2011-12-08)**, with synthetic stock: 4,719 products, of which 762 are high risk, 827 medium and 3,130 low. There are 2,106 reorder suggestions pending approval.

### Limitations

* Stock, lead times and policy parameters are invented, so the risk *levels* say nothing about the real retailer. What the phase demonstrates is the method and the workflow.
* One fixed lead time per product, no open orders, no minimum order quantities or pack sizes, and no supplier capacity.
* The probabilities inherit the forecast's weaknesses (lumpy wholesale demand, peak-season bias) and are calibrated on a single window.

## AI copilot

```bash
export ANTHROPIC_API_KEY=...        # Claude mode (or run with --offline)
python -m src.copilot.cli "Which products are at risk of stockout in the next seven days?"
python -m src.copilot.cli --json "Why has demand for SMALL POPCORN HOLDER changed?"
python -m src.copilot.cli --offline "Which products have unusual sales patterns?"
python -m src.copilot.evaluate --offline       # question-set evaluation (Claude mode: drop --offline)
```

The copilot answers questions such as *which products have the highest forecast demand tomorrow*, *which have unusual sales patterns*, *which are at risk of stockout in the next seven days*, *why a product's demand changed compared with its recent baseline*, and *what reorder quantity is suggested*. **Every number comes from a backend service, not from the language model.**

### How it works

```
question ─► Claude (Opus 5.5) ──tool_use──► validated tools ─► services ─► data / forecasts / synthetic inventory
               ▲    │                                   │
               └────┼───────── tool results (JSON + provenance) ◄┘
                    └─end_turn─► number check ─pass─► answer + evidence
                                     └─fail (once)─► back to Claude with the offending numbers
```

* **LangGraph** runs the loop as a graph (`model → tools → model … → verify`) with explicit routing for tool calls, end of turn, refusals, token limits and a turn limit.
* **Claude** is called with the official `anthropic` SDK: model `claude-opus-5-5` at `medium` effort. Thinking is always on for this model. Tools use strict schemas. The system prompt and tools are prompt-cached, and the history is append-only (responses are passed back unchanged). Server-side refusal fallback (`fallbacks: "default"`) is enabled, so a request declined by the model's safety classifiers is retried on Anthropic's recommended model for that category.
* **Seven tools** (`tools.py`), each a strict JSON schema plus a pydantic model that re-validates the arguments (ranges, stock-code format) before a service runs. Errors go back to Claude as `is_error` tool results so it can correct itself:

  | tool | answers | value types |
  |---|---|---|
  | `top_forecast_demand` | highest forecast demand for horizon 1–7 ("tomorrow" = next trading day) | model prediction (+ observed recent mean) |
  | `unusual_sales` | recent sales furthest from each product's own 56-trading-day baseline | observed |
  | `stockout_risk` | stockout probability, projected stock, suggested order, approval status | synthetic × model prediction |
  | `explain_demand_change` | recent vs baseline: units, order count vs order size, customers, price, returns, largest day | observed (+ forecast) |
  | `reorder_suggestion` | the order-up-to calculation with every input, plus approval status | synthetic × model prediction |
  | `product_info` | resolve names to stock codes (RAG over the catalogue) | observed |
  | `search_docs` | method and policy questions (RAG over README, assumptions, reports) | documentation |

* **Provenance on every result:** `forecast_origin` / `data_as_of` (end of trading day), `value_types` per field (observed / model prediction / synthetic), units, model version, assumptions version, and limitations. The limitations include the measured forecast error (test WAPE 0.989 daily, 0.593 over 7 days), the synthetic-inventory label, the need for human approval, the probability calibration bias, and the simulated-real-time note.
* **Number verification** (`verify.py`): every number in the final answer must match a tool result, allowing rounding and showing a probability as a percentage. Numbers from the question and the horizon numbers 1–7 are also allowed. On a mismatch Claude is told which numbers failed and gets one chance to fix them. If it still fails, the answer is returned **flagged `verified: false`** with the offending numbers, never silently passed.
* **RAG** (`knowledge.py`): BM25 with light stemming over the product catalogue, and over the README, inventory assumptions and evaluation reports split by heading. Retrieval is lexical and deterministic, which suits short keyword-like product names and documentation phrased in the same vocabulary as the questions.
* **Offline mode** (`offline.py`): the same tools, keyword intent routing and templated answers, marked `mode: "offline (no LLM)"`. It is used for tests and CI and when no key is available.

### Verification

* 25 unit tests drive the **real** graph, tools and verifier with a scripted stand-in for the Anthropic client. They cover:
  * the request shape: model, fallback beta, strict tools, no forced `tool_choice`, no thinking or sampling parameters;
  * the append-only history and the tool loop;
  * tool-error recovery;
  * the number-check retry, and flagging when the retry still fails;
  * refusal handling;
  * provenance on every tool and strict JSON (no NaN or Infinity);
  * that the reorder calculation reproduces the suggested quantity;
  * offline routing for each required question;
  * retrieval.
* `python -m src.copilot.evaluate --offline` passes **6/6** on the question set (tools called, numbers verified, labels present).
* **Not yet run against the live Claude API.** This environment has no Anthropic credentials, so no Claude-mode answer has been produced or measured here. Run `python -m src.copilot.evaluate` (which makes real, billed API calls) to measure Claude mode on the same question set.

### Limitations

* Answers are only as good as the services. The forecasts carry the measured errors above, and the inventory is synthetic.
* "Why did demand change" is descriptive (order count vs order size vs price). The data has no promotions, marketing or real stock-outs, so causes can't be identified.
* Retrieval is lexical: synonyms that share no words with the docs may miss.
* Number verification checks that each figure came from a tool, not that the sentence around it uses the figure correctly.

## Data and AI disclosure

Sales figures come from the UCI dataset above. Any inventory, lead-time or safety-stock values introduced in later phases are synthetic and will be labelled as such wherever they appear.
