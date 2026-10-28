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
| 4 | Forecasting models and evaluation | planned |
| 5 | Kafka historical event replay (producer) | planned |
| 6 | Streaming processing pipeline (consumer) | planned |
| 7 | Synthetic inventory and stockout risk | planned |
| 8 | AI copilot (LangGraph tools and RAG) | planned |
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
python -m src.ingestion.explore             # regenerate reports/data_profile.md

pytest                                      # unit tests (no download needed)
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
tests/unit/                         pytest suites on small hand-written fixtures
reports/                            generated, committed: data profile, cleaning and aggregation reports
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

## Data and AI disclosure

Sales figures come from the UCI dataset above. Any inventory, lead-time or safety-stock values introduced in later phases are synthetic and will be labelled as such wherever they appear.
