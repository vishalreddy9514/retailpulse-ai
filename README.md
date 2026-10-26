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
| 1 | Download and inspect UCI Online Retail II | in progress |
| 2 | Data cleaning and quality checks | in progress |
| 3 | Daily product-level demand aggregation | planned |
| 4 | Forecasting models and evaluation | planned |
| 5 | Kafka historical event replay (producer) | planned |
| 6 | Streaming processing pipeline (consumer) | planned |
| 7 | Synthetic inventory and stockout risk | planned |
| 8 | AI copilot (LangGraph tools and RAG) | planned |
| 9 | FastAPI backend and Streamlit dashboard | planned |
| 10 | Docker, CI/CD, logging, monitoring, docs | planned |

## Dataset

| | |
|---|---|
| Name | Online Retail II |
| Source | UCI Machine Learning Repository, <https://archive.ics.uci.edu/dataset/502/online+retail+ii> |
| Citation | Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5CG6D> |
| Licence | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |

The raw data is downloaded by the pipeline and is never committed to this repository.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
