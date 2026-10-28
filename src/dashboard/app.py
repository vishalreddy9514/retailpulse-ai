"""RetailPulse AI dashboard (Streamlit). A thin client of the REST API.

Run (with the API running on :8000)::

    streamlit run src/dashboard/app.py

Every figure comes from the API, which reads the forecasting and inventory services.
Labels say whether a number is observed, a model prediction, or synthetic.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.dashboard.api_client import ApiClient, ApiError

# Validated categorical slots 1-2 (dataviz reference palette; light/dark steps).
SERIES = {
    "light": {"observed": "#2a78d6", "forecast": "#eb6834"},
    "dark": {"observed": "#3987e5", "forecast": "#d95926"},
}
EXAMPLES = [
    "Which products have the highest forecast demand tomorrow?",
    "Which products have unusual sales patterns?",
    "Which products are at risk of stockout in the next seven days?",
    "Why has demand for SMALL POPCORN HOLDER changed compared with its recent baseline?",
    "What reorder quantity is suggested for 22410?",
]

st.set_page_config(page_title="RetailPulse AI", page_icon="📈", layout="wide")


def client() -> ApiClient:
    if "api_client" not in st.session_state:
        st.session_state["api_client"] = ApiClient()
    return st.session_state["api_client"]


def get(path: str, **params) -> dict:
    """GET with a short per-session cache; cleared after any write."""
    cache = st.session_state.setdefault("_cache", {})
    key = (path, tuple(sorted(params.items())))
    if key not in cache:
        cache[key] = client().get(path, **params)
    return cache[key]


def palette() -> dict:
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    return SERIES["dark" if theme == "dark" else "light"]


def limitations(payload: dict) -> None:
    with st.expander("Limitations and provenance"):
        st.markdown("\n".join(f"- {x}" for x in payload.get("limitations", [])))
        if payload.get("value_types"):
            st.caption(
                "Value types: " + "; ".join(f"{k}: {v}" for k, v in payload["value_types"].items())
            )


def line_chart(df: pd.DataFrame, y: str, title: str, y_title: str, color: str) -> alt.Chart:
    """Single-series line: the title names the series, so no legend box."""
    base = alt.Chart(df, title=title).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y(f"{y}:Q", title=y_title),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip(f"{y}:Q", format=",.0f", title=y_title)],
    )
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
    return (
        base.mark_line(strokeWidth=2, color=color)
        + base.mark_point(size=80, filled=True, color=color)
        .encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
        .add_params(hover)
    ).properties(height=260)


def history_forecast_chart(overview: dict) -> alt.Chart:
    colors = palette()
    r = overview["results"]
    hist = pd.DataFrame(r["history"]).assign(
        series="Observed sales", value=lambda d: d["units_sold"]
    )
    fc = pd.DataFrame(r["forecast"]).assign(
        series="Forecast (model prediction)", value=lambda d: d["forecast_units"]
    )
    df = pd.concat(
        [hist[["date", "series", "value"]], fc[["date", "series", "value"]]], ignore_index=True
    )
    df["date"] = pd.to_datetime(df["date"])
    domain = ["Observed sales", "Forecast (model prediction)"]
    enc = {
        "x": alt.X("date:T", title=None),
        "y": alt.Y("value:Q", title="units per trading day"),
        "color": alt.Color("series:N", scale=alt.Scale(domain=domain, range=[colors["observed"], colors["forecast"]]),
                           legend=alt.Legend(orient="top", title=None, labelLimit=400)),
        "strokeDash": alt.StrokeDash("series:N", scale=alt.Scale(domain=domain, range=[[1, 0], [6, 4]]), legend=None),
        "tooltip": [alt.Tooltip("date:T"), alt.Tooltip("series:N"), alt.Tooltip("value:Q", format=",.1f", title="units")],
    }  # fmt: skip
    lines = alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36)).encode(**enc)
    # Direct labels: observed above its last point, forecast below its last point, so they never collide.
    ends = df.sort_values("date").groupby("series").tail(1)
    labels = [
        alt.Chart(ends[ends["series"] == series])
        .mark_text(align="right", dy=dy, fontSize=11, color="#52514e")
        .encode(x="date:T", y="value:Q", text=alt.value(text))
        for series, dy, text in zip(domain, (-14, 14), ("observed", "forecast"), strict=True)
    ]
    origin = (
        alt.Chart(pd.DataFrame({"date": [pd.Timestamp(overview["forecast_origin"][:10])]}))
        .mark_rule(strokeDash=[2, 2], color="#52514e", opacity=0.6)
        .encode(x="date:T", tooltip=[alt.Tooltip("date:T", title="forecast origin")])
    )
    return alt.layer(origin, lines, *labels).properties(
        height=300, title=f"{r['stock_code']} {r['product_description']}"
    )


# -- page ------------------------------------------------------------------------------------
st.title("RetailPulse AI")
try:
    health = get("/health")
except Exception as exc:  # noqa: BLE001 - show any connection problem to the user
    st.error(f"Cannot reach the RetailPulse API ({exc}). Start it with `uvicorn src.api.main:app`.")
    st.stop()

st.info(
    f"**Simulated real-time.** {health['data_note']}  \n"
    f"**Inventory is synthetic:** {health['inventory_note']}.",
    icon="ℹ️",
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast origin (end of day)", health["forecast_origin"])
c2.metric("Products", f"{health['products']:,}")
c3.metric("Copilot modes", ", ".join(health["copilot_modes"]))
c4.metric("Approval writes", "token" if health["write_auth"] == "token" else "open",
          help=f"Write endpoints: {health['write_auth']}")  # fmt: skip
st.caption(
    f"Model {health['model_version']} · inventory assumptions {health['assumptions_version']}"
)

tab_overview, tab_forecast, tab_unusual, tab_risk, tab_copilot = st.tabs(
    ["Overview", "Forecasts", "Unusual sales", "Stockout risk & approvals", "Copilot"]
)

with tab_overview:
    totals = get("/sales/daily-totals", days=120)
    df = pd.DataFrame(totals["results"])
    df["date"] = pd.to_datetime(df["date"])
    last, prev = df.iloc[-1], df.iloc[-2]
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Units sold {last['date']:%d %b %Y} (observed)", f"{int(last['units_sold']):,}",
              f"{int(last['units_sold'] - prev['units_sold']):+,} vs previous trading day")  # fmt: skip
    k2.metric("Gross revenue (observed)", f"£{last['gross_revenue_gbp']:,.0f}")
    k3.metric("Units returned (observed)", f"{int(last['units_returned']):,}")
    blue = palette()["observed"]
    st.altair_chart(line_chart(df, "units_sold", "Units sold per trading day (observed)", "units", blue),
                    use_container_width=True)  # fmt: skip
    st.altair_chart(line_chart(df, "gross_revenue_gbp", "Gross revenue per trading day (observed)", "GBP", blue),
                    use_container_width=True)  # fmt: skip
    limitations(totals)

with tab_forecast:
    left, right = st.columns([1, 2])
    horizon = left.selectbox("Horizon (trading days ahead)", list(range(1, 8)), index=0)
    top = get("/forecasts/top", horizon=horizon, limit=15)
    table = pd.DataFrame(top["results"]).rename(
        columns={
            "forecast_units": "forecast units (prediction)",
            "recent_mean_28_units_per_day": "28-day mean (observed)",
            "units_last_trading_day": "last day (observed)",
        }  # fmt: skip
    )
    left.caption(f"Made at {top['forecast_origin']}")
    left.dataframe(
        table.drop(columns=["product_description"]), hide_index=True, use_container_width=True
    )
    codes = [r["stock_code"] for r in top["results"]]
    names = {r["stock_code"]: r["product_description"] for r in top["results"]}
    code = right.selectbox("Product", codes, format_func=lambda c: f"{c} {names[c]}")
    overview = get(f"/products/{code}", history_days=56)
    right.altair_chart(history_forecast_chart(overview), use_container_width=True)
    with right.expander("Why has demand changed? (observed, last 7 vs previous 28 trading days)"):
        try:
            change = get(f"/products/{code}/demand-change")
            ch = change["results"]
            st.write(
                f"Units/day {ch['recent']['units_per_day']} vs {ch['baseline']['units_per_day']} "
                f"({ch['change']['units_per_day_pct']}%); orders/day {ch['change']['orders_per_day_pct']}%, "
                f"units/order {ch['change']['units_per_order_pct']}%, price {ch['change']['avg_unit_price_pct']}%."
            )
            for d in ch["drivers"]:
                st.write(f"- {d}")
        except ApiError as exc:
            st.warning(exc.detail)
    limitations(top)

with tab_unusual:
    days = st.slider("Recent window (trading days)", 1, 20, 5)
    unusual = get("/sales/unusual", recent_days=days, limit=20)
    st.dataframe(pd.DataFrame(unusual["results"]), hide_index=True, use_container_width=True)
    limitations(unusual)

with tab_risk:
    level = st.radio("Minimum risk level", ["high", "medium", "low"], horizontal=True)
    risk = get("/inventory/risk", min_level=level, limit=50)
    counts = risk["products_by_risk_level"]
    r1, r2, r3 = st.columns(3)
    r1.metric("High risk (≥50%)", f"{counts.get('high', 0):,}")
    r2.metric("Medium risk (≥20%)", f"{counts.get('medium', 0):,}")
    r3.metric("Low risk", f"{counts.get('low', 0):,}")
    st.caption("Stock and lead times are SYNTHETIC; forecasts are model predictions.")
    rdf = pd.DataFrame(risk["results"])
    if not rdf.empty:
        rdf["stockout_probability_7d"] = (rdf["stockout_probability_7d"] * 100).round(1)
        rdf = rdf.rename(columns={"current_stock": "stock (synthetic)",
                                  "stockout_probability_7d": "stockout prob. 7d (%)"})  # fmt: skip
    st.dataframe(
        rdf.drop(columns=["suggestion_id"], errors="ignore"),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Reorder suggestions awaiting approval")
    st.caption(
        "Approving accepts a suggestion for manual ordering. Nothing is ordered automatically."
    )
    pending = get("/approvals", status="pending_approval", limit=1)
    # Offer the highest-risk pending suggestions first (from the risk table above).
    options = {
        x["suggestion_id"]: x
        for x in risk["results"]
        if x["suggestion_id"] and x["suggestion_status"] == "pending_approval"
    }
    if not options:
        st.success(
            f"No pending suggestions among the products shown ({pending['count']:,} pending in total)."
        )
    else:
        sid = st.selectbox(
            f"{pending['count']:,} suggestions pending in total; highest risk first",
            list(options),
            format_func=lambda i: (
                f"{options[i]['stock_code']} {options[i]['product_description']}: "
                f"order {options[i]['suggested_order_qty']:,} units (stockout prob. {options[i]['stockout_probability_7d']:.0%})"
            ),
        )
        detail = get(f"/approvals/{sid}")["details"]
        st.dataframe(
            pd.DataFrame({"field": list(detail), "value": [str(v) for v in detail.values()]}),
            hide_index=True, use_container_width=True,
        )  # fmt: skip
        with st.form("decision"):
            actor = st.text_input("Your name (recorded in the audit log)")
            note = st.text_input("Note (optional)")
            a, b = st.columns(2)
            approve = a.form_submit_button("Approve", type="primary")
            reject = b.form_submit_button("Reject")
        if approve or reject:
            try:
                out = client().post(f"/approvals/{sid}/decision",
                                    {"decision": "approve" if approve else "reject", "actor": actor, "note": note or None})  # fmt: skip
                st.session_state["_cache"] = {}
                st.success(f"{out['stock_code']}: {out['status']} by {out['history'][-1]['actor']}")
            except ApiError as exc:
                st.error(exc.detail)
    limitations(risk)

with tab_copilot:
    modes = ["auto", "offline"] + (["claude"] if "claude" in health["copilot_modes"] else [])
    mode = st.radio("Mode", modes, horizontal=True,
                    help="auto = Claude when an API key is configured, otherwise offline (no LLM)")  # fmt: skip
    example = st.selectbox("Example questions", ["(type your own)", *EXAMPLES])
    question = st.text_area("Question", value="" if example.startswith("(") else example, height=80)
    asked = st.button("Ask", type="primary")  # not disabled: a text area commits on blur
    if asked and len(question.strip()) < 3:
        st.warning("Type a question first.")
    elif asked:
        with st.spinner("Calling the copilot…"):
            try:
                ans = client().post("/copilot/ask", {"question": question, "mode": mode})
            except ApiError as exc:
                st.error(exc.detail)
                ans = None
        if ans:
            badge = "✅ every number matches a tool result" if ans["verified"] else (
                f"⚠️ unverified numbers: {', '.join(ans['unverified_numbers'])}")  # fmt: skip
            st.markdown(ans["answer"])
            st.caption(
                f"{badge} · mode: {ans['mode']}"
                + (f" · model: {ans['model']}" if ans["model"] else "")
            )
            with st.expander("Evidence: tool calls and provenance"):
                st.write("Tools called:", ", ".join(t["tool"] for t in ans["tool_calls"]) or "none")
                st.json(ans["evidence"], expanded=False)
