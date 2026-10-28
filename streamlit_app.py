"""
Streamlit dashboard for the NSE Stock Analysis code.
It imports the functions straight from nse_analysis_api.py (same folder),
so no separate server is needed on Streamlit Community Cloud.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import nse_analysis_api as api

st.set_page_config(page_title="NSE Stock Analysis", page_icon="📈", layout="wide")
st.title("📈 NSE Stock Analysis")
st.caption("Rule-based indicators for learning and research, not investment advice.")


def run(fn, *args, **kwargs):
    """Call an API function and show a friendly error instead of crashing the page."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        st.error(str(getattr(e, "detail", e)))
        return None


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
symbol = st.sidebar.text_input("NSE symbol", "RELIANCE").strip().upper()
months = st.sidebar.slider("Chart history (months)", 1, 24, 6)
atr_mult = st.sidebar.slider("ATR stop-loss multiplier", 1.0, 4.0, 2.0, 0.5)
st.sidebar.caption("Use the plain NSE symbol, e.g. TCS, INFY, M&M.")

tab_overview, tab_chart, tab_risk, tab_rs, tab_compare, tab_screener = st.tabs(
    ["Overview", "Chart", "Risk", "vs Nifty 50", "Compare / Watchlist", "Screener"]
)

# ── OVERVIEW ─────────────────────────────────────────────────────────────────
with tab_overview:
    with st.spinner("Analyzing..."):
        data = run(api.analyze_stock, symbol)
    if data:
        st.subheader(f"{data['company_name']} ({data['symbol']})")
        if data["corporate_action_flag"]:
            st.warning(data["corporate_action_note"])

        t, f, s = data["technical"], data["fundamentals"], data["signal"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Price", t["current_price"])
        c2.metric("Signal", s["label"], f"score {s['score']}")
        c3.metric("RSI (14)", t["rsi_14"])
        c4.metric("MA trend", t["ma_trend"])
        c5.metric("% of 52w high", t["pct_of_52w_high"])

        st.markdown("**Why this score**")
        for reason in s["reasons"]:
            st.write("•", reason)

        col_a, col_b = st.columns(2)
        col_a.markdown("**Technicals**")
        col_a.dataframe(pd.Series(t, name="value").astype(str), use_container_width=True)
        col_b.markdown("**Fundamentals**")
        col_b.dataframe(
            pd.Series({k: v for k, v in f.items() if not k.startswith("_")}, name="value").astype(str),
            use_container_width=True,
        )

# ── CHART ────────────────────────────────────────────────────────────────────
with tab_chart:
    chart = run(api.chart_data, symbol, months)
    if chart:
        df = pd.DataFrame(chart["data"])
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
            subplot_titles=("Price with 50/200-day MA", "RSI (14)", "MACD"),
        )
        fig.add_trace(go.Candlestick(
            x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma_50"], name="MA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma_200"], name="MA 200"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["rsi"], name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", row=2, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["macd_line"], name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"], name="Signal"), row=3, col=1)
        fig.update_layout(height=800, xaxis_rangeslider_visible=False, legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)

# ── RISK ─────────────────────────────────────────────────────────────────────
with tab_risk:
    r = run(api.risk, symbol, atr_mult)
    if r:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", r["price"])
        c2.metric(f"Stop-loss ({atr_mult}x ATR)", r["suggested_stop_loss"])
        c3.metric("Support (20d)", r["support_20d"])
        c4.metric("Resistance (20d)", r["resistance_20d"])
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Annual volatility %", r["annualized_volatility_pct"])
        c6.metric("Max drawdown 2y %", r["max_drawdown_2y_pct"])
        c7.metric("Volume vs 20d avg", r["volume_vs_20d_avg"])
        c8.metric("OBV trend (20d)", r["obv_trend_20d"])
        st.caption(f"Stochastic %K {r['stochastic_k']} / %D {r['stochastic_d']}")

# ── RELATIVE STRENGTH ────────────────────────────────────────────────────────
with tab_rs:
    rs = run(api.relative_strength, symbol)
    if rs:
        table = pd.DataFrame(rs["vs_nifty50"]).T
        st.dataframe(table, use_container_width=True)
        st.bar_chart(table["outperformance_pct"])
        st.caption("Positive outperformance means the stock beat the Nifty 50 over that period.")

# ── COMPARE / WATCHLIST ──────────────────────────────────────────────────────
with tab_compare:
    syms = st.text_input("Symbols (comma separated)", "TCS,INFY,WIPRO,HCLTECH")
    if st.button("Compare and build watchlist"):
        with st.spinner("Fetching..."):
            cmp_res = run(api.compare, syms)
            wl_res = run(api.watchlist, syms)
        if cmp_res:
            st.markdown("**Ranked by signal score**")
            st.dataframe(pd.DataFrame(cmp_res["ranked"]), use_container_width=True)
        if wl_res:
            st.markdown("**Watchlist snapshot**")
            st.dataframe(pd.DataFrame(wl_res["stocks"]), use_container_width=True)

# ── SCREENER ─────────────────────────────────────────────────────────────────
with tab_screener:
    min_score = st.slider("Minimum score", -5, 8, 3)
    custom = st.text_input("Custom universe (optional, comma separated, max 25)", "")
    if st.button("Run screener"):
        with st.spinner("Scanning... the first run can take a minute"):
            res = run(api.screener, min_score, custom)
        if res:
            if res["matches"]:
                st.dataframe(pd.DataFrame(res["matches"]), use_container_width=True)
            else:
                st.info("No stocks met that score. Try lowering the minimum.")
