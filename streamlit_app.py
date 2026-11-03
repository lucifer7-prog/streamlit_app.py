"""
NSE Stock Analysis - Streamlit dashboard (extended version)
Imports the functions straight from nse_analysis_api.py (same folder),
so no separate server is needed on Streamlit Community Cloud.

Tabs: Overview | Chart | Risk & Position size | vs Nifty 50 |
      Compare & Correlation | Screener | Backtest | Portfolio
"""

import math

import numpy as np
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


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# ── BACKTEST ENGINE ──────────────────────────────────────────────────────────

def run_backtest(close, strategy, fast, slow, rsi_buy, rsi_sell, fee_bps):
    """Long-only daily backtest. Signals use today's close and trade from the next day
    (shift by 1) so there is no look-ahead bias. Fees are charged on every position change."""
    close = close.dropna()

    if strategy == "Moving-average crossover":
        fast_ma = close.rolling(fast).mean()
        slow_ma = close.rolling(slow).mean()
        raw = (fast_ma > slow_ma).astype(float)
        raw[slow_ma.isna()] = 0.0
    else:  # RSI mean reversion: buy when oversold, sell when overbought
        rsi = api.calculate_rsi(close)
        pos = pd.Series(np.nan, index=close.index)
        pos[rsi < rsi_buy] = 1.0
        pos[rsi > rsi_sell] = 0.0
        raw = pos.ffill().fillna(0.0)

    position = raw.shift(1).fillna(0.0)
    daily_ret = close.pct_change().fillna(0.0)
    changes = position.diff().abs().fillna(position.abs())
    strat_ret = position * daily_ret - changes * fee_bps / 10000

    equity = (1 + strat_ret).cumprod()
    buy_hold = (1 + daily_ret).cumprod()

    def stats(curve, rets):
        years = max(len(curve) / 252, 1e-9)
        total = curve.iloc[-1] - 1
        cagr = curve.iloc[-1] ** (1 / years) - 1
        max_dd = (curve / curve.cummax() - 1).min()
        vol = rets.std() * math.sqrt(252)
        sharpe = (rets.mean() * 252) / vol if vol else float("nan")
        return {
            "Total return %": round(total * 100, 2),
            "CAGR %": round(cagr * 100, 2),
            "Max drawdown %": round(max_dd * 100, 2),
            "Sharpe (rf=0)": round(sharpe, 2),
        }

    strat_stats = stats(equity, strat_ret)
    strat_stats["Trades (entries)"] = int((position.diff() == 1).sum())
    strat_stats["Time in market %"] = round(position.mean() * 100, 1)
    bh_stats = stats(buy_hold, daily_ret)
    return equity, buy_hold, position, strat_stats, bh_stats


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
symbol = st.sidebar.text_input("NSE symbol", "RELIANCE").strip().upper()
months = st.sidebar.slider("Chart history (months)", 1, 24, 6)
atr_mult = st.sidebar.slider("ATR stop-loss multiplier", 1.0, 4.0, 2.0, 0.5)
st.sidebar.caption("Use the plain NSE symbol, e.g. TCS, INFY, M&M.")

(tab_overview, tab_chart, tab_risk, tab_rs,
 tab_compare, tab_screener, tab_backtest, tab_portfolio) = st.tabs(
    ["Overview", "Chart", "Risk & Position size", "vs Nifty 50",
     "Compare & Correlation", "Screener", "Backtest", "Portfolio"]
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
        col_a.dataframe(pd.Series(t, name="value").astype(str), width="stretch")
        col_b.markdown("**Fundamentals**")
        col_b.dataframe(
            pd.Series({k: v for k, v in f.items() if not k.startswith("_")}, name="value").astype(str),
            width="stretch",
        )

# ── CHART ────────────────────────────────────────────────────────────────────
with tab_chart:
    show_volume = st.checkbox("Show volume panel", value=True)
    chart = run(api.chart_data, symbol, months)
    if chart:
        df = pd.DataFrame(chart["data"])
        rows = 4 if show_volume else 3
        heights = [0.5, 0.15, 0.2, 0.15] if show_volume else [0.6, 0.2, 0.2]
        titles = ["Price with 50/200-day MA"] + (["Volume"] if show_volume else []) + ["RSI (14)", "MACD"]
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                            row_heights=heights, vertical_spacing=0.03, subplot_titles=titles)
        fig.add_trace(go.Candlestick(
            x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma_50"], name="MA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma_200"], name="MA 200"), row=1, col=1)
        r_ = 2
        if show_volume:
            fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="Volume"), row=r_, col=1)
            r_ += 1
        fig.add_trace(go.Scatter(x=df["date"], y=df["rsi"], name="RSI"), row=r_, col=1)
        fig.add_hline(y=70, line_dash="dot", row=r_, col=1)
        fig.add_hline(y=30, line_dash="dot", row=r_, col=1)
        r_ += 1
        fig.add_trace(go.Scatter(x=df["date"], y=df["macd_line"], name="MACD"), row=r_, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"], name="Signal"), row=r_, col=1)
        fig.update_layout(height=850, xaxis_rangeslider_visible=False, legend_orientation="h")
        st.plotly_chart(fig, width="stretch")
        st.download_button("Download chart data (CSV)", to_csv_bytes(df),
                           file_name=f"{symbol}_chart_data.csv", mime="text/csv")

# ── RISK & POSITION SIZE ─────────────────────────────────────────────────────
with tab_risk:
    r = run(api.risk, symbol, atr_mult)
    if r and (r.get("price") is None or r.get("suggested_stop_loss") is None):
        st.warning("Price data for this symbol is incomplete right now. Try again later or try another symbol.")
        r = None
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

        st.divider()
        st.subheader("Position size calculator")
        st.caption("Sizes the trade so that hitting your stop loses only the % of capital you choose.")
        p1, p2, p3, p4 = st.columns(4)
        capital = p1.number_input("Capital (₹)", min_value=1000.0, value=500000.0, step=10000.0)
        risk_pct = p2.number_input("Risk per trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        entry = p3.number_input("Entry price", min_value=0.01, value=float(r["price"]), step=1.0)
        stop = p4.number_input("Stop price", min_value=0.01, value=float(r["suggested_stop_loss"]), step=1.0)
        reward_mult = st.slider("Target as multiple of risk (R)", 1.0, 5.0, 2.0, 0.5)

        if stop >= entry:
            st.warning("Stop must be below entry for a long position.")
        else:
            risk_amount = capital * risk_pct / 100
            per_share = entry - stop
            qty = int(risk_amount // per_share)
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Quantity", f"{qty:,}")
            q2.metric("Position value (₹)", f"{qty * entry:,.0f}")
            q3.metric("% of capital", f"{qty * entry / capital * 100:.1f}%")
            q4.metric(f"Target ({reward_mult}R)", f"{entry + reward_mult * per_share:,.2f}")
            st.caption(f"Max loss if stopped out: about ₹{qty * per_share:,.0f} (before fees and slippage).")
            if qty * entry > capital:
                st.warning("Position value exceeds your capital. Reduce risk % or widen the stop gap.")

# ── RELATIVE STRENGTH ────────────────────────────────────────────────────────
with tab_rs:
    rs = run(api.relative_strength, symbol)
    if rs:
        table = pd.DataFrame(rs["vs_nifty50"]).T
        st.dataframe(table, width="stretch")
        st.bar_chart(table["outperformance_pct"])
        st.caption("Positive outperformance means the stock beat the Nifty 50 over that period.")

# ── COMPARE & CORRELATION ────────────────────────────────────────────────────
with tab_compare:
    syms = st.text_input("Symbols (comma separated, max 10)", "TCS,INFY,WIPRO,HCLTECH")
    if st.button("Compare, snapshot and correlate"):
        with st.spinner("Fetching..."):
            cmp_res = run(api.compare, syms)
            wl_res = run(api.watchlist, syms)
        if cmp_res:
            st.markdown("**Ranked by signal score**")
            st.dataframe(pd.DataFrame(cmp_res["ranked"]), width="stretch")
        if wl_res:
            st.markdown("**Watchlist snapshot**")
            st.dataframe(pd.DataFrame(wl_res["stocks"]), width="stretch")

        # Correlation of daily returns over the last year
        sym_list = [x.strip().upper() for x in syms.split(",") if x.strip()][:10]
        closes = {}
        for sym in sym_list:
            try:
                h = api.get_history(api.to_ticker(sym))
                if not h.empty:
                    closes[sym] = h["Close"].tail(252)
            except Exception:
                continue
        if len(closes) >= 2:
            corr = pd.DataFrame(closes).pct_change().dropna().corr().round(2)
            heat = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                text=corr.values, texttemplate="%{text}", colorscale="RdBu", reversescale=True))
            heat.update_layout(title="Correlation of daily returns (1 year)", height=500)
            st.plotly_chart(heat, width="stretch")
            st.caption("Values near 1 mean the stocks move together, so holding both adds little diversification.")

# ── SCREENER ─────────────────────────────────────────────────────────────────
with tab_screener:
    min_score = st.slider("Minimum score", -5, 8, 3)
    custom = st.text_input("Custom universe (optional, comma separated, max 25)", "")
    if st.button("Run screener"):
        with st.spinner("Scanning... the first run can take a minute"):
            res = run(api.screener, min_score, custom)
        if res:
            if res["matches"]:
                st.dataframe(pd.DataFrame(res["matches"]), width="stretch")
            else:
                st.info("No stocks met that score. Try lowering the minimum.")

# ── BACKTEST ─────────────────────────────────────────────────────────────────
with tab_backtest:
    st.subheader(f"Backtest on {symbol}")
    b1, b2, b3 = st.columns(3)
    strategy = b1.selectbox("Strategy", ["Moving-average crossover", "RSI mean reversion"])
    bt_period = b2.selectbox("History", ["2y", "5y", "10y"], index=1)
    fee_bps = b3.number_input("Cost per trade (bps)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)

    fast, slow, rsi_buy, rsi_sell = 50, 200, 30, 70
    if strategy == "Moving-average crossover":
        m1, m2 = st.columns(2)
        fast = m1.number_input("Fast MA", 5, 100, 50)
        slow = m2.number_input("Slow MA", 20, 300, 200)
    else:
        m1, m2 = st.columns(2)
        rsi_buy = m1.number_input("Buy when RSI below", 5, 50, 30)
        rsi_sell = m2.number_input("Sell when RSI above", 50, 95, 70)

    try:
        hist_bt = api.get_history(api.to_ticker(symbol), period=bt_period)
    except Exception as e:
        hist_bt = None
        st.error(f"Could not load history: {e}")

    if hist_bt is not None and not hist_bt.empty:
        if strategy == "Moving-average crossover" and fast >= slow:
            st.warning("Fast MA must be shorter than slow MA.")
        else:
            equity, buy_hold, position, s_stats, b_stats = run_backtest(
                hist_bt["Close"], strategy, int(fast), int(slow), rsi_buy, rsi_sell, fee_bps)

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=equity.index, y=equity * 100, name="Strategy (start = 100)"))
            fig_bt.add_trace(go.Scatter(x=buy_hold.index, y=buy_hold * 100, name="Buy & hold"))
            fig_bt.update_layout(height=450, yaxis_title="Growth of 100", legend_orientation="h")
            st.plotly_chart(fig_bt, width="stretch")

            comparison = pd.DataFrame({"Strategy": s_stats, "Buy & hold": b_stats})
            st.dataframe(comparison, width="stretch")
            st.caption("Long-only, trades the day after a signal, includes your cost per trade. "
                       "Past results do not predict future returns. Prices are dividend/split adjusted.")
            if s_stats["Trades (entries)"] < 5:
                st.info("Few trades, so these statistics are not very reliable.")

# ── PORTFOLIO ────────────────────────────────────────────────────────────────
with tab_portfolio:
    st.subheader("Portfolio tracker")
    st.caption("Edit the sample rows with your own holdings. Add rows with the + at the bottom of the table. "
               "Nothing is saved between visits.")
    default_pf = pd.DataFrame({
        "symbol": ["TCS", "INFY"],
        "qty": [10, 20],
        "buy_price": [3500.0, 1500.0],
    })
    edited = st.data_editor(default_pf, num_rows="dynamic", width="stretch", key="pf_editor")

    pf_rows = []
    for _, row in edited.dropna(subset=["symbol", "qty", "buy_price"]).iterrows():
        sym = str(row["symbol"]).strip().upper()
        if not sym:
            continue
        try:
            h = api.get_history(api.to_ticker(sym))
            if h.empty:
                raise ValueError("no data")
            price = float(h["Close"].iloc[-1])
        except Exception:
            st.warning(f"Could not fetch a price for {sym}.")
            continue
        invested = float(row["qty"]) * float(row["buy_price"])
        value = float(row["qty"]) * price
        pf_rows.append({
            "Symbol": sym, "Qty": row["qty"], "Buy price": row["buy_price"],
            "Current price": round(price, 2), "Invested": round(invested, 2),
            "Value": round(value, 2), "P&L": round(value - invested, 2),
            "P&L %": round((value / invested - 1) * 100, 2) if invested else None,
        })

    if pf_rows:
        pf = pd.DataFrame(pf_rows)
        tot_inv, tot_val = pf["Invested"].sum(), pf["Value"].sum()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Invested (₹)", f"{tot_inv:,.0f}")
        k2.metric("Current value (₹)", f"{tot_val:,.0f}")
        k3.metric("P&L (₹)", f"{tot_val - tot_inv:,.0f}")
        k4.metric("P&L %", f"{(tot_val / tot_inv - 1) * 100:.2f}%" if tot_inv else "n/a")
        st.dataframe(pf, width="stretch")
        pie = go.Figure(go.Pie(labels=pf["Symbol"], values=pf["Value"], hole=0.4))
        pie.update_layout(title="Allocation by current value", height=400)
        st.plotly_chart(pie, width="stretch")
        st.download_button("Download portfolio (CSV)", to_csv_bytes(pf),
                           file_name="portfolio.csv", mime="text/csv")
    else:
        st.info("Add at least one holding with symbol, quantity and buy price.")
