"""
NSE Stock Analysis API - for n8n integration (full version)
================================================================================
Endpoints
---------
  GET /                              health check
  GET /analyze/{symbol}              technicals + fundamentals + signal score
  GET /chart/{symbol}?months=6       OHLCV + RSI/MACD/MA series for charts
  GET /watchlist?symbols=A,B,C       lightweight multi-stock snapshot
  GET /signal/{symbol}               rule-based score with reasons
  GET /risk/{symbol}?atr_mult=2      ATR stop-loss, support/resistance, volatility
  GET /relative-strength/{symbol}    returns vs Nifty 50 (1m/3m/6m/1y)
  GET /compare?symbols=A,B,C         ranked signal comparison
  GET /screener?min_score=3          scan a universe, return strong scorers

REQUIREMENTS:  pip install fastapi uvicorn yfinance pandas numpy
RUN LOCALLY:   uvicorn nse_analysis_api:app --reload --port 8000
RENDER START:  uvicorn nse_analysis_api:app --host 0.0.0.0 --port $PORT

Note: the signal score is a simple rules-based tool, not investment advice.
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NSE Stock Analysis API")

# Allow any frontend (v0.dev, Lovable, etc.) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── HELPERS: CACHE, TICKER, ROUNDING ─────────────────────────────────────────

_cache = {}


def cached(key, ttl, fn):
    """Tiny in-memory TTL cache so repeated n8n/frontend calls don't hammer Yahoo."""
    now = time.time()
    if key in _cache and now - _cache[key][0] < ttl:
        return _cache[key][1]
    val = fn()
    _cache[key] = (now, val)
    return val


def get_history(ticker, period="2y"):
    """Price history, cached for 15 minutes."""
    def fetch():
        h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        # Yahoo sometimes returns a latest row with a missing Close; drop it
        return h.dropna(subset=["Close"]) if not h.empty else h

    return cached(("hist", ticker, period), 900, fetch)


def to_ticker(symbol):
    s = symbol.upper().strip()
    return s if (s.endswith(".NS") or s.startswith("^")) else f"{s}.NS"


def safe_round(val, decimals=2):
    if val is None or pd.isna(val):
        return None
    return round(float(val), decimals)


def get_company_name(ticker):
    """Best-effort company name; falls back to the ticker if Yahoo blocks .info."""
    def fetch():
        try:
            info = yf.Ticker(ticker).get_info()
            return info.get("longName") or info.get("shortName") or ticker
        except Exception:
            return ticker
    return cached(("name", ticker), 86400, fetch)


# ── INDICATOR FUNCTIONS ──────────────────────────────────────────────────────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_roc(prices, period=12):
    return prices.pct_change(period) * 100


def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def calculate_moving_averages(prices, short_window=50, long_window=200):
    return prices.rolling(short_window).mean(), prices.rolling(long_window).mean()


def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0
    minus_dm[(minus_dm - plus_dm) < 0] = 0
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_52w_high_proximity(prices, window=252):
    return (prices / prices.rolling(window).max()) * 100


def calculate_bollinger_bands(prices, window=20, num_std=2):
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    percent_b = (prices - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle * 100
    return percent_b, bandwidth


# ── NEW INDICATORS ───────────────────────────────────────────────────────────

def calculate_atr(high, low, close, period=14):
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_obv(close, volume):
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


def calculate_stochastic(high, low, close, k=14, d=3):
    lowest, highest = low.rolling(k).min(), high.rolling(k).max()
    pct_k = 100 * (close - lowest) / (highest - lowest)
    return pct_k, pct_k.rolling(d).mean()


# ── FUNDAMENTALS ─────────────────────────────────────────────────────────────

def get_yoy_earnings_growth(net_income, tolerance_days=45):
    """Date-matched YoY growth (robust to irregularly reported quarters)."""
    if len(net_income) < 2:
        return None, None
    latest_date = net_income.index[0]
    latest_val = net_income.iloc[0]
    target_date = latest_date - pd.DateOffset(months=12)
    date_diffs = (net_income.index - target_date).days.to_series(index=net_income.index).abs()
    closest_idx = date_diffs.idxmin()
    if date_diffs[closest_idx] > tolerance_days:
        return None, None
    base_val = net_income.loc[closest_idx]
    if base_val == 0 or pd.isna(base_val):
        return None, None
    return round(((latest_val / base_val) - 1) * 100, 2), str(closest_idx.date())


def get_yoy_revenue_growth(revenue, tolerance_days=45):
    return get_yoy_earnings_growth(revenue, tolerance_days)


def get_fundamentals(stock):
    """Fundamentals computed from income statement and balance sheet.

    Deliberately avoids stock.info for the numbers - Yahoo blocks that endpoint
    on many cloud hosts (Render, Railway, Heroku)."""
    fundamentals = {
        "yoy_earnings_growth_pct": None,
        "earnings_base_quarter": None,
        "yoy_revenue_growth_pct": None,
        "revenue_base_quarter": None,
        "operating_margin_pct": None,
        "net_margin_pct": None,
        "roe_pct": None,
        "debt_to_equity": None,
        "market_cap_cr": None,
    }

    latest_revenue = None
    latest_net_income = None
    latest_operating_income = None

    try:
        income_stmt = stock.quarterly_income_stmt
        if income_stmt is not None:
            if "Net Income" in income_stmt.index:
                net_income = income_stmt.loc["Net Income"].dropna()
                growth, base_date = get_yoy_earnings_growth(net_income)
                fundamentals["yoy_earnings_growth_pct"] = growth
                fundamentals["earnings_base_quarter"] = base_date
                if len(net_income) > 0:
                    latest_net_income = net_income.iloc[0]

            if "Total Revenue" in income_stmt.index:
                revenue = income_stmt.loc["Total Revenue"].dropna()
                growth, base_date = get_yoy_revenue_growth(revenue)
                fundamentals["yoy_revenue_growth_pct"] = growth
                fundamentals["revenue_base_quarter"] = base_date
                if len(revenue) > 0:
                    latest_revenue = revenue.iloc[0]

            if "Operating Income" in income_stmt.index:
                op_income = income_stmt.loc["Operating Income"].dropna()
                if len(op_income) > 0:
                    latest_operating_income = op_income.iloc[0]

        if latest_revenue and latest_revenue != 0:
            if latest_net_income is not None:
                fundamentals["net_margin_pct"] = safe_round((latest_net_income / latest_revenue) * 100)
            if latest_operating_income is not None:
                fundamentals["operating_margin_pct"] = safe_round(
                    (latest_operating_income / latest_revenue) * 100
                )
    except Exception as e:
        fundamentals["_debug_income_stmt_error"] = str(e)

    try:
        balance_sheet = stock.quarterly_balance_sheet
        if balance_sheet is not None:
            equity = None
            total_debt = None

            for label in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if label in balance_sheet.index:
                    series = balance_sheet.loc[label].dropna()
                    if len(series) > 0:
                        equity = series.iloc[0]
                        break

            # Only "Total Debt" - "Net Debt" is a different quantity and would mislabel the ratio
            if "Total Debt" in balance_sheet.index:
                series = balance_sheet.loc["Total Debt"].dropna()
                if len(series) > 0:
                    total_debt = series.iloc[0]

            if equity and equity != 0:
                if total_debt is not None:
                    fundamentals["debt_to_equity"] = safe_round(total_debt / equity)
                # ROE: latest quarter net income annualized (x4) / latest equity (approximation)
                if latest_net_income is not None:
                    fundamentals["roe_pct"] = safe_round(((latest_net_income * 4) / equity) * 100)
    except Exception as e:
        fundamentals["_debug_balance_sheet_error"] = str(e)

    # Market cap via the lighter fast_info endpoint (separate try so failures don't matter)
    try:
        fast_info = stock.fast_info
        mcap = None
        for key in ("market_cap", "marketCap"):
            try:
                mcap = fast_info[key]
                if mcap:
                    break
            except Exception:
                continue
        if mcap:
            fundamentals["market_cap_cr"] = safe_round(mcap / 1e7)
    except Exception:
        pass

    return fundamentals


def get_cached_fundamentals(ticker):
    return cached(("fund", ticker), 3600, lambda: get_fundamentals(yf.Ticker(ticker)))


# ── TECHNICAL SNAPSHOT ───────────────────────────────────────────────────────

def get_technical_snapshot(hist):
    """Core technical calculation, shared by /analyze, /watchlist and /signal."""
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]

    daily_pct_change = close.pct_change()
    corporate_action_flag = bool((daily_pct_change.abs() > 0.15).any())

    rsi = calculate_rsi(close)
    roc = calculate_roc(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    adx = calculate_adx(high, low, close)
    high52 = calculate_52w_high_proximity(close)
    percent_b, bandwidth = calculate_bollinger_bands(close)

    ma_trend = None
    if not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]):
        ma_trend = "bullish" if ma50.iloc[-1] > ma200.iloc[-1] else "bearish"

    technical = {
        "current_price": safe_round(close.iloc[-1]),
        "rsi_14": safe_round(rsi.iloc[-1]),
        "roc_12": safe_round(roc.iloc[-1]),
        "macd_line": safe_round(macd_line.iloc[-1]),
        "macd_signal": safe_round(signal_line.iloc[-1]),
        "macd_histogram": safe_round(macd_hist.iloc[-1]),
        "ma_50": safe_round(ma50.iloc[-1]),
        "ma_200": safe_round(ma200.iloc[-1]),
        "ma_trend": ma_trend,
        "adx_14": safe_round(adx.iloc[-1]),
        "pct_of_52w_high": safe_round(high52.iloc[-1]),
        "bollinger_percent_b": safe_round(percent_b.iloc[-1], 3),
        "bollinger_bandwidth_pct": safe_round(bandwidth.iloc[-1]),
    }
    return technical, corporate_action_flag


# ── RULE-BASED SIGNAL SCORE ──────────────────────────────────────────────────

def build_signal(t, f):
    score = 0
    reasons = []

    def add(pts, text):
        nonlocal score
        score += pts
        reasons.append(f"{'+' if pts > 0 else ''}{pts}: {text}")

    if t["ma_trend"] == "bullish":
        add(2, "50-DMA above 200-DMA")
    elif t["ma_trend"] == "bearish":
        add(-2, "50-DMA below 200-DMA")

    rsi = t["rsi_14"]
    if rsi is not None:
        if rsi > 70:
            add(-1, f"RSI {rsi} is overbought")
        elif rsi < 30:
            add(1, f"RSI {rsi} is oversold (possible rebound)")
        elif 50 <= rsi <= 70:
            add(1, f"RSI {rsi} shows healthy momentum")

    hist_val = t["macd_histogram"]
    if hist_val is not None:
        if hist_val > 0:
            add(1, "MACD histogram positive")
        else:
            add(-1, "MACD histogram negative")

    adx = t["adx_14"]
    if adx is not None and adx > 25 and t["ma_trend"]:
        add(1 if t["ma_trend"] == "bullish" else -1, f"Strong trend (ADX {adx})")

    if t["pct_of_52w_high"] is not None and t["pct_of_52w_high"] >= 90:
        add(1, "Within 10% of 52-week high")

    eg = f.get("yoy_earnings_growth_pct")
    if eg is not None:
        if eg > 10:
            add(1, f"YoY earnings growth {eg}%")
        elif eg < 0:
            add(-1, f"YoY earnings decline {eg}%")

    de = f.get("debt_to_equity")
    if de is not None and de > 2:
        add(-1, f"High debt-to-equity {de}")

    if score >= 5:
        label = "Strong Bullish"
    elif score >= 2:
        label = "Bullish"
    elif score > -2:
        label = "Neutral"
    elif score > -5:
        label = "Bearish"
    else:
        label = "Strong Bearish"

    return {"score": score, "label": label, "reasons": reasons}


# ── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok", "message": "NSE Stock Analysis API is running"}


@app.get("/analyze/{symbol}")
def analyze_stock(symbol: str):
    """Full technical + fundamental analysis + signal score for one NSE stock."""
    ticker = to_ticker(symbol)
    try:
        hist = get_history(ticker)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")

        technical, corporate_action_flag = get_technical_snapshot(hist)
        fundamentals = get_cached_fundamentals(ticker)

        return {
            "symbol": ticker,
            "company_name": get_company_name(ticker),
            "timestamp": datetime.now().isoformat(),
            "corporate_action_flag": corporate_action_flag,
            "corporate_action_note": (
                "A >15% single-day price move was detected in the last 2 years - "
                "moving-average and Bollinger figures may be distorted by a demerger, "
                "split, or similar event. Verify before trusting the technical signals."
                if corporate_action_flag
                else None
            ),
            "technical": technical,
            "fundamentals": fundamentals,
            "signal": build_signal(technical, fundamentals),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing {ticker}: {str(e)}")


@app.get("/chart/{symbol}")
def chart_data(symbol: str, months: int = 6):
    """OHLCV + RSI, MACD, MA50, MA200 per trading day. months: 1-24 (default 6)."""
    ticker = to_ticker(symbol)
    months = min(max(months, 1), 24)
    try:
        hist = get_history(ticker)  # 2y so MA200 has enough lookback
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

        close = hist["Close"]
        rsi = calculate_rsi(close)
        macd_line, signal_line, _ = calculate_macd(close)
        ma50, ma200 = calculate_moving_averages(close)

        recent = hist.tail(months * 21)
        rows = []
        for date, row in recent.iterrows():
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": safe_round(row["Open"]),
                "high": safe_round(row["High"]),
                "low": safe_round(row["Low"]),
                "close": safe_round(row["Close"]),
                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else None,
                "rsi": safe_round(rsi.loc[date]),
                "macd_line": safe_round(macd_line.loc[date]),
                "macd_signal": safe_round(signal_line.loc[date]),
                "ma_50": safe_round(ma50.loc[date]),
                "ma_200": safe_round(ma200.loc[date]),
            })
        return {"symbol": ticker, "months": months, "data": rows}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data for {ticker}: {str(e)}")


@app.get("/watchlist")
def watchlist(symbols: str):
    """Multi-stock snapshot: /watchlist?symbols=RELIANCE,TCS,INFY (max 10)."""
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="Provide at least one symbol, e.g. ?symbols=RELIANCE,TCS")

    results = []
    for symbol in symbol_list:
        ticker = to_ticker(symbol)
        try:
            hist = get_history(ticker)
            if hist.empty:
                results.append({"symbol": ticker, "error": "No data found"})
                continue
            technical, corporate_action_flag = get_technical_snapshot(hist)
            results.append({
                "symbol": ticker,
                "company_name": get_company_name(ticker),
                "current_price": technical["current_price"],
                "rsi_14": technical["rsi_14"],
                "macd_histogram": technical["macd_histogram"],
                "ma_trend": technical["ma_trend"],
                "pct_of_52w_high": technical["pct_of_52w_high"],
                "corporate_action_flag": corporate_action_flag,
            })
        except Exception as e:
            results.append({"symbol": ticker, "error": str(e)})

    return {"count": len(results), "stocks": results}


@app.get("/signal/{symbol}")
def signal(symbol: str):
    """Technical + fundamental score with plain-English reasons."""
    ticker = to_ticker(symbol)
    hist = get_history(ticker)
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")
    technical, flag = get_technical_snapshot(hist)
    fundamentals = get_cached_fundamentals(ticker)
    return {
        "symbol": ticker,
        "company_name": get_company_name(ticker),
        "corporate_action_flag": flag,
        **build_signal(technical, fundamentals),
    }


@app.get("/risk/{symbol}")
def risk(symbol: str, atr_mult: float = 2.0):
    """ATR stop-loss, 20-day support/resistance, volatility, drawdown, volume and OBV."""
    ticker = to_ticker(symbol)
    hist = get_history(ticker)
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

    c, h, l, v = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
    atr = calculate_atr(h, l, c).iloc[-1]
    price = c.iloc[-1]
    obv = calculate_obv(c, v)
    pct_k, pct_d = calculate_stochastic(h, l, c)
    ann_vol = c.pct_change().std() * np.sqrt(252) * 100
    max_dd = ((c / c.cummax()) - 1).min() * 100
    avg_vol = v.tail(20).mean()

    return {
        "symbol": ticker,
        "price": safe_round(price),
        "atr_14": safe_round(atr),
        "atr_multiplier": atr_mult,
        "suggested_stop_loss": safe_round(price - atr_mult * atr),
        "support_20d": safe_round(l.tail(20).min()),
        "resistance_20d": safe_round(h.tail(20).max()),
        "annualized_volatility_pct": safe_round(ann_vol),
        "max_drawdown_2y_pct": safe_round(max_dd),
        "volume_vs_20d_avg": safe_round(v.iloc[-1] / avg_vol) if avg_vol else None,
        "obv_trend_20d": "rising" if obv.iloc[-1] > obv.iloc[-20] else "falling",
        "stochastic_k": safe_round(pct_k.iloc[-1]),
        "stochastic_d": safe_round(pct_d.iloc[-1]),
    }


@app.get("/relative-strength/{symbol}")
def relative_strength(symbol: str):
    """Stock returns vs Nifty 50 over 1m / 3m / 6m / 1y."""
    ticker = to_ticker(symbol)
    stock_hist = get_history(ticker)
    nifty_hist = get_history("^NSEI")
    if stock_hist.empty or nifty_hist.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker} or the Nifty index.")

    s, n = stock_hist["Close"], nifty_hist["Close"]
    out = {}
    for label, days in {"1m": 21, "3m": 63, "6m": 126, "1y": 252}.items():
        if len(s) > days and len(n) > days:
            sr = (s.iloc[-1] / s.iloc[-days - 1] - 1) * 100
            nr = (n.iloc[-1] / n.iloc[-days - 1] - 1) * 100
            out[label] = {
                "stock_pct": safe_round(sr),
                "nifty_pct": safe_round(nr),
                "outperformance_pct": safe_round(sr - nr),
            }
    return {"symbol": ticker, "vs_nifty50": out}


@app.get("/compare")
def compare(symbols: str):
    """Ranked signal comparison: /compare?symbols=TCS,INFY,WIPRO (max 6)."""
    rows = []
    for sym in [x for x in symbols.split(",") if x.strip()][:6]:
        try:
            r = signal(sym)
            rows.append({"symbol": r["symbol"], "score": r["score"], "label": r["label"]})
        except Exception as e:
            rows.append({"symbol": sym, "error": str(getattr(e, "detail", e))})
    return {"ranked": sorted(rows, key=lambda x: x.get("score", -99), reverse=True)}


NIFTY_SAMPLE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL",
    "ITC", "LT", "AXISBANK", "KOTAKBANK", "HINDUNILVR", "MARUTI", "SUNPHARMA", "TITAN",
]


@app.get("/screener")
def screener(min_score: int = 3, symbols: str = ""):
    """Scan a universe (default: sample of Nifty stocks) for stocks scoring >= min_score.
    Pass your own list with ?symbols=A,B,C (max 25). First run is slow; results are cached."""
    universe = [x.strip() for x in symbols.split(",") if x.strip()] or NIFTY_SAMPLE
    hits = []
    for sym in universe[:25]:
        try:
            r = signal(sym)
            if r["score"] >= min_score:
                hits.append({"symbol": r["symbol"], "score": r["score"], "label": r["label"]})
        except Exception:
            continue
    return {"min_score": min_score, "matches": sorted(hits, key=lambda x: -x["score"])}
