import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

# --- Streamlit Setup ---
st.set_page_config(page_title="📊 Stable Stock & Futures Scanner", layout="wide")
st.title("📊 Stable US Stock & Futures Trend Scanner")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Scanner Settings")
tickers_input = st.sidebar.text_input(
    "Enter stock/futures symbols (comma separated):",
    "TSLA, AAPL, NVDA, ES=F, NQ=F"
)
timeframe = st.sidebar.selectbox(
    "Select Timeframe:",
    ["1d", "4h", "1h", "5m"], index=0
)
min_volume = st.sidebar.number_input("Minimum Avg Volume (stocks)", value=2_000_000, step=500_000)

# --- Parse tickers ---
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# --- Cached Data Fetch ---
@st.cache_data(ttl=1800)
def get_data(ticker, interval):
    try:
        period_map = {"1d": "2y", "4h": "180d", "1h": "60d", "5m": "60d"}
        data = yf.download(ticker, period=period_map.get(interval, "2y"), interval=interval,
                           progress=False, auto_adjust=False)
        return data
    except Exception:
        return pd.DataFrame()

# --- Compute RSI ---
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi.dropna()

# --- Trend slope ---
def compute_slope(series, window=45):
    if len(series) < window:
        return np.nan, False
    y = series.tail(window).values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return slope, slope > 0

# --- Analyze ticker ---
def analyze_ticker(ticker):
    data = get_data(ticker, timeframe)
    if data.empty:
        return {
            "Ticker": ticker,
            "Price": "-",
            "Trend": "-",
            "Volume": "-",
            "RSI": "-",
            "Signal": "🟠",
            "Summary": "No data available"
        }

    latest_price = round(float(data["Close"].iloc[-1]), 2)
    avg_volume = float(data["Volume"].mean())
    volume_pass = avg_volume >= min_volume if "=F" not in ticker else True

    slope, slope_pass = compute_slope(data["Close"])
    trend = "Uptrend" if slope_pass else "Downtrend" if slope_pass is not None else "Sideway"

    rsi_series = compute_rsi(data["Close"])
    rsi_last = round(float(rsi_series.iloc[-1]), 2) if not rsi_series.empty else np.nan
    if np.isnan(rsi_last):
        rsi_status = "-"
    elif rsi_last >= 70:
        rsi_status = "Overbought"
    elif rsi_last <= 30:
        rsi_status = "Oversold"
    else:
        rsi_status = "Neutral"

    # Signal & Recommendation
    if slope_pass and volume_pass and (rsi_last < 70):
        signal = "🟢 Bullish"
    elif not slope_pass and volume_pass and (rsi_last > 30):
        signal = "🔴 Bearish"
    else:
        signal = "🟠 Neutral"

    summary_parts = []
    if slope_pass: summary_parts.append("Trend slope positive")
    else: summary_parts.append("Trend slope negative")
    summary_parts.append(f"RSI: {rsi_status}")
    summary_parts.append(f"Avg Volume: {'Pass' if volume_pass else 'Fail'}")
    summary = ", ".join(summary_parts)

    return {
        "Ticker": ticker,
        "Price": latest_price,
        "Trend": trend,
        "Volume": "✅ Pass" if volume_pass else "❌ Fail",
        "RSI": rsi_last,
        "Signal": signal,
        "Summary": summary
    }

# --- Run Scanner ---
if st.button("🔍 Run Scanner"):
    results = [analyze_ticker(t) for t in tickers]
    df = pd.DataFrame(results)
    st.dataframe(df, width='stretch', height=900)
