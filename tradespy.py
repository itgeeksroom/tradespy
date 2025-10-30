import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

# --- Streamlit Setup ---
st.set_page_config(page_title="📊 Stock & Futures Scanner", layout="wide")
st.title("📊 US Stock & Futures Trend & Signal Scanner")

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Scanner Settings")
tickers_input = st.sidebar.text_input(
    "Enter stock/futures symbols (comma separated):",
    "TSLA, AAPL, NVDA, ES=F, NQ=F"
)
timeframe = st.sidebar.selectbox(
    "Select Timeframe:",
    ["1m", "5m", "15m", "1h", "4h", "1d"], index=5
)
min_volume = st.sidebar.number_input("Minimum Avg Volume (stocks)", value=2_000_000, step=500_000)
pct_threshold = st.sidebar.number_input("After-hours % change threshold", value=1.0, step=0.1)

# --- Period mapping ---
period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "180d", "4h": "2y", "1d": "2y"}
period = period_map[timeframe]

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# --- Cached data fetch ---
@st.cache_data(ttl=3600)
def get_data(ticker, period, interval):
    return yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)

# --- Compute RSI ---
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Analyze Ticker ---
def analyze_ticker(ticker):
    try:
        data = get_data(ticker, period, timeframe)

        latest_price = float(data["Close"].iloc[-1]) if not data.empty else np.nan
        prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else latest_price
        pct_change = ((latest_price - prev_close) / prev_close * 100) if prev_close else 0.0

        # Volume check
        avg_volume = float(data["Volume"].mean()) if not data.empty else 0.0
        if "=F" in ticker:
            volume_pass = avg_volume >= 1000
        else:
            volume_pass = avg_volume >= min_volume

        # Trend slope
        slope_pass = False
        slope_val = np.nan
        if len(data["Close"]) >= 45:
            recent_close = data["Close"].tail(45).values
            x = np.arange(len(recent_close))
            slope_val = float(np.polyfit(x, recent_close, 1)[0])
            slope_pass = slope_val > 0

        # RSI calculation
        rsi_series = compute_rsi(data['Close'])
        rsi_last = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else np.nan
        if not np.isnan(rsi_last):
            if rsi_last >= 70:
                rsi_status = "Overbought"
            elif rsi_last <= 30:
                rsi_status = "Oversold"
            else:
                rsi_status = "Neutral"
        else:
            rsi_status = "-"

        # Signal & Summary
        if pct_change >= pct_threshold:
            signal = "🟢 Bullish"
            summary = "Uptrend detected after hours / pre-market"
        elif pct_change <= -pct_threshold:
            signal = "🔴 Bearish"
            summary = "Downtrend detected after hours / pre-market"
        else:
            if slope_pass:
                signal = "🟢 Bullish"
                summary = "Uptrend based on historical momentum"
            elif not slope_pass and not np.isnan(slope_val):
                signal = "🔴 Bearish"
                summary = "Downtrend based on historical momentum"
            else:
                signal = "🟠 Neutral"
                summary = "Sideways / uncertain"

        if signal == "🟢 Bullish" and volume_pass:
            recommendation = "Buy"
        elif signal == "🔴 Bearish" and volume_pass:
            recommendation = "Short"
        else:
            recommendation = "Wait"

        def pf(val):
            return "✅ PASS" if val else "❌ FAIL"

        return {
            "Ticker": ticker,
            "Price": round(latest_price,2) if not np.isnan(latest_price) else "-",
            "Slope": round(slope_val, 2) if not np.isnan(slope_val) else "-",
            "Volume": pf(volume_pass),
            "After-hours %": round(pct_change, 2),
            "RSI": round(rsi_last, 2) if not np.isnan(rsi_last) else "-",
            "RSI Status": rsi_status,
            "Signal": signal,
            "Recommendation": recommendation,
            "Summary": summary
        }

    except Exception as e:
        return {
            "Ticker": ticker,
            "Price": "-",
            "Slope": "-",
            "Volume": "❌ Error",
            "After-hours %": "-",
            "RSI": "-",
            "RSI Status": "-",
            "Signal": "🟠",
            "Recommendation": "Error",
            "Summary": f"Error: {e}"
        }

# --- Session State ---
if "results" not in st.session_state:
    st.session_state.results = None

# --- Run Scanner ---
if st.button("🔍 Run Scanner"):
    with st.spinner("Fetching & analyzing data..."):
        st.session_state.results = [analyze_ticker(t) for t in tickers]

# --- Display Table ---
if st.session_state.results:
    df = pd.DataFrame(st.session_state.results)
    df = df.sort_values("After-hours %", ascending=False).reset_index(drop=True)
    st.dataframe(df, width='stretch', height=800)  # Full table on same screen

st.caption("Built with ❤️ using Streamlit & Yahoo Finance API | Buy/Short/Wait recommendations")
