import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

# --- Streamlit Setup ---
st.set_page_config(page_title="📊 Advanced Stock & Futures Scanner", layout="wide")
st.title("📊 US Stock & Futures Trend & Signal Scanner (Advanced)")

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

# --- Detect Patterns ---
def detect_pattern(data):
    pattern = "-"
    if len(data) >= 20:
        highs = data['High'].tail(20)
        lows = data['Low'].tail(20)
        # Double Top
        if np.isclose(highs.max(), highs.nlargest(2).iloc[-1], rtol=0.01):
            pattern = "Double Top"
        # Double Bottom
        elif np.isclose(lows.min(), lows.nsmallest(2).iloc[-1], rtol=0.01):
            pattern = "Double Bottom"
        # Channel (optional)
        elif highs.max() - lows.min() > 0:
            pattern = "Channel"
    return pattern

# --- Support & Resistance ---
def support_resistance(data, window=20):
    support = data['Low'].tail(window).min()
    resistance = data['High'].tail(window).max()
    return support, resistance

# --- Analyze Ticker ---
def analyze_ticker(ticker):
    try:
        data = get_data(ticker, period, timeframe)
        if data.empty:
            raise ValueError("No data available")

        latest_price = float(data["Close"].iloc[-1])
        prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else latest_price
        pct_change = ((latest_price - prev_close) / prev_close * 100) if prev_close else 0.0

        # Volume check
        avg_volume = float(data["Volume"].mean())
        if "=F" in ticker:
            volume_pass = avg_volume >= 1000
        else:
            volume_pass = avg_volume >= min_volume

        # Trend Slope
        slope_pass = False
        slope_val = np.nan
        if len(data["Close"]) >= 45:
            recent_close = data["Close"].tail(45).values
            x = np.arange(len(recent_close))
            slope_val = float(np.polyfit(x, recent_close, 1)[0])
            slope_pass = slope_val > 0

        # RSI
        rsi_series = compute_rsi(data['Close'])
        rsi_last = float(rsi_series.dropna().iloc[-1])
        if rsi_last >= 70:
            rsi_status = "Overbought"
        elif rsi_last <= 30:
            rsi_status = "Oversold"
        else:
            rsi_status = "Neutral"

        # Pattern
        pattern = detect_pattern(data)

        # Support/Resistance
        support, resistance = support_resistance(data)

        # Trend Strength Score
        score = 0
        micro_signals = []
        if slope_pass: score += 1; micro_signals.append("Slope +")
        if rsi_last <= 30: score += 1; micro_signals.append("RSI oversold")
        if rsi_last >= 70: score -= 1; micro_signals.append("RSI overbought")
        if volume_pass: score +=1; micro_signals.append("Volume ok")
        if pct_change >= pct_threshold: score +=1; micro_signals.append("After-hours up")
        if pct_change <= -pct_threshold: score -=1; micro_signals.append("After-hours down")

        # % bullish / bearish
        bullish_pct = round((sum([1 for m in micro_signals if "+" in m])/len(micro_signals))*100,1) if micro_signals else 0
        bearish_pct = round((sum([1 for m in micro_signals if "-" in m])/len(micro_signals))*100,1) if micro_signals else 0

        # Signal & Recommendation
        if score >= 3:
            signal = "🟢 Bullish"
            recommendation = "Buy"
        elif score <= -2:
            signal = "🔴 Bearish"
            recommendation = "Short"
        else:
            signal = "🟠 Neutral"
            recommendation = "Wait"

        return {
            "Ticker": ticker,
            "Price": round(latest_price,2),
            "Slope": round(slope_val,2) if not np.isnan(slope_val) else "-",
            "Volume": "✅ PASS" if volume_pass else "❌ FAIL",
            "After-hours %": round(pct_change,2),
            "RSI": round(rsi_last,2),
            "RSI Status": rsi_status,
            "Pattern": pattern,
            "Support": round(support,2),
            "Resistance": round(resistance,2),
            "Trend Score": score,
            "Bullish %": bullish_pct,
            "Bearish %": bearish_pct,
            "Signal": signal,
            "Recommendation": recommendation,
            "Summary": ", ".join(micro_signals)
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
            "Pattern": "-",
            "Support": "-",
            "Resistance": "-",
            "Trend Score": "-",
            "Bullish %": "-",
            "Bearish %": "-",
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
    df = df.sort_values("Bullish %", ascending=False).reset_index(drop=True)
    st.dataframe(df, width='stretch', height=800)

st.caption("Built with ❤️ using Streamlit & Yahoo Finance API | Buy/Short/Wait recommendations | Patterns included")
