import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

# --- Streamlit Setup ---
st.set_page_config(page_title="📊 Stock & Futures Scanner", layout="centered")
st.title("📊 US Stock & Futures Trend & Signal Scanner")

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Scanner Settings")

tickers_input = st.sidebar.text_input(
    "Enter stock/futures symbols (comma separated):",
    "TSLA, AAPL, NVDA, ES=F, NQ=F"
)

period = st.sidebar.selectbox("Historical Data Period:", ["3mo", "6mo", "1y"], index=1)
min_volume = st.sidebar.number_input("Minimum Avg Volume (for stocks)", value=2_000_000, step=500_000)
show_chart = st.sidebar.checkbox("Show chart for selected ticker", value=True)
pct_threshold = st.sidebar.number_input("After-hours / pre-market % change threshold", value=1.0, step=0.1)

# --- Parse tickers ---
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# --- Cached data fetch ---
@st.cache_data(ttl=3600)
def get_data(ticker, period):
    """Fetch historical OHLCV data with caching."""
    return yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)

# --- Analysis function ---
def analyze_ticker(ticker):
    try:
        data = get_data(ticker, period)
        tk = yf.Ticker(ticker)
        info = tk.info

        # Latest price (after-hours / pre-market / regular)
        latest_price = info.get("postMarketPrice") or info.get("preMarketPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose")
        pct_change = (latest_price - prev_close) / prev_close * 100 if latest_price and prev_close else 0

        # Average volume
        avg_volume = float(data["Volume"].mean()) if not data.empty else 0
        volume_pass = avg_volume >= min_volume

        # Historical trend slope
        slope_pass = False
        slope = np.nan
        if len(data["Close"]) >= 45:
            recent_close = data["Close"].tail(45).values
            x = np.arange(len(recent_close))
            slope_array = np.polyfit(x, recent_close, 1)  # returns [slope, intercept]
            slope = float(slope_array[0])
            slope_pass = slope > 0

        # Determine signal based on after-hours / slope
        if pct_change >= pct_threshold:
            signal = "🟢 Bullish"
            summary = "Uptrend detected after hours / pre-market"
        elif pct_change <= -pct_threshold:
            signal = "🔴 Bearish"
            summary = "Downtrend detected after hours / pre-market"
        else:
            # Use slope if after-hours move is small
            if slope_pass:
                signal = "🟢 Bullish"
                summary = "Uptrend based on historical momentum"
            elif not slope_pass and not np.isnan(slope):
                signal = "🔴 Bearish"
                summary = "Downtrend based on historical momentum"
            else:
                signal = "🟠 Neutral"
                summary = "Sideways / uncertain"

        # Buy / Short guidance
        if signal == "🟢 Bullish" and volume_pass:
            recommendation = "Buy"
        elif signal == "🔴 Bearish" and volume_pass:
            recommendation = "Short"
        else:
            recommendation = "Wait"

        # Format volume pass/fail
        def pf(val):
            return "✅ PASS" if val else "❌ FAIL"

        return {
            "Ticker": ticker,
            "Latest Price": latest_price or "-",
            "Trend Slope": round(slope, 2) if not np.isnan(slope) else "-",
            "Volume Check": pf(volume_pass),
            "After-hours %": round(pct_change, 2),
            "Signal": signal,
            "Recommendation": recommendation,
            "Summary": summary
        }

    except Exception as e:
        return {
            "Ticker": ticker,
            "Latest Price": "-",
            "Trend Slope": "-",
            "Volume Check": "❌ Error",
            "After-hours %": "-",
            "Signal": "🟠",
            "Recommendation": "Error",
            "Summary": f"Error: {e}"
        }

# --- Run scanner ---
if st.button("🔍 Run Scanner"):
    with st.spinner("Fetching and analyzing data..."):
        results = [analyze_ticker(t) for t in tickers]
        df = pd.DataFrame(results)

        # Sort by After-hours % change
        df = df.sort_values("After-hours %", ascending=False).reset_index(drop=True)
        st.dataframe(df, width='stretch')
        st.success("✅ Scan complete!")

        # Optional chart for selected ticker
        if show_chart:
            selected_ticker = st.selectbox("📈 View chart for:", tickers)
            if selected_ticker:
                chart_data = get_data(selected_ticker, period)
                if not chart_data.empty:
                    st.line_chart(chart_data["Close"])
                else:
                    st.warning("No chart data available for this ticker.")

st.caption("Built with ❤️ using Streamlit & Yahoo Finance API")
