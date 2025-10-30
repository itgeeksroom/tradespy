import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

# --- Streamlit Page Setup ---
st.set_page_config(page_title="📊 Stock & Futures Scanner", layout="centered")
st.title("📊 US Stock & Futures Volatility & Trend Scanner")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Scanner Settings")

tickers_input = st.sidebar.text_input(
    "Enter stock/futures symbols (comma separated):", "TSLA, AAPL, NVDA, ES=F"
)

period = st.sidebar.selectbox("Data Period:", ["3mo", "6mo", "1y"], index=1)
min_volume = st.sidebar.number_input("Minimum Avg Volume", value=2_000_000, step=500_000)
show_chart = st.sidebar.checkbox("Show chart for selected ticker", value=True)

# --- Parse tickers ---
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# --- Cached data fetch ---
@st.cache_data(ttl=3600)
def get_data(ticker, period):
    """Fetch OHLCV data with caching."""
    return yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)

# --- Stock/Futures analysis function ---
def analyze_ticker(ticker):
    try:
        data = get_data(ticker, period)
        if data.empty:
            return {
                "Ticker": ticker,
                "Avg Volume": "❌ N/A",
                "IV30/RV30": "❌ N/A",
                "TS Slope": "❌ N/A",
                "Signal": "🟠",
                "Summary": "No data available",
            }

        # --- Average Volume ---
        avg_volume = float(data["Volume"].mean())
        volume_pass = avg_volume >= min_volume

        # --- Realized Volatility (RV30) & Mock IV30 ---
        rv30 = data["Close"].pct_change().rolling(30).std() * np.sqrt(252)
        iv30 = rv30 * 0.8  # Mock IV

        rv30_last = float(rv30.dropna().iloc[-1]) if not rv30.dropna().empty else np.nan
        iv30_last = float(iv30.dropna().iloc[-1]) if not iv30.dropna().empty else np.nan

        iv_rv_ratio = iv30_last / rv30_last if (not np.isnan(rv30_last) and rv30_last != 0) else np.nan
        iv_rv_pass = bool(iv_rv_ratio > 1) if not np.isnan(iv_rv_ratio) else False

        # --- Trend Slope (last 45 days) ---
        slope_pass = False
        if len(data["Close"]) >= 45:
            recent_close = data["Close"].tail(45).values
            x = np.arange(len(recent_close))
            slope = np.polyfit(x, recent_close, 1)[0]
            slope_pass = slope > 0

        # --- Summary logic & signal ---
        if np.isnan(iv_rv_ratio):
            summary = f"{ticker} data insufficient"
            signal = "🟠"
        elif iv_rv_ratio < 1:
            summary = f"{ticker} low implied volatility"
            signal = "🟠"
        elif slope_pass and volume_pass:
            summary = f"{ticker} bullish momentum"
            signal = "🟢"
        elif not slope_pass and volume_pass:
            summary = f"{ticker} bearish pressure"
            signal = "🔴"
        else:
            summary = f"{ticker} sideways / uncertain"
            signal = "🟠"

        # --- Format pass/fail ---
        def pf(val):
            return "✅ PASS" if val else "❌ FAIL"

        return {
            "Ticker": ticker,
            "Avg Volume": pf(volume_pass),
            "IV30/RV30": pf(iv_rv_pass),
            "TS Slope": pf(slope_pass),
            "Signal": signal,
            "Summary": summary,
        }

    except Exception as e:
        return {
            "Ticker": ticker,
            "Avg Volume": "❌ Error",
            "IV30/RV30": "❌ Error",
            "TS Slope": "❌ Error",
            "Signal": "🟠",
            "Summary": f"Error: {e}",
        }

# --- Run scanner button ---
if st.button("🔍 Run Scanner"):
    with st.spinner("Fetching and analyzing data..."):
        results = [analyze_ticker(t) for t in tickers]
        df = pd.DataFrame(results)

        # --- Display DataFrame ---
        st.dataframe(df, width='stretch')
        st.success("✅ Scan complete!")

        # --- Optional: Chart for selected ticker ---
        if show_chart:
            selected_ticker = st.selectbox("📈 View chart for:", tickers)
            if selected_ticker:
                chart_data = get_data(selected_ticker, period)
                if not chart_data.empty:
                    st.line_chart(chart_data["Close"])
                else:
                    st.warning("No chart data available for this ticker.")

st.caption("Built with ❤️ using Streamlit & Yahoo Finance API")
