import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

# --- Streamlit Page Setup ---
st.set_page_config(page_title="📊 Stock & Futures Scanner", layout="centered")
st.title("📊 US Stock & Futures Scanner — Trend, Volatility & Flow")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Scanner Settings")
tickers_input = st.sidebar.text_input("Enter symbols (comma separated):", "TSLA, AAPL, ES=F")
period = st.sidebar.selectbox("Data Period:", ["3mo", "6mo", "1y"], index=1)
min_volume = st.sidebar.number_input("Minimum Avg Volume", value=2_000_000, step=500_000)
show_chart = st.sidebar.checkbox("Show chart for selected ticker", value=True)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# --- Cached data fetch ---
@st.cache_data(ttl=3600)
def get_data(ticker, period):
    return yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)

# --- Options flow for stocks ---
@st.cache_data(ttl=300)
def get_cp_ratio(ticker):
    if ticker.endswith("=F"):
        return np.nan, "Disabled"
    try:
        tk = yf.Ticker(ticker)
        if not tk.options:
            return np.nan, "No options"
        chain = tk.option_chain(tk.options[0])
        calls, puts = chain.calls, chain.puts
        call_vol = calls["volume"].fillna(0).sum()
        put_vol = puts["volume"].fillna(0).sum()
        cp_ratio = call_vol / put_vol if put_vol > 0 else np.nan
        bias = "Neutral"
        if not np.isnan(cp_ratio):
            if cp_ratio > 1.3:
                bias = "Bullish"
            elif cp_ratio < 0.7:
                bias = "Bearish"
        return cp_ratio, bias
    except:
        return np.nan, "Error"

# --- Analyze ticker ---
def analyze_stock(ticker):
    try:
        df = get_data(ticker, period)
        if df.empty:
            return {"Ticker": ticker, "Price": "-", "Trend": "-", "Volatility": "-", 
                    "Volume": "-", "Flow": "-", "Summary": "No data"}

        # Average volume
        avg_vol = float(df["Volume"].mean())
        volume_pass = avg_vol > min_volume

        # Trend slope (last 45 days)
        slope_pass = False
        if len(df["Close"]) >= 45:
            recent = df["Close"].tail(45)
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent, 1)[0]
            slope_pass = slope > 0

        # Volatility
        if ticker.endswith("=F"):  # Futures: ATR as volatility proxy
            atr = df["Close"].rolling(14).apply(lambda x: x.max()-x.min()).iloc[-1]
            vol_pass = atr > 0
        else:  # Stocks: RV30 & IV30
            rv30 = df["Close"].pct_change().rolling(30).std() * np.sqrt(252)
            iv30 = rv30 * 0.8
            rv_last = float(rv30.dropna().iloc[-1]) if not rv30.dropna().empty else np.nan
            iv_last = float(iv30.dropna().iloc[-1]) if not iv30.dropna().empty else np.nan
            iv_rv_ratio = iv_last / rv_last if (rv_last and not np.isnan(rv_last)) else np.nan
            vol_pass = bool(iv_rv_ratio > 1) if not np.isnan(iv_rv_ratio) else False

        # Volume spike
        vol_ma = df["Volume"].rolling(20).mean().iloc[-1]
        vol_signal = "Volume ↑" if df["Volume"].iloc[-1] > 1.5 * vol_ma else "Volume Normal"

        # Options flow
        cp_ratio, flow_bias = get_cp_ratio(ticker)

        # Summary logic
        summary = "Sideways / uncertain"
        if slope_pass and vol_pass and volume_pass:
            summary = "🔥 Strong Up Move Potential" if flow_bias=="Bullish" else "Uptrend Candidate"
        elif not slope_pass and vol_pass and volume_pass:
            summary = "⚠️ Downside Pressure" if flow_bias=="Bearish" else "Downtrend Candidate"
        elif not slope_pass and not vol_pass:
            summary = "Low volatility / weak trend"

        return {
            "Ticker": ticker,
            "Price": round(df["Close"].iloc[-1],2),
            "Trend": "Uptrend" if slope_pass else "Downtrend",
            "Volatility": "High" if vol_pass else "Low",
            "Volume": vol_signal,
            "Flow": flow_bias,
            "Summary": summary
        }

    except Exception as e:
        return {"Ticker": ticker, "Price": "-", "Trend": "❌ Error", "Volatility": "❌ Error",
                "Volume": "❌ Error", "Flow": "❌ Error", "Summary": f"Error: {e}"}

# --- Run Scanner ---
if st.button("🔍 Run Scanner"):
    if not tickers:
        st.warning("⚠️ Enter at least one symbol")
    else:
        results = [analyze_stock(t) for t in tickers]
        df = pd.DataFrame(results)
        st.subheader("📊 Scan Results")
        st.dataframe(df, width="stretch")

        # Highlight top setups
        bullish = df[df["Summary"].str.contains("🔥|Uptrend Candidate", regex=True, na=False)]
        bearish = df[df["Summary"].str.contains("⚠️|Downtrend Candidate", regex=True, na=False)]
        if not bullish.empty:
            st.success("🔥 Bullish Candidates")
            st.dataframe(bullish, width="stretch")
        if not bearish.empty:
            st.warning("⚠️ Bearish Candidates")
            st.dataframe(bearish, width="stretch")

        # Optional chart
        if show_chart:
            selected = st.selectbox("📈 View chart for:", tickers)
            if selected:
                chart_df = get_data(selected, period)
                if not chart_df.empty:
                    st.line_chart(chart_df["Close"])
                else:
                    st.warning("No chart data for this ticker.")

st.caption("Built with ❤️ using Streamlit & Yahoo Finance API | Stocks + Futures ready")
