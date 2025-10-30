import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import io

# --- Page Setup ---
st.set_page_config(page_title="📊 US Stock & Futures Scanner", layout="centered")
st.title("📊 Stock & Futures Scanner — Trend, Volatility & Flow")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Scanner Settings")
tickers_input = st.sidebar.text_input("Enter symbols (comma separated):", "TSLA, AAPL, ES=F")
period = st.sidebar.selectbox("Data Period:", ["3mo", "6mo", "1y"], index=1)
min_volume = st.sidebar.number_input("Minimum Avg Volume", value=2_000_000, step=500_000)
show_chart = st.sidebar.checkbox("Show chart for selected ticker", value=True)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# --- Cached Data Fetch ---
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
        call_vol = float(calls["volume"].fillna(0).sum())
        put_vol = float(puts["volume"].fillna(0).sum())
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

# --- Analyze Ticker ---
def analyze_stock(ticker):
    try:
        df = get_data(ticker, period)
        if df.empty:
            return {"Ticker": ticker, "Price": "-", "Trend": "-", "Volatility": "-", 
                    "Volume": "-", "Flow": "-", "Summary": "No data"}

        # --- Avg Volume ---
        avg_vol = float(df["Volume"].mean())
        volume_pass = avg_vol > min_volume

        # --- Trend Slope ---
        slope_pass = False
        if len(df["Close"]) >= 20:  # shorter lookback for faster response
            recent = df["Close"].tail(20).to_numpy()
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent, 1)[0]
            slope_pass = slope > 0

        # --- Volatility ---
        if ticker.endswith("=F"):
            recent_close = df["Close"].tail(14).to_numpy()
            atr = recent_close.max() - recent_close.min()
            vol_pass = atr > 0
        else:
            rv30 = df["Close"].pct_change().rolling(30).std() * np.sqrt(252)
            iv30 = rv30 * 0.8
            rv_last = rv30.dropna().to_numpy()[-1] if not rv30.dropna().empty else np.nan
            iv_last = iv30.dropna().to_numpy()[-1] if not iv30.dropna().empty else np.nan
            vol_pass = False
            if not np.isnan(rv_last) and rv_last != 0 and not np.isnan(iv_last):
                vol_pass = (iv_last / rv_last) > 0.8  # more sensitive

        # --- Volume spike ---
        vol_ma = df["Volume"].rolling(20).mean().to_numpy()[-1]
        vol_signal = "Volume ↑" if df["Volume"].to_numpy()[-1] > 1.2 * vol_ma else "Volume Normal"

        # --- Options flow ---
        cp_ratio, flow_bias = get_cp_ratio(ticker)

        # --- Nuanced Summary ---
        if slope_pass and vol_pass and volume_pass:
            summary = "🔥 Strong Up Move Potential" if flow_bias=="Bullish" else "Uptrend Candidate"
        elif slope_pass and (vol_pass or volume_pass):
            summary = "Moderate Uptrend"
        elif not slope_pass and vol_pass and volume_pass:
            summary = "⚠️ Downside Pressure" if flow_bias=="Bearish" else "Downtrend Candidate"
        elif not slope_pass and (vol_pass or volume_pass):
            summary = "Moderate Downtrend"
        else:
            summary = "Sideways / uncertain"

        return {
            "Ticker": ticker,
            "Price": round(float(df["Close"].to_numpy()[-1]),2),
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
        st.session_state["last_scan"] = df  # store results

# --- Display Results ---
if "last_scan" in st.session_state:
    df = st.session_state["last_scan"]
    st.subheader("📊 Scan Results")
    st.dataframe(df, width="stretch")

    # --- Chart Dropdown ---
    if show_chart:
        selected = st.selectbox("📈 View chart for:", tickers)
        if selected:
            chart_df = get_data(selected, period)
            if not chart_df.empty:
                st.line_chart(chart_df["Close"])
            else:
                st.warning("No chart data for this ticker.")
