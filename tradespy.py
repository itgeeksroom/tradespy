import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Daily Stock & Futures Scanner", layout="wide")
st.title("📈 Daily Stock & Futures Scanner — Stocks + Futures Ready")

# ------------------------------------------------------------
# 1️⃣ TICKER INPUT
# ------------------------------------------------------------
tickers_input = st.text_input(
    "Enter stock/futures symbols (comma separated, e.g., TSLA, AAPL, ES=F, NQ=F):",
    "TSLA, AAPL, ES=F"
)

# Parse input into a list
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("⚠️ Please enter at least one symbol to scan.")

# Lookback period
lookback_days = st.number_input("Lookback Days:", min_value=20, max_value=180, value=60, step=1)

# ------------------------------------------------------------
# 2️⃣ HELPER FUNCTIONS
# ------------------------------------------------------------
@st.cache_data(ttl=600)
def get_price_data(ticker):
    end = dt.datetime.now()
    start = end - dt.timedelta(days=lookback_days)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    return df

@st.cache_data(ttl=300)
def get_option_flow(ticker):
    """
    Free call/put ratio flow for stocks. Disabled for futures.
    """
    if ticker.endswith("=F"):  # futures: skip option flow
        return {"call_volume": np.nan, "put_volume": np.nan, "cp_ratio": np.nan, "flow_bias": "Disabled"}
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options
        if not expiries:
            return {"call_volume": np.nan, "put_volume": np.nan, "cp_ratio": np.nan, "flow_bias": "Disabled"}
        chain = tk.option_chain(expiries[0])
        calls = chain.calls
        puts = chain.puts
        call_vol = float(calls["volume"].fillna(0).sum())
        put_vol = float(puts["volume"].fillna(0).sum())
        cp_ratio = call_vol / put_vol if put_vol > 0 else np.nan
        if np.isnan(cp_ratio):
            bias = "Neutral"
        elif cp_ratio > 1.3:
            bias = "Bullish"
        elif cp_ratio < 0.7:
            bias = "Bearish"
        else:
            bias = "Neutral"
        return {"call_volume": call_vol, "put_volume": put_vol, "cp_ratio": cp_ratio, "flow_bias": bias}
    except:
        return {"call_volume": np.nan, "put_volume": np.nan, "cp_ratio": np.nan, "flow_bias": "Error"}

def analyze_trend(df, ticker):
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Close"].pct_change().rolling(20).std() * 100
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    trend = (
        "Uptrend" if latest["SMA20"] > latest["SMA50"]
        else "Downtrend" if latest["SMA20"] < latest["SMA50"]
        else "Sideways"
    )

    vol_expanding = latest["Volatility"] > prev["Volatility"]
    vol_trend = "Expanding" if vol_expanding else "Contracting"

    # Volume signal: only for stocks
    if ticker.endswith("=F"):
        vol_signal = "No Volume Data"
    else:
        vol_signal = "Volume ↑" if latest["Volume"] > latest["Volume_MA20"] else "Volume ↓"

    return trend, vol_trend, vol_signal, latest["Close"]

# ------------------------------------------------------------
# 3️⃣ RUN SCANNER ON BUTTON PRESS
# ------------------------------------------------------------
if st.button("Run Scanner"):

    if not tickers:
        st.warning("⚠️ Please enter at least one symbol to scan.")
    else:
        summary_data = []

        for ticker in tickers:
            st.write(f"🔍 Scanning {ticker} ...")
            try:
                df = get_price_data(ticker)
                trend, vol_trend, vol_signal, last_price = analyze_trend(df, ticker)

                flow_data = get_option_flow(ticker)
                bias = flow_data["flow_bias"]

                # Simple summary interpretation
                summary_parts = [trend, vol_trend, vol_signal, f"Flow: {bias}"]
                if not ticker.endswith("=F"):  # only stock-based flow
                    if bias == "Bullish" and trend == "Uptrend":
                        summary_parts.append("🔥 Strong Up Move Potential")
                    elif bias == "Bearish" and trend == "Downtrend":
                        summary_parts.append("⚠️ Downside Continuation")

                summary = " | ".join(summary_parts)
                summary_data.append([ticker, round(last_price,2), trend, vol_trend, vol_signal, bias, summary])

            except Exception as e:
                summary_data.append([ticker, "-", "❌ Error", "❌ Error", "❌ Error", "❌ Error", f"Error: {e}"])

        # ------------------------------------------------------------
        # 4️⃣ DISPLAY RESULTS
        # ------------------------------------------------------------
        df_summary = pd.DataFrame(summary_data, columns=[
            "Ticker", "Last Price", "Trend", "Volatility", "Volume", "Flow Bias", "Summary"
        ])

        st.subheader("📊 Morning Scan Results")
        st.dataframe(df_summary, use_container_width=True)

        # Highlight best setups
        bullish = df_summary[df_summary["Summary"].str.contains("🔥", na=False)]
        bearish = df_summary[df_summary["Summary"].str.contains("⚠️", na=False)]

        if not bullish.empty:
            st.success("🔥 **Bullish Candidates**")
            st.dataframe(bullish)

        if not bearish.empty:
            st.warning("⚠️ **Bearish Candidates**")
            st.dataframe(bearish)

        st.caption("Data source: Yahoo Finance (free). Options flow only for stocks. Futures handled automatically.")
