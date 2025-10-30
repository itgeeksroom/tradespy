import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
show_chart = st.sidebar.checkbox("Show chart for selected ticker", value=True)
pct_threshold = st.sidebar.number_input("After-hours % change threshold", value=1.0, step=0.1)

# --- Period mapping ---
period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "180d", "4h": "2y", "1d": "2y"}
period = period_map[timeframe]

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# --- Cache data fetch ---
@st.cache_data(ttl=3600)
def get_data(ticker, period, interval):
    return yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)

# --- Analyze Ticker ---
def analyze_ticker(ticker):
    try:
        data = get_data(ticker, period, timeframe)
        tk = yf.Ticker(ticker)
        info = tk.info

        latest_price = info.get("postMarketPrice") or info.get("preMarketPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose")
        pct_change = float((latest_price - prev_close) / prev_close * 100) if latest_price and prev_close else 0.0

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
            "Price": latest_price or "-",
            "Slope": round(slope_val, 2) if not np.isnan(slope_val) else "-",
            "Volume": pf(volume_pass),
            "After-hours %": round(pct_change, 2),
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
            "Signal": "🟠",
            "Recommendation": "Error",
            "Summary": f"Error: {e}"
        }

# --- Session State to preserve results ---
if "results" not in st.session_state:
    st.session_state.results = None

# --- Run Scanner ---
if st.button("🔍 Run Scanner"):
    with st.spinner("Fetching & analyzing data..."):
        st.session_state.results = [analyze_ticker(t) for t in tickers]

# --- Display Table ---
if st.session_state.results is not None:
    df = pd.DataFrame(st.session_state.results)
    df = df.sort_values("After-hours %", ascending=False).reset_index(drop=True)
    st.dataframe(df, width='stretch', height=600)

    # --- Chart ---
    if show_chart:
        selected_ticker = st.selectbox("📈 View chart for:", tickers)
        if selected_ticker:
            chart_data = get_data(selected_ticker, period, timeframe)
            if not chart_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=chart_data.index,
                    open=chart_data['Open'],
                    high=chart_data['High'],
                    low=chart_data['Low'],
                    close=chart_data['Close'],
                    name='Price'
                ))

                # Add Buy/Sell markers based on recommendation
                row = df[df["Ticker"] == selected_ticker]
                rec = row["Recommendation"].values[0]
                price = row["Price"].values[0]
                if rec == "Buy":
                    fig.add_trace(go.Scatter(
                        x=[chart_data.index[-1]],
                        y=[price],
                        mode="markers+text",
                        marker=dict(color="green", size=15, symbol="triangle-up"),
                        text=["Buy"],
                        textposition="top center",
                        name="Buy Signal"
                    ))
                elif rec == "Short":
                    fig.add_trace(go.Scatter(
                        x=[chart_data.index[-1]],
                        y=[price],
                        mode="markers+text",
                        marker=dict(color="red", size=15, symbol="triangle-down"),
                        text=["Short"],
                        textposition="bottom center",
                        name="Sell Signal"
                    ))

                fig.update_layout(
                    title=f"{selected_ticker} Price Chart with Buy/Sell",
                    yaxis=dict(title='Price'),
                    height=600,
                    legend=dict(orientation='h')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No chart data available for this ticker.")

st.caption("Built with ❤️ using Streamlit & Yahoo Finance API | Buy/Sell signals shown on chart")
