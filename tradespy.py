import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Streamlit Setup ---
st.set_page_config(page_title="📊 Multi-Timeframe Stock & Futures Scanner", layout="wide")
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

# --- Period mapping for yfinance ---
period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "180d", "4h": "2y", "1d": "2y"}
period = period_map[timeframe]

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# --- Cached data fetch ---
@st.cache_data(ttl=3600)
def get_data(ticker, period, interval):
    return yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)

# --- RSI Calculation ---
def compute_rsi(data, period=14):
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- MACD Calculation ---
def compute_macd(data):
    exp1 = data["Close"].ewm(span=12, adjust=False).mean()
    exp2 = data["Close"].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

# --- Analyze Function ---
def analyze_ticker(ticker):
    try:
        data = get_data(ticker, period, timeframe)
        tk = yf.Ticker(ticker)
        info = tk.info

        latest_price = info.get("postMarketPrice") or info.get("preMarketPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose")
        pct_change = float((latest_price - prev_close) / prev_close * 100) if latest_price and prev_close else 0.0

        # --- Volume check (different for futures) ---
        avg_volume = float(data["Volume"].mean()) if not data.empty else 0.0
        if "=F" in ticker:
            volume_pass = avg_volume >= 1000
        else:
            volume_pass = avg_volume >= min_volume

        # --- Trend slope ---
        slope_pass = False
        slope_val = np.nan
        if len(data["Close"]) >= 45:
            recent_close = data["Close"].tail(45).values
            x = np.arange(len(recent_close))
            slope_val = float(np.polyfit(x, recent_close, 1)[0])
            slope_pass = slope_val > 0

        # --- MACD & RSI ---
        macd, macd_signal, macd_hist = compute_macd(data)
        rsi = compute_rsi(data)
        macd_val = float(macd_hist.dropna().iloc[-1]) if not macd_hist.dropna().empty else np.nan
        macd_status = "Bullish" if macd_val > 0 else "Bearish"
        rsi_val = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else np.nan
        if np.isnan(rsi_val):
            rsi_status = "-"
        elif rsi_val > 70:
            rsi_status = "Overbought"
        elif rsi_val < 30:
            rsi_status = "Oversold"
        else:
            rsi_status = "Neutral"

        # --- Signal & Summary ---
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
            "MACD": macd_status,
            "RSI": rsi_status,
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
            "MACD": "-",
            "RSI": "-",
            "Signal": "🟠",
            "Recommendation": "Error",
            "Summary": f"Error: {e}"
        }

# --- Run Scanner ---
if st.button("🔍 Run Scanner"):
    with st.spinner("Fetching & analyzing data..."):
        results = [analyze_ticker(t) for t in tickers]
        df = pd.DataFrame(results)
        df = df.sort_values("After-hours %", ascending=False).reset_index(drop=True)
        st.dataframe(df, width='stretch', height=600)  # Single screen output

        # --- Chart ---
        if show_chart:
            selected_ticker = st.selectbox("📈 View chart for:", tickers)
            if selected_ticker:
                chart_data = get_data(selected_ticker, period, timeframe)
                if not chart_data.empty:
                    rsi = compute_rsi(chart_data)
                    macd, macd_signal, macd_hist = compute_macd(chart_data)

                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=chart_data.index,
                        open=chart_data['Open'],
                        high=chart_data['High'],
                        low=chart_data['Low'],
                        close=chart_data['Close'],
                        name='Price'
                    ))
                    if not rsi.dropna().empty:
                        fig.add_trace(go.Scatter(
                            x=chart_data.index,
                            y=rsi,
                            line=dict(color='orange', width=2),
                            name='RSI',
                            yaxis='y2'
                        ))
                    if not macd_hist.dropna().empty:
                        fig.add_trace(go.Bar(
                            x=chart_data.index,
                            y=macd_hist,
                            marker_color='blue',
                            name='MACD Hist',
                            yaxis='y3'
                        ))

                    fig.update_layout(
                        title=f"{selected_ticker} Price, RSI & MACD",
                        yaxis=dict(title='Price', side='left'),
                        yaxis2=dict(title='RSI', overlaying='y', side='right', range=[0, 100]),
                        yaxis3=dict(title='MACD Hist', overlaying='y', side='right', position=0.85),
                        height=600,
                        legend=dict(orientation='h')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No chart data available for this ticker.")

st.caption("Built with ❤️ using Streamlit & Yahoo Finance API | MACD & RSI included | Volume check fixed for futures")
