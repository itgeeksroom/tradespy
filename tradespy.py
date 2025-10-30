import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Streamlit Setup ---
st.set_page_config(page_title="📊 Multi-Timeframe Stock & Futures Scanner", layout="centered")
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

period_map = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "1h": "180d",
    "4h": "2y",
    "1d": "2y"
}
period = period_map[timeframe]

min_volume = st.sidebar.number_input("Minimum Avg Volume (for stocks)", value=2_000_000, step=500_000)
show_chart = st.sidebar.checkbox("Show chart for selected ticker", value=True)
pct_threshold = st.sidebar.number_input("After-hours / pre-market % change threshold", value=1.0, step=0.1)

# --- Parse tickers ---
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

# --- Analysis Function ---
def analyze_ticker(ticker):
    try:
        data = get_data(ticker, period, timeframe)
        tk = yf.Ticker(ticker)
        info = tk.info

        # Latest price and % change
        latest_price = info.get("postMarketPrice") or info.get("preMarketPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose")
        pct_change = (latest_price - prev_close) / prev_close * 100 if latest_price and prev_close else 0

        # Average volume
        avg_volume = float(data["Volume"].mean()) if not data.empty else 0
        volume_pass = avg_volume >= min_volume

        # Trend slope
        slope_pass = False
        slope = np.nan
        lookback_bars = 45
        if len(data["Close"]) >= lookback_bars:
            recent_close = data["Close"].tail(lookback_bars).values
            x = np.arange(len(recent_close))
            slope = float(np.polyfit(x, recent_close, 1)[0])
            slope_pass = slope > 0

        # MACD & RSI
        macd, macd_signal, macd_hist = compute_macd(data)
        rsi = compute_rsi(data)

        # MACD signal
        macd_status = "Bullish" if macd_hist.iloc[-1] > 0 else "Bearish"

        # RSI signal
        if rsi.iloc[-1] > 70:
            rsi_status = "Overbought"
        elif rsi.iloc[-1] < 30:
            rsi_status = "Oversold"
        else:
            rsi_status = "Neutral"

        # Signal and recommendation
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
            elif not slope_pass and not np.isnan(slope):
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
            "Latest Price": latest_price or "-",
            "Trend Slope": round(slope, 2) if not np.isnan(slope) else "-",
            "Volume Check": pf(volume_pass),
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
            "Latest Price": "-",
            "Trend Slope": "-",
            "Volume Check": "❌ Error",
            "After-hours %": "-",
            "MACD": "-",
            "RSI": "-",
            "Signal": "🟠",
            "Recommendation": "Error",
            "Summary": f"Error: {e}"
        }

# --- Run Scanner ---
if st.button("🔍 Run Scanner"):
    with st.spinner("Fetching and analyzing data..."):
        results = [analyze_ticker(t) for t in tickers]
        df = pd.DataFrame(results)
        df = df.sort_values("After-hours %", ascending=False).reset_index(drop=True)
        st.dataframe(df, width='stretch')
        st.success("✅ Scan complete!")

        # Optional Chart with RSI & MACD
        if show_chart:
            selected_ticker = st.selectbox("📈 View chart for:", tickers)
            if selected_ticker:
                chart_data = get_data(selected_ticker, period, timeframe)
                if not chart_data.empty:
                    rsi = compute_rsi(chart_data)
                    macd, macd_signal, macd_hist = compute_macd(chart_data)

                    fig = go.Figure()
                    # Price candles / line
                    fig.add_trace(go.Candlestick(
                        x=chart_data.index,
                        open=chart_data['Open'],
                        high=chart_data['High'],
                        low=chart_data['Low'],
                        close=chart_data['Close'],
                        name='Price'
                    ))
                    # RSI line
                    fig.add_trace(go.Scatter(
                        x=chart_data.index,
                        y=rsi,
                        line=dict(color='orange', width=2),
                        name='RSI',
                        yaxis='y2'
                    ))
                    # MACD histogram
                    fig.add_trace(go.Bar(
                        x=chart_data.index,
                        y=macd_hist,
                        marker_color='blue',
                        name='MACD Hist',
                        yaxis='y3'
                    ))

                    fig.update_layout(
                        yaxis2=dict(
                            overlaying='y',
                            side='right',
                            range=[0, 100],
                            title='RSI'
                        ),
                        yaxis3=dict(
                            overlaying='y',
                            side='left',
                            position=0.15,
                            title='MACD Hist'
                        ),
                        height=600,
                        title=f"{selected_ticker} Price, RSI & MACD",
                        legend=dict(orientation="h")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No chart data available for this ticker.")

st.caption("Built with ❤️ using Streamlit & Yahoo Finance API | Includes MACD & RSI")
