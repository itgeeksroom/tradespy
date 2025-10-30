import yfinance as yf
import pandas as pd

# --- Load S&P 500 tickers (free source) ---
sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
tickers = sp500['Symbol'].tolist()

# --- Parameters ---
min_pct_change = 1  # Minimum % change after hours to flag
top_n = 20          # Top N movers to report

results = []

for ticker in tickers:
    try:
        tk = yf.Ticker(ticker)
        info = tk.info

        # Current price after hours / pre-market
        pre_price = info.get("preMarketPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose")
        if pre_price is None or prev_close is None:
            continue

        pct_change = (pre_price - prev_close) / prev_close * 100

        # Determine signal
        if pct_change >= min_pct_change:
            signal = "🟢 Bullish"
        elif pct_change <= -min_pct_change:
            signal = "🔴 Bearish"
        else:
            signal = "🟠 Neutral"

        results.append({
            "Ticker": ticker,
            "Prev Close": prev_close,
            "After Hours": pre_price,
            "% Change": round(pct_change, 2),
            "Signal": signal
        })

    except Exception as e:
        # Skip any tickers with missing data
        continue

# --- Convert to DataFrame & sort ---
df = pd.DataFrame(results)
df = df.sort_values("% Change", ascending=False).reset_index(drop=True)

# --- Show top movers ---
print(f"\nTop {top_n} S&P 500 After-Hours / Pre-Market Movers:\n")
print(df.head(top_n))
