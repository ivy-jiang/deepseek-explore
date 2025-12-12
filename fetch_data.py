import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import datetime

# --- Configuration ---
START_DATE = "2005-01-01"
END_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

print(f"Fetching data from {START_DATE} to {END_DATE}...")

# 1. Fetch Market Data (Yahoo)
tickers = ["QQQ", "^VIX"]
market_data = yf.download(tickers, start=START_DATE, end=END_DATE)["Close"]
market_data.columns = ["QQQ", "VIX"]

# 2. Fetch Economic Data (FRED)
# DGS2: 2-Year Treasury Constant Maturity Rate
# CPIAUCSL: Consumer Price Index for All Urban Consumers: All Items
# UNRATE: Unemployment Rate
economic_tickers = ["DGS2", "CPIAUCSL", "UNRATE"]
try:
    econ_data = web.DataReader(economic_tickers, "fred", START_DATE, END_DATE)
except Exception as e:
    print(f"Error fetching FRED data: {e}")
    # Fallback or exit? For now, let's assume it works or create dummy if it fails hard
    # But usually FRED is reliable.
    exit(1)

# 3. Merge Data
# Economic data is often Monthly (CPI, UNRATE) or Daily (DGS2).
# We align everything to the Market Data index (Daily).
full_df = market_data.join(econ_data, how="left")

# Forward fill monthly economic data to daily
full_df = full_df.ffill()

# 4. Feature Engineering
print("Calculating Technical Indicators...")

# QQQ Returns (Log Return)
full_df["QQQ_Log_Return"] = np.log(full_df["QQQ"] / full_df["QQQ"].shift(1))

# RSI
full_df["RSI"] = calculate_rsi(full_df["QQQ"])

# MACD
macd, signal = calculate_macd(full_df["QQQ"])
full_df["MACD"] = macd
full_df["MACD_Signal"] = signal

# Transformations for Stationarity/Normalization
# VIX is already a "score" (0-100 approx)
# Yields (DGS2) are percents.
# CPI: Calculate YoY Inflation Rate if not already done, or MoM change
# CPIAUCSL is an Index. We need the % change.
full_df["CPI_YoY"] = full_df["CPIAUCSL"].pct_change(periods=252) * 100 # Approx 1 year (252 trading days)
full_df["Unemployment"] = full_df["UNRATE"] # Already a rate

# Drop NaN values (created by lags/diffs)
full_df = full_df.dropna()

print(f"Data Shape: {full_df.shape}")
print(full_df.head())
print(full_df.tail())

# Save
full_df.to_csv("market_data.csv")
print("Saved to market_data.csv")
