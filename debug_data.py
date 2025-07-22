#!/usr/bin/env python3

import yfinance as yf
import pandas as pd

# Test data download to see structure
ticker = "AAPL"
data = yf.download(ticker, start="2023-01-01", end="2023-02-01", progress=False)

print("Data columns:", data.columns)
print("Data index:", data.index)
print("Data shape:", data.shape)
print("Data head:")
print(data.head())

# Check if MultiIndex
if isinstance(data.columns, pd.MultiIndex):
    print("MultiIndex columns - levels:", data.columns.levels)
else:
    print("Single level columns")

print("\nColumn types:")
for col in data.columns:
    print(f"  {col}: {type(col)}")