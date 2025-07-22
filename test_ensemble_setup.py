#!/usr/bin/env python3
"""
Simplified FinRL Ensemble Strategy Setup Test
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Test basic imports
try:
    import stable_baselines3
    print("[OK] Stable Baselines3 imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import Stable Baselines3: {e}")

try:
    import gymnasium as gym
    print("[OK] Gymnasium imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import Gymnasium: {e}")

# DOW 30 Tickers for testing
DOW_30_TICKER = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "TSLA", "META", "UNH", "XOM", "JNJ",
    "JPM", "V", "PG", "AVGO", "HD", "CVX", "LLY", "ABBV", "BAC", "ASML",
    "WMT", "KO", "COST", "MRK", "PEP", "NFLX", "ADBE", "TMO", "ACN", "CSCO"
]

def test_data_download():
    """Test Yahoo Finance data download"""
    print("\nTesting data download...")
    
    try:
        # Test with a small subset first
        test_tickers = ["AAPL", "MSFT", "GOOGL"]
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        # Download data
        data = yf.download(test_tickers, start=start_date, end=end_date)
        
        print(f"[OK] Downloaded data shape: {data.shape}")
        print(f"[OK] Date range: {data.index[0]} to {data.index[-1]}")
        print(f"[OK] Columns: {data.columns.names}")
        
        return data
        
    except Exception as e:
        print(f"[ERROR] Data download failed: {e}")
        return None

def preprocess_data(data, tickers):
    """Basic data preprocessing"""
    print("\nPreprocessing data...")
    
    try:
        # Reshape data for FinRL format
        df_list = []
        
        for ticker in tickers:
            ticker_data = data.xs(ticker, level=1, axis=1).copy()
            ticker_data['tic'] = ticker
            ticker_data['date'] = ticker_data.index
            ticker_data = ticker_data.reset_index(drop=True)
            
            # Rename columns to match FinRL format
            ticker_data = ticker_data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adjcp'
            })
            
            df_list.append(ticker_data)
        
        # Combine all tickers
        processed_df = pd.concat(df_list, ignore_index=True)
        processed_df = processed_df.sort_values(['date', 'tic'])
        
        print(f"[OK] Processed data shape: {processed_df.shape}")
        print(f"[OK] Unique tickers: {len(processed_df['tic'].unique())}")
        print(f"[OK] Date range: {processed_df['date'].min()} to {processed_df['date'].max()}")
        
        return processed_df
        
    except Exception as e:
        print(f"[ERROR] Data preprocessing failed: {e}")
        return None

def add_technical_indicators(df):
    """Add basic technical indicators"""
    print("\nAdding technical indicators...")
    
    try:
        df = df.copy()
        
        # Add basic indicators for each ticker
        for ticker in df['tic'].unique():
            ticker_data = df[df['tic'] == ticker].copy()
            
            # Moving averages
            ticker_data['sma_10'] = ticker_data['close'].rolling(10).mean()
            ticker_data['sma_30'] = ticker_data['close'].rolling(30).mean()
            
            # RSI calculation
            delta = ticker_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            ticker_data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = ticker_data['close'].ewm(span=12).mean()
            exp2 = ticker_data['close'].ewm(span=26).mean()
            ticker_data['macd'] = exp1 - exp2
            
            # Update original dataframe
            df.loc[df['tic'] == ticker, ['sma_10', 'sma_30', 'rsi', 'macd']] = ticker_data[['sma_10', 'sma_30', 'rsi', 'macd']].values
        
        # Fill NaN values
        df = df.fillna(0)
        
        print(f"[OK] Added technical indicators")
        print(f"[OK] Final data shape: {df.shape}")
        
        return df
        
    except Exception as e:
        print(f"[ERROR] Technical indicators failed: {e}")
        return None

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = [
        'data',
        'trained_models', 
        'results',
        'tensorboard_log'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[OK] Created directory: {directory}")

def main():
    print("FinRL Ensemble Strategy Setup Test")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Test data download
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    data = test_data_download()
    
    if data is not None:
        # Preprocess data
        processed_data = preprocess_data(data, test_tickers)
        
        if processed_data is not None:
            # Add technical indicators
            final_data = add_technical_indicators(processed_data)
            
            if final_data is not None:
                # Save processed data
                final_data.to_csv('data/processed_data.csv', index=False)
                print(f"[OK] Saved processed data to data/processed_data.csv")
                
                print("\n" + "=" * 50)
                print("[SUCCESS] Setup completed successfully!")
                print("[SUCCESS] Ready to run FinRL Ensemble Strategy")
                print("\nNext steps:")
                print("1. Run the full ensemble strategy")
                print("2. Train models with A2C, PPO, and DDPG")
                print("3. Backtest and evaluate results")
                
                return True
    
    print("\n" + "=" * 50)
    print("[FAILED] Setup failed. Please check the errors above.")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)