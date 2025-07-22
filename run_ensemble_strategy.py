#!/usr/bin/env python3
"""
FinRL Ensemble Strategy Implementation
Simplified version without full FinRL library dependencies
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
import torch

# Stable Baselines3 imports
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

# Configure device
def setup_device():
    """Setup and verify CUDA device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0):.0f} bytes")
        
        # Test GPU functionality
        try:
            test_tensor = torch.randn(1000, 1000).to(device)
            result = torch.matmul(test_tensor, test_tensor.T)
            print(f"[OK] GPU functionality test passed")
            del test_tensor, result
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[WARNING] GPU test failed: {e}")
            device = torch.device("cpu")
            print("Falling back to CPU")
    else:
        print("[WARNING] CUDA not available. Using CPU for training.")
        print("For GPU acceleration, install PyTorch with CUDA support:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    return device

device = setup_device()

# Configuration
DOW_30_TICKER = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "TSLA", "META", "UNH", "XOM", "JNJ",
    "JPM", "V", "PG", "AVGO", "HD", "CVX", "LLY", "ABBV", "BAC", "WMT",
    "KO", "COST", "MRK", "PEP", "NFLX", "ADBE", "TMO", "ACN", "CSCO", "DIS"
]

# For testing, use a subset
SELECTED_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "JPM"]

TRAIN_START_DATE = '2022-01-01'
TRAIN_END_DATE = '2023-06-01' 
TEST_START_DATE = '2023-06-01'
TEST_END_DATE = '2024-01-01'

INITIAL_AMOUNT = 1000000  # $1M
TRANSACTION_COST = 0.001  # 0.1%

class StockTradingEnv(gym.Env):
    """Simplified Stock Trading Environment"""
    
    def __init__(self, df, initial_amount=1000000, transaction_cost=0.001):
        super(StockTradingEnv, self).__init__()
        
        self.df = df.copy()
        self.stock_dim = len(df['tic'].unique())
        self.initial_amount = initial_amount
        self.transaction_cost = transaction_cost
        
        # Action space: portfolio weights for each stock (-1 to 1)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32
        )
        
        # State space: [balance, prices, holdings, indicators]
        self.state_dim = 1 + 2 * self.stock_dim + 4 * self.stock_dim  # balance + prices + holdings + 4 indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_amount
        self.holdings = np.zeros(self.stock_dim)
        
        # Get unique dates and tickers
        self.dates = sorted(self.df['date'].unique())
        self.tickers = sorted(self.df['tic'].unique())
        
        self.portfolio_values = [self.initial_amount]
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current state observation"""
        if self.current_step >= len(self.dates):
            # Return last valid observation
            self.current_step = len(self.dates) - 1
        
        current_date = self.dates[self.current_step]
        current_data = self.df[self.df['date'] == current_date].sort_values('tic')
        
        # Prices
        prices = current_data['close'].values
        if len(prices) != self.stock_dim:
            prices = np.zeros(self.stock_dim)
        
        # Technical indicators
        indicators = []
        for indicator in ['sma_10', 'sma_30', 'rsi', 'macd']:
            values = current_data[indicator].values
            if len(values) != self.stock_dim:
                values = np.zeros(self.stock_dim)
            indicators.append(values)
        
        indicators = np.concatenate(indicators)
        
        # Combine state: [balance, prices, holdings, indicators]
        state = np.concatenate([
            [self.balance], 
            prices, 
            self.holdings, 
            indicators
        ])
        
        return state.astype(np.float32)
    
    def step(self, action):
        """Execute trading action"""
        if self.current_step >= len(self.dates) - 1:
            return self._get_observation(), 0, True, True, {}
        
        current_date = self.dates[self.current_step]
        current_data = self.df[self.df['date'] == current_date].sort_values('tic')
        prices = current_data['close'].values
        
        if len(prices) != self.stock_dim:
            prices = np.ones(self.stock_dim) * 100  # Default price
        
        # Calculate portfolio value before action
        portfolio_value_before = self.balance + np.sum(self.holdings * prices)
        
        # Execute trading action (normalized to portfolio weights)
        action = np.clip(action, -1, 1)  # Ensure action is in valid range
        
        # Calculate target holdings based on action
        total_value = self.balance + np.sum(self.holdings * prices)
        target_values = action * total_value * 0.5  # Max 50% in any position
        target_holdings = target_values / (prices + 1e-8)  # Avoid division by zero
        
        # Execute trades
        holdings_change = target_holdings - self.holdings
        
        for i in range(self.stock_dim):
            if holdings_change[i] > 0:  # Buy
                cost = holdings_change[i] * prices[i] * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.balance -= cost
                    self.holdings[i] += holdings_change[i]
            elif holdings_change[i] < 0:  # Sell
                proceeds = abs(holdings_change[i]) * prices[i] * (1 - self.transaction_cost)
                if self.holdings[i] >= abs(holdings_change[i]):
                    self.balance += proceeds
                    self.holdings[i] += holdings_change[i]
        
        # Move to next time step
        self.current_step += 1
        
        # Calculate portfolio value after action
        next_prices = prices  # Use same prices for simplicity
        if self.current_step < len(self.dates):
            next_date = self.dates[self.current_step]
            next_data = self.df[self.df['date'] == next_date].sort_values('tic')
            if len(next_data) > 0:
                next_prices = next_data['close'].values
                if len(next_prices) != self.stock_dim:
                    next_prices = prices
        
        portfolio_value_after = self.balance + np.sum(self.holdings * next_prices)
        
        # Calculate reward (portfolio return)
        reward = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        
        self.portfolio_values.append(portfolio_value_after)
        
        # Check if done
        done = self.current_step >= len(self.dates) - 1
        truncated = False
        
        return self._get_observation(), reward, done, truncated, {
            'portfolio_value': portfolio_value_after,
            'balance': self.balance,
            'holdings': self.holdings.copy()
        }

def download_and_process_data():
    """Download and process stock data"""
    print("Downloading stock data...")
    
    # Download data
    all_data = []
    for ticker in SELECTED_TICKERS:
        try:
            data = yf.download(ticker, start=TRAIN_START_DATE, end=TEST_END_DATE, 
                             progress=False, auto_adjust=False)
            
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                # For single ticker, flatten the MultiIndex
                data.columns = data.columns.get_level_values(0)
            
            # Reset index to get Date as a column
            data = data.reset_index()
            
            # Add ticker information
            data['tic'] = ticker
            data['date'] = data['Date']
            
            # Rename columns to match expected format
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low', 
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adjcp'
            })
            
            # Select only needed columns
            data = data[['date', 'open', 'high', 'low', 'close', 'volume', 'adjcp', 'tic']]
            
            all_data.append(data)
            print(f"[OK] Downloaded {ticker}: {len(data)} rows")
            
        except Exception as e:
            print(f"[ERROR] Failed to download {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_data:
        raise ValueError("No data downloaded successfully")
    
    # Combine data
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values(['date', 'tic'])
    
    # Add technical indicators
    print("Adding technical indicators...")
    
    # Process each ticker separately and rebuild dataframe
    processed_data = []
    
    for ticker in df['tic'].unique():
        ticker_data = df[df['tic'] == ticker].copy().reset_index(drop=True)
        
        # Technical indicators
        ticker_data['sma_10'] = ticker_data['close'].rolling(10).mean()
        ticker_data['sma_30'] = ticker_data['close'].rolling(30).mean()
        
        # RSI
        delta = ticker_data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        ticker_data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = ticker_data['close'].ewm(span=12).mean()
        exp2 = ticker_data['close'].ewm(span=26).mean()
        ticker_data['macd'] = exp1 - exp2
        
        processed_data.append(ticker_data)
    
    # Rebuild dataframe
    df = pd.concat(processed_data, ignore_index=True)
    df = df.sort_values(['date', 'tic'])
    
    # Fill NaN values
    df = df.fillna(0)
    
    print(f"[OK] Data processed: {df.shape} rows, {len(df['tic'].unique())} stocks")
    return df

def split_data(df):
    """Split data into train and test sets"""
    train_data = df[(df['date'] >= TRAIN_START_DATE) & (df['date'] < TEST_START_DATE)]
    test_data = df[(df['date'] >= TEST_START_DATE) & (df['date'] <= TEST_END_DATE)]
    
    print(f"[OK] Train data: {len(train_data)} rows")
    print(f"[OK] Test data: {len(test_data)} rows")
    
    return train_data, test_data

def train_ensemble_models(train_data):
    """Train ensemble of RL models with hardware acceleration"""
    print(f"\nTraining ensemble models on {device}...")
    
    # Create environment
    env = DummyVecEnv([lambda: StockTradingEnv(train_data)])
    
    models = {}
    
    # Set up training parameters optimized for available hardware
    is_gpu = str(device) != "cpu" and str(device) != "auto"
    training_timesteps = 100000 if is_gpu else 50000  # More timesteps for GPU
    batch_size_multiplier = 4 if is_gpu else 1
    
    # Train A2C with hardware optimization
    print(f"Training A2C on {device} ({training_timesteps:,} timesteps)...")
    a2c_model = A2C(
        'MlpPolicy', 
        env, 
        verbose=1, 
        seed=42,
        device=device,
        tensorboard_log="./tensorboard_log/a2c/",
        n_steps=5,
        learning_rate=0.0007
    )
    a2c_model.learn(total_timesteps=training_timesteps, progress_bar=True)
    a2c_model.save("trained_models/a2c_model")
    models['A2C'] = a2c_model
    
    # Train PPO with hardware optimization
    print(f"Training PPO on {device} ({training_timesteps:,} timesteps)...")
    ppo_model = PPO(
        'MlpPolicy', 
        env, 
        verbose=1, 
        seed=42,
        device=device,
        tensorboard_log="./tensorboard_log/ppo/",
        batch_size=64 * batch_size_multiplier,
        n_epochs=10,
        learning_rate=0.0003
    )
    ppo_model.learn(total_timesteps=training_timesteps, progress_bar=True)
    ppo_model.save("trained_models/ppo_model")
    models['PPO'] = ppo_model
    
    # Train DDPG with hardware optimization
    print(f"Training DDPG on {device} ({training_timesteps:,} timesteps)...")
    ddpg_model = DDPG(
        'MlpPolicy', 
        env, 
        verbose=1, 
        seed=42,
        device=device,
        tensorboard_log="./tensorboard_log/ddpg/",
        batch_size=64 * batch_size_multiplier,
        buffer_size=100000 if not is_gpu else 200000,
        learning_rate=0.001
    )
    ddpg_model.learn(total_timesteps=training_timesteps, progress_bar=True)
    ddpg_model.save("trained_models/ddpg_model")
    models['DDPG'] = ddpg_model
    
    # Print hardware usage statistics
    if device_available and torch.cuda.is_available() and is_gpu:
        try:
            torch.cuda.synchronize()
            print(f"\n[GPU] Training completed with GPU acceleration:")
            print(f"[GPU] Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"[GPU] Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
            print(f"[GPU] Max memory used: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
            torch.cuda.empty_cache()
        except:
            pass
    
    training_mode = "GPU acceleration" if is_gpu else "CPU optimization"
    print(f"[OK] All models trained successfully with {training_mode}")
    print(f"[OK] Training timesteps: {training_timesteps:,} per model")
    return models

def backtest_ensemble(models, test_data):
    """Backtest ensemble strategy"""
    print("\nBacktesting ensemble strategy...")
    
    # Create test environment
    env = StockTradingEnv(test_data)
    
    # Individual model results
    model_results = {}
    
    for name, model in models.items():
        print(f"Testing {name}...")
        
        obs, _ = env.reset()
        portfolio_values = [INITIAL_AMOUNT]
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            portfolio_values.append(info.get('portfolio_value', portfolio_values[-1]))
        
        model_results[name] = portfolio_values
    
    # Ensemble strategy (average of all models)
    print("Running ensemble strategy...")
    obs, _ = env.reset()
    ensemble_portfolio = [INITIAL_AMOUNT]
    done = False
    
    while not done:
        # Get predictions from all models
        actions = []
        for model in models.values():
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action)
        
        # Average the actions
        ensemble_action = np.mean(actions, axis=0)
        
        obs, reward, done, truncated, info = env.step(ensemble_action)
        ensemble_portfolio.append(info.get('portfolio_value', ensemble_portfolio[-1]))
    
    model_results['Ensemble'] = ensemble_portfolio
    
    return model_results

def calculate_performance_metrics(portfolio_values):
    """Calculate performance metrics"""
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    max_drawdown = (pd.Series(portfolio_values) / pd.Series(portfolio_values).cummax() - 1).min()
    
    return {
        'Total Return': f"{total_return:.2%}",
        'Annual Return': f"{annual_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}"
    }

def plot_results(results):
    """Plot backtest results"""
    print("\nPlotting results...")
    
    plt.figure(figsize=(12, 8))
    
    for name, portfolio_values in results.items():
        plt.plot(portfolio_values, label=name, linewidth=2)
    
    plt.title('FinRL Ensemble Strategy Backtest Results')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('results/ensemble_backtest_results.png', dpi=300, bbox_inches='tight')
    print("[OK] Results saved to results/ensemble_backtest_results.png")
    
    # Show performance metrics
    print("\nPerformance Summary:")
    print("=" * 60)
    
    for name, portfolio_values in results.items():
        metrics = calculate_performance_metrics(portfolio_values)
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

def main():
    """Main execution function"""
    print("FinRL Ensemble Strategy")
    print("=" * 50)
    
    try:
        # Download and process data
        df = download_and_process_data()
        
        # Split data
        train_data, test_data = split_data(df)
        
        # Train models
        models = train_ensemble_models(train_data)
        
        # Backtest
        results = backtest_ensemble(models, test_data)
        
        # Plot and analyze results
        plot_results(results)
        
        print("\n" + "=" * 50)
        print("[SUCCESS] Ensemble strategy completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Strategy failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)