#!/usr/bin/env python3
"""
FinRL Ensemble Strategy with GPU Acceleration
Optimized for CUDA training with CPU fallback
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

# First try to setup PyTorch with CUDA support
try:
    import torch
    device_available = True
except ImportError:
    print("[WARNING] PyTorch not available. Please install: pip install torch")
    device_available = False
    device = "cpu"

# Stable Baselines3 imports
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
from gymnasium import spaces

def setup_device():
    """Setup and verify CUDA device with comprehensive fallbacks"""
    if not device_available:
        print("[INFO] Using CPU-only mode")
        return "cpu"
        
    try:
        print(f"PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            device = torch.device("cuda")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Test GPU with simple operations
            try:
                test_tensor = torch.randn(100, 100, device=device)
                result = torch.matmul(test_tensor, test_tensor.T)
                print("[OK] GPU functionality test passed")
                
                # Clean up test tensors
                del test_tensor, result
                torch.cuda.empty_cache()
                
                return device
                
            except Exception as e:
                print(f"[WARNING] GPU test failed: {e}")
                print("[INFO] Falling back to CPU")
                return torch.device("cpu")
        else:
            print("[INFO] CUDA not available, using CPU")
            return torch.device("cpu")
            
    except Exception as e:
        print(f"[WARNING] Device setup failed: {e}")
        print("[INFO] Using CPU as fallback")
        return "cpu"

# Setup device
if device_available:
    device = setup_device()
else:
    device = "cpu"

# Configuration
SELECTED_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]  # GPU-friendly subset
TRAIN_START_DATE = '2022-01-01'
TRAIN_END_DATE = '2023-06-01' 
TEST_START_DATE = '2023-06-01'
TEST_END_DATE = '2024-01-01'

INITIAL_AMOUNT = 1000000
TRANSACTION_COST = 0.001

class GPUStockTradingEnv(gym.Env):
    """GPU-optimized Stock Trading Environment"""
    
    def __init__(self, df, initial_amount=1000000, transaction_cost=0.001):
        super().__init__()
        
        self.df = df.copy()
        self.stock_dim = len(df['tic'].unique())
        self.initial_amount = initial_amount
        self.transaction_cost = transaction_cost
        
        # Action space: continuous portfolio weights
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32
        )
        
        # State space: optimized for GPU training
        self.state_dim = 1 + 2 * self.stock_dim + 4 * self.stock_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = float(self.initial_amount)
        self.holdings = np.zeros(self.stock_dim, dtype=np.float32)
        
        self.dates = sorted(self.df['date'].unique())
        self.tickers = sorted(self.df['tic'].unique())
        
        self.portfolio_values = [self.initial_amount]
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current state - optimized for GPU"""
        if self.current_step >= len(self.dates):
            self.current_step = len(self.dates) - 1
        
        current_date = self.dates[self.current_step]
        current_data = self.df[self.df['date'] == current_date].sort_values('tic')
        
        # Vectorized operations for speed
        prices = current_data['close'].values
        if len(prices) != self.stock_dim:
            prices = np.ones(self.stock_dim, dtype=np.float32) * 100
        
        # Technical indicators (vectorized)
        indicators = []
        for indicator in ['sma_10', 'sma_30', 'rsi', 'macd']:
            values = current_data[indicator].values
            if len(values) != self.stock_dim:
                values = np.zeros(self.stock_dim, dtype=np.float32)
            indicators.append(values)
        
        indicators = np.concatenate(indicators).astype(np.float32)
        
        # Combine state
        state = np.concatenate([
            [self.balance], 
            prices.astype(np.float32), 
            self.holdings, 
            indicators
        ])
        
        return state
    
    def step(self, action):
        """Execute trading step - optimized"""
        if self.current_step >= len(self.dates) - 1:
            return self._get_observation(), 0.0, True, True, {}
        
        current_date = self.dates[self.current_step]
        current_data = self.df[self.df['date'] == current_date].sort_values('tic')
        prices = current_data['close'].values.astype(np.float32)
        
        if len(prices) != self.stock_dim:
            prices = np.ones(self.stock_dim, dtype=np.float32) * 100
        
        # Portfolio value before action
        portfolio_value_before = self.balance + np.sum(self.holdings * prices)
        
        # Execute trades (vectorized for speed)
        action = np.clip(action, -1, 1)
        total_value = portfolio_value_before
        
        # Target allocation based on action
        target_values = action * total_value * 0.3  # Max 30% per position
        target_holdings = target_values / (prices + 1e-8)
        
        # Execute trades
        holdings_change = target_holdings - self.holdings
        
        # Vectorized buy/sell operations
        buy_mask = holdings_change > 0
        sell_mask = holdings_change < 0
        
        # Buy orders
        buy_costs = holdings_change[buy_mask] * prices[buy_mask] * (1 + self.transaction_cost)
        total_buy_cost = np.sum(buy_costs)
        
        if total_buy_cost <= self.balance:
            self.balance -= total_buy_cost
            self.holdings[buy_mask] += holdings_change[buy_mask]
        
        # Sell orders
        sell_proceeds = np.abs(holdings_change[sell_mask]) * prices[sell_mask] * (1 - self.transaction_cost)
        valid_sells = self.holdings[sell_mask] >= np.abs(holdings_change[sell_mask])
        
        self.balance += np.sum(sell_proceeds[valid_sells])
        self.holdings[sell_mask] = np.where(valid_sells, 
                                          self.holdings[sell_mask] + holdings_change[sell_mask],
                                          self.holdings[sell_mask])
        
        # Next step
        self.current_step += 1
        
        # Calculate reward
        if self.current_step < len(self.dates):
            next_date = self.dates[self.current_step]
            next_data = self.df[self.df['date'] == next_date].sort_values('tic')
            next_prices = next_data['close'].values.astype(np.float32)
            if len(next_prices) != self.stock_dim:
                next_prices = prices
        else:
            next_prices = prices
        
        portfolio_value_after = self.balance + np.sum(self.holdings * next_prices)
        reward = (portfolio_value_after - portfolio_value_before) / (portfolio_value_before + 1e-8)
        
        self.portfolio_values.append(portfolio_value_after)
        
        done = self.current_step >= len(self.dates) - 1
        
        return self._get_observation(), float(reward), done, False, {
            'portfolio_value': portfolio_value_after,
            'balance': self.balance,
            'holdings': self.holdings.copy()
        }

def download_and_process_data():
    """Download and process data - same as before"""
    print("Downloading stock data...")
    
    all_data = []
    for ticker in SELECTED_TICKERS:
        try:
            data = yf.download(ticker, start=TRAIN_START_DATE, end=TEST_END_DATE, 
                             progress=False, auto_adjust=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            data = data.reset_index()
            data['tic'] = ticker
            data['date'] = data['Date']
            
            data = data.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adjcp'
            })
            
            data = data[['date', 'open', 'high', 'low', 'close', 'volume', 'adjcp', 'tic']]
            all_data.append(data)
            print(f"[OK] Downloaded {ticker}: {len(data)} rows")
            
        except Exception as e:
            print(f"[ERROR] Failed to download {ticker}: {e}")
    
    if not all_data:
        raise ValueError("No data downloaded successfully")
    
    # Combine data
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values(['date', 'tic'])
    
    # Add technical indicators (vectorized)
    print("Adding technical indicators...")
    processed_data = []
    
    for ticker in df['tic'].unique():
        ticker_data = df[df['tic'] == ticker].copy().reset_index(drop=True)
        
        # Vectorized technical indicators
        ticker_data['sma_10'] = ticker_data['close'].rolling(10).mean()
        ticker_data['sma_30'] = ticker_data['close'].rolling(30).mean()
        
        # RSI
        delta = ticker_data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        ticker_data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = ticker_data['close'].ewm(span=12).mean()
        exp2 = ticker_data['close'].ewm(span=26).mean()
        ticker_data['macd'] = exp1 - exp2
        
        processed_data.append(ticker_data)
    
    df = pd.concat(processed_data, ignore_index=True)
    df = df.sort_values(['date', 'tic'])
    df = df.fillna(0)
    
    print(f"[OK] Data processed: {df.shape} rows, {len(df['tic'].unique())} stocks")
    return df

def train_gpu_ensemble(train_data):
    """Train ensemble with GPU optimization"""
    print(f"\n{'='*50}")
    print(f"Training ensemble models on: {device}")
    print(f"{'='*50}")
    
    # Create environment
    env = DummyVecEnv([lambda: GPUStockTradingEnv(train_data)])
    
    models = {}
    training_timesteps = 100000  # Increased for GPU
    
    # Training parameters optimized for GPU
    gpu_batch_size = 256 if str(device) != "cpu" else 64
    
    # Train A2C
    print(f"\n[1/3] Training A2C on {device}...")
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
    print(f"Training A2C for {training_timesteps:,} timesteps...")
    a2c_model.learn(total_timesteps=training_timesteps, progress_bar=True)
    a2c_model.save("trained_models/gpu_a2c_model")
    models['A2C'] = a2c_model
    
    if device_available and torch.cuda.is_available():
        print(f"[GPU] A2C training memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        torch.cuda.empty_cache()
    
    # Train PPO
    print(f"\n[2/3] Training PPO on {device}...")
    ppo_model = PPO(
        'MlpPolicy', 
        env, 
        verbose=1,
        seed=42,
        device=device,
        tensorboard_log="./tensorboard_log/ppo/",
        batch_size=gpu_batch_size,
        n_epochs=10,
        learning_rate=0.0003
    )
    print(f"Training PPO for {training_timesteps:,} timesteps...")
    ppo_model.learn(total_timesteps=training_timesteps, progress_bar=True)
    ppo_model.save("trained_models/gpu_ppo_model")
    models['PPO'] = ppo_model
    
    if device_available and torch.cuda.is_available():
        print(f"[GPU] PPO training memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        torch.cuda.empty_cache()
    
    # Train DDPG
    print(f"\n[3/3] Training DDPG on {device}...")
    ddpg_model = DDPG(
        'MlpPolicy', 
        env, 
        verbose=1,
        seed=42,
        device=device,
        tensorboard_log="./tensorboard_log/ddpg/",
        batch_size=gpu_batch_size,
        buffer_size=200000,
        learning_rate=0.001
    )
    print(f"Training DDPG for {training_timesteps:,} timesteps...")
    ddpg_model.learn(total_timesteps=training_timesteps, progress_bar=True)
    ddpg_model.save("trained_models/gpu_ddpg_model")
    models['DDPG'] = ddpg_model
    
    # Final GPU memory report
    if device_available and torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"\n[GPU] Final training statistics:")
        print(f"[GPU] Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"[GPU] Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"[GPU] Max memory allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        torch.cuda.empty_cache()
    
    print(f"\n[SUCCESS] All models trained successfully!")
    return models

def backtest_and_analyze(models, test_data):
    """Backtest with detailed analysis"""
    print("\n" + "="*50)
    print("BACKTESTING ENSEMBLE STRATEGY")
    print("="*50)
    
    env = GPUStockTradingEnv(test_data)
    model_results = {}
    
    # Test each model
    for name, model in models.items():
        print(f"\nTesting {name}...")
        obs, _ = env.reset()
        portfolio_values = [INITIAL_AMOUNT]
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            portfolio_values.append(info.get('portfolio_value', portfolio_values[-1]))
        
        model_results[name] = portfolio_values
    
    # Ensemble strategy
    print("\nRunning ensemble strategy...")
    obs, _ = env.reset()
    ensemble_portfolio = [INITIAL_AMOUNT]
    done = False
    
    while not done:
        actions = []
        for model in models.values():
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action)
        
        ensemble_action = np.mean(actions, axis=0)
        obs, reward, done, truncated, info = env.step(ensemble_action)
        ensemble_portfolio.append(info.get('portfolio_value', ensemble_portfolio[-1]))
    
    model_results['Ensemble'] = ensemble_portfolio
    
    # Performance analysis
    print("\n" + "="*60)
    print("PERFORMANCE RESULTS")
    print("="*60)
    
    for name, portfolio_values in model_results.items():
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        max_drawdown = (pd.Series(portfolio_values) / pd.Series(portfolio_values).cummax() - 1).min()
        
        print(f"\n{name}:")
        print(f"  Total Return:     {total_return:7.2%}")
        print(f"  Annual Return:    {annual_return:7.2%}")
        print(f"  Volatility:       {volatility:7.2%}")
        print(f"  Sharpe Ratio:     {sharpe_ratio:7.2f}")
        print(f"  Max Drawdown:     {max_drawdown:7.2%}")
        print(f"  Final Value:      ${portfolio_values[-1]:,.0f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    for name, portfolio_values in model_results.items():
        style = '--' if name == 'Ensemble' else '-'
        width = 3 if name == 'Ensemble' else 2
        plt.plot(portfolio_values, label=name, linestyle=style, linewidth=width)
    
    plt.title('GPU-Accelerated FinRL Ensemble Strategy Results', fontsize=16)
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('results/gpu_ensemble_results.png', dpi=300, bbox_inches='tight')
    print(f"\n[OK] Results saved to results/gpu_ensemble_results.png")
    
    return model_results

def main():
    """Main execution with GPU optimization"""
    print("FinRL GPU-Accelerated Ensemble Strategy")
    print("="*60)
    print(f"Device: {device}")
    print(f"Selected tickers: {SELECTED_TICKERS}")
    print("="*60)
    
    try:
        # Download and process data
        df = download_and_process_data()
        
        # Split data
        train_data = df[(df['date'] >= TRAIN_START_DATE) & (df['date'] < TEST_START_DATE)]
        test_data = df[(df['date'] >= TEST_START_DATE) & (df['date'] <= TEST_END_DATE)]
        
        print(f"[OK] Train data: {len(train_data)} rows")
        print(f"[OK] Test data: {len(test_data)} rows")
        
        # Train models
        models = train_gpu_ensemble(train_data)
        
        # Backtest
        results = backtest_and_analyze(models, test_data)
        
        print("\n" + "="*60)
        print("🚀 GPU-ACCELERATED ENSEMBLE STRATEGY COMPLETED!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Strategy failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)