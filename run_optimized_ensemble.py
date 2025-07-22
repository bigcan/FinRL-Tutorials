#!/usr/bin/env python3
"""
FinRL Ensemble Strategy - Optimized Version
CPU/GPU optimized with fallback handling
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

# Try to import PyTorch for GPU detection
try:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_available = True
    print(f"PyTorch detected - Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
except:
    pytorch_available = False
    device = "auto"  # Let Stable Baselines3 decide
    print("PyTorch not available or CUDA issues detected")
    print("Using CPU with automatic device selection")

# Stable Baselines3 imports
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

# Configuration - Optimized for performance
SELECTED_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "JPM", "V", "META"]  # Extended list
TRAIN_START_DATE = '2022-01-01'
TRAIN_END_DATE = '2023-06-01' 
TEST_START_DATE = '2023-06-01'
TEST_END_DATE = '2024-01-01'

INITIAL_AMOUNT = 1000000
TRANSACTION_COST = 0.001

class OptimizedStockTradingEnv(gym.Env):
    """Optimized Stock Trading Environment for better performance"""
    
    def __init__(self, df, initial_amount=1000000, transaction_cost=0.001):
        super().__init__()
        
        # Preprocess and cache data for faster access
        self.df = df.copy()
        self.stock_dim = len(df['tic'].unique())
        self.initial_amount = initial_amount
        self.transaction_cost = transaction_cost
        
        # Action space: portfolio weights
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32
        )
        
        # State space: [balance, prices, holdings, technical_indicators]
        self.state_dim = 1 + 2 * self.stock_dim + 4 * self.stock_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Pre-compute data structures for faster access
        self.dates = sorted(self.df['date'].unique())
        self.tickers = sorted(self.df['tic'].unique())
        
        # Create lookup dictionary for faster data access
        self.data_lookup = {}
        for date in self.dates:
            date_data = self.df[self.df['date'] == date].sort_values('tic')
            self.data_lookup[date] = {
                'prices': date_data['close'].values.astype(np.float32),
                'indicators': np.concatenate([
                    date_data['sma_10'].values,
                    date_data['sma_30'].values, 
                    date_data['rsi'].values,
                    date_data['macd'].values
                ]).astype(np.float32)
            }
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = float(self.initial_amount)
        self.holdings = np.zeros(self.stock_dim, dtype=np.float32)
        self.portfolio_values = [self.initial_amount]
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Fast observation using pre-computed lookup"""
        if self.current_step >= len(self.dates):
            self.current_step = len(self.dates) - 1
        
        current_date = self.dates[self.current_step]
        data = self.data_lookup[current_date]
        
        # Get prices and indicators
        prices = data['prices']
        if len(prices) != self.stock_dim:
            prices = np.ones(self.stock_dim, dtype=np.float32) * 100
        
        indicators = data['indicators']
        if len(indicators) != 4 * self.stock_dim:
            indicators = np.zeros(4 * self.stock_dim, dtype=np.float32)
        
        # Combine state
        state = np.concatenate([
            [self.balance], 
            prices, 
            self.holdings, 
            indicators
        ]).astype(np.float32)
        
        return state
    
    def step(self, action):
        """Optimized step function"""
        if self.current_step >= len(self.dates) - 1:
            return self._get_observation(), 0.0, True, True, {}
        
        # Get current prices
        current_date = self.dates[self.current_step]
        prices = self.data_lookup[current_date]['prices']
        
        if len(prices) != self.stock_dim:
            prices = np.ones(self.stock_dim, dtype=np.float32) * 100
        
        # Portfolio value before action
        portfolio_value_before = self.balance + np.sum(self.holdings * prices)
        
        # Execute trades with vectorized operations
        action = np.clip(action, -1, 1)
        total_value = portfolio_value_before
        
        # Calculate target positions (max 25% per stock)
        target_values = action * total_value * 0.25
        target_holdings = target_values / (prices + 1e-8)
        
        # Calculate trades needed
        holdings_change = target_holdings - self.holdings
        
        # Execute buy orders
        buy_mask = holdings_change > 0
        if np.any(buy_mask):
            buy_costs = holdings_change[buy_mask] * prices[buy_mask] * (1 + self.transaction_cost)
            total_buy_cost = np.sum(buy_costs)
            
            if total_buy_cost <= self.balance:
                self.balance -= total_buy_cost
                self.holdings[buy_mask] += holdings_change[buy_mask]
        
        # Execute sell orders
        sell_mask = holdings_change < 0
        if np.any(sell_mask):
            sell_amounts = np.abs(holdings_change[sell_mask])
            valid_sells = self.holdings[sell_mask] >= sell_amounts
            
            if np.any(valid_sells):
                sell_proceeds = sell_amounts[valid_sells] * prices[sell_mask][valid_sells] * (1 - self.transaction_cost)
                self.balance += np.sum(sell_proceeds)
                self.holdings[sell_mask] = np.where(
                    valid_sells,
                    self.holdings[sell_mask] + holdings_change[sell_mask],
                    self.holdings[sell_mask]
                )
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        if self.current_step < len(self.dates):
            next_date = self.dates[self.current_step]
            next_prices = self.data_lookup[next_date]['prices']
            if len(next_prices) != self.stock_dim:
                next_prices = prices
        else:
            next_prices = prices
        
        portfolio_value_after = self.balance + np.sum(self.holdings * next_prices)
        
        # Reward is percentage change in portfolio value
        reward = (portfolio_value_after - portfolio_value_before) / (portfolio_value_before + 1e-8)
        
        self.portfolio_values.append(portfolio_value_after)
        
        done = self.current_step >= len(self.dates) - 1
        
        return self._get_observation(), float(reward), done, False, {
            'portfolio_value': portfolio_value_after,
            'balance': self.balance,
            'holdings': self.holdings.copy()
        }

def download_and_process_data():
    """Download and process stock data efficiently"""
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
    
    # Combine and sort data
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values(['date', 'tic'])
    
    # Add technical indicators efficiently
    print("Adding technical indicators...")
    processed_data = []
    
    for ticker in df['tic'].unique():
        ticker_data = df[df['tic'] == ticker].copy().reset_index(drop=True)
        
        # Calculate technical indicators
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
    df = df.sort_values(['date', 'tic']).fillna(0)
    
    print(f"[OK] Processed {df.shape[0]} rows for {len(df['tic'].unique())} stocks")
    return df

def train_optimized_ensemble(train_data):
    """Train ensemble with hardware optimization"""
    print(f"\n{'='*60}")
    print(f"TRAINING ENSEMBLE MODELS")
    print(f"Device: {device}")
    print(f"Stocks: {len(train_data['tic'].unique())}")
    print(f"Training period: {len(train_data)} rows")
    print(f"{'='*60}")
    
    # Create environment
    env = DummyVecEnv([lambda: OptimizedStockTradingEnv(train_data)])
    
    models = {}
    
    # Determine training parameters based on available hardware
    if pytorch_available and torch.cuda.is_available():
        training_timesteps = 150000  # More timesteps for GPU
        batch_size_base = 128
        buffer_size = 300000
        print("[INFO] GPU acceleration detected - using intensive training")
    else:
        training_timesteps = 75000   # Moderate for CPU
        batch_size_base = 64
        buffer_size = 150000
        print("[INFO] Using CPU optimization")
    
    # 1. Train A2C
    print(f"\n[1/3] Training A2C ({training_timesteps:,} timesteps)...")
    a2c_model = A2C(
        'MlpPolicy',
        env,
        verbose=1,
        seed=42,
        device=device,
        tensorboard_log="./tensorboard_log/a2c/",
        n_steps=5,
        learning_rate=0.0007,
        ent_coef=0.01
    )
    
    print("Starting A2C training...")
    a2c_model.learn(total_timesteps=training_timesteps, progress_bar=True)
    a2c_model.save("trained_models/optimized_a2c")
    models['A2C'] = a2c_model
    print("✓ A2C training completed")
    
    # 2. Train PPO
    print(f"\n[2/3] Training PPO ({training_timesteps:,} timesteps)...")
    ppo_model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        seed=42,
        device=device,
        tensorboard_log="./tensorboard_log/ppo/",
        batch_size=batch_size_base,
        n_epochs=10,
        learning_rate=0.0003,
        ent_coef=0.01,
        clip_range=0.2
    )
    
    print("Starting PPO training...")
    ppo_model.learn(total_timesteps=training_timesteps, progress_bar=True)
    ppo_model.save("trained_models/optimized_ppo")
    models['PPO'] = ppo_model
    print("✓ PPO training completed")
    
    # 3. Train DDPG
    print(f"\n[3/3] Training DDPG ({training_timesteps:,} timesteps)...")
    ddpg_model = DDPG(
        'MlpPolicy',
        env,
        verbose=1,
        seed=42,
        device=device,
        tensorboard_log="./tensorboard_log/ddpg/",
        batch_size=batch_size_base,
        buffer_size=buffer_size,
        learning_rate=0.001,
        tau=0.005
    )
    
    print("Starting DDPG training...")
    ddpg_model.learn(total_timesteps=training_timesteps, progress_bar=True)
    ddpg_model.save("trained_models/optimized_ddpg")
    models['DDPG'] = ddpg_model
    print("✓ DDPG training completed")
    
    # Memory cleanup for GPU
    if pytorch_available and torch.cuda.is_available():
        try:
            print(f"\n[GPU] Peak memory usage: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
            torch.cuda.empty_cache()
        except:
            pass
    
    print(f"\n✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"📊 Total training timesteps: {training_timesteps * 3:,}")
    
    return models

def backtest_and_evaluate(models, test_data):
    """Comprehensive backtesting and evaluation"""
    print(f"\n{'='*60}")
    print("BACKTESTING ENSEMBLE STRATEGY")
    print(f"{'='*60}")
    
    env = OptimizedStockTradingEnv(test_data)
    results = {}
    
    # Test individual models
    for name, model in models.items():
        print(f"\nTesting {name}...")
        
        obs, _ = env.reset()
        portfolio_values = [INITIAL_AMOUNT]
        actions_taken = []
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions_taken.append(action.copy())
            obs, reward, done, truncated, info = env.step(action)
            portfolio_values.append(info.get('portfolio_value', portfolio_values[-1]))
        
        results[name] = {
            'portfolio_values': portfolio_values,
            'actions': actions_taken
        }
        
        final_value = portfolio_values[-1]
        total_return = (final_value / INITIAL_AMOUNT) - 1
        print(f"✓ {name}: ${final_value:,.0f} ({total_return:+.2%})")
    
    # Ensemble strategy
    print(f"\nTesting Ensemble Strategy...")
    obs, _ = env.reset()
    ensemble_portfolio = [INITIAL_AMOUNT]
    done = False
    
    while not done:
        # Get predictions from all models
        actions = []
        for model in models.values():
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action)
        
        # Average the actions for ensemble
        ensemble_action = np.mean(actions, axis=0)
        
        obs, reward, done, truncated, info = env.step(ensemble_action)
        ensemble_portfolio.append(info.get('portfolio_value', ensemble_portfolio[-1]))
    
    results['Ensemble'] = {'portfolio_values': ensemble_portfolio, 'actions': []}
    
    final_value = ensemble_portfolio[-1]
    total_return = (final_value / INITIAL_AMOUNT) - 1
    print(f"✓ Ensemble: ${final_value:,.0f} ({total_return:+.2%})")
    
    # Performance analysis
    print(f"\n{'='*70}")
    print("DETAILED PERFORMANCE ANALYSIS")
    print(f"{'='*70}")
    
    performance_data = []
    
    for name, data in results.items():
        portfolio_values = data['portfolio_values']
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        metrics = {
            'Strategy': name,
            'Final Value': portfolio_values[-1],
            'Total Return': (portfolio_values[-1] / portfolio_values[0]) - 1,
            'Annualized Return': (portfolio_values[-1] / portfolio_values[0]) ** (252 / len(portfolio_values)) - 1,
            'Volatility': returns.std() * np.sqrt(252),
            'Sharpe Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            'Max Drawdown': (pd.Series(portfolio_values) / pd.Series(portfolio_values).cummax() - 1).min(),
            'Win Rate': (returns > 0).mean()
        }
        
        performance_data.append(metrics)
        
        print(f"\n{name.upper()}:")
        print(f"  Final Value:      ${metrics['Final Value']:,.0f}")
        print(f"  Total Return:     {metrics['Total Return']:7.2%}")
        print(f"  Annualized:       {metrics['Annualized Return']:7.2%}")
        print(f"  Volatility:       {metrics['Volatility']:7.2%}")
        print(f"  Sharpe Ratio:     {metrics['Sharpe Ratio']:7.2f}")
        print(f"  Max Drawdown:     {metrics['Max Drawdown']:7.2%}")
        print(f"  Win Rate:         {metrics['Win Rate']:7.2%}")
    
    # Create comprehensive plot
    plt.figure(figsize=(16, 12))
    
    # Main performance plot
    plt.subplot(2, 2, 1)
    for name, data in results.items():
        style = '--' if name == 'Ensemble' else '-'
        width = 3 if name == 'Ensemble' else 2
        plt.plot(data['portfolio_values'], label=name, linestyle=style, linewidth=width)
    
    plt.title('Portfolio Performance Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Returns distribution
    plt.subplot(2, 2, 2)
    for name, data in results.items():
        returns = pd.Series(data['portfolio_values']).pct_change().dropna()
        plt.hist(returns, alpha=0.6, label=name, bins=30)
    
    plt.title('Daily Returns Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Daily Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Drawdown analysis
    plt.subplot(2, 2, 3)
    for name, data in results.items():
        portfolio_values = pd.Series(data['portfolio_values'])
        drawdown = (portfolio_values / portfolio_values.cummax() - 1) * 100
        plt.plot(drawdown, label=name)
    
    plt.title('Drawdown Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Rolling Sharpe ratio
    plt.subplot(2, 2, 4)
    window = 30
    for name, data in results.items():
        returns = pd.Series(data['portfolio_values']).pct_change().dropna()
        if len(returns) > window:
            rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
            plt.plot(rolling_sharpe, label=name)
    
    plt.title(f'Rolling {window}-Day Sharpe Ratio', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comprehensive_ensemble_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Comprehensive analysis saved to results/comprehensive_ensemble_analysis.png")
    
    # Save performance data
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv('results/performance_metrics.csv', index=False)
    print(f"📋 Performance metrics saved to results/performance_metrics.csv")
    
    return results

def main():
    """Main execution function"""
    print("🚀 FINRL OPTIMIZED ENSEMBLE STRATEGY")
    print("="*70)
    print(f"Selected tickers: {', '.join(SELECTED_TICKERS)}")
    print(f"Training period: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
    print(f"Testing period: {TEST_START_DATE} to {TEST_END_DATE}")
    print(f"Initial capital: ${INITIAL_AMOUNT:,}")
    print("="*70)
    
    try:
        # Download and process data
        df = download_and_process_data()
        
        # Split data
        train_data = df[(df['date'] >= TRAIN_START_DATE) & (df['date'] < TEST_START_DATE)]
        test_data = df[(df['date'] >= TEST_START_DATE) & (df['date'] <= TEST_END_DATE)]
        
        print(f"\n📊 Data Summary:")
        print(f"  Training data: {len(train_data)} rows ({len(train_data['tic'].unique())} stocks)")
        print(f"  Testing data: {len(test_data)} rows ({len(test_data['tic'].unique())} stocks)")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Train ensemble models
        models = train_optimized_ensemble(train_data)
        
        # Backtest and analyze
        results = backtest_and_evaluate(models, test_data)
        
        print(f"\n{'='*70}")
        print("🎉 ENSEMBLE STRATEGY COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("📁 Generated files:")
        print("  • results/comprehensive_ensemble_analysis.png")
        print("  • results/performance_metrics.csv") 
        print("  • trained_models/ (A2C, PPO, DDPG models)")
        print("  • tensorboard_log/ (training logs)")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Strategy failed - {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)