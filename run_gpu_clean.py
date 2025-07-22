#!/usr/bin/env python3
"""
FinRL Ensemble Strategy - GPU Accelerated Clean Version
No Unicode characters for Windows compatibility
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import torch

# GPU Configuration and Testing
print("FinRL GPU-Accelerated Ensemble Strategy")
print("=" * 60)

# Test GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    use_gpu = True
else:
    device = torch.device("cpu")
    print("GPU NOT AVAILABLE - Using CPU")
    use_gpu = False

print("=" * 60)

# Stable Baselines3 imports
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

# Configuration - Extended for GPU power
SELECTED_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", 
    "JPM", "V", "META", "AMZN", "UNH"  # More stocks for GPU
]

TRAIN_START = '2021-01-01'  # Longer training period for GPU
TRAIN_END = '2023-06-01' 
TEST_START = '2023-06-01'
TEST_END = '2024-01-01'

INITIAL_CAPITAL = 1000000
TRANSACTION_COST = 0.001

class GPUOptimizedTradingEnv(gym.Env):
    """GPU-optimized trading environment"""
    
    def __init__(self, df, initial_amount=1000000, transaction_cost=0.001):
        super().__init__()
        
        self.df = df.copy()
        self.stock_dim = len(df['tic'].unique())
        self.initial_amount = initial_amount
        self.transaction_cost = transaction_cost
        
        # Action space: portfolio weights
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32
        )
        
        # State space
        self.state_dim = 1 + 2 * self.stock_dim + 4 * self.stock_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Pre-compute data for faster access
        self.dates = sorted(self.df['date'].unique())
        self.tickers = sorted(self.df['tic'].unique())
        
        # Create data cache
        self.data_cache = {}
        for date in self.dates:
            date_data = self.df[self.df['date'] == date].sort_values('tic')
            if len(date_data) == self.stock_dim:
                self.data_cache[date] = {
                    'prices': date_data['close'].values.astype(np.float32),
                    'sma_10': date_data['sma_10'].values.astype(np.float32),
                    'sma_30': date_data['sma_30'].values.astype(np.float32),
                    'rsi': date_data['rsi'].values.astype(np.float32),
                    'macd': date_data['macd'].values.astype(np.float32)
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
        if self.current_step >= len(self.dates):
            self.current_step = len(self.dates) - 1
        
        current_date = self.dates[self.current_step]
        
        if current_date in self.data_cache:
            data = self.data_cache[current_date]
            prices = data['prices']
            indicators = np.concatenate([
                data['sma_10'], data['sma_30'], 
                data['rsi'], data['macd']
            ])
        else:
            prices = np.ones(self.stock_dim, dtype=np.float32) * 100
            indicators = np.zeros(4 * self.stock_dim, dtype=np.float32)
        
        state = np.concatenate([
            [self.balance], prices, self.holdings, indicators
        ]).astype(np.float32)
        
        return state
    
    def step(self, action):
        if self.current_step >= len(self.dates) - 1:
            return self._get_observation(), 0.0, True, True, {}
        
        current_date = self.dates[self.current_step]
        
        if current_date in self.data_cache:
            prices = self.data_cache[current_date]['prices']
        else:
            prices = np.ones(self.stock_dim, dtype=np.float32) * 100
        
        portfolio_before = self.balance + np.sum(self.holdings * prices)
        
        # Process trading actions
        action = np.clip(action, -1, 1)
        total_value = portfolio_before
        
        # Target positions (max 30% per stock for GPU training)
        target_values = action * total_value * 0.3
        target_holdings = target_values / (prices + 1e-8)
        holdings_change = target_holdings - self.holdings
        
        # Execute trades
        buy_mask = holdings_change > 0
        if np.any(buy_mask):
            buy_costs = holdings_change[buy_mask] * prices[buy_mask] * (1 + self.transaction_cost)
            total_cost = np.sum(buy_costs)
            if total_cost <= self.balance:
                self.balance -= total_cost
                self.holdings[buy_mask] += holdings_change[buy_mask]
        
        sell_mask = holdings_change < 0
        if np.any(sell_mask):
            sell_amounts = np.abs(holdings_change[sell_mask])
            can_sell = self.holdings[sell_mask] >= sell_amounts
            if np.any(can_sell):
                proceeds = sell_amounts[can_sell] * prices[sell_mask][can_sell] * (1 - self.transaction_cost)
                self.balance += np.sum(proceeds)
                self.holdings[sell_mask] = np.where(
                    can_sell,
                    self.holdings[sell_mask] + holdings_change[sell_mask],
                    self.holdings[sell_mask]
                )
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        if self.current_step < len(self.dates):
            next_date = self.dates[self.current_step]
            if next_date in self.data_cache:
                next_prices = self.data_cache[next_date]['prices']
            else:
                next_prices = prices
        else:
            next_prices = prices
        
        portfolio_after = self.balance + np.sum(self.holdings * next_prices)
        reward = (portfolio_after - portfolio_before) / (portfolio_before + 1e-8)
        
        self.portfolio_values.append(portfolio_after)
        done = self.current_step >= len(self.dates) - 1
        
        return self._get_observation(), float(reward), done, False, {
            'portfolio_value': portfolio_after,
            'balance': self.balance,
            'holdings': self.holdings.copy()
        }

def download_market_data():
    """Download market data efficiently"""
    print("\\n[DATA] Downloading market data...")
    
    all_data = []
    for i, ticker in enumerate(SELECTED_TICKERS, 1):
        try:
            print(f"  [{i}/{len(SELECTED_TICKERS)}] {ticker}...", end=" ")
            
            data = yf.download(ticker, start=TRAIN_START, end=TEST_END, 
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
            print(f"OK ({len(data)} rows)")
            
        except Exception as e:
            print(f"ERROR - {e}")
    
    if not all_data:
        raise ValueError("No data downloaded")
    
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values(['date', 'tic'])
    
    print(f"\\n[TECH] Processing technical indicators...")
    processed_data = []
    
    for ticker in df['tic'].unique():
        ticker_data = df[df['tic'] == ticker].copy().reset_index(drop=True)
        
        # Technical indicators
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
    
    print(f"[OK] Data processed: {len(df)} rows, {len(df['tic'].unique())} stocks")
    return df

def train_gpu_ensemble(train_data):
    """Train ensemble with GPU acceleration"""
    print(f"\\n[TRAIN] ENSEMBLE TRAINING - GPU ACCELERATED")
    print("=" * 60)
    print(f"Training data: {len(train_data)} rows")
    print(f"Stocks: {train_data['tic'].nunique()}")
    print("=" * 60)
    
    env = DummyVecEnv([lambda: GPUOptimizedTradingEnv(train_data)])
    
    models = {}
    
    # GPU-optimized parameters
    if use_gpu:
        training_timesteps = 200000  # More timesteps for GPU
        batch_size_base = 256       # Larger batches for GPU
        buffer_size = 500000        # Larger buffer for DDPG
        print(f"[GPU] GPU ACCELERATION ENABLED")
        print(f"   Training timesteps: {training_timesteps:,} per model")
        print(f"   Batch size: {batch_size_base}")
    else:
        training_timesteps = 100000
        batch_size_base = 128
        buffer_size = 200000
        print(f"[CPU] Using CPU optimization")
    
    # 1. Train A2C
    print(f"\\n[1/3] Training A2C ({training_timesteps:,} steps)")
    a2c = A2C(
        'MlpPolicy', env, verbose=1, seed=42, device=device,
        tensorboard_log="./tensorboard_log/gpu_a2c/",
        n_steps=5, learning_rate=0.0007, ent_coef=0.01
    )
    
    a2c.learn(total_timesteps=training_timesteps, progress_bar=True)
    a2c.save("trained_models/gpu_a2c")
    models['A2C'] = a2c
    print("      [OK] A2C training completed")
    
    # 2. Train PPO  
    print(f"\\n[2/3] Training PPO ({training_timesteps:,} steps)")
    ppo = PPO(
        'MlpPolicy', env, verbose=1, seed=42, device=device,
        tensorboard_log="./tensorboard_log/gpu_ppo/",
        batch_size=batch_size_base, n_epochs=10, 
        learning_rate=0.0003, ent_coef=0.01, clip_range=0.2
    )
    
    ppo.learn(total_timesteps=training_timesteps, progress_bar=True)
    ppo.save("trained_models/gpu_ppo")
    models['PPO'] = ppo
    print("      [OK] PPO training completed")
    
    # 3. Train DDPG
    print(f"\\n[3/3] Training DDPG ({training_timesteps:,} steps)")
    ddpg = DDPG(
        'MlpPolicy', env, verbose=1, seed=42, device=device,
        tensorboard_log="./tensorboard_log/gpu_ddpg/",
        batch_size=batch_size_base, buffer_size=buffer_size,
        learning_rate=0.001, tau=0.005, gamma=0.99
    )
    
    ddpg.learn(total_timesteps=training_timesteps, progress_bar=True)
    ddpg.save("trained_models/gpu_ddpg")
    models['DDPG'] = ddpg
    print("      [OK] DDPG training completed")
    
    # GPU memory cleanup
    if use_gpu:
        try:
            print(f"\\n[GPU] Peak memory usage: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
            torch.cuda.empty_cache()
        except:
            pass
    
    print(f"\\n[OK] Total timesteps: {training_timesteps * 3:,}")
    print(f"[OK] All models trained successfully!")
    
    return models

def comprehensive_backtest(models, test_data):
    """Comprehensive backtesting"""
    print(f"\\n[TEST] BACKTESTING ENSEMBLE STRATEGY")
    print("=" * 60)
    
    env = GPUOptimizedTradingEnv(test_data)
    results = {}
    
    # Test individual models
    for name, model in models.items():
        print(f"\\n[{name}] Testing {name}...")
        
        obs, _ = env.reset()
        portfolio_values = [INITIAL_CAPITAL]
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            portfolio_values.append(info.get('portfolio_value', portfolio_values[-1]))
        
        results[name] = portfolio_values
        final_value = portfolio_values[-1]
        total_return = (final_value / INITIAL_CAPITAL - 1) * 100
        print(f"    Final: ${final_value:,.0f} ({total_return:+.1f}%)")
    
    # Ensemble strategy
    print(f"\\n[ENSEMBLE] Testing Ensemble...")
    obs, _ = env.reset()
    ensemble_values = [INITIAL_CAPITAL]
    done = False
    
    while not done:
        actions = [model.predict(obs, deterministic=True)[0] for model in models.values()]
        ensemble_action = np.mean(actions, axis=0)
        
        obs, reward, done, truncated, info = env.step(ensemble_action)
        ensemble_values.append(info.get('portfolio_value', ensemble_values[-1]))
    
    results['Ensemble'] = ensemble_values
    final_value = ensemble_values[-1]
    total_return = (final_value / INITIAL_CAPITAL - 1) * 100
    print(f"    [RESULT] Total Return: {total_return:+.2f}%")
    
    return results

def analyze_performance(results):
    """Detailed performance analysis"""
    print(f"\\n[ANALYSIS] PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    performance_data = []
    
    for name, portfolio_values in results.items():
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # Metrics
        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        annual_return = ((portfolio_values[-1] / portfolio_values[0]) ** (252 / len(portfolio_values)) - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Drawdown
        peak = pd.Series(portfolio_values).cummax()
        drawdown = (pd.Series(portfolio_values) - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1
        sortino = (returns.mean() * 252) / downside_std if downside_std > 0 else 0
        
        win_rate = (returns > 0).mean() * 100
        
        metrics = {
            'Strategy': name,
            'Final_Value': portfolio_values[-1],
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe,
            'Sortino_Ratio': sortino,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Days': len(portfolio_values) - 1
        }
        
        performance_data.append(metrics)
        
        print(f"\\n{name.upper()}:")
        print(f"  [VAL] Final Value:      ${metrics['Final_Value']:,.0f}")
        print(f"  [RET] Total Return:     {metrics['Total_Return']:6.2f}%")
        print(f"  [ANN] Annual Return:    {metrics['Annual_Return']:7.2f}%")
        print(f"  [VOL] Volatility:       {metrics['Volatility']:7.2f}%")
        print(f"  [SHP] Sharpe Ratio:     {metrics['Sharpe_Ratio']:7.2f}")
        print(f"  [SOR] Sortino Ratio:    {metrics['Sortino_Ratio']:7.2f}")
        print(f"  [DD]  Max Drawdown:     {metrics['Max_Drawdown']:7.2f}%")
        print(f"  [WIN] Win Rate:         {metrics['Win_Rate']:7.2f}%")
        print(f"  [DAYS] Trading Days:     {metrics['Days']}")
    
    # Save data
    df = pd.DataFrame(performance_data)
    df.to_csv('results/gpu_performance_metrics.csv', index=False)
    print(f"\\n[SAVE] Performance data saved: results/gpu_performance_metrics.csv")
    
    return performance_data

def create_visualizations(results):
    """Create performance visualizations"""
    print(f"\\n[VIZ] Creating visualizations...")
    
    # Main performance chart
    plt.figure(figsize=(15, 10))
    
    # Performance comparison
    plt.subplot(2, 2, 1)
    for name, values in results.items():
        style = '--' if name == 'Ensemble' else '-'
        width = 3 if name == 'Ensemble' else 2
        plt.plot(values, label=name, linestyle=style, linewidth=width)
    
    plt.title('Portfolio Performance Comparison', fontsize=14, weight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Returns distribution
    plt.subplot(2, 2, 2)
    for name, values in results.items():
        returns = pd.Series(values).pct_change().dropna()
        if len(returns) > 0:
            plt.hist(returns, alpha=0.6, label=name, bins=30, density=True)
    
    plt.title('Daily Returns Distribution', fontsize=14, weight='bold')
    plt.xlabel('Daily Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Drawdown
    plt.subplot(2, 2, 3)
    for name, values in results.items():
        portfolio = pd.Series(values)
        peak = portfolio.cummax()
        drawdown = (portfolio - peak) / peak * 100
        plt.plot(drawdown, label=name)
    
    plt.title('Drawdown Analysis', fontsize=14, weight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Performance bar chart
    plt.subplot(2, 2, 4)
    names = list(results.keys())
    final_returns = [(results[name][-1] / INITIAL_CAPITAL - 1) * 100 for name in names]
    
    bars = plt.bar(names, final_returns)
    plt.title('Total Returns Comparison', fontsize=14, weight='bold')
    plt.ylabel('Total Return (%)')
    plt.xticks(rotation=45)
    
    # Color bars
    for bar, return_val in zip(bars, final_returns):
        if return_val > 20:
            bar.set_color('darkgreen')
        elif return_val > 10:
            bar.set_color('green')
        elif return_val > 0:
            bar.set_color('lightgreen')
        else:
            bar.set_color('red')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/gpu_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("  [OK] Comprehensive analysis: results/gpu_comprehensive_analysis.png")
    
    # Summary performance chart
    plt.figure(figsize=(12, 8))
    for name, values in results.items():
        style = '--' if name == 'Ensemble' else '-'
        width = 4 if name == 'Ensemble' else 2
        alpha = 1.0 if name == 'Ensemble' else 0.7
        plt.plot(values, label=name, linestyle=style, linewidth=width, alpha=alpha)
    
    plt.title('FinRL GPU Ensemble Strategy - Performance Results', fontsize=16, weight='bold')
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add performance annotation
    if 'Ensemble' in results:
        final_value = results['Ensemble'][-1]
        total_return = (final_value / INITIAL_CAPITAL - 1) * 100
        plt.text(0.02, 0.98, 
                f'[GPU] Ensemble Performance:\\n[VAL] Final: ${final_value:,.0f}\\n[RET] Return: {total_return:+.1f}%\\n[HW] GPU Accelerated', 
                transform=plt.gca().transAxes, fontsize=12, weight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('results/gpu_ensemble_performance.png', dpi=300, bbox_inches='tight')
    print("  [OK] Performance chart: results/gpu_ensemble_performance.png")

def main():
    """Main execution with GPU optimization"""
    print("\\n[GPU] FINRL GPU-ACCELERATED ENSEMBLE STRATEGY")
    print("=" * 80)
    print(f"[TIME] Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[HW] Hardware: {torch.cuda.get_device_name(0) if use_gpu else 'CPU Only'}")
    print(f"[TRAIN] Training: {TRAIN_START} to {TRAIN_END}")
    print(f"[TEST] Testing: {TEST_START} to {TEST_END}")
    print(f"[CAPITAL] Capital: ${INITIAL_CAPITAL:,}")
    print(f"[STOCKS] Universe: {len(SELECTED_TICKERS)} stocks")
    print("=" * 80)
    
    try:
        overall_start = datetime.now()
        
        # Download and process data
        df = download_market_data()
        
        # Split data
        train_data = df[(df['date'] >= TRAIN_START) & (df['date'] < TEST_START)]
        test_data = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)]
        
        print(f"\\n[DATA] DATA SUMMARY:")
        print(f"  [TRAIN] Training: {len(train_data)} rows")
        print(f"  [TEST] Testing:  {len(test_data)} rows")
        print(f"  [STOCKS] Stocks:   {train_data['tic'].nunique()}")
        print(f"  [PERIOD] Period:   {df['date'].min()} to {df['date'].max()}")
        
        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("Insufficient data")
        
        # Train models
        models = train_gpu_ensemble(train_data)
        
        # Backtest
        results = comprehensive_backtest(models, test_data)
        
        # Analyze performance
        performance_data = analyze_performance(results)
        
        # Create visualizations
        create_visualizations(results)
        
        # Final summary
        end_time = datetime.now()
        runtime = (end_time - overall_start).total_seconds() / 60
        
        print(f"\\n{'='*80}")
        print("[SUCCESS] GPU ENSEMBLE STRATEGY COMPLETED!")
        print(f"{'='*80}")
        
        if 'Ensemble' in results:
            final_value = results['Ensemble'][-1]
            total_return = (final_value / INITIAL_CAPITAL - 1) * 100
            trading_days = len(results['Ensemble']) - 1
            
            print(f"\\n[RESULTS] ENSEMBLE PERFORMANCE:")
            print(f"  [INIT] Initial Capital:    ${INITIAL_CAPITAL:,}")
            print(f"  [FINAL] Final Value:        ${final_value:,.0f}")
            print(f"  [RETURN] Total Return:       {total_return:+.2f}%")
            print(f"  [DAYS] Trading Period:     {trading_days} days")
            print(f"  [DAILY] Avg Daily Return:   {total_return/trading_days:+.3f}%")
            print(f"  [TIME] Runtime:            {runtime:.1f} minutes")
        
        print(f"\\n[FILES] GENERATED FILES:")
        print("  • results/gpu_comprehensive_analysis.png")
        print("  • results/gpu_ensemble_performance.png") 
        print("  • results/gpu_performance_metrics.csv")
        print("  • trained_models/ (GPU-trained models)")
        print("  • tensorboard_log/ (training logs)")
        
        print(f"\\n[NEXT] NEXT STEPS FOR PRODUCTION:")
        print("  • Scale to full DOW 30 universe")
        print("  • Implement risk management overlays") 
        print("  • Connect to live trading APIs")
        print("  • Deploy to cloud GPU instances")
        print("  • Add real-time market data feeds")
        
        return True
        
    except Exception as e:
        print(f"\\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Create directories
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("tensorboard_log", exist_ok=True)
    
    success = main()
    sys.exit(0 if success else 1)