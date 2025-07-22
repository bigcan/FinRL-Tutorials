#!/usr/bin/env python3
"""
FinRL Ensemble Strategy - GPU Accelerated Version
Optimized for NVIDIA RTX 4060 with CUDA 11.8
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
        
        # Action space
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32
        )
        
        # State space
        self.state_dim = 1 + 2 * self.stock_dim + 4 * self.stock_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Optimize data access
        self.dates = sorted(self.df['date'].unique())
        self.tickers = sorted(self.df['tic'].unique())
        
        # Create fast lookup cache
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
        
        # Calculate portfolio value before trading
        portfolio_before = self.balance + np.sum(self.holdings * prices)
        
        # Process actions with vectorized operations
        action = np.clip(action, -1, 1)
        total_value = portfolio_before
        
        # Target positions (15% max per stock for diversification)
        target_values = action * total_value * 0.15
        target_holdings = target_values / (prices + 1e-8)
        holdings_change = target_holdings - self.holdings
        
        # Vectorized trading execution
        # Buy orders
        buy_mask = holdings_change > 0
        if np.any(buy_mask):
            buy_costs = holdings_change[buy_mask] * prices[buy_mask] * (1 + self.transaction_cost)
            total_cost = np.sum(buy_costs)
            if total_cost <= self.balance:
                self.balance -= total_cost
                self.holdings[buy_mask] += holdings_change[buy_mask]
        
        # Sell orders
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
        
        # Move to next timestep
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
    """Download extended market data for GPU training"""
    print(f"\n[STEP 1] Downloading market data for {len(SELECTED_TICKERS)} stocks...")
    
    all_data = []
    for i, ticker in enumerate(SELECTED_TICKERS, 1):
        try:
            print(f"  [{i:2d}/{len(SELECTED_TICKERS)}] {ticker:5s} ... ", end="")
            
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
            print(f"FAILED: {e}")
    
    if not all_data:
        raise ValueError("Data download failed")
    
    # Process technical indicators
    print(f"\n[STEP 2] Computing technical indicators...")
    
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values(['date', 'tic'])
    
    processed_data = []
    for ticker in df['tic'].unique():
        ticker_data = df[df['tic'] == ticker].copy().reset_index(drop=True)
        
        # Technical indicators (vectorized for speed)
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
    
    print(f"  ✓ Processed {len(df)} rows for {len(df['tic'].unique())} stocks")
    return df

def train_gpu_ensemble(train_data):
    """Train ensemble with GPU acceleration"""
    print(f"\n[STEP 3] GPU-ACCELERATED TRAINING")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Training samples: {len(train_data)}")
    print(f"Stocks: {train_data['tic'].nunique()}")
    
    if use_gpu:
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB available")
    
    print("=" * 50)
    
    # Create environment
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
        batch_size_base = 64
        buffer_size = 200000
        print(f"💻 CPU MODE")
        print(f"   Training timesteps: {training_timesteps:,} per model")
    
    start_time = datetime.now()
    
    # 1. Train A2C with GPU acceleration
    print(f"\n[1/3] Training A2C...")
    print(f"      Algorithm: Actor-Critic")
    print(f"      Device: {device}")
    print(f"      Timesteps: {training_timesteps:,}")
    
    a2c_model = A2C(
        'MlpPolicy',
        env,
        verbose=1,
        seed=42,
        device=device,
        n_steps=5,
        learning_rate=0.0007,
        ent_coef=0.01,
        vf_coef=0.25
    )
    
    print("      Status: Training in progress...")
    a2c_model.learn(total_timesteps=training_timesteps, progress_bar=True)
    a2c_model.save("trained_models/gpu_a2c")
    models['A2C'] = a2c_model
    
    if use_gpu:
        print(f"      GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
        torch.cuda.empty_cache()
    
    print("      ✅ A2C training completed")
    
    # 2. Train PPO with GPU acceleration  
    print(f"\n[2/3] Training PPO...")
    print(f"      Algorithm: Proximal Policy Optimization")
    print(f"      Device: {device}")
    print(f"      Timesteps: {training_timesteps:,}")
    
    ppo_model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        seed=42,
        device=device,
        batch_size=batch_size_base,
        n_epochs=10,
        learning_rate=0.0003,
        ent_coef=0.01,
        clip_range=0.2
    )
    
    print("      Status: Training in progress...")
    ppo_model.learn(total_timesteps=training_timesteps, progress_bar=True)
    ppo_model.save("trained_models/gpu_ppo")
    models['PPO'] = ppo_model
    
    if use_gpu:
        print(f"      GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
        torch.cuda.empty_cache()
    
    print("      ✅ PPO training completed")
    
    # 3. Train DDPG with GPU acceleration
    print(f"\n[3/3] Training DDPG...")
    print(f"      Algorithm: Deep Deterministic Policy Gradient")
    print(f"      Device: {device}")
    print(f"      Timesteps: {training_timesteps:,}")
    
    ddpg_model = DDPG(
        'MlpPolicy',
        env,
        verbose=1,
        seed=42,
        device=device,
        batch_size=batch_size_base,
        buffer_size=buffer_size,
        learning_rate=0.001,
        tau=0.005,
        gamma=0.99
    )
    
    print("      Status: Training in progress...")
    ddpg_model.learn(total_timesteps=training_timesteps, progress_bar=True)
    ddpg_model.save("trained_models/gpu_ddpg")
    models['DDPG'] = ddpg_model
    
    # Training completed
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds() / 60
    
    print(f"\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"⏱️  Total training time: {training_time:.1f} minutes")
    print(f"📊 Total timesteps: {training_timesteps * 3:,}")
    
    if use_gpu:
        torch.cuda.synchronize()
        print(f"🔥 GPU Performance Summary:")
        print(f"   Peak GPU memory: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        print(f"   Current allocation: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"   GPU utilization: Optimized for {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    return models

def comprehensive_backtest(models, test_data):
    """Advanced backtesting with performance analysis"""
    print(f"\n[STEP 4] COMPREHENSIVE BACKTESTING")
    print("=" * 50)
    
    env = GPUOptimizedTradingEnv(test_data)
    results = {}
    
    print(f"Test period: {len(test_data)} rows ({test_data['tic'].nunique()} stocks)")
    
    # Individual model testing
    for name, model in models.items():
        print(f"\n🧪 Testing {name}...")
        
        obs, _ = env.reset()
        portfolio_values = [INITIAL_CAPITAL]
        daily_actions = []
        done = False
        
        step_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            daily_actions.append(action.copy())
            
            obs, reward, done, truncated, info = env.step(action)
            portfolio_value = info.get('portfolio_value', portfolio_values[-1])
            portfolio_values.append(portfolio_value)
            step_count += 1
        
        results[name] = {
            'portfolio_values': portfolio_values,
            'actions': daily_actions,
            'steps': step_count
        }
        
        final_value = portfolio_values[-1]
        total_return = (final_value / INITIAL_CAPITAL - 1) * 100
        print(f"    Final Value: ${final_value:,.0f}")
        print(f"    Total Return: {total_return:+.2f}%")
        print(f"    Trading Days: {len(portfolio_values)-1}")
    
    # Ensemble Strategy
    print(f"\n🤝 Testing ENSEMBLE STRATEGY...")
    
    obs, _ = env.reset()
    ensemble_portfolio = [INITIAL_CAPITAL]
    ensemble_actions = []
    done = False
    
    while not done:
        # Get predictions from all models
        individual_actions = []
        for model in models.values():
            action, _ = model.predict(obs, deterministic=True)
            individual_actions.append(action)
        
        # Ensemble method: weighted average
        # You can experiment with different weighting schemes
        weights = [1.0, 1.0, 1.0]  # Equal weights for A2C, PPO, DDPG
        weighted_actions = []
        
        for i, action in enumerate(individual_actions):
            weighted_actions.append(action * weights[i])
        
        ensemble_action = np.mean(weighted_actions, axis=0)
        ensemble_actions.append(ensemble_action.copy())
        
        obs, reward, done, truncated, info = env.step(ensemble_action)
        portfolio_value = info.get('portfolio_value', ensemble_portfolio[-1])
        ensemble_portfolio.append(portfolio_value)
    
    results['Ensemble'] = {
        'portfolio_values': ensemble_portfolio,
        'actions': ensemble_actions,
        'steps': len(ensemble_portfolio) - 1
    }
    
    final_value = ensemble_portfolio[-1]
    total_return = (final_value / INITIAL_CAPITAL - 1) * 100
    print(f"    🏆 Final Value: ${final_value:,.0f}")
    print(f"    🚀 Total Return: {total_return:+.2f}%")
    print(f"    📅 Trading Days: {len(ensemble_portfolio)-1}")
    
    return results

def advanced_performance_analysis(results):
    """Advanced performance metrics and analysis"""
    print(f"\n[STEP 5] ADVANCED PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    performance_metrics = []
    
    for name, data in results.items():
        portfolio_values = data['portfolio_values']
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # Core metrics
        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        trading_days = len(portfolio_values) - 1
        annual_return = ((portfolio_values[-1] / portfolio_values[0]) ** (252 / trading_days) - 1) * 100
        
        # Risk metrics
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252) * 100
        sharpe_ratio = (returns.mean() * 252) / (daily_vol * np.sqrt(252)) if daily_vol > 0 else 0
        
        # Drawdown analysis
        cumulative = pd.Series(portfolio_values)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Advanced metrics
        win_rate = (returns > 0).mean() * 100
        profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if (returns < 0).sum() != 0 else np.inf
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return / 100) / (downside_deviation) if downside_deviation > 0 else 0
        
        # Calmar ratio (return/max drawdown)
        calmar_ratio = (annual_return / 100) / abs(max_drawdown / 100) if max_drawdown != 0 else 0
        
        metrics = {
            'Strategy': name,
            'Final_Value': portfolio_values[-1],
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Volatility': annual_vol,
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Calmar_Ratio': calmar_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Profit_Factor': profit_factor,
            'Trading_Days': trading_days
        }
        
        performance_metrics.append(metrics)
        
        # Print detailed metrics
        print(f"\n{name.upper()} PERFORMANCE:")
        print(f"  💰 Final Value:      ${metrics['Final_Value']:,.0f}")
        print(f"  📈 Total Return:     {metrics['Total_Return']:7.2f}%")
        print(f"  📊 Annual Return:    {metrics['Annual_Return']:7.2f}%")
        print(f"  📉 Volatility:       {metrics['Volatility']:7.2f}%")
        print(f"  ⚡ Sharpe Ratio:     {metrics['Sharpe_Ratio']:7.2f}")
        print(f"  🎯 Sortino Ratio:    {metrics['Sortino_Ratio']:7.2f}")
        print(f"  🏆 Calmar Ratio:     {metrics['Calmar_Ratio']:7.2f}")
        print(f"  🔻 Max Drawdown:     {metrics['Max_Drawdown']:7.2f}%")
        print(f"  ✅ Win Rate:         {metrics['Win_Rate']:7.2f}%")
        print(f"  💎 Profit Factor:    {metrics['Profit_Factor']:7.2f}")
        print(f"  📅 Trading Days:     {metrics['Trading_Days']}")
    
    # Save detailed metrics
    df_metrics = pd.DataFrame(performance_metrics)
    df_metrics.to_csv('results/gpu_performance_detailed.csv', index=False)
    print(f"\n💾 Detailed metrics saved: results/gpu_performance_detailed.csv")
    
    return performance_metrics

def create_advanced_visualizations(results):
    """Create comprehensive performance visualizations"""
    print(f"\n[STEP 6] Creating advanced visualizations...")
    
    # Create comprehensive dashboard
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Main Performance Chart
    plt.subplot(3, 4, 1)
    for name, data in results.items():
        style = '--' if name == 'Ensemble' else '-'
        width = 4 if name == 'Ensemble' else 2
        alpha = 1.0 if name == 'Ensemble' else 0.8
        plt.plot(data['portfolio_values'], label=name, linestyle=style, linewidth=width, alpha=alpha)
    
    plt.title('Portfolio Performance Comparison', fontsize=14, weight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style='plain', axis='y')
    
    # 2. Returns Distribution
    plt.subplot(3, 4, 2)
    for name, data in results.items():
        returns = pd.Series(data['portfolio_values']).pct_change().dropna()
        if len(returns) > 0:
            plt.hist(returns, alpha=0.6, label=name, bins=50, density=True)
    
    plt.title('Daily Returns Distribution', fontsize=14, weight='bold')
    plt.xlabel('Daily Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Drawdown Analysis
    plt.subplot(3, 4, 3)
    for name, data in results.items():
        portfolio = pd.Series(data['portfolio_values'])
        peak = portfolio.expanding().max()
        drawdown = (portfolio - peak) / peak * 100
        plt.plot(drawdown, label=name, alpha=0.8)
    
    plt.title('Drawdown Analysis', fontsize=14, weight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Rolling Sharpe Ratio
    plt.subplot(3, 4, 4)
    window = 30
    for name, data in results.items():
        returns = pd.Series(data['portfolio_values']).pct_change().dropna()
        if len(returns) > window:
            rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
            plt.plot(rolling_sharpe, label=name, alpha=0.8)
    
    plt.title(f'Rolling {window}-Day Sharpe Ratio', fontsize=14, weight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Cumulative Returns
    plt.subplot(3, 4, 5)
    for name, data in results.items():
        returns = pd.Series(data['portfolio_values']).pct_change().dropna()
        if len(returns) > 0:
            cumulative_returns = (1 + returns).cumprod() - 1
            plt.plot(cumulative_returns * 100, label=name, alpha=0.8)
    
    plt.title('Cumulative Returns', fontsize=14, weight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Risk-Return Scatter
    plt.subplot(3, 4, 6)
    for name, data in results.items():
        returns = pd.Series(data['portfolio_values']).pct_change().dropna()
        if len(returns) > 0:
            annual_return = returns.mean() * 252 * 100
            volatility = returns.std() * np.sqrt(252) * 100
            size = 200 if name == 'Ensemble' else 100
            plt.scatter(volatility, annual_return, s=size, label=name, alpha=0.7)
            plt.annotate(name, (volatility, annual_return), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10)
    
    plt.title('Risk-Return Analysis', fontsize=14, weight='bold')
    plt.xlabel('Volatility (% annually)')
    plt.ylabel('Return (% annually)')
    plt.grid(True, alpha=0.3)
    
    # 7. Final Returns Bar Chart
    plt.subplot(3, 4, 7)
    names = list(results.keys())
    final_returns = [(results[name]['portfolio_values'][-1] / INITIAL_CAPITAL - 1) * 100 for name in names]
    
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']
    bars = plt.bar(names, final_returns, color=colors[:len(names)])
    plt.title('Total Returns Comparison', fontsize=14, weight='bold')
    plt.ylabel('Total Return (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, return_val in zip(bars, final_returns):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{return_val:.1f}%', ha='center', fontsize=10, weight='bold')
    
    # 8. Portfolio Allocation (Ensemble)
    plt.subplot(3, 4, 8)
    if 'Ensemble' in results and results['Ensemble']['actions']:
        actions = np.array(results['Ensemble']['actions'])
        if len(actions) > 0:
            avg_weights = np.mean(np.abs(actions), axis=0)
            stock_names = SELECTED_TICKERS[:len(avg_weights)]
            
            plt.pie(avg_weights, labels=stock_names, autopct='%1.1f%%', startangle=90)
            plt.title('Average Portfolio Allocation\n(Ensemble Strategy)', fontsize=12, weight='bold')
    
    # 9-12. Additional analysis plots
    # Monthly returns heatmap, volatility analysis, etc.
    for i in range(9, 13):
        plt.subplot(3, 4, i)
        # Placeholder for additional analysis
        plt.text(0.5, 0.5, f'Advanced\nAnalysis\n#{i-8}', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=12)
        plt.title(f'Advanced Analysis #{i-8}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/gpu_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("  🎨 Comprehensive dashboard: results/gpu_comprehensive_analysis.png")
    
    # Create focused ensemble performance chart
    plt.figure(figsize=(14, 8))
    
    for name, data in results.items():
        style = '--' if name == 'Ensemble' else '-'
        width = 4 if name == 'Ensemble' else 2
        alpha = 1.0 if name == 'Ensemble' else 0.7
        plt.plot(data['portfolio_values'], label=name, linestyle=style, linewidth=width, alpha=alpha)
    
    plt.title('GPU-Accelerated FinRL Ensemble Strategy - Performance Results', fontsize=16, weight='bold')
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add performance annotations
    if 'Ensemble' in results:
        final_value = results['Ensemble']['portfolio_values'][-1]
        total_return = (final_value / INITIAL_CAPITAL - 1) * 100
        
        plt.text(0.02, 0.98, 
                f'🏆 Ensemble Performance:\n💰 Final: ${final_value:,.0f}\n📈 Return: {total_return:+.1f}%\n🔥 GPU Accelerated', 
                transform=plt.gca().transAxes, fontsize=12, weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('results/gpu_ensemble_performance.png', dpi=300, bbox_inches='tight')
    print("  🚀 Focused performance chart: results/gpu_ensemble_performance.png")

def main():
    """Main execution with GPU optimization"""
    print("\n[GPU] FINRL GPU-ACCELERATED ENSEMBLE STRATEGY")
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
        
        print(f"\n📊 DATA SUMMARY:")
        print(f"  🏋️  Training samples: {len(train_data):,}")
        print(f"  🧪 Testing samples:  {len(test_data):,}")
        print(f"  📈 Stocks analyzed:  {train_data['tic'].nunique()}")
        print(f"  📅 Date range:       {df['date'].min()} to {df['date'].max()}")
        
        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("Insufficient data for analysis")
        
        # Train models with GPU acceleration
        models = train_gpu_ensemble(train_data)
        
        # Comprehensive backtesting
        results = comprehensive_backtest(models, test_data)
        
        # Advanced performance analysis
        metrics = advanced_performance_analysis(results)
        
        # Create visualizations
        create_advanced_visualizations(results)
        
        # Final summary
        overall_end = datetime.now()
        total_runtime = (overall_end - overall_start).total_seconds() / 60
        
        print(f"\n{'🎉' * 25}")
        print("GPU-ACCELERATED ENSEMBLE STRATEGY COMPLETED!")
        print(f"{'🎉' * 25}")
        
        if 'Ensemble' in results:
            final_value = results['Ensemble']['portfolio_values'][-1]
            total_return = (final_value / INITIAL_CAPITAL - 1) * 100
            trading_days = len(results['Ensemble']['portfolio_values']) - 1
            daily_avg = total_return / trading_days
            
            print(f"\n🏆 ENSEMBLE STRATEGY RESULTS:")
            print(f"  💰 Initial Capital:    ${INITIAL_CAPITAL:,}")
            print(f"  💎 Final Value:        ${final_value:,.0f}")
            print(f"  📈 Total Return:       {total_return:+.2f}%")
            print(f"  📅 Trading Period:     {trading_days} days")
            print(f"  ⚡ Daily Average:      {daily_avg:+.3f}%")
            print(f"  🕐 Total Runtime:      {total_runtime:.1f} minutes")
            
            if use_gpu:
                print(f"  🔥 GPU Utilization:    {torch.cuda.get_device_name(0)}")
                print(f"  💾 Peak GPU Memory:    {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        
        print(f"\n📁 GENERATED FILES:")
        print("  • results/gpu_comprehensive_analysis.png")
        print("  • results/gpu_ensemble_performance.png")
        print("  • results/gpu_performance_detailed.csv")
        print("  • trained_models/ (GPU-optimized models)")
        
        print(f"\n🚀 NEXT STEPS FOR PRODUCTION:")
        print("  1. Scale to full market universe (S&P 500, Russell 2000)")
        print("  2. Implement real-time data feeds")
        print("  3. Add advanced risk management")
        print("  4. Connect to live trading APIs")
        print("  5. Deploy to cloud infrastructure")
        print("  6. Implement automated rebalancing")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)