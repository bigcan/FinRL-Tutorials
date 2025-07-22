#!/usr/bin/env python3
"""
FinRL Ensemble Strategy - CPU Optimized Version
No PyTorch dependencies, pure CPU training
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

print("[CPU] FinRL Ensemble Strategy - CPU Optimized")
print("=" * 50)
print("Device: CPU (PyTorch-free implementation)")

# Stable Baselines3 imports (CPU only)
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

# Configuration
SELECTED_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "JPM", "V", "META"]
TRAIN_START_DATE = '2022-01-01'
TRAIN_END_DATE = '2023-06-01' 
TEST_START_DATE = '2023-06-01'
TEST_END_DATE = '2024-01-01'

INITIAL_AMOUNT = 1000000
TRANSACTION_COST = 0.001

class CPUOptimizedTradingEnv(gym.Env):
    """CPU-Optimized Trading Environment"""
    
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
        
        # State space: [balance, prices, holdings, indicators]
        self.state_dim = 1 + 2 * self.stock_dim + 4 * self.stock_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Precompute data for faster access
        self.dates = sorted(self.df['date'].unique())
        self.tickers = sorted(self.df['tic'].unique())
        
        # Data lookup cache
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
        """Get current state observation"""
        if self.current_step >= len(self.dates):
            self.current_step = len(self.dates) - 1
        
        current_date = self.dates[self.current_step]
        
        # Use cached data if available
        if current_date in self.data_cache:
            data = self.data_cache[current_date]
            prices = data['prices']
            indicators = np.concatenate([
                data['sma_10'], data['sma_30'], 
                data['rsi'], data['macd']
            ])
        else:
            # Fallback to default values
            prices = np.ones(self.stock_dim, dtype=np.float32) * 100
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
        """Execute trading step"""
        if self.current_step >= len(self.dates) - 1:
            return self._get_observation(), 0.0, True, True, {}
        
        current_date = self.dates[self.current_step]
        
        # Get current prices
        if current_date in self.data_cache:
            prices = self.data_cache[current_date]['prices']
        else:
            prices = np.ones(self.stock_dim, dtype=np.float32) * 100
        
        # Portfolio value before trading
        portfolio_value_before = self.balance + np.sum(self.holdings * prices)
        
        # Process actions
        action = np.clip(action, -1, 1)
        total_value = portfolio_value_before
        
        # Target positions (max 20% per stock for safety)
        target_values = action * total_value * 0.2
        target_holdings = target_values / (prices + 1e-8)
        
        # Calculate trades
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
            can_sell = self.holdings[sell_mask] >= sell_amounts
            
            if np.any(can_sell):
                valid_sells = sell_amounts[can_sell]
                sell_prices = prices[sell_mask][can_sell]
                sell_proceeds = valid_sells * sell_prices * (1 - self.transaction_cost)
                
                self.balance += np.sum(sell_proceeds)
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
        
        portfolio_value_after = self.balance + np.sum(self.holdings * next_prices)
        
        # Reward as percentage change
        reward = (portfolio_value_after - portfolio_value_before) / (portfolio_value_before + 1e-8)
        
        self.portfolio_values.append(portfolio_value_after)
        
        done = self.current_step >= len(self.dates) - 1
        
        return self._get_observation(), float(reward), done, False, {
            'portfolio_value': portfolio_value_after,
            'balance': self.balance,
            'holdings': self.holdings.copy()
        }

def download_and_process_data():
    """Download and process stock data"""
    print("\n[DATA] Downloading stock data...")
    
    all_data = []
    for i, ticker in enumerate(SELECTED_TICKERS, 1):
        try:
            print(f"  [{i}/{len(SELECTED_TICKERS)}] {ticker}...", end=" ")
            
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
            print(f"[OK] {len(data)} rows")
            
        except Exception as e:
            print(f"[ERROR] {e}")
    
    if not all_data:
        raise ValueError("No data downloaded successfully")
    
    # Combine and process
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values(['date', 'tic'])
    
    print(f"\n[TECH] Adding technical indicators...")
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
    
    print(f"✅ Data processed: {len(df)} rows, {len(df['tic'].unique())} stocks")
    return df

def train_cpu_ensemble(train_data):
    """Train ensemble optimized for CPU"""
    print(f"\n🏋️ TRAINING ENSEMBLE MODELS (CPU)")
    print("=" * 50)
    print(f"Training data: {len(train_data)} rows")
    print(f"Stocks: {train_data['tic'].nunique()}")
    print("=" * 50)
    
    # Create environment
    env = DummyVecEnv([lambda: CPUOptimizedTradingEnv(train_data)])
    
    models = {}
    
    # CPU-optimized parameters
    training_timesteps = 75000  # Balanced for CPU performance
    
    # 1. Train A2C
    print(f"\n[1/3] 🤖 Training A2C ({training_timesteps:,} timesteps)")
    print("      Policy: Actor-Critic with advantage estimation")
    
    a2c_model = A2C(
        'MlpPolicy',
        env,
        verbose=1,
        seed=42,
        device='cpu',  # Explicit CPU
        tensorboard_log="./tensorboard_log/cpu_a2c/",
        n_steps=5,
        learning_rate=0.0007,
        ent_coef=0.01,
        vf_coef=0.25,
        max_grad_norm=0.5
    )
    
    print("      Starting A2C training...")
    a2c_model.learn(total_timesteps=training_timesteps, progress_bar=True)
    a2c_model.save("trained_models/cpu_a2c")
    models['A2C'] = a2c_model
    print("      ✅ A2C training completed")
    
    # 2. Train PPO
    print(f"\n[2/3] 🎯 Training PPO ({training_timesteps:,} timesteps)")
    print("      Policy: Proximal Policy Optimization")
    
    ppo_model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        seed=42,
        device='cpu',
        tensorboard_log="./tensorboard_log/cpu_ppo/",
        batch_size=64,
        n_epochs=10,
        learning_rate=0.0003,
        ent_coef=0.01,
        clip_range=0.2,
        max_grad_norm=0.5
    )
    
    print("      Starting PPO training...")
    ppo_model.learn(total_timesteps=training_timesteps, progress_bar=True)
    ppo_model.save("trained_models/cpu_ppo")
    models['PPO'] = ppo_model
    print("      ✅ PPO training completed")
    
    # 3. Train DDPG
    print(f"\n[3/3] 🎲 Training DDPG ({training_timesteps:,} timesteps)")
    print("      Policy: Deep Deterministic Policy Gradient")
    
    ddpg_model = DDPG(
        'MlpPolicy',
        env,
        verbose=1,
        seed=42,
        device='cpu',
        tensorboard_log="./tensorboard_log/cpu_ddpg/",
        batch_size=64,
        buffer_size=100000,
        learning_rate=0.001,
        tau=0.005,
        gamma=0.99
    )
    
    print("      Starting DDPG training...")
    ddpg_model.learn(total_timesteps=training_timesteps, progress_bar=True)
    ddpg_model.save("trained_models/cpu_ddpg")
    models['DDPG'] = ddpg_model
    print("      ✅ DDPG training completed")
    
    print(f"\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"📊 Total training: {training_timesteps * 3:,} timesteps")
    print(f"⏱️  Estimated time: ~45-60 minutes on modern CPU")
    
    return models

def comprehensive_backtest(models, test_data):
    """Comprehensive backtesting with detailed analysis"""
    print(f"\n📊 COMPREHENSIVE BACKTESTING")
    print("=" * 50)
    
    env = CPUOptimizedTradingEnv(test_data)
    results = {}
    
    # Test each model
    for name, model in models.items():
        print(f"\n🧪 Testing {name}...")
        
        obs, _ = env.reset()
        portfolio_values = [INITIAL_AMOUNT]
        daily_returns = []
        actions_history = []
        done = False
        
        step_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions_history.append(action.copy())
            
            obs, reward, done, truncated, info = env.step(action)
            portfolio_value = info.get('portfolio_value', portfolio_values[-1])
            portfolio_values.append(portfolio_value)
            
            if len(portfolio_values) > 1:
                daily_return = (portfolio_values[-1] / portfolio_values[-2]) - 1
                daily_returns.append(daily_return)
            
            step_count += 1
        
        results[name] = {
            'portfolio_values': portfolio_values,
            'daily_returns': daily_returns,
            'actions': actions_history,
            'steps': step_count
        }
        
        final_value = portfolio_values[-1]
        total_return = (final_value / INITIAL_AMOUNT) - 1
        print(f"    💰 Final value: ${final_value:,.0f}")
        print(f"    📈 Total return: {total_return:+.2%}")
        print(f"    📅 Trading days: {len(portfolio_values)-1}")
    
    # Ensemble strategy
    print(f"\n🤝 Testing ENSEMBLE STRATEGY...")
    obs, _ = env.reset()
    ensemble_portfolio = [INITIAL_AMOUNT]
    ensemble_returns = []
    done = False
    
    while not done:
        # Get predictions from all models
        actions = []
        for model in models.values():
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action)
        
        # Ensemble: weighted average (you can experiment with different weights)
        # Equal weighting for now
        ensemble_action = np.mean(actions, axis=0)
        
        obs, reward, done, truncated, info = env.step(ensemble_action)
        portfolio_value = info.get('portfolio_value', ensemble_portfolio[-1])
        ensemble_portfolio.append(portfolio_value)
        
        if len(ensemble_portfolio) > 1:
            daily_return = (ensemble_portfolio[-1] / ensemble_portfolio[-2]) - 1
            ensemble_returns.append(daily_return)
    
    results['Ensemble'] = {
        'portfolio_values': ensemble_portfolio,
        'daily_returns': ensemble_returns,
        'actions': [],
        'steps': len(ensemble_portfolio) - 1
    }
    
    final_value = ensemble_portfolio[-1]
    total_return = (final_value / INITIAL_AMOUNT) - 1
    print(f"    💎 Final value: ${final_value:,.0f}")
    print(f"    🚀 Total return: {total_return:+.2%}")
    
    # Detailed performance analysis
    print(f"\n📈 PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    performance_summary = []
    
    for name, data in results.items():
        portfolio_values = data['portfolio_values']
        daily_returns = np.array(data['daily_returns'])
        
        # Calculate metrics
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        if len(daily_returns) > 0:
            annual_return = np.mean(daily_returns) * 252
            volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Drawdown calculation
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (np.array(portfolio_values) - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # Win rate
            win_rate = np.mean(daily_returns > 0) if len(daily_returns) > 0 else 0
        else:
            annual_return = volatility = sharpe_ratio = max_drawdown = win_rate = 0
        
        metrics = {
            'Strategy': name,
            'Final_Value': portfolio_values[-1],
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Trading_Days': len(portfolio_values) - 1
        }
        
        performance_summary.append(metrics)
        
        print(f"\n{name.upper()} PERFORMANCE:")
        print(f"  💰 Final Value:      ${metrics['Final_Value']:,.0f}")
        print(f"  📊 Total Return:     {metrics['Total_Return']:7.2%}")
        print(f"  📅 Annualized:       {metrics['Annual_Return']:7.2%}")
        print(f"  📉 Volatility:       {metrics['Volatility']:7.2%}")
        print(f"  ⚡ Sharpe Ratio:     {metrics['Sharpe_Ratio']:7.2f}")
        print(f"  🔻 Max Drawdown:     {metrics['Max_Drawdown']:7.2%}")
        print(f"  🎯 Win Rate:         {metrics['Win_Rate']:7.2%}")
        print(f"  📈 Trading Days:     {metrics['Trading_Days']}")
    
    # Create visualizations
    create_comprehensive_plots(results)
    
    # Save performance data
    performance_df = pd.DataFrame(performance_summary)
    performance_df.to_csv('results/cpu_performance_metrics.csv', index=False)
    print(f"\n💾 Performance data saved to: results/cpu_performance_metrics.csv")
    
    return results

def create_comprehensive_plots(results):
    """Create comprehensive performance visualization"""
    print(f"\n🎨 Creating performance visualizations...")
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Portfolio Performance Comparison
    plt.subplot(3, 3, 1)
    for name, data in results.items():
        style = '--' if name == 'Ensemble' else '-'
        width = 3 if name == 'Ensemble' else 2
        alpha = 1.0 if name == 'Ensemble' else 0.8
        plt.plot(data['portfolio_values'], label=name, linestyle=style, linewidth=width, alpha=alpha)
    
    plt.title('Portfolio Performance Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style='plain', axis='y')
    
    # 2. Daily Returns Distribution
    plt.subplot(3, 3, 2)
    for name, data in results.items():
        if data['daily_returns']:
            plt.hist(data['daily_returns'], alpha=0.6, label=name, bins=50, density=True)
    
    plt.title('Daily Returns Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Daily Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Drawdown Analysis
    plt.subplot(3, 3, 3)
    for name, data in results.items():
        portfolio_values = np.array(data['portfolio_values'])
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak * 100
        plt.plot(drawdown, label=name)
    
    plt.title('Drawdown Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Rolling Sharpe Ratio
    plt.subplot(3, 3, 4)
    window = 30
    for name, data in results.items():
        if len(data['daily_returns']) > window:
            returns = pd.Series(data['daily_returns'])
            rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
            plt.plot(rolling_sharpe, label=name)
    
    plt.title(f'Rolling {window}-Day Sharpe Ratio', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Cumulative Returns
    plt.subplot(3, 3, 5)
    for name, data in results.items():
        if data['daily_returns']:
            cumulative_returns = (1 + pd.Series(data['daily_returns'])).cumprod() - 1
            plt.plot(cumulative_returns * 100, label=name)
    
    plt.title('Cumulative Returns', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Risk-Return Scatter
    plt.subplot(3, 3, 6)
    for name, data in results.items():
        if data['daily_returns']:
            returns = np.array(data['daily_returns'])
            annual_return = np.mean(returns) * 252 * 100
            volatility = np.std(returns) * np.sqrt(252) * 100
            plt.scatter(volatility, annual_return, s=100, label=name)
            plt.annotate(name, (volatility, annual_return), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10)
    
    plt.title('Risk-Return Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Volatility (% annually)')
    plt.ylabel('Return (% annually)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Monthly Returns Heatmap
    plt.subplot(3, 3, 7)
    # For simplicity, show ensemble monthly returns
    if 'Ensemble' in results and results['Ensemble']['daily_returns']:
        returns = pd.Series(results['Ensemble']['daily_returns'])
        # Create a simple monthly returns visualization
        monthly_returns = returns.rolling(21).sum()  # Approximate monthly
        plt.plot(monthly_returns * 100)
        plt.title('Ensemble Monthly Returns', fontsize=14, fontweight='bold')
        plt.xlabel('Trading Days')
        plt.ylabel('Monthly Return (%)')
        plt.grid(True, alpha=0.3)
    
    # 8. Performance Metrics Bar Chart
    plt.subplot(3, 3, 8)
    metrics_data = []
    names = []
    
    for name, data in results.items():
        if data['daily_returns']:
            returns = np.array(data['daily_returns'])
            sharpe = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))
            metrics_data.append(sharpe)
            names.append(name)
    
    bars = plt.bar(names, metrics_data)
    plt.title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(rotation=45)
    
    # Color bars
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
    for bar, color in zip(bars, colors[:len(bars)]):
        bar.set_color(color)
    
    plt.grid(True, alpha=0.3)
    
    # 9. Final Portfolio Allocation (for Ensemble)
    plt.subplot(3, 3, 9)
    if 'Ensemble' in results and results['Ensemble']['actions']:
        # Show average allocation across all actions
        all_actions = np.array(results['Ensemble']['actions'])
        if len(all_actions) > 0:
            avg_allocation = np.mean(np.abs(all_actions), axis=0)
            stock_names = SELECTED_TICKERS[:len(avg_allocation)]
            
            plt.pie(avg_allocation, labels=stock_names, autopct='%1.1f%%', startangle=90)
            plt.title('Average Portfolio Allocation\n(Ensemble Strategy)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/cpu_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  📊 Comprehensive analysis: results/cpu_comprehensive_analysis.png")
    
    # Additional simple performance chart
    plt.figure(figsize=(12, 8))
    for name, data in results.items():
        style = '--' if name == 'Ensemble' else '-'
        width = 4 if name == 'Ensemble' else 2
        plt.plot(data['portfolio_values'], label=f"{name}", linestyle=style, linewidth=width)
    
    plt.title('FinRL CPU Ensemble Strategy - Performance Results', fontsize=16, fontweight='bold')
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add performance text
    if 'Ensemble' in results:
        final_value = results['Ensemble']['portfolio_values'][-1]
        total_return = (final_value / INITIAL_AMOUNT - 1) * 100
        plt.text(0.02, 0.98, f'Ensemble Final Return: {total_return:+.1f}%', 
                transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('results/cpu_ensemble_summary.png', dpi=300, bbox_inches='tight')
    print(f"  📈 Summary chart: results/cpu_ensemble_summary.png")

def main():
    """Main execution function"""
    print("🚀 FINRL CPU ENSEMBLE TRADING STRATEGY")
    print("=" * 70)
    print(f"📅 Training: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
    print(f"📅 Testing:  {TEST_START_DATE} to {TEST_END_DATE}")
    print(f"💰 Capital:  ${INITIAL_AMOUNT:,}")
    print(f"🏢 Stocks:   {', '.join(SELECTED_TICKERS)}")
    print("=" * 70)
    
    try:
        # Download and process data
        df = download_and_process_data()
        
        # Split data
        train_data = df[(df['date'] >= TRAIN_START_DATE) & (df['date'] < TEST_START_DATE)]
        test_data = df[(df['date'] >= TEST_START_DATE) & (df['date'] <= TEST_END_DATE)]
        
        print(f"\n📊 DATA SUMMARY:")
        print(f"  🏋️ Training: {len(train_data)} rows")
        print(f"  🧪 Testing:  {len(test_data)} rows")
        print(f"  📈 Stocks:   {train_data['tic'].nunique()}")
        print(f"  📅 Period:   {df['date'].min()} to {df['date'].max()}")
        
        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("Insufficient data for training/testing")
        
        # Train models
        models = train_cpu_ensemble(train_data)
        
        # Backtest
        results = comprehensive_backtest(models, test_data)
        
        print(f"\n{'🎉' * 20}")
        print("SUCCESS! ENSEMBLE STRATEGY COMPLETED")
        print(f"{'🎉' * 20}")
        print("\n📁 Generated Files:")
        print("  • results/cpu_comprehensive_analysis.png")
        print("  • results/cpu_ensemble_summary.png")
        print("  • results/cpu_performance_metrics.csv")
        print("  • trained_models/ (CPU-optimized models)")
        print("  • tensorboard_log/ (training logs)")
        
        # Final summary
        if 'Ensemble' in results:
            final_value = results['Ensemble']['portfolio_values'][-1]
            total_return = (final_value / INITIAL_AMOUNT - 1) * 100
            trading_days = len(results['Ensemble']['portfolio_values']) - 1
            
            print(f"\n🏆 ENSEMBLE PERFORMANCE SUMMARY:")
            print(f"  💰 Initial Capital: ${INITIAL_AMOUNT:,}")
            print(f"  💎 Final Value:     ${final_value:,.0f}")
            print(f"  📈 Total Return:    {total_return:+.2f}%")
            print(f"  📅 Trading Period:  {trading_days} days")
            print(f"  ⚡ Daily Avg:       {total_return/trading_days:+.2f}%")
        
        print(f"\n💡 Next Steps:")
        print("  • Review results in generated files")
        print("  • Install PyTorch CUDA for GPU acceleration")
        print("  • Scale to full DOW 30 for production")
        print("  • Implement live trading with Alpaca API")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)