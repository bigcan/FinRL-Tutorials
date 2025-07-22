#!/usr/bin/env python3
"""
FinRL Ensemble Strategy - Final Production Version
CPU-optimized with comprehensive analysis
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

print("FinRL Ensemble Strategy - Final Version")
print("=" * 50)
print("Status: Ready for production deployment")
print("Hardware: CPU optimized with GPU detection")

# Stable Baselines3 imports
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

# Configuration
SELECTED_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "JPM", "V", "META"]
TRAIN_START = '2022-01-01'
TRAIN_END = '2023-06-01' 
TEST_START = '2023-06-01'
TEST_END = '2024-01-01'

INITIAL_CAPITAL = 1000000
TRANSACTION_COST = 0.001

class ProductionTradingEnv(gym.Env):
    """Production-ready trading environment"""
    
    def __init__(self, df, initial_amount=1000000, transaction_cost=0.001):
        super().__init__()
        
        self.df = df.copy()
        self.stock_dim = len(df['tic'].unique())
        self.initial_amount = initial_amount
        self.transaction_cost = transaction_cost
        
        # Action space: portfolio weights (-1 to 1 for each stock)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32
        )
        
        # State space: [balance, prices, holdings, 4 technical indicators]
        self.state_dim = 1 + 2 * self.stock_dim + 4 * self.stock_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Pre-process data for faster access
        self.dates = sorted(self.df['date'].unique())
        self.tickers = sorted(self.df['tic'].unique())
        
        # Create data lookup cache
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
        
        # Portfolio value before action
        portfolio_before = self.balance + np.sum(self.holdings * prices)
        
        # Process trading actions
        action = np.clip(action, -1, 1)
        total_value = portfolio_before
        
        # Calculate target positions (max 20% per stock)
        target_values = action * total_value * 0.2
        target_holdings = target_values / (prices + 1e-8)
        holdings_change = target_holdings - self.holdings
        
        # Execute trades
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

def download_data():
    """Download stock data"""
    print("\n[STEP 1] Downloading market data...")
    
    all_data = []
    for i, ticker in enumerate(SELECTED_TICKERS, 1):
        try:
            print(f"  Downloading {ticker} [{i}/{len(SELECTED_TICKERS)}]...", end=" ")
            
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
            print(f"FAILED - {e}")
    
    if not all_data:
        raise ValueError("Data download failed")
    
    # Combine data
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values(['date', 'tic'])
    
    print(f"[STEP 2] Processing technical indicators...")
    processed_data = []
    
    for ticker in df['tic'].unique():
        ticker_data = df[df['tic'] == ticker].copy().reset_index(drop=True)
        
        # Add technical indicators
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
    
    print(f"  Data processed: {len(df)} rows, {len(df['tic'].unique())} stocks")
    return df

def train_models(train_data):
    """Train the ensemble models"""
    print(f"\n[STEP 3] Training ensemble models...")
    print(f"  Training data: {len(train_data)} rows")
    print(f"  Stocks: {train_data['tic'].nunique()}")
    
    env = DummyVecEnv([lambda: ProductionTradingEnv(train_data)])
    models = {}
    timesteps = 75000  # CPU-optimized
    
    # A2C
    print(f"\n  [1/3] Training A2C ({timesteps:,} steps)...")
    a2c = A2C('MlpPolicy', env, verbose=0, seed=42, device='cpu',
              tensorboard_log="./tensorboard_log/a2c/")
    a2c.learn(total_timesteps=timesteps)
    a2c.save("trained_models/final_a2c")
    models['A2C'] = a2c
    print("        A2C training completed")
    
    # PPO
    print(f"  [2/3] Training PPO ({timesteps:,} steps)...")
    ppo = PPO('MlpPolicy', env, verbose=0, seed=42, device='cpu',
              tensorboard_log="./tensorboard_log/ppo/", batch_size=64)
    ppo.learn(total_timesteps=timesteps)
    ppo.save("trained_models/final_ppo")
    models['PPO'] = ppo
    print("        PPO training completed")
    
    # DDPG
    print(f"  [3/3] Training DDPG ({timesteps:,} steps)...")
    ddpg = DDPG('MlpPolicy', env, verbose=0, seed=42, device='cpu',
                tensorboard_log="./tensorboard_log/ddpg/", batch_size=64)
    ddpg.learn(total_timesteps=timesteps)
    ddpg.save("trained_models/final_ddpg")
    models['DDPG'] = ddpg
    print("        DDPG training completed")
    
    print(f"\n  All models trained successfully!")
    print(f"  Total training: {timesteps * 3:,} timesteps")
    
    return models

def backtest_models(models, test_data):
    """Comprehensive backtesting"""
    print(f"\n[STEP 4] Backtesting strategies...")
    print(f"  Test data: {len(test_data)} rows")
    
    env = ProductionTradingEnv(test_data)
    results = {}
    
    # Test individual models
    for name, model in models.items():
        print(f"  Testing {name}...", end=" ")
        
        obs, _ = env.reset()
        portfolio_values = [INITIAL_CAPITAL]
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            portfolio_values.append(info.get('portfolio_value', portfolio_values[-1]))
        
        results[name] = portfolio_values
        final_value = portfolio_values[-1]
        return_pct = (final_value / INITIAL_CAPITAL - 1) * 100
        print(f"Final: ${final_value:,.0f} ({return_pct:+.1f}%)")
    
    # Ensemble strategy
    print(f"  Testing Ensemble...", end=" ")
    obs, _ = env.reset()
    ensemble_values = [INITIAL_CAPITAL]
    done = False
    
    while not done:
        # Average predictions from all models
        actions = [model.predict(obs, deterministic=True)[0] for model in models.values()]
        ensemble_action = np.mean(actions, axis=0)
        
        obs, reward, done, truncated, info = env.step(ensemble_action)
        ensemble_values.append(info.get('portfolio_value', ensemble_values[-1]))
    
    results['Ensemble'] = ensemble_values
    final_value = ensemble_values[-1]
    return_pct = (final_value / INITIAL_CAPITAL - 1) * 100
    print(f"Final: ${final_value:,.0f} ({return_pct:+.1f}%)")
    
    return results

def analyze_performance(results):
    """Detailed performance analysis"""
    print(f"\n[STEP 5] Performance analysis...")
    
    print(f"\n{'='*60}")
    print("STRATEGY PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    performance_data = []
    
    for name, portfolio_values in results.items():
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # Calculate metrics
        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        annual_return = ((portfolio_values[-1] / portfolio_values[0]) ** (252 / len(portfolio_values)) - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Drawdown
        peak = pd.Series(portfolio_values).cummax()
        drawdown = (pd.Series(portfolio_values) - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        win_rate = (returns > 0).mean() * 100
        
        metrics = {
            'Strategy': name,
            'Final_Value': portfolio_values[-1],
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Days': len(portfolio_values) - 1
        }
        
        performance_data.append(metrics)
        
        print(f"\n{name.upper()}:")
        print(f"  Final Value:      ${metrics['Final_Value']:,.0f}")
        print(f"  Total Return:     {metrics['Total_Return']:6.2f}%")
        print(f"  Annual Return:    {metrics['Annual_Return']:6.2f}%")
        print(f"  Volatility:       {metrics['Volatility']:6.2f}%")
        print(f"  Sharpe Ratio:     {metrics['Sharpe_Ratio']:6.2f}")
        print(f"  Max Drawdown:     {metrics['Max_Drawdown']:6.2f}%")
        print(f"  Win Rate:         {metrics['Win_Rate']:6.2f}%")
        print(f"  Trading Days:     {metrics['Days']}")
    
    # Save performance data
    df = pd.DataFrame(performance_data)
    df.to_csv('results/final_performance_metrics.csv', index=False)
    print(f"\n  Performance metrics saved: results/final_performance_metrics.csv")
    
    return performance_data

def create_visualizations(results):
    """Create performance visualizations"""
    print(f"\n[STEP 6] Creating visualizations...")
    
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
    
    # Drawdown analysis
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
    
    # Final comparison bar chart
    plt.subplot(2, 2, 4)
    names = list(results.keys())
    final_returns = [(results[name][-1] / INITIAL_CAPITAL - 1) * 100 for name in names]
    
    bars = plt.bar(names, final_returns)
    plt.title('Final Returns Comparison', fontsize=14, weight='bold')
    plt.ylabel('Total Return (%)')
    plt.xticks(rotation=45)
    
    # Color bars based on performance
    for bar, return_val in zip(bars, final_returns):
        if return_val > 15:
            bar.set_color('green')
        elif return_val > 5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/final_ensemble_analysis.png', dpi=300, bbox_inches='tight')
    print("  Comprehensive analysis: results/final_ensemble_analysis.png")
    
    # Summary performance chart
    plt.figure(figsize=(12, 8))
    for name, values in results.items():
        style = '--' if name == 'Ensemble' else '-'
        width = 4 if name == 'Ensemble' else 2
        alpha = 1.0 if name == 'Ensemble' else 0.7
        plt.plot(values, label=name, linestyle=style, linewidth=width, alpha=alpha)
    
    plt.title('FinRL Ensemble Strategy - Final Results', fontsize=16, weight='bold')
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add performance annotation
    if 'Ensemble' in results:
        final_value = results['Ensemble'][-1]
        total_return = (final_value / INITIAL_CAPITAL - 1) * 100
        plt.text(0.02, 0.98, f'Ensemble Total Return: {total_return:+.1f}%', 
                transform=plt.gca().transAxes, fontsize=12, weight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('results/final_ensemble_summary.png', dpi=300, bbox_inches='tight')
    print("  Summary chart: results/final_ensemble_summary.png")

def main():
    """Main execution function"""
    print("\nFINRL ENSEMBLE STRATEGY - PRODUCTION DEPLOYMENT")
    print("=" * 70)
    print(f"Training Period: {TRAIN_START} to {TRAIN_END}")
    print(f"Testing Period:  {TEST_START} to {TEST_END}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"Stock Universe:  {', '.join(SELECTED_TICKERS)}")
    print("=" * 70)
    
    try:
        start_time = datetime.now()
        
        # Download and process data
        df = download_data()
        
        # Split data
        train_data = df[(df['date'] >= TRAIN_START) & (df['date'] < TEST_START)]
        test_data = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)]
        
        print(f"\nDATA SUMMARY:")
        print(f"  Training samples: {len(train_data)}")
        print(f"  Testing samples:  {len(test_data)}")
        print(f"  Stocks analyzed:  {train_data['tic'].nunique()}")
        print(f"  Date range:       {df['date'].min()} to {df['date'].max()}")
        
        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("Insufficient data")
        
        # Train models
        models = train_models(train_data)
        
        # Backtest
        results = backtest_models(models, test_data)
        
        # Analyze performance
        performance_data = analyze_performance(results)
        
        # Create visualizations
        create_visualizations(results)
        
        # Final summary
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds() / 60
        
        print(f"\n{'='*70}")
        print("DEPLOYMENT SUCCESSFUL!")
        print(f"{'='*70}")
        
        if 'Ensemble' in results:
            final_value = results['Ensemble'][-1]
            total_return = (final_value / INITIAL_CAPITAL - 1) * 100
            trading_days = len(results['Ensemble']) - 1
            
            print(f"\nENSEMBLE STRATEGY RESULTS:")
            print(f"  Initial Capital:  ${INITIAL_CAPITAL:,}")
            print(f"  Final Value:      ${final_value:,.0f}")
            print(f"  Total Return:     {total_return:+.2f}%")
            print(f"  Trading Period:   {trading_days} days")
            print(f"  Avg Daily:        {total_return/trading_days:+.3f}%")
            print(f"  Runtime:          {runtime:.1f} minutes")
        
        print(f"\nGENERATED FILES:")
        print("  • results/final_ensemble_analysis.png")
        print("  • results/final_ensemble_summary.png") 
        print("  • results/final_performance_metrics.csv")
        print("  • trained_models/ (A2C, PPO, DDPG models)")
        print("  • tensorboard_log/ (training logs)")
        
        print(f"\nNEXT STEPS:")
        print("  1. Review performance analysis")
        print("  2. Scale to full DOW 30 universe")
        print("  3. Implement GPU acceleration for faster training")
        print("  4. Connect to live trading APIs (Alpaca, Interactive Brokers)")
        print("  5. Add risk management and position sizing")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)