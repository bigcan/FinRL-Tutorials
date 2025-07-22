# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the FinRL-Tutorials repository, a collection of hands-on tutorials for deep reinforcement learning (DRL) applied to quantitative finance. It demonstrates how to use the FinRL ecosystem for algorithmic trading.

## Directory Structure

### Core Tutorial Directories
- **1-Introduction**: Beginner-friendly notebooks covering stock trading, portfolio allocation, fundamental analysis
- **2-Advance**: Advanced techniques including ensemble strategies, explainable DRL, multi-library comparisons
- **3-Practical**: Real-world applications including paper trading, multi-crypto trading, China A-share market
- **4-Optimization**: Hyperparameter tuning using Optuna, RayTune, and other optimization frameworks
- **5-Others**: Additional examples and experimental notebooks

### Key Components
- **DQN-DDPG_Stock_Trading**: Legacy implementation using OpenAI baselines (now integrated into FinRL)
- **3-Practical/FinRL-Meta**: Complete FinRL-Meta framework implementation with agents, environments, and utilities

## Commands and Development Workflow

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (for FinRL-Meta)
cd 3-Practical/FinRL-Meta
pip install -r requirements.txt
pip install -e .
```

### Running Notebooks
Most tutorials are in Jupyter notebook format (.ipynb). Key notebooks include:
- `Stock_NeurIPS2018_SB3.ipynb`: Stock trading with Stable Baselines3
- `FinRL_PortfolioAllocation_NeurIPS_2020.ipynb`: Portfolio optimization
- `FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb`: Ensemble trading strategies

### Running Python Scripts
Standalone Python versions are available for most notebooks:
```bash
# Example: Portfolio allocation
python 1-Introduction/FinRL_PortfolioAllocation_NeurIPS_2020.py

# Example: Forex trading
python 1-Introduction/ForexTrading_Demo.py
```

### FinRL-Meta Usage
```bash
cd 3-Practical/FinRL-Meta

# Main entry point (creates necessary directories)
python main.py --mode=train

# Training
python train.py

# Trading/Testing
python trade.py

# Plotting results
python plot.py
```

### Testing
```bash
# Run unit tests (from FinRL-Meta)
cd 3-Practical/FinRL-Meta
python -m pytest unit_tests/

# Or use unittest
python -m unittest discover
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit
pre-commit install
```

## Key Libraries and Dependencies

### Core Dependencies
- **stable-baselines3**: Primary DRL library for most tutorials
- **gym**: OpenAI Gym environments
- **pandas, numpy**: Data manipulation
- **matplotlib**: Plotting and visualization
- **yfinance**: Yahoo Finance data fetching
- **alpaca_trade_api**: Alpaca trading API for paper trading
- **ccxt**: Cryptocurrency exchange connectivity

### Optional Dependencies
- **elegantrl**: Alternative DRL library
- **ray[rllib]**: RLlib for distributed training
- **optuna**: Hyperparameter optimization
- **tensorboard**: Training visualization

## Architecture Patterns

### FinRL Pipeline
1. **Data Layer**: Fetching and preprocessing market data
2. **Environment Layer**: Gym-compatible trading environments
3. **Agent Layer**: DRL algorithms (PPO, A2C, DDPG, SAC, TD3)
4. **Training Layer**: Model training with various libraries
5. **Backtesting Layer**: Performance evaluation

### Common Workflow
1. Data download and preprocessing
2. Feature engineering (technical indicators, fundamental data)
3. Environment setup with action/observation spaces
4. Agent training with selected algorithms
5. Backtesting and performance analysis
6. (Optional) Paper trading deployment

## Data Sources
- **Yahoo Finance**: Default for US equities
- **Alpaca**: Real-time data and paper trading
- **Tushare**: China A-share market data
- **CCXT**: Cryptocurrency exchanges
- **WRDS**: Academic financial data (requires subscription)

## Important Configuration Files
- `3-Practical/FinRL-Meta/meta/config.py`: Main configuration for FinRL-Meta
- `3-Practical/FinRL-Meta/meta/config_tickers.py`: Ticker lists (DOW 30, etc.)
- Environment configs in respective tutorial directories

## Common Issues and Solutions
- **GPU Support**: Most tutorials work with CPU; GPU improves training speed
- **Data Download**: Yahoo Finance may have rate limits; implement retry logic
- **Memory Usage**: Large datasets may require data chunking or cloud resources
- **Library Conflicts**: Use virtual environments to isolate dependencies

## Best Practices
- Start with Introduction tutorials before moving to Advanced
- Each notebook is self-contained with its own data download
- Check individual notebook requirements for specific dependencies
- Use paper trading for validation before live trading
- Monitor training with TensorBoard when available