# Thesis: ML-Enhanced Portfolio Optimization

Industrial Engineering thesis project — macroeconomic factor-based dynamic portfolio optimization using machine learning.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set FRED API key
# Option 1: Environment variable
export FRED_API_KEY=your_key_here

# Option 2: .env file (auto-loaded)
echo "FRED_API_KEY=your_key_here" > .env
```

## Project Structure

```
thesis_portfolio_opt/
├── src/                          # Core Source Code
│   ├── config.py                 # Central configuration (tickers, dates, hyperparams, stress scenarios)
│   ├── utils.py                  # Logging, validation, financial metrics, LaTeX/Excel export
│   ├── data/
│   │   ├── fetcher.py            # FRED & YFinance data fetchers
│   │   └── preprocessor.py       # Feature engineering, VIF, stationarity tests
│   ├── models/
│   │   ├── trainer.py            # 8 ML models, time-series CV, feature importance
│   │   └── predict.py            # Predictions, ensemble, per-date inference
│   ├── optimization/
│   │   ├── optimizer.py          # MV, max Sharpe, min var, risk parity, Black-Litterman
│   │   └── backtester.py         # Rolling backtest, stress tests, strategy comparison
│   └── visualization/
│       └── plots.py              # 15 thesis-quality figure functions
├── notebooks/
│   ├── 01_eda_visualization.ipynb      # 25 cells — full EDA
│   ├── 02_model_experimentation.ipynb  # 21 cells — model training & comparison
│   ├── 03_optimization_analysis.ipynb  # 27 cells — optimization & backtesting
│   └── 04_thesis_figures.ipynb         # 27 cells — all publication figures
├── app/
│   └── dashboard.py              # 5-page Streamlit dashboard (500+ lines)
├── tests/
│   ├── conftest.py               # Shared fixtures (synthetic data)
│   ├── test_data_integrity.py    # Data quality tests
│   └── test_optimization.py      # Optimizer, backtester, preprocessor, trainer tests
├── run_pipeline.py               # CLI: python run_pipeline.py --step all
├── Makefile                      # make pipeline, make test, make dashboard
├── pyproject.toml                # pytest & ruff configuration
├── requirements.txt              # All Python dependencies
└── .env                          # FRED API key (gitignored)
```

## Quick Start

### Option A: Full pipeline (recommended)
```bash
make pipeline    # or: python run_pipeline.py --step all
```

### Option B: Step by step
```bash
make fetch       # Download FRED + YFinance data
make features    # Build feature matrix
make train       # Train 8 ML models for each asset
make backtest    # Run rolling-window backtests
make figures     # Generate 14 thesis figures (PDF)
```

### Option C: Python API
```python
from src.data import fetch_all, build_features
from src.models import train_all_models, predict_returns
from src.optimization import backtest_strategy, compare_strategies

prices, macro = fetch_all()
features = build_features(prices, macro)
```

## Models

| Model | Type |
|-------|------|
| Ridge / Lasso / ElasticNet | Regularized linear |
| Random Forest | Ensemble (bagging) |
| Gradient Boosting | Ensemble (boosting) |
| XGBoost | Gradient boosting |
| LightGBM | Gradient boosting |
| SVR | Support vector regression |

## Optimization Strategies

| Strategy | Method |
|----------|--------|
| Mean-Variance | CVXPY quadratic programming (configurable λ) |
| Max Sharpe | Cornuejols-Tutuncu transformation |
| Min Variance | Risk-only optimization |
| Risk Parity | Equal risk contribution (iterative) |
| Black-Litterman | Bayesian prior + investor views |
| Inverse Volatility | Heuristic benchmark |

## Dashboard

```bash
make dashboard   # or: streamlit run app/dashboard.py
```

5 pages: Overview, Optimization, Backtesting, Stress Testing, Model Insights.

## Tests

```bash
make test        # Full test suite
make test-unit   # Unit tests only (no data required)
```

## Team Roles

| Role | Focus Area | Primary Files |
|------|-----------|---------------|
| Person A (Tech Lead) | `src/` modules, pipeline, dashboard | config.py, fetcher.py, optimizer.py |
| Person C (Research) | Thesis figures, documentation | notebooks/04_thesis_figures.ipynb |
| Person E (Quant) | Model validation, statistical tests | notebooks/02_model_experimentation.ipynb |
