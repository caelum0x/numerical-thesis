# Autonomous Portfolio Research — program.md

Adapted from Karpathy's autoresearch pattern for macro-based portfolio optimization.

## Goal

Autonomously experiment with different model configurations, feature sets, and optimization strategies to find the best-performing portfolio. The metric is **out-of-sample Sharpe ratio** and **Information Coefficient (IC)**.

## Setup

The thesis project lives in `/Users/arhansubasi/thesis/thesis_portfolio_opt/`.

Key files:
- `src/config.py` — all parameters (do not modify during experiments)
- `data/raw/prices.csv` — 12 ETFs, 2005-2024 (already fetched)
- `data/raw/macro.csv` — 18 FRED series (already fetched)
- `data/processed/features.csv` — 212 features (already built)
- `research/experiment.py` — the file the agent modifies and runs

## What you CAN modify

Only `research/experiment.py`. This file contains:
- Feature selection logic
- Model choice and hyperparameters
- Target variable construction
- Optimization strategy
- Backtest configuration

## What you CANNOT modify

- Raw data files
- `src/config.py`
- `src/` modules (they are the library)

## The experiment loop

LOOP FOREVER:

1. Read current `research/experiment.py` and `research/results.tsv`
2. Form a hypothesis (e.g. "adding RSI features will improve IC for equities")
3. Modify `experiment.py` with the change
4. Run: `cd /Users/arhansubasi/thesis/thesis_portfolio_opt && source venv/bin/activate && python research/experiment.py > research/run.log 2>&1`
5. Read results: `grep "^RESULT:" research/run.log`
6. Log to `research/results.tsv`
7. If Sharpe or IC improved, keep the change. If not, revert.
8. Repeat.

## Metrics (printed by experiment.py)

```
RESULT: sharpe=X.XXX ic=X.XXX dir_acc=X.X% ann_return=X.X% max_dd=X.X% description="what was tried"
```

## Ideas to try

- Different feature subsets (macro only, momentum only, all features)
- Different models (Lasso, XGBoost, LightGBM, ensemble)
- Different prediction horizons (5d, 21d, 63d)
- Different optimization objectives (MV, CVaR, risk parity)
- Different risk aversion levels
- Shrinkage of predictions toward historical mean
- Transaction cost sensitivity
- Rolling vs expanding window
- Feature selection via mutual information
- PCA dimensionality reduction before modeling
