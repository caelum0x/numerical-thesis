"""
Autonomous Portfolio Research Experiment
========================================
This file is modified by the research agent. Each run tests a hypothesis
and prints a RESULT line that gets logged to results.tsv.

Current experiment: BASELINE — Lasso predictions with MV optimization (λ=5)
"""

import sys, os, warnings, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone
import cvxpy as cp

from src.config import RAW_DIR, PROCESSED_DIR, RESULTS_DIR, TICKER_LIST, TICKERS, RANDOM_STATE

# ============================================================
# EXPERIMENT CONFIGURATION — MODIFY THIS SECTION
# ============================================================
EXPERIMENT_NAME = "baseline_lasso_mv5"
DESCRIPTION = "Baseline: Lasso model, MV optimization lambda=5, monthly rebalance"

# Model
MODEL = Lasso(alpha=0.001, max_iter=5000)

# Feature selection
def select_features(features_df):
    """Select which columns to use as predictors."""
    # Exclude all return columns to avoid leakage
    feat_cols = [c for c in features_df.columns if '_ret_' not in c]
    return feat_cols

# Target
PREDICTION_HORIZON = 21  # days

# Optimization
RISK_AVERSION = 5.0
MAX_WEIGHT = 0.40
REBALANCE_FREQ = 21  # trading days

# Backtest
TRAIN_WINDOW = 252 * 3  # 3 years
TEST_WINDOW = 63
CV_GAP = 21
N_CV_FOLDS = 5
BACKTEST_START = 252 * 3
LOOKBACK = 252
TC_BPS = 10

# ============================================================
# PIPELINE — DO NOT MODIFY BELOW THIS LINE (unless you know what you're doing)
# ============================================================

def main():
    start_time = time.time()

    # Load data
    prices = pd.read_csv(RAW_DIR / 'prices.csv', index_col=0, parse_dates=True).ffill().bfill()
    features = pd.read_csv(PROCESSED_DIR / 'features.csv', index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()

    tickers = list(prices.columns)
    feat_cols = select_features(features)

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Features: {len(feat_cols)} columns")
    print(f"Model: {type(MODEL).__name__}")
    print(f"Horizon: {PREDICTION_HORIZON}d, Risk aversion: {RISK_AVERSION}")

    # Build forward-looking targets
    fwd_returns = prices.shift(-PREDICTION_HORIZON) / prices - 1

    # ============================================================
    # STEP 1: Train models with time-series CV
    # ============================================================
    print("\n--- STEP 1: Model Training ---")
    models = {}
    cv_metrics = []

    for ticker in tickers:
        if ticker not in fwd_returns.columns:
            continue

        X = features[feat_cols]
        y = fwd_returns[ticker].reindex(X.index)
        mask = X.notna().all(axis=1) & y.notna()
        X_c, y_c = X[mask], y[mask]

        if len(X_c) < TRAIN_WINDOW + TEST_WINDOW + CV_GAP:
            continue

        fold_ics, fold_das = [], []

        for i in range(N_CV_FOLDS):
            tr_end = TRAIN_WINDOW + i * TEST_WINDOW
            te_start = tr_end + CV_GAP
            te_end = te_start + TEST_WINDOW
            if te_end > len(X_c):
                break

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_c.iloc[:tr_end])
            X_te = scaler.transform(X_c.iloc[te_start:te_end])

            m = clone(MODEL)
            m.fit(X_tr, y_c.iloc[:tr_end])
            pred = m.predict(X_te)
            actual = y_c.iloc[te_start:te_end].values

            ic = np.corrcoef(actual, pred)[0, 1] if len(actual) > 2 else 0
            da = np.mean(np.sign(actual) == np.sign(pred))
            fold_ics.append(ic)
            fold_das.append(da)

        avg_ic = np.mean(fold_ics) if fold_ics else 0
        avg_da = np.mean(fold_das) if fold_das else 0.5

        cv_metrics.append({'ticker': ticker, 'ic': avg_ic, 'dir_acc': avg_da})

        # Train final model on all data
        scaler = StandardScaler()
        X_all = scaler.fit_transform(X_c)
        final = clone(MODEL)
        final.fit(X_all, y_c)
        models[ticker] = {'model': final, 'scaler': scaler, 'feat_cols': feat_cols}

    cv_df = pd.DataFrame(cv_metrics)
    avg_ic = cv_df['ic'].mean()
    avg_da = cv_df['dir_acc'].mean()
    print(f"  Avg IC: {avg_ic:+.3f}, Avg DirAcc: {avg_da:.1%}")
    print(f"  Models trained: {len(models)}/{len(tickers)}")

    # ============================================================
    # STEP 2: Backtest with ML predictions
    # ============================================================
    print("\n--- STEP 2: Backtesting ---")

    daily_ret = prices.pct_change().dropna()
    n_assets = daily_ret.shape[1]
    dates = daily_ret.index
    w = np.ones(n_assets) / n_assets
    port_rets = []

    for t in range(BACKTEST_START, len(dates)):
        day_ret = daily_ret.iloc[t].values

        if (t - BACKTEST_START) % REBALANCE_FREQ == 0:
            # Get ML predictions
            date = dates[t]
            mu = np.zeros(n_assets)

            if date in features.index:
                row = features.loc[[date], feat_cols]
            else:
                idx = features.index.get_indexer([date], method='nearest')[0]
                row = features.iloc[[idx]][feat_cols]

            for j, ticker in enumerate(tickers):
                if ticker in models:
                    m = models[ticker]
                    X_s = m['scaler'].transform(row)
                    mu[j] = m['model'].predict(X_s)[0] * (252 / PREDICTION_HORIZON)

            # Covariance
            window = daily_ret.iloc[max(0, t-LOOKBACK):t]
            from sklearn.covariance import LedoitWolf
            cov = LedoitWolf().fit(window.dropna()).covariance_ * 252

            # Optimize
            ww = cp.Variable(n_assets)
            prob = cp.Problem(
                cp.Maximize(mu @ ww - (RISK_AVERSION/2) * cp.quad_form(ww, cov)),
                [cp.sum(ww)==1, ww>=0, ww<=MAX_WEIGHT]
            )
            prob.solve(solver=cp.OSQP, verbose=False)

            if prob.status in ('optimal', 'optimal_inaccurate') and ww.value is not None:
                new_w = ww.value
                turnover = np.sum(np.abs(new_w - w))
                tc = turnover * TC_BPS / 10000
                w = new_w
            else:
                tc = 0
        else:
            tc = 0

        port_ret = np.sum(w * day_ret) - tc
        port_rets.append({'date': dates[t], 'return': port_ret})
        w = w * (1 + day_ret)
        w /= w.sum()

    rets = pd.DataFrame(port_rets).set_index('date')['return']

    # ============================================================
    # STEP 3: Compute metrics
    # ============================================================
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + rets).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    sortino_down = rets[rets < 0].std() * np.sqrt(252)
    sortino = ann_ret / sortino_down if sortino_down > 0 else 0
    total = cum.iloc[-1] - 1

    elapsed = time.time() - start_time

    print(f"\n--- RESULTS ---")
    print(f"  Ann. Return: {ann_ret:+.2%}")
    print(f"  Ann. Vol:    {ann_vol:.2%}")
    print(f"  Sharpe:      {sharpe:.3f}")
    print(f"  Sortino:     {sortino:.3f}")
    print(f"  Max DD:      {max_dd:.2%}")
    print(f"  Total:       {total:+.1%}")
    print(f"  Avg IC:      {avg_ic:+.3f}")
    print(f"  Avg DirAcc:  {avg_da:.1%}")
    print(f"  Runtime:     {elapsed:.0f}s")

    # Machine-readable result line
    print(f'\nRESULT: sharpe={sharpe:.4f} ic={avg_ic:.4f} dir_acc={avg_da:.4f} ann_return={ann_ret:.4f} max_dd={max_dd:.4f} description="{DESCRIPTION}"')

    # Save results
    rets.to_csv(RESULTS_DIR / f'experiment_{EXPERIMENT_NAME}.csv')


if __name__ == '__main__':
    main()
