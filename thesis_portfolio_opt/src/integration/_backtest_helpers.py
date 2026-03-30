"""
Backtest helpers for the integrated pipeline.

Provides:
    load_best_models()         — load trained .pkl models from results dir
    run_integrated_backtest()  — walk-forward backtest with optional risk scaling
    compute_metrics()          — standard performance metrics from return series
"""

from __future__ import annotations

import glob
import logging
import os
import pickle
from typing import Callable, Optional

import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

PREDICTION_HORIZON = 21


def load_best_models(results_dir, tickers: list[str]) -> dict:
    """Load the best trained model for each ticker from .pkl files.

    Scans for model_*.pkl, groups by ticker, picks the most recent.
    Returns dict: ticker → {'model': model, 'scaler': scaler}.
    """
    models = {}
    pattern = os.path.join(str(results_dir), 'model_*.pkl')
    files = glob.glob(pattern)

    for fpath in files:
        fname = os.path.basename(fpath)
        # Pattern: model_{algo}_{ticker}_{target}.pkl
        parts = fname.replace('.pkl', '').split('_')
        if len(parts) < 3:
            continue
        ticker = parts[2].upper()
        if ticker not in tickers:
            continue

        try:
            with open(fpath, 'rb') as f:
                bundle = pickle.load(f)
            if isinstance(bundle, dict) and 'model' in bundle and 'scaler' in bundle:
                models[ticker] = bundle
            else:
                # Might be just the model object
                models[ticker] = {'model': bundle, 'scaler': None}
        except Exception as e:
            logger.debug("Failed to load %s: %s", fpath, e)

    logger.info("Loaded %d/%d models", len(models), len(tickers))
    return models


def _optimize(mu: np.ndarray, cov: np.ndarray,
              risk_aversion: float, max_weight: float) -> np.ndarray:
    """Mean-variance optimization via CVXPY."""
    n = len(mu)
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Maximize(mu @ w - (risk_aversion / 2) * cp.quad_form(w, cov)),
        [cp.sum(w) == 1, w >= 0, w <= max_weight]
    )
    prob.solve(solver=cp.OSQP, verbose=False)
    if prob.status in ('optimal', 'optimal_inaccurate') and w.value is not None:
        return np.array(w.value).flatten()
    return np.ones(n) / n


def run_integrated_backtest(
    prices: pd.DataFrame,
    daily_ret: pd.DataFrame,
    models: dict,
    features: pd.DataFrame,
    tickers: list[str],
    risk_aversion: float = 5.0,
    max_weight: float = 0.35,
    rebalance_freq: int = 21,
    tc_bps: int = 10,
    shrinkage: float = 0.0,
    risk_scale_fn: Optional[Callable] = None,
) -> pd.Series:
    """Run walk-forward backtest with optional MiroFish risk scaling.

    Parameters
    ----------
    risk_scale_fn : callable, optional
        Function(date) → float ∈ [0, 1].  When provided, portfolio weights
        are scaled by this factor, with the remainder going to equal-weight.
        This implements the MiroFish agreement-based risk overlay.
    """
    n = len(tickers)
    # Historical mean for shrinkage
    train_prices = prices[prices.index < pd.Timestamp('2022-01-01')]
    hist_mu = train_prices.pct_change().dropna().mean().values * 252

    w = np.ones(n) / n
    port_rets = []

    # Feature columns (exclude swarm_ for model prediction, they go to optimizer)
    base_feat_cols = [c for c in features.columns if not c.startswith('swarm_')]

    for t in range(21, len(daily_ret)):
        day_ret = daily_ret.iloc[t].values

        if t % rebalance_freq == 0:
            date = daily_ret.index[t]
            mu = np.zeros(n)

            # Get features for this date
            if date in features.index:
                row = features.loc[[date]]
            else:
                idx = features.index.get_indexer([date], method='nearest')[0]
                row = features.iloc[[idx]]

            # Predict returns per asset
            for j, ticker in enumerate(tickers):
                if ticker in models:
                    m = models[ticker]
                    try:
                        model_cols = base_feat_cols
                        if m.get('scaler') is not None:
                            scaler = m['scaler']
                            if hasattr(scaler, 'feature_names_in_'):
                                model_cols = list(scaler.feature_names_in_)
                            available = [c for c in model_cols if c in row.columns]
                            X = row[available].fillna(0)
                            if len(available) < len(model_cols):
                                for mc in model_cols:
                                    if mc not in available:
                                        X[mc] = 0.0
                                X = X[model_cols]
                            X_s = scaler.transform(X)
                        else:
                            X_s = row[base_feat_cols].fillna(0).values
                        mu[j] = m['model'].predict(X_s)[0] * (252 / PREDICTION_HORIZON)
                    except Exception:
                        mu[j] = 0.0

            # Shrinkage toward historical mean
            if shrinkage > 0:
                mu = (1 - shrinkage) * mu + shrinkage * hist_mu

            # Covariance
            window = daily_ret.iloc[max(0, t - 252):t]
            cov = LedoitWolf().fit(window.dropna()).covariance_ * 252

            # Optimize
            new_w = _optimize(mu, cov, risk_aversion, max_weight)

            # Apply MiroFish risk overlay
            if risk_scale_fn is not None:
                scale = risk_scale_fn(date)
                # Blend: scale * optimizer_weights + (1-scale) * equal_weight
                ew = np.ones(n) / n
                new_w = scale * new_w + (1 - scale) * ew

            turnover = np.sum(np.abs(new_w - w))
            tc = turnover * tc_bps / 10000
            w = new_w
        else:
            tc = 0

        port_rets.append(np.sum(w * day_ret) - tc)
        w = w * (1 + day_ret)
        total = w.sum()
        if total > 0:
            w = w / total

    return pd.Series(port_rets, index=daily_ret.index[21:])


def compute_metrics(returns: pd.Series) -> dict:
    """Compute standard portfolio performance metrics."""
    if returns.empty or returns.std() == 0:
        return {
            'sharpe': 0, 'sortino': 0, 'ann_return': 0,
            'ann_vol': 0, 'max_dd': 0, 'calmar': 0, 'total_return': 0,
        }

    ann_return = float(returns.mean() * 252)
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_return / downside if downside > 0 else 0

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = float(drawdown.min())

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    total_return = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0

    return {
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'ann_return': round(ann_return, 4),
        'ann_vol': round(ann_vol, 4),
        'max_dd': round(max_dd, 4),
        'calmar': round(calmar, 4),
        'total_return': round(total_return, 4),
    }
