"""
Portfolio AutoResearch — Data preparation and evaluation infrastructure.
Adapted from Karpathy's autoresearch for financial time series prediction.

This file is READ-ONLY during experiments. The agent only modifies train.py.

Usage:
    python prepare.py              # prepare data (download + features)
    python prepare.py --no-fetch   # skip fetching, just rebuild features
"""

import os
import sys
import time
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.covariance import LedoitWolf
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_LOOKBACK = 252 * 3      # 3 years of history for model training
TIME_BUDGET = 120            # max 2 minutes per experiment
PREDICTION_HORIZON = 21      # predict 21-day forward returns
TRAIN_END = '2021-12-31'     # strict temporal cutoff
OOS_START = '2022-01-01'     # out-of-sample begins here

# Paths
THESIS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'thesis_portfolio_opt')
RAW_DIR = os.path.join(THESIS_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(THESIS_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(THESIS_ROOT, 'data', 'results')
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_portfolio")

for d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# Assets
TICKERS = {
    "SPY": "US Large Cap Equity", "IWM": "US Small Cap Equity",
    "EFA": "Intl Developed Equity", "EEM": "Emerging Markets",
    "AGG": "US Agg Bonds", "TLT": "US Long Treasuries",
    "LQD": "US IG Corporate", "HYG": "US High Yield",
    "GLD": "Gold", "VNQ": "US REITs",
    "DBC": "Commodities", "TIP": "TIPS",
}
TICKER_LIST = list(TICKERS.keys())
N_ASSETS = len(TICKER_LIST)

# FRED series
FRED_SERIES = {
    "DFF": "Fed Funds Rate", "DGS2": "2Y Treasury",
    "DGS10": "10Y Treasury", "T10Y2Y": "10Y-2Y Spread",
    "T10Y3M": "10Y-3M Spread", "VIXCLS": "VIX",
    "BAMLH0A0HYM2": "HY OAS", "BAMLC0A4CBBB": "BBB Spread",
    "DTWEXBGS": "USD Index", "UMCSENT": "Consumer Sentiment",
    "UNRATE": "Unemployment", "ICSA": "Jobless Claims",
    "CPIAUCSL": "CPI", "T10YIE": "Breakeven Inflation",
    "PPIACO": "PPI", "M2SL": "M2 Money Supply",
    "USSLIND": "Leading Index", "HOUST": "Housing Starts",
}

# Benchmarks (computed once, cached)
_benchmark_cache = {}

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_prices(start='2005-01-01', end='2024-12-31'):
    """Fetch ETF prices from Yahoo Finance."""
    import yfinance as yf
    path = os.path.join(RAW_DIR, 'prices.csv')
    if os.path.exists(path):
        print(f"Prices already exist at {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    print(f"Fetching prices for {len(TICKER_LIST)} ETFs...")
    data = yf.download(TICKER_LIST, start=start, end=end, auto_adjust=True)
    prices = data['Close']
    prices.to_csv(path)
    print(f"Saved: {path} — {prices.shape}")
    return prices


def fetch_macro(start='2005-01-01', end='2024-12-31'):
    """Fetch FRED macro data."""
    from fredapi import Fred
    path = os.path.join(RAW_DIR, 'macro.csv')
    if os.path.exists(path):
        print(f"Macro data already exists at {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    api_key = os.environ.get('FRED_API_KEY', 'd6995d762b3aed1ddd40e8ae0bdeb08a')
    fred = Fred(api_key=api_key)
    frames = {}
    for sid, desc in FRED_SERIES.items():
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            frames[sid] = s
            print(f"  {sid}: {len(s)} obs")
        except Exception as e:
            print(f"  {sid}: FAILED — {e}")
    macro = pd.DataFrame(frames)
    macro.index = pd.to_datetime(macro.index)
    macro.to_csv(path)
    print(f"Saved: {path} — {macro.shape}")
    return macro


# ---------------------------------------------------------------------------
# Feature engineering (fixed, do not modify)
# ---------------------------------------------------------------------------

def build_features(prices, macro):
    """
    Build the complete feature matrix. This is fixed infrastructure.

    Includes:
    - Returns at multiple horizons (1d, 5d, 21d, 63d)
    - Annualized volatility at multiple windows (21d, 63d, 126d)
    - Momentum at multiple horizons (21d, 63d, 126d, 252d)
    - RSI (14-day)
    - Bollinger Band %B (20-day, 2 std)
    - MACD histogram (12/26/9)
    - Rolling beta vs SPY (63-day)
    - Cross-sectional return dispersion
    - VIX z-score (252-day rolling)
    - Yield curve slope change (5d diff of T10Y2Y)
    - Macro variables (raw + lagged)
    """
    path = os.path.join(PROCESSED_DIR, 'features.csv')
    if os.path.exists(path):
        print(f"Features already exist at {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    prices = prices.ffill().bfill()
    all_dfs = []

    # Returns at multiple horizons
    for h in [1, 5, 21, 63]:
        ret = prices.pct_change(h)
        ret.columns = [f'{c}_ret_{h}d' for c in ret.columns]
        all_dfs.append(ret)

    # Volatility
    daily_ret = prices.pct_change()
    for w in [21, 63, 126]:
        vol = daily_ret.rolling(w).std() * np.sqrt(252)
        vol.columns = [f'{c}_vol_{w}d' for c in vol.columns]
        all_dfs.append(vol)

    # Momentum
    for w in [21, 63, 126, 252]:
        mom = prices.pct_change(w)
        mom.columns = [f'{c}_mom_{w}d' for c in mom.columns]
        all_dfs.append(mom)

    # RSI
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi.columns = [f'{c}_rsi_14' for c in rsi.columns]
    all_dfs.append(rsi)

    # -----------------------------------------------------------------------
    # Advanced feature engineering
    # -----------------------------------------------------------------------

    # Bollinger Band %B for each asset (20-day window, 2 standard deviations)
    bb_window = 20
    bb_std_mult = 2
    rolling_mean = prices.rolling(bb_window).mean()
    rolling_std = prices.rolling(bb_window).std()
    bb_upper = rolling_mean + bb_std_mult * rolling_std
    bb_lower = rolling_mean - bb_std_mult * rolling_std
    # %B = (price - lower) / (upper - lower); 0 = at lower band, 1 = at upper
    bb_pctb = (prices - bb_lower) / (bb_upper - bb_lower)
    bb_pctb.columns = [f'{c}_bb_pctb' for c in bb_pctb.columns]
    all_dfs.append(bb_pctb)

    # MACD histogram for each asset (fast=12, slow=26, signal=9)
    ema_fast = prices.ewm(span=12, adjust=False).mean()
    ema_slow = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - macd_signal
    # Normalize by price to make cross-asset comparable
    macd_hist_norm = macd_histogram / prices
    macd_hist_norm.columns = [f'{c}_macd_hist' for c in macd_hist_norm.columns]
    all_dfs.append(macd_hist_norm)

    # Rolling beta vs SPY for each asset (63-day rolling window)
    spy_ret = daily_ret['SPY']
    beta_window = 63
    betas = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    spy_var = spy_ret.rolling(beta_window).var()
    for ticker in prices.columns:
        asset_ret = daily_ret[ticker]
        cov_with_spy = asset_ret.rolling(beta_window).cov(spy_ret)
        betas[ticker] = cov_with_spy / spy_var.replace(0, np.nan)
    betas.columns = [f'{c}_beta_spy_63d' for c in betas.columns]
    all_dfs.append(betas)

    # Cross-sectional return dispersion (std of returns across all assets)
    for h in [1, 5, 21]:
        cross_ret = prices.pct_change(h)
        dispersion = cross_ret.std(axis=1)
        dispersion_df = pd.DataFrame(
            {f'cross_disp_{h}d': dispersion},
            index=prices.index
        )
        all_dfs.append(dispersion_df)

    # VIX z-score (deviation from 252-day rolling mean, in units of rolling std)
    macro_daily = macro.reindex(prices.index).ffill().bfill()
    if 'VIXCLS' in macro_daily.columns:
        vix = macro_daily['VIXCLS']
        vix_mean_252 = vix.rolling(252).mean()
        vix_std_252 = vix.rolling(252).std()
        vix_zscore = (vix - vix_mean_252) / vix_std_252.replace(0, np.nan)
        vix_zscore_df = pd.DataFrame(
            {'vix_zscore_252d': vix_zscore},
            index=prices.index
        )
        all_dfs.append(vix_zscore_df)

    # Yield curve slope change (5-day difference of T10Y2Y spread)
    if 'T10Y2Y' in macro_daily.columns:
        t10y2y = macro_daily['T10Y2Y']
        yc_slope_chg = t10y2y.diff(5)
        yc_slope_chg_df = pd.DataFrame(
            {'yc_slope_chg_5d': yc_slope_chg},
            index=prices.index
        )
        all_dfs.append(yc_slope_chg_df)

    # Macro: forward fill to daily, lag
    good = macro_daily.columns[macro_daily.isnull().mean() < 0.3]
    macro_daily = macro_daily[good]
    all_dfs.append(macro_daily)
    for lag in [1, 5, 21]:
        lagged = macro_daily.shift(lag)
        lagged.columns = [f'{c}_lag{lag}' for c in lagged.columns]
        all_dfs.append(lagged)

    features = pd.concat(all_dfs, axis=1)
    price_cols = [c for c in features.columns if any(t in c for t in TICKER_LIST)]
    features = features.dropna(subset=price_cols).ffill().bfill().dropna()

    features.to_csv(path)
    print(f"Saved: {path} — {features.shape}")
    return features


# ---------------------------------------------------------------------------
# Evaluation (the ground truth metric — do not modify)
# ---------------------------------------------------------------------------

def evaluate_oos(strategy_returns: pd.Series) -> dict:
    """
    Compute out-of-sample portfolio metrics. This is the ground truth evaluation.
    Analogous to evaluate_bpb in the original autoresearch.

    Primary metric: OOS Sharpe ratio
    Secondary: IC, directional accuracy, max drawdown
    """
    if len(strategy_returns) == 0:
        return {'sharpe': -999, 'ann_return': 0, 'ann_vol': 0, 'max_dd': 0, 'total': 0}

    ann_r = strategy_returns.mean() * 252
    ann_v = strategy_returns.std() * np.sqrt(252)
    sharpe = ann_r / ann_v if ann_v > 0 else 0
    cum = (1 + strategy_returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    total = cum.iloc[-1] - 1

    down = strategy_returns[strategy_returns < 0].std() * np.sqrt(252)
    sortino = ann_r / down if down > 0 else 0

    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'ann_return': ann_r,
        'ann_vol': ann_v,
        'max_dd': max_dd,
        'total': total,
    }


# ---------------------------------------------------------------------------
# Detailed OOS evaluation with extended risk/return decomposition
# ---------------------------------------------------------------------------

def evaluate_oos_detailed(strategy_returns: pd.Series,
                          benchmark_returns: pd.Series) -> dict:
    """
    Extended out-of-sample evaluation that supplements evaluate_oos with
    relative and distributional risk metrics.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily portfolio returns (already aligned to OOS period).
    benchmark_returns : pd.Series
        Daily benchmark returns (e.g. SPY or equal-weight) on the same dates.

    Returns
    -------
    dict with keys:
        -- all keys from evaluate_oos --
        tracking_error, information_ratio, beta, alpha,
        up_capture, down_capture, var_95, cvar_95,
        skewness, kurtosis, winning_pct, avg_win, avg_loss, win_loss_ratio
    """
    # Start with the base metrics
    base = evaluate_oos(strategy_returns)

    # Align the two series on their common dates
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    strat = strategy_returns.reindex(common_idx).dropna()
    bench = benchmark_returns.reindex(common_idx).dropna()
    common_idx = strat.index.intersection(bench.index)
    strat = strat.loc[common_idx]
    bench = bench.loc[common_idx]

    # Tracking error (annualized)
    excess = strat - bench
    tracking_error = excess.std() * np.sqrt(252) if len(excess) > 1 else 0.0

    # Information ratio
    info_ratio = (excess.mean() * 252) / tracking_error if tracking_error > 0 else 0.0

    # Beta vs benchmark (OLS-style: cov / var)
    if bench.std() > 0 and len(bench) > 2:
        cov_sb = strat.cov(bench)
        var_b = bench.var()
        beta = cov_sb / var_b if var_b > 0 else 1.0
    else:
        beta = 1.0

    # Alpha (annualized Jensen's alpha)
    alpha = (strat.mean() - beta * bench.mean()) * 252

    # Up-capture and down-capture ratios
    up_days = bench > 0
    down_days = bench < 0
    if up_days.sum() > 0:
        up_capture = (strat[up_days].mean() / bench[up_days].mean()) * 100
    else:
        up_capture = np.nan
    if down_days.sum() > 0:
        down_capture = (strat[down_days].mean() / bench[down_days].mean()) * 100
    else:
        down_capture = np.nan

    # Value-at-Risk (historical, 95%)
    var_95 = np.percentile(strat, 5) if len(strat) > 0 else 0.0

    # Conditional VaR (Expected Shortfall, 95%)
    below_var = strat[strat <= var_95]
    cvar_95 = below_var.mean() if len(below_var) > 0 else var_95

    # Distributional stats
    skewness = float(scipy_stats.skew(strat)) if len(strat) > 2 else 0.0
    kurtosis = float(scipy_stats.kurtosis(strat, fisher=True)) if len(strat) > 3 else 0.0

    # Win/loss analysis
    wins = strat[strat > 0]
    losses = strat[strat < 0]
    winning_pct = len(wins) / len(strat) * 100 if len(strat) > 0 else 0.0
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    # Merge everything
    detailed = {
        **base,
        'tracking_error': tracking_error,
        'information_ratio': info_ratio,
        'beta': beta,
        'alpha': alpha,
        'up_capture': up_capture,
        'down_capture': down_capture,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'winning_pct': winning_pct,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
    }
    return detailed


# ---------------------------------------------------------------------------
# Model-level prediction evaluation
# ---------------------------------------------------------------------------

def evaluate_model_predictions(y_true: np.ndarray,
                               y_pred: np.ndarray) -> dict:
    """
    Evaluate quality of return predictions at the model level.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,) or (n_samples, n_assets)
        Actual forward returns.
    y_pred : array-like, shape (n_samples,) or (n_samples, n_assets)
        Predicted forward returns.

    Returns
    -------
    dict with keys:
        rmse, mae, r2, ic (Spearman rank correlation),
        directional_accuracy, hit_rate_top_3, decile_spread
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Flatten for scalar metrics if needed
    yt_flat = y_true.ravel()
    yp_flat = y_pred.ravel()

    # Remove NaN pairs
    valid = np.isfinite(yt_flat) & np.isfinite(yp_flat)
    yt_flat = yt_flat[valid]
    yp_flat = yp_flat[valid]

    n = len(yt_flat)
    if n == 0:
        return {
            'rmse': np.nan, 'mae': np.nan, 'r2': np.nan,
            'ic': np.nan, 'directional_accuracy': np.nan,
            'hit_rate_top_3': np.nan, 'decile_spread': np.nan,
        }

    # RMSE
    rmse = np.sqrt(np.mean((yt_flat - yp_flat) ** 2))

    # MAE
    mae = np.mean(np.abs(yt_flat - yp_flat))

    # R-squared
    ss_res = np.sum((yt_flat - yp_flat) ** 2)
    ss_tot = np.sum((yt_flat - np.mean(yt_flat)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Information Coefficient (Spearman rank correlation)
    if n > 2:
        ic, _ = scipy_stats.spearmanr(yt_flat, yp_flat)
    else:
        ic = 0.0

    # Directional accuracy (predicted sign matches actual sign)
    same_sign = np.sign(yt_flat) == np.sign(yp_flat)
    directional_accuracy = np.mean(same_sign)

    # Hit rate top 3 — evaluated cross-sectionally when 2D input is provided
    hit_rate_top_3 = _compute_hit_rate_top_k(y_true, y_pred, k=3)

    # Decile spread — return of top predicted decile minus bottom predicted decile
    decile_spread = _compute_decile_spread(yt_flat, yp_flat)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'ic': ic,
        'directional_accuracy': directional_accuracy,
        'hit_rate_top_3': hit_rate_top_3,
        'decile_spread': decile_spread,
    }


def _compute_hit_rate_top_k(y_true, y_pred, k=3):
    """
    For each cross-sectional row, check whether the top-k predicted assets
    overlap with the top-k actual assets. Returns the average overlap ratio.

    Works with 2D arrays (n_periods x n_assets). For 1D input, falls back
    to a single-row evaluation.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)

    n_periods, n_assets = y_true.shape
    if n_assets < k:
        k = max(1, n_assets)

    hits = 0.0
    valid_periods = 0
    for i in range(n_periods):
        row_t = y_true[i]
        row_p = y_pred[i]
        if np.any(np.isnan(row_t)) or np.any(np.isnan(row_p)):
            continue
        top_k_true = set(np.argsort(row_t)[-k:])
        top_k_pred = set(np.argsort(row_p)[-k:])
        overlap = len(top_k_true & top_k_pred) / k
        hits += overlap
        valid_periods += 1

    return hits / valid_periods if valid_periods > 0 else np.nan


def _compute_decile_spread(y_true_flat, y_pred_flat):
    """
    Compute return spread between top and bottom predicted deciles.

    Sorts observations by predicted return, splits into 10 buckets,
    and returns mean actual return of top bucket minus bottom bucket.
    """
    n = len(y_true_flat)
    if n < 10:
        # Not enough data for deciles; split in half instead
        if n < 2:
            return 0.0
        order = np.argsort(y_pred_flat)
        mid = n // 2
        bottom_ret = np.mean(y_true_flat[order[:mid]])
        top_ret = np.mean(y_true_flat[order[mid:]])
        return top_ret - bottom_ret

    order = np.argsort(y_pred_flat)
    decile_size = n // 10
    bottom_decile = order[:decile_size]
    top_decile = order[-decile_size:]
    spread = np.mean(y_true_flat[top_decile]) - np.mean(y_true_flat[bottom_decile])
    return spread


# ---------------------------------------------------------------------------
# Benchmarks and portfolio optimization
# ---------------------------------------------------------------------------

def get_benchmarks(prices: pd.DataFrame) -> dict:
    """Compute benchmark returns for comparison. Cached after first call."""
    global _benchmark_cache
    if _benchmark_cache:
        return _benchmark_cache

    oos_prices = prices[prices.index >= pd.Timestamp(OOS_START)]
    daily_ret = oos_prices.pct_change().dropna()

    # Equal weight
    eq = daily_ret.iloc[21:].mean(axis=1)

    # SPY buy-and-hold
    spy = daily_ret.iloc[21:]['SPY']

    # 60/40
    sf = daily_ret.iloc[21:][['SPY', 'AGG']].dot([0.6, 0.4])

    _benchmark_cache = {
        'equal_weight': evaluate_oos(eq),
        'spy': evaluate_oos(spy),
        'sixty_forty': evaluate_oos(sf),
        'eq_returns': eq,
        'spy_returns': spy,
    }

    return _benchmark_cache


def optimize_portfolio(mu, cov, risk_aversion=5.0, max_weight=0.4):
    """Mean-variance optimization. Fixed infrastructure."""
    import cvxpy as cp
    n = len(mu)
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Maximize(mu @ w - (risk_aversion / 2) * cp.quad_form(w, cov)),
        [cp.sum(w) == 1, w >= 0, w <= max_weight]
    )
    prob.solve(solver=cp.OSQP, verbose=False)
    if prob.status in ('optimal', 'optimal_inaccurate') and w.value is not None:
        return w.value
    return np.ones(n) / n


def run_oos_backtest(prices, models, features, feat_cols,
                     risk_aversion=5.0, max_weight=0.4,
                     rebalance_freq=21, tc_bps=10,
                     shrinkage=0.0, prediction_horizon=PREDICTION_HORIZON):
    """
    Run strict OOS backtest. Fixed infrastructure.
    The agent controls what goes into `models` and `feat_cols`.
    """
    tickers = list(prices.columns)
    n = len(tickers)

    # Historical mean for shrinkage
    hist = prices[prices.index <= pd.Timestamp(TRAIN_END)]
    hist_mu = hist.pct_change().dropna().mean().values * 252

    oos_prices = prices[prices.index >= pd.Timestamp(OOS_START)]
    daily_ret = oos_prices.pct_change().dropna()
    w = np.ones(n) / n
    port_rets = []

    for t in range(21, len(daily_ret.index)):
        day_ret = daily_ret.iloc[t].values

        if t % rebalance_freq == 0:
            date = daily_ret.index[t]
            mu = np.zeros(n)

            if date in features.index:
                row = features.loc[[date], feat_cols]
            else:
                idx = features.index.get_indexer([date], method='nearest')[0]
                row = features.iloc[[idx]][feat_cols]

            for j, ticker in enumerate(tickers):
                if ticker in models:
                    m = models[ticker]
                    X_s = m['scaler'].transform(row)
                    mu[j] = m['model'].predict(X_s)[0] * (252 / prediction_horizon)

            if shrinkage > 0:
                mu = (1 - shrinkage) * mu + shrinkage * hist_mu

            window = daily_ret.iloc[max(0, t - 252):t]
            cov = LedoitWolf().fit(window.dropna()).covariance_ * 252
            new_w = optimize_portfolio(mu, cov, risk_aversion, max_weight)
            turnover = np.sum(np.abs(new_w - w))
            tc = turnover * tc_bps / 10000
            w = new_w
        else:
            tc = 0

        port_rets.append(np.sum(w * day_ret) - tc)
        w = w * (1 + day_ret)
        w /= w.sum()

    return pd.Series(port_rets, index=daily_ret.index[21:])


# ---------------------------------------------------------------------------
# Walk-forward backtest with quarterly retraining
# ---------------------------------------------------------------------------

def run_walk_forward_backtest(prices, features, model_template, feat_cols,
                              risk_aversion=5.0, max_weight=0.4,
                              rebalance_freq=21, tc_bps=10,
                              shrinkage=0.0,
                              prediction_horizon=PREDICTION_HORIZON,
                              retrain_freq_months=3):
    """
    Walk-forward backtest that retrains models every quarter using an
    expanding window (all data up to the current quarter boundary).

    This is more realistic than a fixed train/test split because:
    - Models see only past data at each point in time
    - Retraining captures regime changes
    - Expanding window grows the training set over time

    Parameters
    ----------
    prices : pd.DataFrame
        Full price history (train + OOS).
    features : pd.DataFrame
        Full feature matrix aligned to prices.
    model_template : callable
        A function(X_train, y_train) -> fitted model object that has
        a .predict(X) method. This is called fresh at each retrain point.
    feat_cols : list[str]
        Feature column names used for training.
    risk_aversion : float
        Risk aversion for portfolio optimization.
    max_weight : float
        Maximum single-asset weight.
    rebalance_freq : int
        Rebalance every N trading days.
    tc_bps : float
        Transaction cost in basis points (one-way).
    shrinkage : float
        Blend of model predictions with historical mean (0 = pure model).
    prediction_horizon : int
        Forward return horizon in trading days.
    retrain_freq_months : int
        How often to retrain (default 3 = quarterly).

    Returns
    -------
    pd.Series of daily portfolio returns in the OOS period.
    """
    from sklearn.preprocessing import StandardScaler as _Scaler

    tickers = list(prices.columns)
    n_assets = len(tickers)

    # Compute forward returns for training targets
    fwd_returns = prices.shift(-prediction_horizon) / prices - 1

    # Historical mean for shrinkage
    hist = prices[prices.index <= pd.Timestamp(TRAIN_END)]
    hist_mu = hist.pct_change().dropna().mean().values * 252

    # Build the OOS time grid
    oos_prices = prices[prices.index >= pd.Timestamp(OOS_START)]
    daily_ret = oos_prices.pct_change().dropna()

    # Determine retrain boundaries: every retrain_freq_months months
    oos_dates = daily_ret.index[21:]
    if len(oos_dates) == 0:
        return pd.Series(dtype=float)

    # Quarter boundaries in the OOS period
    retrain_dates = pd.date_range(
        start=oos_dates[0],
        end=oos_dates[-1],
        freq=f'{retrain_freq_months}MS'
    )
    retrain_dates = retrain_dates.append(pd.DatetimeIndex([oos_dates[-1] + pd.Timedelta(days=1)]))

    # Storage for models trained at each boundary
    current_models = {}
    w = np.ones(n_assets) / n_assets
    port_rets = []

    for t_idx, t in enumerate(range(21, len(daily_ret.index))):
        date = daily_ret.index[t]
        day_ret = daily_ret.iloc[t].values

        # Check if we need to retrain
        needs_retrain = False
        if t_idx == 0:
            needs_retrain = True
        else:
            for rd in retrain_dates:
                prev_date = daily_ret.index[t - 1] if t > 0 else date
                if prev_date < rd <= date:
                    needs_retrain = True
                    break

        if needs_retrain:
            # Expanding window: all data strictly before current date
            train_mask = features.index < date
            train_feat = features.loc[train_mask, feat_cols]
            current_models = {}

            for j, ticker in enumerate(tickers):
                target_col = fwd_returns[ticker]
                # Align features and targets
                common = train_feat.index.intersection(target_col.dropna().index)
                if len(common) < 50:
                    continue

                X_train = train_feat.loc[common].values
                y_train = target_col.loc[common].values

                # Remove rows with NaN
                valid = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
                X_train = X_train[valid]
                y_train = y_train[valid]
                if len(X_train) < 50:
                    continue

                scaler = _Scaler()
                X_scaled = scaler.fit_transform(X_train)

                try:
                    fitted_model = model_template(X_scaled, y_train)
                    current_models[ticker] = {
                        'model': fitted_model,
                        'scaler': scaler,
                    }
                except Exception:
                    pass  # skip this asset if model fails

        # Portfolio rebalancing
        if t % rebalance_freq == 0:
            mu = np.zeros(n_assets)

            if date in features.index:
                row = features.loc[[date], feat_cols]
            else:
                idx = features.index.get_indexer([date], method='nearest')[0]
                row = features.iloc[[idx]][feat_cols]

            for j, ticker in enumerate(tickers):
                if ticker in current_models:
                    m = current_models[ticker]
                    X_s = m['scaler'].transform(row)
                    mu[j] = m['model'].predict(X_s)[0] * (252 / prediction_horizon)

            if shrinkage > 0:
                mu = (1 - shrinkage) * mu + shrinkage * hist_mu

            window = daily_ret.iloc[max(0, t - 252):t]
            cov = LedoitWolf().fit(window.dropna()).covariance_ * 252
            new_w = optimize_portfolio(mu, cov, risk_aversion, max_weight)
            turnover = np.sum(np.abs(new_w - w))
            tc = turnover * tc_bps / 10000
            w = new_w
        else:
            tc = 0

        port_rets.append(np.sum(w * day_ret) - tc)
        # Drift weights with realized returns
        w = w * (1 + day_ret)
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
        else:
            w = np.ones(n_assets) / n_assets

    return pd.Series(port_rets, index=daily_ret.index[21:])


# ---------------------------------------------------------------------------
# Benchmark comparison table
# ---------------------------------------------------------------------------

def compare_to_benchmarks(strategy_returns: pd.Series,
                          prices: pd.DataFrame,
                          strategy_name: str = "Strategy") -> pd.DataFrame:
    """
    Build a formatted comparison table between the strategy and standard
    benchmarks (equal weight, SPY, 60/40).

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily strategy returns in the OOS period.
    prices : pd.DataFrame
        Full price history for computing benchmarks.
    strategy_name : str
        Label for the strategy column.

    Returns
    -------
    pd.DataFrame with one column per strategy/benchmark and one row per metric.
    Also prints the table.
    """
    benchmarks = get_benchmarks(prices)
    strat_metrics = evaluate_oos(strategy_returns)

    rows = {
        'Sharpe Ratio': 'sharpe',
        'Sortino Ratio': 'sortino',
        'Ann. Return': 'ann_return',
        'Ann. Volatility': 'ann_vol',
        'Max Drawdown': 'max_dd',
        'Total Return': 'total',
    }

    data = {}
    data[strategy_name] = {label: strat_metrics[key] for label, key in rows.items()}
    for bm_name, bm_key in [('Equal Weight', 'equal_weight'),
                              ('SPY', 'spy'),
                              ('60/40', 'sixty_forty')]:
        data[bm_name] = {label: benchmarks[bm_key][key] for label, key in rows.items()}

    df = pd.DataFrame(data)

    # Pretty-print
    print(f"\n{'='*72}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*72}")
    fmt_funcs = {
        'Sharpe Ratio': lambda x: f"{x:.4f}",
        'Sortino Ratio': lambda x: f"{x:.4f}",
        'Ann. Return': lambda x: f"{x:+.2%}",
        'Ann. Volatility': lambda x: f"{x:.2%}",
        'Max Drawdown': lambda x: f"{x:.2%}",
        'Total Return': lambda x: f"{x:+.1%}",
    }

    header = f"{'Metric':<20}" + "".join(f"{col:>16}" for col in df.columns)
    print(header)
    print("-" * len(header))
    for metric_label in rows.keys():
        row_str = f"{metric_label:<20}"
        for col in df.columns:
            val = df.loc[metric_label, col]
            formatter = fmt_funcs.get(metric_label, lambda x: f"{x:.4f}")
            row_str += f"{formatter(val):>16}"
        print(row_str)
    print(f"{'='*72}\n")

    return df


# ---------------------------------------------------------------------------
# LaTeX report generation
# ---------------------------------------------------------------------------

def generate_latex_report(metrics: dict, filename: str,
                          caption: str = "Portfolio Performance Metrics",
                          label: str = "tab:portfolio_metrics"):
    """
    Export a dictionary of metrics as a LaTeX table.

    Parameters
    ----------
    metrics : dict
        Keys are metric names, values are numeric values.
        Can also be a dict of dicts for multi-column tables
        (outer keys = column headers, inner keys = metric names).
    filename : str
        Output file path (e.g. 'results/metrics.tex').
    caption : str
        LaTeX table caption.
    label : str
        LaTeX table label for cross-referencing.
    """
    # Determine if single-column or multi-column
    first_val = next(iter(metrics.values()))
    multi_column = isinstance(first_val, dict)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")

    if multi_column:
        col_names = list(metrics.keys())
        # Collect all metric names across columns
        all_metric_names = []
        for col_dict in metrics.values():
            for k in col_dict.keys():
                if k not in all_metric_names:
                    all_metric_names.append(k)

        n_cols = len(col_names)
        col_spec = "l" + "r" * n_cols
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        header = "Metric & " + " & ".join(col_names) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        for metric_name in all_metric_names:
            vals = []
            for col in col_names:
                v = metrics[col].get(metric_name, None)
                if v is None:
                    vals.append("--")
                elif isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            row = f"{metric_name} & " + " & ".join(vals) + " \\\\"
            lines.append(row)
    else:
        lines.append("\\begin{tabular}{lr}")
        lines.append("\\toprule")
        lines.append("Metric & Value \\\\")
        lines.append("\\midrule")
        for k, v in metrics.items():
            if isinstance(v, float):
                val_str = f"{v:.4f}"
            else:
                val_str = str(v)
            lines.append(f"{k} & {val_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_str = "\n".join(lines)

    # Ensure directory exists
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(filename, 'w') as f:
        f.write(latex_str)

    print(f"LaTeX table saved to: {filename}")
    return latex_str


# ---------------------------------------------------------------------------
# Rolling metrics computation
# ---------------------------------------------------------------------------

def compute_rolling_metrics(returns: pd.Series,
                            window: int = 252) -> pd.DataFrame:
    """
    Compute rolling portfolio risk/return metrics over a trailing window.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    window : int
        Rolling window in trading days (default 252 = ~1 year).

    Returns
    -------
    pd.DataFrame with columns:
        rolling_return, rolling_vol, rolling_sharpe, rolling_drawdown
    """
    if len(returns) < window:
        print(f"Warning: series length ({len(returns)}) < window ({window}). "
              f"Returning partial results.")

    # Rolling annualized return
    rolling_return = returns.rolling(window).mean() * 252

    # Rolling annualized volatility
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)

    # Rolling Sharpe ratio
    rolling_sharpe = rolling_return / rolling_vol.replace(0, np.nan)

    # Rolling maximum drawdown
    cum = (1 + returns).cumprod()
    rolling_dd = pd.Series(index=returns.index, dtype=float)
    for i in range(window - 1, len(returns)):
        window_cum = cum.iloc[i - window + 1: i + 1]
        peak = window_cum.cummax()
        dd = (window_cum / peak - 1).min()
        rolling_dd.iloc[i] = dd

    result = pd.DataFrame({
        'rolling_return': rolling_return,
        'rolling_vol': rolling_vol,
        'rolling_sharpe': rolling_sharpe,
        'rolling_drawdown': rolling_dd,
    }, index=returns.index)

    return result


# ---------------------------------------------------------------------------
# Regime-based evaluation
# ---------------------------------------------------------------------------

def regime_split_evaluation(returns: pd.Series,
                            prices: pd.DataFrame,
                            macro: pd.DataFrame,
                            vol_threshold_pctile: int = 75,
                            vol_lookback: int = 63) -> dict:
    """
    Split the OOS period into high-volatility and low-volatility regimes
    and compute separate performance metrics for each.

    The regime is determined by comparing the trailing realized volatility
    of SPY to a rolling percentile threshold. Days where trailing vol
    exceeds the threshold are classified as 'high_vol', and the rest as
    'low_vol'.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns (OOS).
    prices : pd.DataFrame
        Full price history (must contain 'SPY').
    macro : pd.DataFrame
        Macro data (used for VIX if available, currently informational).
    vol_threshold_pctile : int
        Percentile of rolling vol used to split regimes (default 75).
    vol_lookback : int
        Lookback window for realized vol calculation (default 63 = ~3 months).

    Returns
    -------
    dict with keys:
        'high_vol': dict of evaluate_oos metrics for high-vol days
        'low_vol': dict of evaluate_oos metrics for low-vol days
        'high_vol_pct': fraction of days in high-vol regime
        'low_vol_pct': fraction of days in low-vol regime
        'regime_series': pd.Series of 'high_vol'/'low_vol' labels
    """
    # Compute trailing realized vol for SPY
    if 'SPY' not in prices.columns:
        raise ValueError("prices must contain 'SPY' for regime classification")

    spy_daily = prices['SPY'].pct_change()
    trailing_vol = spy_daily.rolling(vol_lookback).std() * np.sqrt(252)

    # Compute percentile threshold on the full history (not just OOS)
    vol_threshold = trailing_vol.quantile(vol_threshold_pctile / 100.0)

    # Classify each day
    regime = pd.Series(index=trailing_vol.index, dtype=str)
    regime[trailing_vol >= vol_threshold] = 'high_vol'
    regime[trailing_vol < vol_threshold] = 'low_vol'

    # Align to the returns index
    common_idx = returns.index.intersection(regime.index)
    returns_aligned = returns.loc[common_idx]
    regime_aligned = regime.loc[common_idx]

    high_vol_mask = regime_aligned == 'high_vol'
    low_vol_mask = regime_aligned == 'low_vol'

    high_vol_returns = returns_aligned[high_vol_mask]
    low_vol_returns = returns_aligned[low_vol_mask]

    high_vol_metrics = evaluate_oos(high_vol_returns) if len(high_vol_returns) > 5 else {
        'sharpe': np.nan, 'sortino': np.nan, 'ann_return': np.nan,
        'ann_vol': np.nan, 'max_dd': np.nan, 'total': np.nan,
    }
    low_vol_metrics = evaluate_oos(low_vol_returns) if len(low_vol_returns) > 5 else {
        'sharpe': np.nan, 'sortino': np.nan, 'ann_return': np.nan,
        'ann_vol': np.nan, 'max_dd': np.nan, 'total': np.nan,
    }

    n_total = len(returns_aligned)
    high_vol_pct = high_vol_mask.sum() / n_total if n_total > 0 else 0.0
    low_vol_pct = low_vol_mask.sum() / n_total if n_total > 0 else 0.0

    # Print summary
    print(f"\n{'='*60}")
    print("REGIME-SPLIT EVALUATION")
    print(f"{'='*60}")
    print(f"  Volatility threshold: {vol_threshold:.2%} "
          f"(p{vol_threshold_pctile} of trailing {vol_lookback}d vol)")
    print(f"  High-vol days: {high_vol_mask.sum()} ({high_vol_pct:.1%})")
    print(f"  Low-vol days:  {low_vol_mask.sum()} ({low_vol_pct:.1%})")
    print()
    print(f"  {'Metric':<18} {'High-Vol':>12} {'Low-Vol':>12}")
    print(f"  {'-'*42}")
    for key in ['sharpe', 'sortino', 'ann_return', 'ann_vol', 'max_dd', 'total']:
        hv = high_vol_metrics.get(key, np.nan)
        lv = low_vol_metrics.get(key, np.nan)
        if key in ('ann_return', 'ann_vol', 'max_dd', 'total'):
            print(f"  {key:<18} {hv:>11.2%} {lv:>11.2%}")
        else:
            print(f"  {key:<18} {hv:>12.4f} {lv:>12.4f}")
    print(f"{'='*60}\n")

    return {
        'high_vol': high_vol_metrics,
        'low_vol': low_vol_metrics,
        'high_vol_pct': high_vol_pct,
        'low_vol_pct': low_vol_pct,
        'regime_series': regime_aligned,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_results(metrics, benchmarks, description=""):
    """Print formatted results summary."""
    print(f"\n{'='*60}")
    print(f"RESULTS{' — ' + description if description else ''}")
    print(f"{'='*60}")
    print(f"  Sharpe:      {metrics['sharpe']:.4f}  (EqWt={benchmarks['equal_weight']['sharpe']:.3f}, SPY={benchmarks['spy']['sharpe']:.3f}, 60/40={benchmarks['sixty_forty']['sharpe']:.3f})")
    print(f"  Sortino:     {metrics['sortino']:.4f}")
    print(f"  Ann Return:  {metrics['ann_return']:+.2%}")
    print(f"  Ann Vol:     {metrics['ann_vol']:.2%}")
    print(f"  Max DD:      {metrics['max_dd']:.2%}")
    print(f"  Total:       {metrics['total']:+.1%}")

    beats_eq = metrics['sharpe'] > benchmarks['equal_weight']['sharpe']
    beats_spy = metrics['sharpe'] > benchmarks['spy']['sharpe']
    beats_sf = metrics['sharpe'] > benchmarks['sixty_forty']['sharpe']
    print(f"  vs EqWt:     {'BEATS' if beats_eq else 'loses'}")
    print(f"  vs SPY:      {'BEATS' if beats_spy else 'loses'}")
    print(f"  vs 60/40:    {'BEATS' if beats_sf else 'loses'}")


def print_detailed_results(detailed_metrics: dict, label: str = ""):
    """
    Pretty-print the full set of detailed metrics produced by
    evaluate_oos_detailed.
    """
    print(f"\n{'='*60}")
    print(f"DETAILED RESULTS{' — ' + label if label else ''}")
    print(f"{'='*60}")

    sections = {
        'Return/Risk': ['sharpe', 'sortino', 'ann_return', 'ann_vol', 'max_dd', 'total'],
        'Relative': ['tracking_error', 'information_ratio', 'beta', 'alpha'],
        'Capture': ['up_capture', 'down_capture'],
        'Tail Risk': ['var_95', 'cvar_95'],
        'Distribution': ['skewness', 'kurtosis'],
        'Win/Loss': ['winning_pct', 'avg_win', 'avg_loss', 'win_loss_ratio'],
    }

    for section_name, keys in sections.items():
        print(f"\n  [{section_name}]")
        for k in keys:
            v = detailed_metrics.get(k, None)
            if v is None:
                continue
            if k in ('ann_return', 'ann_vol', 'max_dd', 'total',
                     'tracking_error', 'var_95', 'cvar_95', 'avg_win', 'avg_loss'):
                print(f"    {k:<22} {v:>10.4%}")
            elif k in ('up_capture', 'down_capture', 'winning_pct'):
                print(f"    {k:<22} {v:>10.1f}%")
            else:
                print(f"    {k:<22} {v:>10.4f}")

    print(f"{'='*60}\n")


def print_model_eval(model_metrics: dict, label: str = ""):
    """
    Pretty-print the model-level prediction metrics from
    evaluate_model_predictions.
    """
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION{' — ' + label if label else ''}")
    print(f"{'='*60}")
    for k, v in model_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<25} {v:>10.6f}")
        else:
            print(f"  {k:<25} {str(v):>10}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_data():
    """Load all prepared data. Returns (prices, features, fwd_returns)."""
    prices = pd.read_csv(os.path.join(RAW_DIR, 'prices.csv'), index_col=0, parse_dates=True).ffill().bfill()
    features = pd.read_csv(os.path.join(PROCESSED_DIR, 'features.csv'), index_col=0, parse_dates=True)
    fwd_returns = prices.shift(-PREDICTION_HORIZON) / prices - 1
    return prices, features, fwd_returns


def load_macro():
    """Load raw macro data. Returns pd.DataFrame."""
    path = os.path.join(RAW_DIR, 'macro.csv')
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(
            f"Macro data not found at {path}. Run 'python prepare.py' first."
        )


# ---------------------------------------------------------------------------
# Main: one-time data preparation
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-fetch', action='store_true', help='Skip fetching, just rebuild features')
    args = parser.parse_args()

    os.environ.setdefault('FRED_API_KEY', 'd6995d762b3aed1ddd40e8ae0bdeb08a')

    if not args.no_fetch:
        prices = fetch_prices()
        macro = fetch_macro()
    else:
        prices = pd.read_csv(os.path.join(RAW_DIR, 'prices.csv'), index_col=0, parse_dates=True)
        macro = pd.read_csv(os.path.join(RAW_DIR, 'macro.csv'), index_col=0, parse_dates=True)

    features = build_features(prices, macro)
    print(f"\nData ready. Features: {features.shape}")
    print(f"Train period: up to {TRAIN_END}")
    print(f"OOS period: {OOS_START} onwards")
    print(f"\nRun experiments with: python train.py")
