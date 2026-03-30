"""
Feature Importance, Transaction Cost Sensitivity, and OOS Robustness Analysis
=============================================================================
Part 1: Feature importance deep-dive for LightGBM macro-only model
Part 2: Transaction cost sensitivity analysis
Part 3: Robustness across different OOS periods

Author : Arhan Subasi
"""

import sys, os, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.covariance import LedoitWolf
from lightgbm import LGBMRegressor
import cvxpy as cp

from src.config import (
    RAW_DIR, PROCESSED_DIR, RESULTS_DIR, TICKER_LIST,
    RANDOM_STATE, FRED_SERIES, LATEX_FONT_SETTINGS, FIGURE_DPI,
)

plt.rcParams.update(LATEX_FONT_SETTINGS)

# ============================================================
# GLOBAL CONFIG — best strategy: LightGBM macro, lambda=5, maxW=0.5
# ============================================================
FOCUS_TICKERS = ['SPY', 'AGG', 'GLD', 'TLT']
PREDICTION_HORIZON = 21
RISK_AVERSION = 5.0
MAX_WEIGHT = 0.50
REBALANCE_FREQ = 21
TRAIN_WINDOW = 252 * 3
TEST_WINDOW = 63
CV_GAP = 21
N_CV_FOLDS = 5
BACKTEST_START = 252 * 3
LOOKBACK = 252
SPY_SHARPE_BENCHMARK = 0.671  # SPY buy-and-hold Sharpe (2022-2024 OOS)

MODEL_TEMPLATE = LGBMRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    num_leaves=31, verbose=-1, random_state=RANDOM_STATE,
)

# ============================================================
# HELPER: identify macro-only feature columns
# ============================================================
FRED_IDS = list(FRED_SERIES.keys())

def get_macro_cols(features_df):
    """Return only macro-related columns (FRED series + their lags)."""
    cols = features_df.columns.tolist()
    macro = [c for c in cols if any(c.startswith(fid) for fid in FRED_IDS)
             and '_ret_' not in c]
    return macro

def get_fred_display_name(col_name):
    """Convert feature column name to readable FRED indicator name."""
    # Strip lag suffix
    base = col_name
    for suffix in ['_lag1', '_lag5', '_lag21']:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            lag = suffix.replace('_lag', 'L')
            return f"{FRED_SERIES.get(base, base)} ({lag})"
    return FRED_SERIES.get(base, base)


# ============================================================
# HELPER: train LightGBM per-asset on given date split
# ============================================================
def train_models_for_period(features, prices, train_end_date, tickers=None):
    """Train LightGBM macro-only models for each ticker using data up to train_end_date.

    Returns dict: {ticker: {'model', 'scaler', 'feat_cols', 'ic'}}
    """
    if tickers is None:
        tickers = list(prices.columns)

    macro_cols = get_macro_cols(features)
    fwd_returns = prices.shift(-PREDICTION_HORIZON) / prices - 1

    models = {}
    for ticker in tickers:
        if ticker not in fwd_returns.columns:
            continue

        X = features[macro_cols]
        y = fwd_returns[ticker].reindex(X.index)
        mask = X.notna().all(axis=1) & y.notna()
        X_c, y_c = X[mask], y[mask]

        # Only use training data up to train_end_date
        train_mask = X_c.index <= train_end_date
        X_train = X_c[train_mask]
        y_train = y_c[train_mask]

        if len(X_train) < TRAIN_WINDOW:
            continue

        # Cross-validation for IC
        fold_ics = []
        for i in range(N_CV_FOLDS):
            tr_end = TRAIN_WINDOW + i * TEST_WINDOW
            te_start = tr_end + CV_GAP
            te_end = te_start + TEST_WINDOW
            if te_end > len(X_train):
                break
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train.iloc[:tr_end])
            X_te = scaler.transform(X_train.iloc[te_start:te_end])
            m = clone(MODEL_TEMPLATE)
            m.fit(X_tr, y_train.iloc[:tr_end])
            pred = m.predict(X_te)
            actual = y_train.iloc[te_start:te_end].values
            ic = np.corrcoef(actual, pred)[0, 1] if len(actual) > 2 else 0
            fold_ics.append(ic)

        avg_ic = np.mean(fold_ics) if fold_ics else 0

        # Train final model on all training data
        scaler = StandardScaler()
        X_all = scaler.fit_transform(X_train)
        final = clone(MODEL_TEMPLATE)
        final.fit(X_all, y_train)

        models[ticker] = {
            'model': final, 'scaler': scaler, 'feat_cols': macro_cols,
            'ic': avg_ic, 'n_train': len(X_train),
        }

    return models


# ============================================================
# HELPER: backtest with given TC
# ============================================================
def run_backtest(features, prices, models, oos_start, oos_end, tc_bps):
    """Run walk-forward backtest with ML predictions and MV optimization.

    Returns: pd.Series of daily portfolio returns.
    """
    macro_cols = get_macro_cols(features)
    tickers = list(prices.columns)
    daily_ret = prices.pct_change().dropna()

    # Filter to OOS period
    oos_dates = daily_ret.loc[oos_start:oos_end].index
    if len(oos_dates) == 0:
        return pd.Series(dtype=float)

    n_assets = len(tickers)
    w = np.ones(n_assets) / n_assets
    port_rets = []

    # Find start index in daily_ret
    all_dates = daily_ret.index
    start_idx = all_dates.get_indexer([oos_dates[0]], method='nearest')[0]
    end_idx = all_dates.get_indexer([oos_dates[-1]], method='nearest')[0]

    for t in range(start_idx, end_idx + 1):
        day_ret = daily_ret.iloc[t].values
        date = all_dates[t]

        if (t - start_idx) % REBALANCE_FREQ == 0:
            # Get ML predictions
            mu = np.zeros(n_assets)
            if date in features.index:
                row = features.loc[[date], macro_cols]
            else:
                idx = features.index.get_indexer([date], method='nearest')[0]
                row = features.iloc[[idx]][macro_cols]

            for j, ticker in enumerate(tickers):
                if ticker in models:
                    m = models[ticker]
                    X_s = m['scaler'].transform(row)
                    mu[j] = m['model'].predict(X_s)[0] * (252 / PREDICTION_HORIZON)

            # Covariance
            window = daily_ret.iloc[max(0, t - LOOKBACK):t]
            try:
                cov = LedoitWolf().fit(window.dropna()).covariance_ * 252
            except Exception:
                cov = window.cov().values * 252

            # MV Optimize
            ww = cp.Variable(n_assets)
            prob = cp.Problem(
                cp.Maximize(mu @ ww - (RISK_AVERSION / 2) * cp.quad_form(ww, cov)),
                [cp.sum(ww) == 1, ww >= 0, ww <= MAX_WEIGHT]
            )
            prob.solve(solver=cp.OSQP, verbose=False)

            if prob.status in ('optimal', 'optimal_inaccurate') and ww.value is not None:
                new_w = ww.value
                turnover = np.sum(np.abs(new_w - w))
                tc = turnover * tc_bps / 10000
                w = new_w
            else:
                tc = 0
        else:
            tc = 0

        port_ret = np.sum(w * day_ret) - tc
        port_rets.append({'date': all_dates[t], 'return': port_ret})
        w = w * (1 + day_ret)
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum

    rets = pd.DataFrame(port_rets).set_index('date')['return']
    return rets


def compute_metrics(rets):
    """Compute key backtest metrics."""
    if len(rets) == 0:
        return {'sharpe': np.nan, 'ann_return': np.nan, 'total_return': np.nan,
                'ann_vol': np.nan, 'max_dd': np.nan, 'sortino': np.nan}
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + rets).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    sortino_down = rets[rets < 0].std() * np.sqrt(252)
    sortino = ann_ret / sortino_down if sortino_down > 0 else 0
    total = cum.iloc[-1] - 1
    return {
        'sharpe': sharpe, 'ann_return': ann_ret, 'total_return': total,
        'ann_vol': ann_vol, 'max_dd': max_dd, 'sortino': sortino,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()

    # Load data
    print("Loading data...")
    prices = pd.read_csv(RAW_DIR / 'prices.csv', index_col=0, parse_dates=True).ffill().bfill()
    features = pd.read_csv(PROCESSED_DIR / 'features.csv', index_col=0, parse_dates=True)
    macro_cols = get_macro_cols(features)
    print(f"  Prices: {prices.shape}, Features: {features.shape}")
    print(f"  Macro features: {len(macro_cols)}")

    # ================================================================
    # PART 1: Feature Importance Deep-Dive
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 1: FEATURE IMPORTANCE DEEP-DIVE")
    print("=" * 70)

    train_end = '2021-12-31'
    models_fi = train_models_for_period(features, prices, train_end, tickers=FOCUS_TICKERS)

    # Extract gain-based feature importance
    importance_dict = {}
    for ticker in FOCUS_TICKERS:
        if ticker not in models_fi:
            print(f"  WARNING: No model for {ticker}")
            continue
        m = models_fi[ticker]['model']
        imp = m.feature_importances_  # gain-based by default
        imp_df = pd.DataFrame({
            'feature': macro_cols,
            'importance': imp,
        }).sort_values('importance', ascending=False)
        importance_dict[ticker] = imp_df
        print(f"  {ticker}: IC={models_fi[ticker]['ic']:+.4f}, "
              f"top feature={imp_df.iloc[0]['feature']} "
              f"(gain={imp_df.iloc[0]['importance']:.0f})")

    # Average importance across assets
    all_imp = pd.DataFrame({'feature': macro_cols})
    for ticker in FOCUS_TICKERS:
        if ticker in importance_dict:
            imp_vals = importance_dict[ticker].set_index('feature')['importance']
            all_imp[ticker] = all_imp['feature'].map(imp_vals)
    asset_cols = [c for c in all_imp.columns if c != 'feature']
    all_imp['avg_importance'] = all_imp[asset_cols].mean(axis=1)
    all_imp = all_imp.sort_values('avg_importance', ascending=False)

    print("\n  Top 10 most important FRED indicators (avg across assets):")
    for i, row in all_imp.head(10).iterrows():
        name = get_fred_display_name(row['feature'])
        print(f"    {name:40s}  avg_gain={row['avg_importance']:.1f}")

    # --- Figure 24: Feature Importance ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, ticker in enumerate(FOCUS_TICKERS):
        ax = axes[idx]
        if ticker not in importance_dict:
            ax.set_title(f"{ticker} — no model")
            continue
        imp_df = importance_dict[ticker].head(15).iloc[::-1]  # reverse for horizontal bar
        labels = [get_fred_display_name(f) for f in imp_df['feature']]
        ic_val = models_fi[ticker]['ic']

        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(imp_df)))[::-1]
        ax.barh(range(len(imp_df)), imp_df['importance'].values, color=colors)
        ax.set_yticks(range(len(imp_df)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Feature Importance (Gain)')
        ax.set_title(f'{ticker}  (IC = {ic_val:+.3f})', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle('Feature Importance — LightGBM Macro-Only Model (Top 15 per Asset)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'fig24_feature_importance.pdf',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved fig24_feature_importance.pdf")

    # ================================================================
    # PART 2: Transaction Cost Sensitivity
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 2: TRANSACTION COST SENSITIVITY")
    print("=" * 70)

    tc_levels = [0, 2, 5, 10, 15, 20, 30, 50]
    oos_start = '2022-01-01'
    oos_end = '2024-12-31'

    # Train models for OOS period
    models_tc = train_models_for_period(features, prices, train_end, tickers=list(prices.columns))
    print(f"  Models trained: {len(models_tc)}")

    tc_results = []
    for tc in tc_levels:
        print(f"  Running backtest with TC = {tc} bps...", end='')
        rets = run_backtest(features, prices, models_tc, oos_start, oos_end, tc)
        m = compute_metrics(rets)
        tc_results.append({
            'tc_bps': tc,
            'sharpe': m['sharpe'],
            'ann_return': m['ann_return'],
            'total_return': m['total_return'],
            'ann_vol': m['ann_vol'],
            'max_dd': m['max_dd'],
            'sortino': m['sortino'],
        })
        print(f" Sharpe={m['sharpe']:.3f}, Return={m['ann_return']:.2%}")

    tc_df = pd.DataFrame(tc_results)
    print(f"\n  Transaction Cost Sensitivity Results:")
    print(tc_df.to_string(index=False))

    # Find break-even TC where Sharpe drops below SPY
    breakeven_tc = None
    for i in range(len(tc_df) - 1):
        if tc_df.iloc[i]['sharpe'] >= SPY_SHARPE_BENCHMARK and tc_df.iloc[i + 1]['sharpe'] < SPY_SHARPE_BENCHMARK:
            # Linear interpolation
            tc1, s1 = tc_df.iloc[i]['tc_bps'], tc_df.iloc[i]['sharpe']
            tc2, s2 = tc_df.iloc[i + 1]['tc_bps'], tc_df.iloc[i + 1]['sharpe']
            if s2 != s1:
                breakeven_tc = tc1 + (SPY_SHARPE_BENCHMARK - s1) * (tc2 - tc1) / (s2 - s1)
            break

    if breakeven_tc is None:
        # Check if always above or always below
        if tc_df['sharpe'].iloc[0] < SPY_SHARPE_BENCHMARK:
            breakeven_tc = 0
            print(f"\n  Strategy never beats SPY benchmark (Sharpe {SPY_SHARPE_BENCHMARK})")
        else:
            breakeven_tc = tc_df['tc_bps'].max()
            print(f"\n  Strategy beats SPY at all tested TC levels! Break-even > {breakeven_tc} bps")
    else:
        print(f"\n  Break-even TC: ~{breakeven_tc:.1f} bps (where Sharpe drops below SPY's {SPY_SHARPE_BENCHMARK})")

    # --- Figure 25: TC Sensitivity ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tc_df['tc_bps'], tc_df['sharpe'], 'b-o', linewidth=2, markersize=8,
            label='LightGBM Macro Strategy', zorder=3)
    ax.axhline(y=SPY_SHARPE_BENCHMARK, color='red', linestyle='--', linewidth=1.5,
               label=f'SPY Buy-and-Hold (Sharpe = {SPY_SHARPE_BENCHMARK:.3f})', zorder=2)

    if breakeven_tc is not None and 0 < breakeven_tc < tc_df['tc_bps'].max():
        # Interpolate Sharpe at breakeven point
        ax.plot(breakeven_tc, SPY_SHARPE_BENCHMARK, 'r*', markersize=15, zorder=5,
                label=f'Break-even ({breakeven_tc:.0f} bps)')
        ax.annotate(f'{breakeven_tc:.0f} bps',
                    xy=(breakeven_tc, SPY_SHARPE_BENCHMARK),
                    xytext=(breakeven_tc + 3, SPY_SHARPE_BENCHMARK + 0.05),
                    fontsize=10, fontweight='bold', color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.fill_between(tc_df['tc_bps'], tc_df['sharpe'], SPY_SHARPE_BENCHMARK,
                    where=tc_df['sharpe'] >= SPY_SHARPE_BENCHMARK,
                    alpha=0.15, color='green', label='Outperformance region')
    ax.fill_between(tc_df['tc_bps'], tc_df['sharpe'], SPY_SHARPE_BENCHMARK,
                    where=tc_df['sharpe'] < SPY_SHARPE_BENCHMARK,
                    alpha=0.15, color='red', label='Underperformance region')

    ax.set_xlabel('Transaction Cost (bps)', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Transaction Cost Sensitivity — LightGBM Macro Strategy',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, max(tc_levels) + 2)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'fig25_tc_sensitivity.pdf',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved fig25_tc_sensitivity.pdf")

    # --- Table: TC Sensitivity (LaTeX) ---
    tex_rows = []
    tex_rows.append(r"\begin{table}[htbp]")
    tex_rows.append(r"\centering")
    tex_rows.append(r"\caption{Transaction Cost Sensitivity — LightGBM Macro Strategy (OOS 2022--2024)}")
    tex_rows.append(r"\label{tab:tc_sensitivity}")
    tex_rows.append(r"\begin{tabular}{rrrrrr}")
    tex_rows.append(r"\toprule")
    tex_rows.append(r"TC (bps) & Sharpe & Ann.\ Return & Total Return & Ann.\ Vol & Max DD \\")
    tex_rows.append(r"\midrule")
    for _, row in tc_df.iterrows():
        tex_rows.append(
            f"{int(row['tc_bps'])} & {row['sharpe']:.3f} & "
            f"{row['ann_return']:.2%} & {row['total_return']:.2%} & "
            f"{row['ann_vol']:.2%} & {row['max_dd']:.2%} \\\\"
        )
    tex_rows.append(r"\midrule")
    if breakeven_tc is not None:
        tex_rows.append(f"\\multicolumn{{6}}{{l}}{{Break-even TC $\\approx$ {breakeven_tc:.0f} bps "
                        f"(SPY Sharpe = {SPY_SHARPE_BENCHMARK:.3f})}} \\\\")
    tex_rows.append(r"\bottomrule")
    tex_rows.append(r"\end{tabular}")
    tex_rows.append(r"\end{table}")

    tex_content = "\n".join(tex_rows)
    with open(RESULTS_DIR / 'table_tc_sensitivity.tex', 'w') as f:
        f.write(tex_content)
    print(f"  Saved table_tc_sensitivity.tex")

    # ================================================================
    # PART 3: Robustness — Different OOS Periods
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 3: ROBUSTNESS — DIFFERENT OOS PERIODS")
    print("=" * 70)

    periods = {
        'A: 2018-2021\n(Pre-COVID + Recovery)': ('2018-01-01', '2021-12-31', '2017-12-31'),
        'B: 2020-2022\n(COVID + Rate Hikes)': ('2020-01-01', '2022-12-31', '2019-12-31'),
        'C: 2022-2024\n(Main OOS)': ('2022-01-01', '2024-12-31', '2021-12-31'),
    }

    TC_BPS_ROBUST = 10  # use 10 bps for robustness tests

    robustness_results = []
    for period_name, (oos_s, oos_e, train_e) in periods.items():
        print(f"\n  Period {period_name.split(chr(10))[0]}: train until {train_e}, test {oos_s} to {oos_e}")

        # Train models for this period
        models_r = train_models_for_period(features, prices, train_e, tickers=list(prices.columns))
        print(f"    Models trained: {len(models_r)}")

        # Run backtest
        rets = run_backtest(features, prices, models_r, oos_s, oos_e, TC_BPS_ROBUST)
        m = compute_metrics(rets)

        # SPY benchmark for same period
        spy_rets = prices['SPY'].pct_change().dropna()
        spy_oos = spy_rets.loc[oos_s:oos_e]
        spy_m = compute_metrics(spy_oos)

        robustness_results.append({
            'period': period_name,
            'period_short': period_name.split('\n')[0],
            'oos_start': oos_s,
            'oos_end': oos_e,
            'sharpe': m['sharpe'],
            'ann_return': m['ann_return'],
            'total_return': m['total_return'],
            'ann_vol': m['ann_vol'],
            'max_dd': m['max_dd'],
            'sortino': m['sortino'],
            'spy_sharpe': spy_m['sharpe'],
            'spy_ann_return': spy_m['ann_return'],
            'spy_total_return': spy_m['total_return'],
            'excess_sharpe': m['sharpe'] - spy_m['sharpe'],
        })
        print(f"    Strategy: Sharpe={m['sharpe']:.3f}, Return={m['ann_return']:.2%}")
        print(f"    SPY:      Sharpe={spy_m['sharpe']:.3f}, Return={spy_m['ann_return']:.2%}")
        print(f"    Excess Sharpe: {m['sharpe'] - spy_m['sharpe']:+.3f}")

    robust_df = pd.DataFrame(robustness_results)
    robust_df.to_csv(RESULTS_DIR / 'robustness_results.csv', index=False)
    print(f"\n  Saved robustness_results.csv")

    # --- Figure 26: Robustness Periods ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(robust_df))
    width = 0.35

    bars1 = ax.bar(x - width / 2, robust_df['sharpe'], width,
                   label='LightGBM Macro Strategy', color='#2171b5', edgecolor='white', zorder=3)
    bars2 = ax.bar(x + width / 2, robust_df['spy_sharpe'], width,
                   label='SPY Buy-and-Hold', color='#d94801', alpha=0.7, edgecolor='white', zorder=3)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#d94801')

    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Strategy Robustness Across OOS Periods (TC = 10 bps)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(robust_df['period'].values, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylim(bottom=min(0, robust_df[['sharpe', 'spy_sharpe']].min().min() - 0.1))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'fig26_robustness_periods.pdf',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved fig26_robustness_periods.pdf")

    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"ALL DONE in {elapsed:.0f}s")
    print("=" * 70)
    print(f"  fig24_feature_importance.pdf")
    print(f"  fig25_tc_sensitivity.pdf")
    print(f"  fig26_robustness_periods.pdf")
    print(f"  table_tc_sensitivity.tex")
    print(f"  robustness_results.csv")


if __name__ == '__main__':
    main()
