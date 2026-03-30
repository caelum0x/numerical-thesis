"""
Turnover-Constrained Portfolio Optimization
============================================
Tests the best strategy (LightGBM macro, lambda=5, maxW=0.5) under
progressively tighter turnover constraints:
    - No constraint (baseline)
    - Max 30% turnover per rebalance
    - Max 20% turnover
    - Max 10% turnover
    - Max 5% turnover

Implements the L1 turnover constraint in CVXPY:
    cp.norm(w_new - w_old, 1) <= max_turnover

Outputs
-------
    fig28_turnover_constraint.pdf  – dual-axis Sharpe vs avg turnover
    turnover_results.csv           – full results table
    table_turnover.tex             – LaTeX table

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
PREDICTION_HORIZON = 21
RISK_AVERSION = 5.0
MAX_WEIGHT = 0.50
REBALANCE_FREQ = 21
TRAIN_WINDOW = 252 * 3
LOOKBACK = 252
TC_BPS = 10  # 10 bps transaction cost

MODEL_TEMPLATE = LGBMRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    num_leaves=31, verbose=-1, random_state=RANDOM_STATE,
)

# Turnover constraint levels to test (L1 norm)
# None = unconstrained; otherwise the total absolute weight change limit
TURNOVER_LEVELS = [None, 0.30, 0.20, 0.10, 0.05]
TURNOVER_LABELS = ['None', '30%', '20%', '10%', '5%']

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


# ============================================================
# HELPER: train LightGBM per-asset
# ============================================================
def train_models(features, prices, train_end_date):
    """Train LightGBM macro-only models for each ticker up to train_end_date."""
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

        train_mask = X_c.index <= train_end_date
        X_train = X_c[train_mask]
        y_train = y_c[train_mask]

        if len(X_train) < TRAIN_WINDOW:
            continue

        scaler = StandardScaler()
        X_all = scaler.fit_transform(X_train)
        final = clone(MODEL_TEMPLATE)
        final.fit(X_all, y_train)

        models[ticker] = {
            'model': final, 'scaler': scaler, 'feat_cols': macro_cols,
        }

    return models


# ============================================================
# BACKTEST with turnover constraint
# ============================================================
def run_backtest_turnover(features, prices, models, oos_start, oos_end,
                          tc_bps, max_turnover=None):
    """Walk-forward backtest with ML predictions, MV optimization, and
    an optional L1 turnover constraint.

    Parameters
    ----------
    max_turnover : float or None
        If None, no turnover constraint is applied (baseline).
        Otherwise, the L1-norm of weight changes is capped at this value.

    Returns
    -------
    dict with keys: 'returns', 'turnovers', 'tc_paid', 'weight_history'
    """
    macro_cols = get_macro_cols(features)
    tickers = list(prices.columns)
    daily_ret = prices.pct_change().dropna()

    oos_dates = daily_ret.loc[oos_start:oos_end].index
    if len(oos_dates) == 0:
        return {'returns': pd.Series(dtype=float), 'turnovers': [],
                'tc_paid': [], 'weight_history': []}

    n_assets = len(tickers)
    w = np.ones(n_assets) / n_assets   # start equal-weight
    port_rets = []
    turnovers = []
    tc_paid_list = []

    all_dates = daily_ret.index
    start_idx = all_dates.get_indexer([oos_dates[0]], method='nearest')[0]
    end_idx = all_dates.get_indexer([oos_dates[-1]], method='nearest')[0]

    for t in range(start_idx, end_idx + 1):
        day_ret = daily_ret.iloc[t].values
        date = all_dates[t]

        if (t - start_idx) % REBALANCE_FREQ == 0:
            # --- ML predictions ---
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

            # --- Covariance (Ledoit-Wolf, annualised) ---
            window = daily_ret.iloc[max(0, t - LOOKBACK):t]
            try:
                cov = LedoitWolf().fit(window.dropna()).covariance_ * 252
            except Exception:
                cov = window.cov().values * 252

            # --- CVXPY optimisation with optional turnover constraint ---
            ww = cp.Variable(n_assets)
            objective = cp.Maximize(
                mu @ ww - (RISK_AVERSION / 2) * cp.quad_form(ww, cov)
            )
            constraints = [
                cp.sum(ww) == 1,
                ww >= 0,
                ww <= MAX_WEIGHT,
            ]

            # Add turnover constraint if specified
            if max_turnover is not None:
                constraints.append(cp.norm(ww - w, 1) <= max_turnover)

            prob = cp.Problem(objective, constraints)
            try:
                prob.solve(solver=cp.SCS, verbose=False)
            except Exception:
                pass

            if prob.status in ('optimal', 'optimal_inaccurate') and ww.value is not None:
                new_w = np.array(ww.value).flatten()
                # Clip small numerical artefacts
                new_w = np.maximum(new_w, 0.0)
                if new_w.sum() > 0:
                    new_w /= new_w.sum()
                turnover = np.sum(np.abs(new_w - w))
                tc = turnover * tc_bps / 10000.0
                w = new_w
                turnovers.append(turnover)
                tc_paid_list.append(tc)
            else:
                # Solver failed — keep current weights (no rebalance)
                turnovers.append(0.0)
                tc_paid_list.append(0.0)
                tc = 0.0
        else:
            tc = 0.0

        port_ret = np.sum(w * day_ret) - tc
        port_rets.append({'date': date, 'return': port_ret})

        # Drift weights
        w = w * (1 + day_ret)
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum

    rets = pd.DataFrame(port_rets).set_index('date')['return']
    return {
        'returns': rets,
        'turnovers': turnovers,
        'tc_paid': tc_paid_list,
    }


def compute_metrics(result_dict):
    """Compute key backtest metrics from the result dictionary."""
    rets = result_dict['returns']
    turnovers = result_dict['turnovers']
    tc_paid = result_dict['tc_paid']

    if len(rets) == 0:
        return {k: np.nan for k in [
            'sharpe', 'ann_return', 'ann_vol', 'max_dd',
            'avg_turnover', 'total_tc_paid', 'total_return', 'sortino']}

    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum = (1 + rets).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    downside = rets[rets < 0]
    down_std = downside.std() * np.sqrt(252) if len(downside) > 1 else np.nan
    sortino = ann_ret / down_std if (down_std and down_std > 0) else 0.0

    total_ret = cum.iloc[-1] - 1
    avg_to = np.mean(turnovers) if turnovers else 0.0
    total_tc = np.sum(tc_paid) if tc_paid else 0.0

    return {
        'sharpe': sharpe,
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'max_dd': max_dd,
        'avg_turnover': avg_to,
        'total_tc_paid': total_tc,
        'total_return': total_ret,
        'sortino': sortino,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    print("Loading data...")
    prices = pd.read_csv(RAW_DIR / 'prices.csv', index_col=0, parse_dates=True).ffill().bfill()
    features = pd.read_csv(PROCESSED_DIR / 'features.csv', index_col=0, parse_dates=True)
    macro_cols = get_macro_cols(features)
    print(f"  Prices: {prices.shape}, Features: {features.shape}")
    print(f"  Macro features: {len(macro_cols)}")
    print(f"  Tickers: {list(prices.columns)}")

    # ----------------------------------------------------------
    # Train models (once — same models for all turnover tests)
    # ----------------------------------------------------------
    train_end = '2021-12-31'
    oos_start = '2022-01-01'
    oos_end   = '2024-12-31'

    print(f"\nTraining LightGBM macro-only models (train <= {train_end})...")
    models = train_models(features, prices, train_end)
    print(f"  Models trained for {len(models)} assets: {list(models.keys())}")

    # ----------------------------------------------------------
    # Run backtests for each turnover constraint level
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("TURNOVER-CONSTRAINED OPTIMIZATION — OOS 2022-2024")
    print("=" * 70)
    print(f"  Strategy: LightGBM macro, lambda={RISK_AVERSION}, maxW={MAX_WEIGHT}")
    print(f"  TC: {TC_BPS} bps per trade")
    print(f"  Rebalance: every {REBALANCE_FREQ} days")
    print(f"  Constraint: cp.norm(w_new - w_old, 1) <= max_turnover")
    print()

    all_results = []

    for level, label in zip(TURNOVER_LEVELS, TURNOVER_LABELS):
        constraint_str = f"max_turnover={label}"
        print(f"  Running: {constraint_str} ...", end='', flush=True)

        result = run_backtest_turnover(
            features, prices, models, oos_start, oos_end,
            tc_bps=TC_BPS, max_turnover=level,
        )
        metrics = compute_metrics(result)

        metrics['constraint'] = label
        metrics['max_turnover_value'] = level if level is not None else np.inf
        all_results.append(metrics)

        print(f"  Sharpe={metrics['sharpe']:.3f}  "
              f"Return={metrics['ann_return']:.2%}  "
              f"AvgTO={metrics['avg_turnover']:.3f}  "
              f"TotalTC={metrics['total_tc_paid']:.4f}")

    results_df = pd.DataFrame(all_results)

    # ----------------------------------------------------------
    # Print summary table
    # ----------------------------------------------------------
    print("\n" + "-" * 80)
    print("SUMMARY TABLE")
    print("-" * 80)
    display_cols = ['constraint', 'sharpe', 'ann_return', 'ann_vol',
                    'max_dd', 'avg_turnover', 'total_tc_paid']
    print(results_df[display_cols].to_string(index=False, float_format='{:.4f}'.format))

    # ----------------------------------------------------------
    # Save CSV
    # ----------------------------------------------------------
    csv_path = RESULTS_DIR / 'turnover_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path}")

    # ----------------------------------------------------------
    # Figure 28: Dual-axis — Sharpe (left) and Avg Turnover (right)
    # ----------------------------------------------------------
    print("\nGenerating fig28_turnover_constraint.pdf ...")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x_labels = TURNOVER_LABELS
    x_pos = np.arange(len(x_labels))

    sharpes = results_df['sharpe'].values
    avg_tos = results_df['avg_turnover'].values

    # Left axis — Sharpe
    color_sharpe = '#1f77b4'
    ax1.bar(x_pos - 0.18, sharpes, width=0.35, color=color_sharpe,
            alpha=0.85, label='Sharpe Ratio', zorder=3)
    ax1.set_xlabel('Turnover Constraint (max L1 change per rebalance)', fontsize=12)
    ax1.set_ylabel('Sharpe Ratio', fontsize=12, color=color_sharpe)
    ax1.tick_params(axis='y', labelcolor=color_sharpe)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=11)
    ax1.grid(axis='y', alpha=0.3, zorder=0)

    # Annotate Sharpe values on bars
    for i, v in enumerate(sharpes):
        ax1.text(x_pos[i] - 0.18, v + 0.01, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold',
                 color=color_sharpe)

    # Right axis — Avg Turnover
    ax2 = ax1.twinx()
    color_to = '#d62728'
    ax2.bar(x_pos + 0.18, avg_tos, width=0.35, color=color_to,
            alpha=0.65, label='Avg Turnover', zorder=3)
    ax2.set_ylabel('Average Turnover per Rebalance', fontsize=12, color=color_to)
    ax2.tick_params(axis='y', labelcolor=color_to)

    # Annotate turnover values on bars
    for i, v in enumerate(avg_tos):
        ax2.text(x_pos[i] + 0.18, v + 0.005, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold',
                 color=color_to)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    ax1.set_title('Turnover Constraint Trade-off: Sharpe Ratio vs. Average Turnover\n'
                   '(LightGBM Macro, $\\lambda=5$, maxW=0.5, OOS 2022--2024)',
                   fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig_path = RESULTS_DIR / 'fig28_turnover_constraint.pdf'
    fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fig_path}")

    # ----------------------------------------------------------
    # LaTeX table
    # ----------------------------------------------------------
    print("\nGenerating table_turnover.tex ...")
    tex_rows = []
    tex_rows.append(r"\begin{table}[htbp]")
    tex_rows.append(r"\centering")
    tex_rows.append(r"\caption{Turnover-Constrained Optimization --- LightGBM Macro Strategy (OOS 2022--2024)}")
    tex_rows.append(r"\label{tab:turnover_constraint}")
    tex_rows.append(r"\begin{tabular}{lrrrrrr}")
    tex_rows.append(r"\toprule")
    tex_rows.append(r"Constraint & Sharpe & Ann.\ Return & Ann.\ Vol & Max DD & Avg Turnover & Total TC \\")
    tex_rows.append(r"\midrule")
    for _, row in results_df.iterrows():
        tex_rows.append(
            f"{row['constraint']} & {row['sharpe']:.3f} & "
            f"{row['ann_return']:.2%} & {row['ann_vol']:.2%} & "
            f"{row['max_dd']:.2%} & {row['avg_turnover']:.3f} & "
            f"{row['total_tc_paid']:.4f} \\\\"
        )
    tex_rows.append(r"\bottomrule")
    tex_rows.append(r"\end{tabular}")
    tex_rows.append(r"\begin{tablenotes}")
    tex_rows.append(r"\small")
    tex_rows.append(r"\item Turnover constraint: $\|w_{\text{new}} - w_{\text{old}}\|_1 \le \tau$. "
                    r"TC = 10 bps per unit turnover. Monthly rebalancing.")
    tex_rows.append(r"\end{tablenotes}")
    tex_rows.append(r"\end{table}")

    tex_path = RESULTS_DIR / 'table_turnover.tex'
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_rows))
    print(f"  Saved {tex_path}")

    # ----------------------------------------------------------
    # Wrap-up
    # ----------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print("=" * 70)
    print("OUTPUT FILES:")
    print(f"  {RESULTS_DIR / 'fig28_turnover_constraint.pdf'}")
    print(f"  {RESULTS_DIR / 'turnover_results.csv'}")
    print(f"  {RESULTS_DIR / 'table_turnover.tex'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
