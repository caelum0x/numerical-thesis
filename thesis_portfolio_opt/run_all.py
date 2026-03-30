"""
End-to-end orchestrator — connects autoresearch, MiroFish, and thesis pipeline.

Architecture:
    ┌─────────────────┐
    │  1. AutoResearch │  Autonomous ML experiment search
    │  train.py --batch│  → batch_results.csv (best model config)
    └────────┬────────┘
             │ best hyperparams
             ▼
    ┌─────────────────┐
    │  2. MiroFish     │  Multi-agent swarm simulation
    │  financial_sim   │  → agreement signal, regime, weights
    └────────┬────────┘
             │ risk overlay + features
             ▼
    ┌─────────────────┐
    │  3. Pipeline     │  Walk-forward backtest with:
    │  thesis_portfolio│    - AutoResearch best config
    │  _opt            │    - MiroFish risk scaling
    │                  │    - MiroFish swarm features
    └────────┬────────┘
             │ results
             ▼
    ┌─────────────────┐
    │  4. Comparison   │  ML-only vs ML+Swarm vs Swarm-only vs SPY
    │  & Figures       │  → LaTeX tables, PDFs, thesis_summary
    └─────────────────┘

Usage:
    python run_all.py                    # full end-to-end
    python run_all.py --autoresearch     # step 1 only
    python run_all.py --mirofish         # step 2 only
    python run_all.py --pipeline         # step 3 only
    python run_all.py --compare          # step 4 only
    python run_all.py --no-autoresearch  # skip step 1 (use cached results)
    python run_all.py --no-mirofish      # skip step 2 (use cached results)
"""

import os
import sys
import json
import time
import argparse
import subprocess
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
AUTORESEARCH_DIR = os.path.join(PROJECT_ROOT, '..', 'autoresearch')
MIROFISH_DIR = os.path.join(PROJECT_ROOT, '..', 'MiroFish')

sys.path.insert(0, PROJECT_ROOT)

from src.config import RAW_DIR, PROCESSED_DIR, RESULTS_DIR, TICKER_LIST


# ============================================================================
# Helpers
# ============================================================================

def run_command(cmd: str, cwd: str, description: str) -> subprocess.CompletedProcess:
    """Run a shell command with progress output."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for line in lines[-20:]:
            print(f"  {line}")
        print(f"  [{elapsed:.1f}s]")
    else:
        print(f"  ERROR (exit {result.returncode}):")
        print(result.stderr[-500:] if result.stderr else result.stdout[-500:])
    return result


def find_python() -> str:
    """Find Python executable — prefer project venv, fall back to system."""
    venv_py = os.path.join(PROJECT_ROOT, 'venv', 'bin', 'python3')
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable


# ============================================================================
# Step 1: AutoResearch — autonomous model search
# ============================================================================

def step_autoresearch() -> dict:
    """Run autoresearch batch experiments and return best config.

    AutoResearch iterates over 12+ model/feature/hyperparameter combinations,
    each running a strict OOS backtest (train≤2021, test 2022-2024).
    The best config is passed to the thesis pipeline.
    """
    print("\n" + "=" * 70)
    print("  STEP 1: AUTORESEARCH — Autonomous Model Search")
    print("=" * 70)

    py = find_python()
    run_command(f"{py} train.py --batch", cwd=AUTORESEARCH_DIR,
                description="Running 12 autonomous experiments (~2 min each)")

    # Read results via bridge
    from src.integration.autoresearch_bridge import AutoResearchBridge
    bridge = AutoResearchBridge()
    print(f"\n{bridge.summary()}")

    best = bridge.get_best_config()
    if best:
        return {
            'model_type': best.model_type,
            'risk_aversion': best.risk_aversion,
            'max_weight': best.max_weight,
            'rebalance_freq': best.rebalance_freq,
            'tc_bps': best.tc_bps,
            'shrinkage': best.shrinkage,
            'feature_set': best.feature_set,
            'sharpe': best.sharpe,
            'experiment': best.experiment_name,
        }
    return {}


# ============================================================================
# Step 2: MiroFish — multi-agent swarm simulation
# ============================================================================

def step_mirofish() -> dict:
    """Run MiroFish financial simulation and return swarm intelligence data.

    MiroFish spawns 14 agents (momentum, contrarian, macro, ML, adaptive,
    regime, noise) that generate predictions. Their agreement level serves
    as a real-time risk indicator for the optimizer.
    """
    print("\n" + "=" * 70)
    print("  STEP 2: MIROFISH — Multi-Agent Swarm Simulation")
    print("=" * 70)

    py = find_python()
    sim_script = os.path.join('backend', 'app', 'services', 'financial_simulator.py')
    run_command(f"{py} {sim_script}", cwd=MIROFISH_DIR,
                description="Running 14-agent financial simulation (35 rounds)")

    from src.integration.mirofish_bridge import MiroFishBridge
    bridge = MiroFishBridge()

    if bridge.is_available:
        print(f"\n{bridge.summary()}")
        return {
            'available': True,
            'mean_agreement': float(bridge.get_agreement_series().mean()),
            'mean_risk_scale': float(bridge.get_risk_scale_series().mean()),
        }

    print("  MiroFish simulation data not found — continuing without swarm overlay")
    return {'available': False}


# ============================================================================
# Step 3: Integrated Pipeline — walk-forward with both inputs
# ============================================================================

def step_pipeline(autoresearch_config: dict, mirofish_info: dict) -> pd.DataFrame:
    """Run the thesis walk-forward backtest with autoresearch + MiroFish inputs.

    Three strategies are compared:
        1. ML-only:      Best autoresearch model, standard optimizer
        2. ML+Swarm:     Best model + MiroFish risk overlay + swarm features
        3. Swarm-only:   MiroFish agent consensus weights directly
    """
    print("\n" + "=" * 70)
    print("  STEP 3: INTEGRATED PIPELINE — Walk-Forward Backtest")
    print("=" * 70)

    from src.integration.autoresearch_bridge import AutoResearchBridge
    from src.integration.mirofish_bridge import MiroFishBridge

    ar_bridge = AutoResearchBridge()
    mf_bridge = MiroFishBridge()

    # Load data
    prices = pd.read_csv(RAW_DIR / 'prices.csv', index_col=0, parse_dates=True).ffill().bfill()
    macro = pd.read_csv(RAW_DIR / 'macro.csv', index_col=0, parse_dates=True).ffill()
    features = pd.read_csv(PROCESSED_DIR / 'features.csv', index_col=0, parse_dates=True)

    # --- Strategy configs from autoresearch ---
    best = ar_bridge.get_best_config()
    risk_aversion = best.risk_aversion if best else 5.0
    max_weight = best.max_weight if best else 0.35
    rebalance_freq = best.rebalance_freq if best else 21
    tc_bps = best.tc_bps if best else 10
    shrinkage = best.shrinkage if best else 0.0

    print(f"\n  AutoResearch config: λ={risk_aversion}, maxW={max_weight}, "
          f"rebal={rebalance_freq}d, tc={tc_bps}bps, shrink={shrinkage}")

    # --- Inject MiroFish features ---
    if mf_bridge.is_available:
        features_enhanced = mf_bridge.inject_features(features)
        print(f"  Features: {features.shape[1]} base + "
              f"{features_enhanced.shape[1] - features.shape[1]} swarm = "
              f"{features_enhanced.shape[1]} total")
    else:
        features_enhanced = features

    # --- Run three backtests ---
    from sklearn.covariance import LedoitWolf

    oos_start = pd.Timestamp('2022-01-01')
    oos_prices = prices[prices.index >= oos_start]
    daily_ret = oos_prices.pct_change().dropna()
    tickers = list(prices.columns)
    n_assets = len(tickers)

    # Load trained models (from autoresearch or thesis pipeline)
    from src.integration._backtest_helpers import (
        load_best_models,
        run_integrated_backtest,
        compute_metrics,
    )

    models = load_best_models(RESULTS_DIR, tickers)

    # Strategy 1: ML-only (autoresearch best config, no swarm)
    print("\n  Running Strategy 1: ML-only (AutoResearch best)...")
    ml_returns = run_integrated_backtest(
        prices, daily_ret, models, features, tickers,
        risk_aversion=risk_aversion, max_weight=max_weight,
        rebalance_freq=rebalance_freq, tc_bps=tc_bps, shrinkage=shrinkage,
        risk_scale_fn=None,
    )

    # Strategy 2: ML + Swarm (autoresearch config + MiroFish risk overlay)
    if mf_bridge.is_available:
        print("  Running Strategy 2: ML + Swarm overlay...")
        ml_swarm_returns = run_integrated_backtest(
            prices, daily_ret, models, features_enhanced, tickers,
            risk_aversion=risk_aversion, max_weight=max_weight,
            rebalance_freq=rebalance_freq, tc_bps=tc_bps, shrinkage=shrinkage,
            risk_scale_fn=mf_bridge.get_risk_scale_at,
        )
    else:
        ml_swarm_returns = ml_returns.copy()

    # Strategy 3: Swarm-only (MiroFish consensus weights)
    if mf_bridge.is_available:
        print("  Running Strategy 3: Swarm-only (MiroFish weights)...")
        swarm_weights_df = mf_bridge.get_swarm_weights()
        swarm_returns = _run_swarm_only_backtest(daily_ret, swarm_weights_df, tickers, tc_bps)
    else:
        swarm_returns = pd.Series(dtype=float)

    # Benchmark: SPY buy-and-hold
    spy_returns = daily_ret['SPY'] if 'SPY' in daily_ret.columns else pd.Series(dtype=float)

    # Benchmark: Equal weight
    ew_returns = daily_ret.mean(axis=1)

    # --- Compile results ---
    all_strategies = {
        'ML_Only': ml_returns,
        'ML_Swarm': ml_swarm_returns,
    }
    if not swarm_returns.empty:
        all_strategies['Swarm_Only'] = swarm_returns
    all_strategies['SPY'] = spy_returns
    all_strategies['EqualWeight'] = ew_returns

    results = []
    for name, rets in all_strategies.items():
        if rets.empty:
            continue
        m = compute_metrics(rets)
        m['strategy'] = name
        results.append(m)

    results_df = pd.DataFrame(results).set_index('strategy')
    results_df.to_csv(RESULTS_DIR / 'integrated_comparison.csv')

    print("\n  INTEGRATED RESULTS:")
    print("  " + "=" * 65)
    for _, row in results_df.iterrows():
        star = ' ★' if row.name == results_df['sharpe'].idxmax() else '  '
        print(f"  {star} {row.name:15s}  Sharpe={row['sharpe']:+.3f}  "
              f"Return={row['ann_return']:+.1%}  MaxDD={row['max_dd']:.1%}  "
              f"Vol={row['ann_vol']:.1%}")

    return results_df


def _run_swarm_only_backtest(
    daily_ret: pd.DataFrame,
    swarm_weights_df: pd.DataFrame,
    tickers: list[str],
    tc_bps: int,
) -> pd.Series:
    """Backtest using only MiroFish swarm weights."""
    n = len(tickers)
    w = np.ones(n) / n
    port_rets = []

    for t in range(len(daily_ret)):
        date = daily_ret.index[t]
        day_ret = daily_ret.iloc[t].values

        # Check if we should rebalance (new swarm weights available)
        valid = swarm_weights_df.index[swarm_weights_df.index <= date]
        if len(valid) > 0:
            latest_row = swarm_weights_df.loc[valid[-1]]
            new_w = np.array([latest_row.get(ticker, 0.0) for ticker in tickers])
            total = new_w.sum()
            if total > 0:
                new_w = new_w / total
                turnover = np.sum(np.abs(new_w - w))
                tc = turnover * tc_bps / 10000
                w = new_w
            else:
                tc = 0
        else:
            tc = 0

        port_rets.append(np.sum(w * day_ret) - tc)
        w = w * (1 + day_ret)
        total = w.sum()
        if total > 0:
            w = w / total

    return pd.Series(port_rets, index=daily_ret.index)


# ============================================================================
# Step 4: Comparison figures
# ============================================================================

def step_compare() -> None:
    """Generate integrated comparison figures and LaTeX tables."""
    print("\n" + "=" * 70)
    print("  STEP 4: COMPARISON FIGURES & TABLES")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='whitegrid', palette='deep')

    # Load integrated results
    comp_path = RESULTS_DIR / 'integrated_comparison.csv'
    if not comp_path.exists():
        print("  No integrated results found. Run --pipeline first.")
        return

    results = pd.read_csv(comp_path, index_col=0)

    # --- Figure: Strategy comparison bar chart ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, title in zip(axes,
                                  ['sharpe', 'ann_return', 'max_dd'],
                                  ['Sharpe Ratio', 'Annualized Return', 'Max Drawdown']):
        vals = results[metric]
        colors = ['#2ecc71' if v == vals.max() else '#3498db' for v in vals]
        if metric == 'max_dd':
            colors = ['#2ecc71' if v == vals.max() else '#e74c3c' for v in vals]
        ax.barh(results.index, vals, color=colors)
        ax.set_title(title, fontsize=13)
        ax.axvline(x=0, color='gray', linewidth=0.5)

    fig.suptitle('Integrated Strategy Comparison (2022-2024 OOS)', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'fig35_integrated_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  fig35_integrated_comparison.pdf")

    # --- Figure: AutoResearch batch results ---
    from src.integration.autoresearch_bridge import AutoResearchBridge
    ar = AutoResearchBridge()
    ar_results = ar.get_all_results()
    if not ar_results.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        top20 = ar_results.head(20).sort_values('sharpe', ascending=True)
        colors = ['#2ecc71' if s > 0.671 else '#3498db' if s > 0.202 else '#e74c3c'
                  for s in top20['sharpe']]
        ax.barh(top20['experiment'], top20['sharpe'], color=colors)
        ax.axvline(x=0.202, color='gray', linestyle='--', linewidth=1, label='Equal Weight')
        ax.axvline(x=0.671, color='black', linestyle='--', linewidth=1, label='SPY')
        ax.set_title('AutoResearch: All Experiments Ranked by OOS Sharpe', fontsize=14)
        ax.legend()
        fig.savefig(RESULTS_DIR / 'fig36_autoresearch_all.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("  fig36_autoresearch_all.pdf")

    # --- Figure: MiroFish agreement timeline ---
    from src.integration.mirofish_bridge import MiroFishBridge
    mf = MiroFishBridge()
    if mf.is_available:
        agreement = mf.get_agreement_series()
        risk_scale = mf.get_risk_scale_series()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        ax1.plot(agreement.index, agreement.values, 'o-', color='steelblue', markersize=4)
        ax1.axhline(y=agreement.mean(), color='red', linestyle='--',
                     label=f'Mean: {agreement.mean():.3f}')
        ax1.fill_between(agreement.index, agreement.values, alpha=0.2, color='steelblue')
        ax1.set_ylabel('Agent Agreement')
        ax1.set_title('MiroFish Swarm Intelligence Signals (2022-2024)', fontsize=14)
        ax1.legend()

        ax2.plot(risk_scale.index, risk_scale.values, 's-', color='darkorange', markersize=4)
        ax2.axhline(y=1.0, color='gray', linestyle=':', label='Full allocation')
        ax2.axhline(y=risk_scale.mean(), color='red', linestyle='--',
                     label=f'Mean: {risk_scale.mean():.3f}')
        ax2.fill_between(risk_scale.index, risk_scale.values, alpha=0.2, color='darkorange')
        ax2.set_ylabel('Risk Scale Factor')
        ax2.set_xlabel('Date')
        ax2.legend()

        fig.tight_layout()
        fig.savefig(RESULTS_DIR / 'fig37_mirofish_signals.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("  fig37_mirofish_signals.pdf")

    # --- LaTeX table ---
    latex = results.to_latex(
        float_format='%.3f',
        caption='Integrated Strategy Comparison (Out-of-Sample 2022-2024)',
        label='tab:integrated_comparison',
    )
    with open(RESULTS_DIR / 'table_integrated_comparison.tex', 'w') as f:
        f.write(latex)
    print("  table_integrated_comparison.tex")

    print("\n  All integrated figures generated.")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description='End-to-end thesis orchestrator')
    parser.add_argument('--autoresearch', action='store_true', help='Run autoresearch only')
    parser.add_argument('--mirofish', action='store_true', help='Run MiroFish only')
    parser.add_argument('--pipeline', action='store_true', help='Run integrated backtest only')
    parser.add_argument('--compare', action='store_true', help='Generate comparison figures only')
    parser.add_argument('--no-autoresearch', action='store_true', help='Skip autoresearch (use cached)')
    parser.add_argument('--no-mirofish', action='store_true', help='Skip MiroFish (use cached)')
    args = parser.parse_args()

    run_specific = args.autoresearch or args.mirofish or args.pipeline or args.compare
    t0 = time.time()

    # Step 1: AutoResearch
    ar_config = {}
    if args.autoresearch or (not run_specific and not args.no_autoresearch):
        ar_config = step_autoresearch()
    elif not run_specific or args.pipeline:
        # Load cached results
        from src.integration.autoresearch_bridge import AutoResearchBridge
        bridge = AutoResearchBridge()
        best = bridge.get_best_config()
        if best:
            ar_config = {'model_type': best.model_type, 'sharpe': best.sharpe}
            print(f"\n  Using cached AutoResearch: {best.experiment_name} (Sharpe={best.sharpe:.3f})")

    # Step 2: MiroFish
    mf_info = {'available': False}
    if args.mirofish or (not run_specific and not args.no_mirofish):
        mf_info = step_mirofish()
    elif not run_specific or args.pipeline:
        from src.integration.mirofish_bridge import MiroFishBridge
        mf = MiroFishBridge()
        if mf.is_available:
            mf_info = {'available': True}
            print(f"\n  Using cached MiroFish: {mf.summary()}")

    # Step 3: Integrated pipeline
    if args.pipeline or not run_specific:
        step_pipeline(ar_config, mf_info)

    # Step 4: Comparison
    if args.compare or not run_specific:
        step_compare()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  ALL DONE ({elapsed:.0f}s)")
    print("=" * 70)

    # Final summary
    comp_path = RESULTS_DIR / 'integrated_comparison.csv'
    if comp_path.exists():
        results = pd.read_csv(comp_path, index_col=0)
        best_strategy = results['sharpe'].idxmax()
        best_sharpe = results.loc[best_strategy, 'sharpe']
        print(f"\n  Best strategy: {best_strategy} (Sharpe={best_sharpe:.3f})")

    pdfs = list(RESULTS_DIR.glob('fig*.pdf'))
    print(f"  Figures: {len(pdfs)} PDFs in {RESULTS_DIR}")


if __name__ == '__main__':
    main()
