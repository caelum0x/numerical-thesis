"""
Extended AutoResearch — 30 experiments guided by feedback loop.
Estimated runtime: 40-60 minutes (~2 min per experiment).

Experiments organized in 6 themes:
  1. Feedback-driven: elastic net, SVR, regime, lower λ (from feedback/latest.json)
  2. LGBM tuning: depth, n_estimators, learning_rate sweeps around best config
  3. Feature engineering: macro-only, momentum, PCA, feature selection
  4. Ensemble variations: weighted, stacked, top-3
  5. Constraint sweeps: max_weight, shrinkage, turnover
  6. Model zoo: GBR, AdaBoost, Ridge variants, MLP

Best so far: A1_lgbm_maxw50_tc5 Sharpe=0.832
Target: beat 0.832 with a robust, simple config
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Ensure imports work
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
)
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from train import run_experiment
from prepare import load_data, get_benchmarks, TICKER_LIST

# Feature selectors
all_feat = lambda df: [c for c in df.columns if '_ret_' not in c]
macro_only = lambda df: [c for c in df.columns
                         if '_ret_' not in c and '_vol_' not in c
                         and '_mom_' not in c and '_rsi_' not in c]
mom_vol = lambda df: [c for c in df.columns
                      if ('_vol_' in c or '_mom_' in c or '_rsi_' in c)]
macro_mom = lambda df: [c for c in df.columns if '_ret_' not in c and '_rsi_' not in c]
technical = lambda df: [c for c in df.columns
                        if ('_rsi_' in c or '_mom_' in c or '_vol_' in c
                            or '_beta_' in c)]


def build_experiments():
    """Build the full experiment list — 30 experiments."""
    experiments = []

    # ══════════════════════════════════════════════════════════
    # Theme 1: FEEDBACK-DRIVEN (from feedback/latest.json)
    # ══════════════════════════════════════════════════════════

    # C1: ElasticNet — untried model from feedback
    experiments.append(
        ("C1_elastic_macro", "ElasticNet macro λ=5 maxW=0.5",
         [("elastic", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000))],
         macro_only, 5.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # C2: ElasticNet tuned
    experiments.append(
        ("C2_elastic_macro_l1_80", "ElasticNet l1=0.8 macro λ=5 maxW=0.5",
         [("elastic", ElasticNet(alpha=0.0005, l1_ratio=0.8, max_iter=5000))],
         macro_only, 5.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # C3: SVR — untried model
    experiments.append(
        ("C3_svr_macro", "SVR macro λ=5 maxW=0.5",
         [("svr", SVR(kernel='rbf', C=1.0, epsilon=0.01))],
         macro_only, 5.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # C4: Lower lambda with best model (feedback: swarm reduces DD)
    experiments.append(
        ("C4_lgbm_lam2_maxw50", "LGBM λ=2 maxW=0.5 (aggressive, swarm-safe)",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  num_leaves=31, subsample=0.8, random_state=42, verbose=-1))],
         macro_only, 2.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # C5: Tighter constraints (feedback: push Sharpe with constraints)
    experiments.append(
        ("C5_lgbm_maxw30_shrink20", "LGBM maxW=0.3 shrink=0.2 (tight)",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbose=-1))],
         macro_only, 5.0, 0.30, 21, 10, 0.2, None, False, 20, False, 50))

    # ══════════════════════════════════════════════════════════
    # Theme 2: LGBM HYPERPARAMETER TUNING
    # ══════════════════════════════════════════════════════════

    # C6: More trees, lower LR
    experiments.append(
        ("C6_lgbm_n500_lr02", "LGBM n=500 lr=0.02 macro maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.02,
                                  num_leaves=31, subsample=0.8, random_state=42, verbose=-1))],
         macro_only, 5.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # C7: Shallower trees
    experiments.append(
        ("C7_lgbm_d3_n300", "LGBM depth=3 n=300 macro maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                  num_leaves=15, subsample=0.8, random_state=42, verbose=-1))],
         macro_only, 5.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # C8: More leaves
    experiments.append(
        ("C8_lgbm_leaves63", "LGBM leaves=63 macro maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                  num_leaves=63, subsample=0.7, random_state=42, verbose=-1))],
         macro_only, 5.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # C9: Bagging fraction
    experiments.append(
        ("C9_lgbm_bag60", "LGBM subsample=0.6 macro maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  num_leaves=31, subsample=0.6, colsample_bytree=0.7,
                                  random_state=42, verbose=-1))],
         macro_only, 5.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # C10: Higher LR, fewer trees
    experiments.append(
        ("C10_lgbm_n150_lr10", "LGBM n=150 lr=0.1 macro maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=150, max_depth=4, learning_rate=0.1,
                                  num_leaves=20, subsample=0.8, random_state=42, verbose=-1))],
         macro_only, 5.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # ══════════════════════════════════════════════════════════
    # Theme 3: FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════

    # C11: Macro + momentum combined
    experiments.append(
        ("C11_lgbm_macro_mom", "LGBM macro+momentum λ=5 maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbose=-1))],
         macro_mom, 5.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # C12: PCA fixed (was B10 crash — now should work)
    experiments.append(
        ("C12_lgbm_pca20_fixed", "LGBM all→PCA(20) λ=5 maxW=0.5 [PCA bug fixed]",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbose=-1))],
         all_feat, 5.0, 0.50, 21, 10, 0.0, None, True, 20, False, 50))

    # C13: PCA with fewer components
    experiments.append(
        ("C13_lgbm_pca10", "LGBM all→PCA(10) λ=5 maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbose=-1))],
         all_feat, 5.0, 0.50, 21, 10, 0.0, None, True, 10, False, 50))

    # C14: Feature selection top-30 by MI
    experiments.append(
        ("C14_lgbm_featsel30", "LGBM top-30 MI features λ=5 maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbose=-1))],
         all_feat, 5.0, 0.50, 21, 10, 0.0, None, False, 20, True, 30))

    # C15: Momentum-only features
    experiments.append(
        ("C15_lgbm_momentum", "LGBM momentum+vol features λ=5 maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbose=-1))],
         mom_vol, 5.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # C16: Technical indicators
    experiments.append(
        ("C16_lgbm_technical", "LGBM technical indicators λ=5 maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbose=-1))],
         technical, 5.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # ══════════════════════════════════════════════════════════
    # Theme 4: ENSEMBLE VARIATIONS
    # ══════════════════════════════════════════════════════════

    # C17: LGBM + Ridge ensemble (feedback: IC-weighted)
    experiments.append(
        ("C17_ens_lgbm_ridge", "Ensemble LGBM+Ridge macro λ=5 maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbose=-1)),
          ("ridge", Ridge(alpha=1.0))],
         macro_only, 5.0, 0.50, 21, 10, 0.0,
         {"lgbm": 0.7, "ridge": 0.3}, False, 20, False, 50))

    # C18: LGBM + XGB ensemble
    experiments.append(
        ("C18_ens_lgbm_xgb", "Ensemble LGBM+XGB macro λ=5 maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbose=-1)),
          ("xgb", XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                               random_state=42, verbosity=0))],
         macro_only, 5.0, 0.50, 21, 10, 0.0,
         {"lgbm": 0.6, "xgb": 0.4}, False, 20, False, 50))

    # C19: Triple ensemble LGBM + Ridge + ElasticNet
    experiments.append(
        ("C19_ens_triple", "Ensemble LGBM+Ridge+Elastic macro λ=5 maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbose=-1)),
          ("ridge", Ridge(alpha=1.0)),
          ("elastic", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000))],
         macro_only, 5.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # C20: Ensemble with shrinkage
    experiments.append(
        ("C20_ens_lgbm_ridge_shrink", "Ensemble LGBM+Ridge shrink=0.15 maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbose=-1)),
          ("ridge", Ridge(alpha=1.0))],
         macro_only, 5.0, 0.50, 21, 10, 0.15,
         {"lgbm": 0.7, "ridge": 0.3}, False, 20, False, 50))

    # ══════════════════════════════════════════════════════════
    # Theme 5: CONSTRAINT & PARAMETER SWEEPS
    # ══════════════════════════════════════════════════════════

    # C21: Max weight sweep 0.4
    experiments.append(
        ("C21_lgbm_maxw40", "LGBM macro λ=5 maxW=0.4",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbose=-1))],
         macro_only, 5.0, 0.40, 21, 10, 0.0, None, False, 20, False, 50))

    # C22: Max weight sweep 0.6
    experiments.append(
        ("C22_lgbm_maxw60", "LGBM macro λ=5 maxW=0.6",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbose=-1))],
         macro_only, 5.0, 0.60, 21, 10, 0.0, None, False, 20, False, 50))

    # C23: Shrinkage sweep 0.1
    experiments.append(
        ("C23_lgbm_shrink10_maxw50", "LGBM macro shrink=0.1 maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbose=-1))],
         macro_only, 5.0, 0.50, 21, 10, 0.1, None, False, 20, False, 50))

    # C24: Shrinkage sweep 0.3
    experiments.append(
        ("C24_lgbm_shrink30_maxw50", "LGBM macro shrink=0.3 maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbose=-1))],
         macro_only, 5.0, 0.50, 21, 10, 0.3, None, False, 20, False, 50))

    # C25: Lambda sweep 3
    experiments.append(
        ("C25_lgbm_lam3_maxw50_tc5", "LGBM λ=3 maxW=0.5 tc=5bps",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  num_leaves=31, subsample=0.8, random_state=42, verbose=-1))],
         macro_only, 3.0, 0.50, 21, 5, 0.0, None, False, 20, False, 50))

    # C26: Lambda sweep 7
    experiments.append(
        ("C26_lgbm_lam7_maxw50", "LGBM λ=7 maxW=0.5",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbose=-1))],
         macro_only, 7.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # ══════════════════════════════════════════════════════════
    # Theme 6: MODEL ZOO — unexplored algorithms
    # ══════════════════════════════════════════════════════════

    # C27: Random Forest tuned
    experiments.append(
        ("C27_rf_n200_d8", "RF n=200 depth=8 macro maxW=0.5",
         [("rf", RandomForestRegressor(n_estimators=200, max_depth=8,
                                        min_samples_leaf=10, random_state=42, n_jobs=-1))],
         macro_only, 5.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # C28: GBR tuned
    experiments.append(
        ("C28_gbr_n200", "GBR n=200 macro maxW=0.5",
         [("gbr", GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                             learning_rate=0.05, random_state=42))],
         macro_only, 5.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # C29: AdaBoost + Ridge base
    experiments.append(
        ("C29_adaboost", "AdaBoost macro maxW=0.5",
         [("ada", AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=42))],
         macro_only, 5.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50))

    # C30: Best config + tc=3bps (best case scenario)
    experiments.append(
        ("C30_lgbm_maxw50_tc3", "LGBM macro maxW=0.5 tc=3bps (best case)",
         [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  num_leaves=31, subsample=0.8, random_state=42, verbose=-1))],
         macro_only, 5.0, 0.50, 21, 3, 0.0, None, False, 20, False, 50))

    return experiments


def main():
    t0 = time.time()
    experiments = build_experiments()

    print("=" * 90)
    print(f"EXTENDED AUTORESEARCH: {len(experiments)} experiments")
    print(f"Estimated runtime: {len(experiments) * 2:.0f} minutes")
    print(f"Current best: A1_lgbm_maxw50_tc5 Sharpe=0.832")
    print("=" * 90)

    results = []
    best_sharpe = 0.832  # Start from known best

    for i, (name, desc, mods, feat_fn, lam, mw, rb, tc, sh, ew, pca, pca_n, fs, fsk) in enumerate(experiments):
        elapsed = time.time() - t0
        remaining = (len(experiments) - i) * 2
        print(f"\n>>> [{i+1}/{len(experiments)}] {name}  ({elapsed/60:.0f}m elapsed, ~{remaining:.0f}m remaining)")

        try:
            r = run_experiment(name, desc, mods, feat_fn, lam, mw, rb, tc, sh, ew, pca, pca_n, fs, fsk)
            results.append(r)

            if r['sharpe'] > best_sharpe:
                best_sharpe = r['sharpe']
                print(f"  ★★★ NEW ALL-TIME BEST: Sharpe={best_sharpe:.4f} ★★★")
            elif r['sharpe'] > 0.5:
                print(f"  ✓ Decent: Sharpe={r['sharpe']:.4f}")
        except Exception as e:
            print(f"  CRASH: {e}")
            results.append({
                'experiment': name, 'sharpe': 0, 'ic': 0, 'dir_acc': 0,
                'ann_return': 0, 'max_dd': 0, 'time_s': 0,
                'description': f'CRASH: {e}',
            })

    # Save results
    results_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    results_df.to_csv('extended_results.csv', index=False)

    # Append to results.tsv
    with open('results.tsv', 'a') as f:
        for r in results:
            status = 'keep' if r['sharpe'] == best_sharpe else 'discard'
            f.write(f"{r.get('experiment','')}\t{r.get('sharpe',0):.4f}\t"
                    f"{r.get('ic',0):.4f}\t{r.get('dir_acc',0):.4f}\t"
                    f"{status}\t{r.get('description','')}\n")

    total_time = time.time() - t0

    print("\n" + "=" * 90)
    print(f"EXTENDED RESULTS — {len(experiments)} experiments in {total_time/60:.1f} minutes")
    print("=" * 90)
    print(f"\n  {'Rank':>4s}  {'Experiment':<35s}  {'Sharpe':>8s}  {'Return':>8s}  {'MaxDD':>8s}  {'IC':>7s}")
    print("  " + "-" * 80)
    for rank, (_, r) in enumerate(results_df.iterrows(), 1):
        star = '★' if r['sharpe'] == best_sharpe else ' '
        print(f"  {star}{rank:3d}  {r.get('experiment',''):35s}  "
              f"{r.get('sharpe',0):+8.4f}  {r.get('ann_return',0):+7.1%}  "
              f"{r.get('max_dd',0):7.1%}  {r.get('ic',0):+6.4f}")

    print(f"\n  Best: {results_df.iloc[0].get('experiment','')} "
          f"Sharpe={results_df.iloc[0].get('sharpe',0):.4f}")
    print(f"  Saved to: extended_results.csv")


if __name__ == '__main__':
    main()
