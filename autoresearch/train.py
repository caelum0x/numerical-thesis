"""
Portfolio AutoResearch — train.py (THE file the agent modifies)
================================================================
Adapted from Karpathy's autoresearch for macro-based portfolio optimization.

Usage: python train.py                  # run single experiment
       python train.py --batch          # run batch of experiments
       python train.py --sweep lambda   # parameter sweep

The agent iterates on the EXPERIMENT section. The pipeline below is fixed.
"""

import os
import sys
import time
import warnings
import argparse
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

from prepare import (
    load_data, evaluate_oos, get_benchmarks, run_oos_backtest,
    print_results, TRAIN_END, OOS_START, PREDICTION_HORIZON,
    TICKER_LIST, N_ASSETS, TIME_BUDGET,
)

# ===========================================================================
# EXPERIMENT CONFIGURATION — THIS IS WHAT THE AGENT MODIFIES
# ===========================================================================

EXPERIMENT_NAME = "E13_ensemble_macro_shrink30"
DESCRIPTION = "Ensemble (Lasso+LightGBM+Ridge) macro-only, lambda=5, 30% shrinkage"

# --- Models (can be single or list for ensemble) ---
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from xgboost import XGBRegressor

MODELS = [
    ("lasso", Lasso(alpha=0.001, max_iter=5000)),
    ("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                            num_leaves=31, subsample=0.8, random_state=42, verbose=-1)),
    ("ridge", Ridge(alpha=1.0)),
]

# Ensemble weights (None = equal weight, or dict like {"lasso": 0.4, "lgbm": 0.4, "ridge": 0.2})
ENSEMBLE_WEIGHTS = None

# --- Feature selection ---
USE_PCA = False
PCA_COMPONENTS = 20

USE_FEATURE_SELECTION = False
FEATURE_SELECTION_TOP_K = 50

def select_features(features_df):
    """Choose which columns to use as predictors."""
    cols = [c for c in features_df.columns
            if '_ret_' not in c
            and '_vol_' not in c
            and '_mom_' not in c
            and '_rsi_' not in c]
    return cols

# --- Portfolio optimization parameters ---
RISK_AVERSION = 5.0
MAX_WEIGHT = 0.35
REBALANCE_FREQ = 21
TC_BPS = 10
SHRINKAGE = 0.3

# ===========================================================================
# FIXED PIPELINE — Infrastructure (do not modify during experiment loop)
# ===========================================================================

def train_single_model(model_template, X_train, y_train, X_test, y_test):
    """Train one model and return predictions + metrics."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    m = clone(model_template)
    m.fit(X_tr_s, y_train)

    pred_oos = m.predict(X_te_s)
    actual = y_test.values
    ic = np.corrcoef(actual, pred_oos)[0, 1] if len(actual) > 2 else 0
    da = np.mean(np.sign(actual) == np.sign(pred_oos))

    return m, scaler, pred_oos, ic, da


def train_ensemble(model_list, X_train, y_train, X_test, y_test, weights=None):
    """Train multiple models and ensemble their predictions."""
    trained = []
    all_preds = []
    all_ics = []

    for name, template in model_list:
        m, scaler, pred, ic, da = train_single_model(template, X_train, y_train, X_test, y_test)
        trained.append((name, m, scaler))
        all_preds.append(pred)
        all_ics.append(ic)

    if weights is None:
        w = np.ones(len(all_preds)) / len(all_preds)
    else:
        w = np.array([weights.get(name, 1.0) for name, _, _ in trained])
        w = w / w.sum()

    ensemble_pred = np.average(all_preds, axis=0, weights=w)
    actual = y_test.values
    ensemble_ic = np.corrcoef(actual, ensemble_pred)[0, 1] if len(actual) > 2 else 0
    ensemble_da = np.mean(np.sign(actual) == np.sign(ensemble_pred))

    return trained, ensemble_pred, ensemble_ic, ensemble_da


def apply_feature_transform(X_train, X_test, y_train, feat_cols):
    """Apply PCA or feature selection if configured."""
    if USE_FEATURE_SELECTION:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        mi = mutual_info_regression(X_tr_s, y_train, random_state=42)
        top_idx = np.argsort(mi)[-FEATURE_SELECTION_TOP_K:]
        selected_cols = [feat_cols[i] for i in top_idx]
        return X_train[selected_cols], X_test[selected_cols], selected_cols

    if USE_PCA:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        X_tr_pca = pd.DataFrame(pca.fit_transform(X_tr_s), index=X_train.index,
                                 columns=[f'PC{i}' for i in range(PCA_COMPONENTS)])
        X_te_pca = pd.DataFrame(pca.transform(X_te_s), index=X_test.index,
                                 columns=[f'PC{i}' for i in range(PCA_COMPONENTS)])
        return X_tr_pca, X_te_pca, list(X_tr_pca.columns)

    return X_train, X_test, feat_cols


def run_experiment(experiment_name, description, model_list, feat_fn, risk_aversion,
                   max_weight, rebalance_freq, tc_bps, shrinkage, ensemble_weights=None,
                   use_pca=False, pca_n=20, use_feat_sel=False, feat_sel_k=50):
    """Run a complete experiment. Returns metrics dict."""
    global USE_PCA, PCA_COMPONENTS, USE_FEATURE_SELECTION, FEATURE_SELECTION_TOP_K
    USE_PCA = use_pca
    PCA_COMPONENTS = pca_n
    USE_FEATURE_SELECTION = use_feat_sel
    FEATURE_SELECTION_TOP_K = feat_sel_k

    t0 = time.time()
    prices, features, fwd_returns = load_data()
    tickers = list(prices.columns)
    feat_cols = feat_fn(features)
    benchmarks = get_benchmarks(prices)

    is_ensemble = len(model_list) > 1
    models = {}
    oos_ics, oos_das = [], []

    for ticker in tickers:
        X = features[feat_cols]
        y = fwd_returns[ticker].reindex(X.index)
        mask = X.notna().all(axis=1) & y.notna()
        X_c, y_c = X[mask], y[mask]

        train_mask = X_c.index <= pd.Timestamp(TRAIN_END)
        test_mask = X_c.index >= pd.Timestamp(OOS_START)
        X_train, y_train = X_c[train_mask], y_c[train_mask]
        X_test, y_test = X_c[test_mask], y_c[test_mask]

        if len(X_train) < 500 or len(X_test) < 50:
            continue

        X_train_t, X_test_t, used_cols = apply_feature_transform(
            X_train, X_test, y_train, feat_cols
        )

        if is_ensemble:
            trained, pred, ic, da = train_ensemble(
                model_list, X_train_t, y_train, X_test_t, y_test, ensemble_weights
            )
            # Store all models for backtest prediction
            models[ticker] = {
                'ensemble': trained,
                'scaler': StandardScaler().fit(X_train_t),
                'feat_cols': used_cols,
            }
        else:
            name, template = model_list[0]
            m, scaler, pred, ic, da = train_single_model(
                template, X_train_t, y_train, X_test_t, y_test
            )
            models[ticker] = {'model': m, 'scaler': scaler, 'feat_cols': used_cols}

        oos_ics.append(ic)
        oos_das.append(da)

    avg_ic = np.mean(oos_ics) if oos_ics else 0
    avg_da = np.mean(oos_das) if oos_das else 0.5

    # Build models dict compatible with run_oos_backtest
    bt_models = {}
    for ticker, md in models.items():
        if 'ensemble' in md:
            # Create a wrapper that averages ensemble predictions
            class EnsembleWrapper:
                def __init__(self, trained_models, ew):
                    self.trained = trained_models
                    self.weights = ew
                def predict(self, X):
                    preds = [m.predict(X) for _, m, _ in self.trained]
                    if self.weights:
                        w = np.array([self.weights.get(n, 1.0) for n, _, _ in self.trained])
                        w = w / w.sum()
                    else:
                        w = np.ones(len(preds)) / len(preds)
                    return np.average(preds, axis=0, weights=w)

            bt_models[ticker] = {
                'model': EnsembleWrapper(md['ensemble'], ensemble_weights),
                'scaler': md['scaler'],
            }
        else:
            bt_models[ticker] = {'model': md['model'], 'scaler': md['scaler']}

    # Backtest
    rets = run_oos_backtest(
        prices, bt_models, features, feat_cols,
        risk_aversion=risk_aversion, max_weight=max_weight,
        rebalance_freq=rebalance_freq, tc_bps=tc_bps, shrinkage=shrinkage,
    )

    metrics = evaluate_oos(rets)
    elapsed = time.time() - t0

    result = {
        'experiment': experiment_name,
        'sharpe': metrics['sharpe'],
        'sortino': metrics['sortino'],
        'ic': avg_ic,
        'dir_acc': avg_da,
        'ann_return': metrics['ann_return'],
        'ann_vol': metrics['ann_vol'],
        'max_dd': metrics['max_dd'],
        'total': metrics['total'],
        'time_s': elapsed,
        'description': description,
    }

    print_results(metrics, benchmarks, description)
    print(f"  OOS IC:      {avg_ic:+.4f}")
    print(f"  OOS DirAcc:  {avg_da:.1%}")
    print(f"  Runtime:     {elapsed:.1f}s")
    print(f'\nRESULT: sharpe={metrics["sharpe"]:.4f} ic={avg_ic:.4f} dir_acc={avg_da:.4f} ann_return={metrics["ann_return"]:.4f} max_dd={metrics["max_dd"]:.4f} description="{description}"')

    return result


def run_batch():
    """Run a batch of experiments automatically — autonomous research mode."""
    all_feat = lambda df: [c for c in df.columns if '_ret_' not in c]
    macro_only = lambda df: [c for c in df.columns if '_ret_' not in c and '_vol_' not in c and '_mom_' not in c and '_rsi_' not in c]
    mom_vol = lambda df: [c for c in df.columns if ('_vol_' in c or '_mom_' in c or '_rsi_' in c)]

    experiments = [
        # (name, description, models, feat_fn, lambda, max_w, rebal, tc, shrink, ens_w, pca, pca_n, feat_sel, feat_k)
        ("B1_lasso_macro", "Lasso macro-only λ=5", [("lasso", Lasso(alpha=0.001, max_iter=5000))], macro_only, 5.0, 0.35, 21, 10, 0.0, None, False, 20, False, 50),
        ("B2_lgbm_macro", "LightGBM macro-only λ=5", [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1))], macro_only, 5.0, 0.35, 21, 10, 0.0, None, False, 20, False, 50),
        ("B3_ensemble_macro", "Ensemble(Lasso+LGBM+Ridge) macro λ=5", [("lasso", Lasso(alpha=0.001, max_iter=5000)), ("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)), ("ridge", Ridge(alpha=1.0))], macro_only, 5.0, 0.35, 21, 10, 0.0, None, False, 20, False, 50),
        ("B4_ensemble_shrink30", "Ensemble macro λ=5 shrink=0.3", [("lasso", Lasso(alpha=0.001, max_iter=5000)), ("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)), ("ridge", Ridge(alpha=1.0))], macro_only, 5.0, 0.35, 21, 10, 0.3, None, False, 20, False, 50),
        ("B5_lgbm_all_feat", "LightGBM all features λ=5", [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1))], all_feat, 5.0, 0.35, 21, 10, 0.0, None, False, 20, False, 50),
        ("B6_lgbm_macro_lam10", "LightGBM macro λ=10 (conservative)", [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1))], macro_only, 10.0, 0.30, 21, 10, 0.0, None, False, 20, False, 50),
        ("B7_lgbm_macro_weekly", "LightGBM macro λ=5 weekly rebal", [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1))], macro_only, 5.0, 0.35, 5, 10, 0.0, None, False, 20, False, 50),
        ("B8_lgbm_macro_quarterly", "LightGBM macro λ=5 quarterly rebal", [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1))], macro_only, 5.0, 0.35, 63, 10, 0.0, None, False, 20, False, 50),
        ("B9_xgb_macro", "XGBoost macro-only λ=5", [("xgb", XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42, verbosity=0))], macro_only, 5.0, 0.35, 21, 10, 0.0, None, False, 20, False, 50),
        ("B10_lgbm_pca20", "LightGBM all→PCA(20) λ=5", [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1))], all_feat, 5.0, 0.35, 21, 10, 0.0, None, True, 20, False, 50),
        ("B11_lgbm_macro_tc5", "LightGBM macro λ=5 tc=5bps", [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1))], macro_only, 5.0, 0.35, 21, 5, 0.0, None, False, 20, False, 50),
        ("B12_lgbm_macro_maxw50", "LightGBM macro λ=5 maxW=0.5", [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1))], macro_only, 5.0, 0.50, 21, 10, 0.0, None, False, 20, False, 50),
    ]

    print("=" * 90)
    print(f"BATCH MODE: {len(experiments)} experiments")
    print("=" * 90)

    results = []
    best_sharpe = -999

    for name, desc, mods, feat_fn, lam, mw, rb, tc, sh, ew, pca, pca_n, fs, fsk in experiments:
        print(f"\n>>> {name}")
        try:
            r = run_experiment(name, desc, mods, feat_fn, lam, mw, rb, tc, sh, ew, pca, pca_n, fs, fsk)
            results.append(r)

            status = 'keep' if r['sharpe'] > best_sharpe else 'discard'
            if r['sharpe'] > best_sharpe:
                best_sharpe = r['sharpe']
                print(f"  ★ NEW BEST: Sharpe={best_sharpe:.4f}")
        except Exception as e:
            print(f"  CRASH: {e}")
            results.append({'experiment': name, 'sharpe': 0, 'ic': 0, 'dir_acc': 0,
                           'ann_return': 0, 'max_dd': 0, 'time_s': 0, 'description': f'CRASH: {e}'})

    # Save results
    results_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    results_df.to_csv('batch_results.csv', index=False)

    # Append to results.tsv
    with open('results.tsv', 'a') as f:
        for r in results:
            status = 'keep' if r['sharpe'] == best_sharpe else 'discard'
            f.write(f"{r['experiment']}\t{r.get('sharpe',0):.4f}\t{r.get('ic',0):.4f}\t{r.get('dir_acc',0):.4f}\t{status}\t{r.get('description','')}\n")

    print("\n" + "=" * 90)
    print("BATCH RESULTS — RANKED BY OOS SHARPE")
    print("=" * 90)
    for _, r in results_df.iterrows():
        star = '★' if r['sharpe'] == best_sharpe else ' '
        print(f"  {star} {r['experiment']:30s} Sharpe={r['sharpe']:+.4f}  IC={r.get('ic',0):+.4f}  Return={r.get('ann_return',0):+.2%}  DD={r.get('max_dd',0):.2%}")

    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', action='store_true', help='Run batch of experiments')
    args = parser.parse_args()

    if args.batch:
        run_batch()
    else:
        run_experiment(
            EXPERIMENT_NAME, DESCRIPTION, MODELS, select_features,
            RISK_AVERSION, MAX_WEIGHT, REBALANCE_FREQ, TC_BPS, SHRINKAGE,
            ENSEMBLE_WEIGHTS, USE_PCA, PCA_COMPONENTS,
            USE_FEATURE_SELECTION, FEATURE_SELECTION_TOP_K,
        )


if __name__ == '__main__':
    main()
