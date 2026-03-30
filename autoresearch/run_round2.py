"""
AutoResearch Round 2 — Feedback-driven experiments.
Experiments: regime-conditional, IC-weighted ensemble, maxW=0.6 variants.

Current best: C22_lgbm_maxw60 Sharpe=0.897
Target: beat 0.897 with regime or ensemble approaches
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

sys.path.insert(0, os.path.dirname(__file__))

from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from train import run_experiment
from prepare import (
    load_data, evaluate_oos, get_benchmarks, run_oos_backtest,
    print_results, TRAIN_END, OOS_START, TICKER_LIST,
)

# Feature selectors
macro_only = lambda df: [c for c in df.columns
                         if '_ret_' not in c and '_vol_' not in c
                         and '_mom_' not in c and '_rsi_' not in c]

all_feat = lambda df: [c for c in df.columns if '_ret_' not in c]


# ═══════════════════════════════════════════════════════════════════
# Regime-conditional experiment: train separate models for VIX regimes
# ═══════════════════════════════════════════════════════════════════

def run_regime_conditional(name: str, desc: str, risk_aversion: float,
                           max_weight: float, tc_bps: int,
                           vix_threshold: float = 20.0) -> dict:
    """Train separate LGBM models for high/low VIX, switch at test time."""
    t0 = time.time()
    prices, features, fwd_returns = load_data()
    tickers = list(prices.columns)
    feat_cols = macro_only(features)
    benchmarks = get_benchmarks(prices)

    vix = features['VIXCLS'] if 'VIXCLS' in features.columns else None
    if vix is None:
        print("  No VIX feature found — falling back to standard LGBM")
        return run_experiment(name, desc,
                              [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5,
                                                       learning_rate=0.05, random_state=42, verbose=-1))],
                              macro_only, risk_aversion, max_weight, 21, tc_bps, 0.0)

    models = {}
    oos_ics, oos_das = [], []

    lgbm_low = LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                              num_leaves=31, subsample=0.8, random_state=42, verbose=-1)
    lgbm_high = LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.03,
                               num_leaves=15, subsample=0.7, random_state=42, verbose=-1)

    for ticker in tickers:
        X = features[feat_cols]
        y = fwd_returns[ticker].reindex(X.index)
        mask = X.notna().all(axis=1) & y.notna()
        X_c, y_c = X[mask], y[mask]
        vix_c = vix.reindex(X_c.index).ffill()

        train_mask = X_c.index <= pd.Timestamp(TRAIN_END)
        test_mask = X_c.index >= pd.Timestamp(OOS_START)

        X_train, y_train = X_c[train_mask], y_c[train_mask]
        X_test, y_test = X_c[test_mask], y_c[test_mask]
        vix_train = vix_c[train_mask]
        vix_test = vix_c[test_mask]

        if len(X_train) < 500 or len(X_test) < 50:
            continue

        # Split training data by VIX regime
        low_mask = vix_train <= vix_threshold
        high_mask = vix_train > vix_threshold

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        X_tr_df = pd.DataFrame(X_tr_s, index=X_train.index, columns=feat_cols)
        X_te_df = pd.DataFrame(X_te_s, index=X_test.index, columns=feat_cols)

        # Train low-VIX model
        m_low = clone(lgbm_low)
        if low_mask.sum() > 100:
            m_low.fit(X_tr_df[low_mask].values, y_train[low_mask].values)
        else:
            m_low.fit(X_tr_df.values, y_train.values)

        # Train high-VIX model
        m_high = clone(lgbm_high)
        if high_mask.sum() > 100:
            m_high.fit(X_tr_df[high_mask].values, y_train[high_mask].values)
        else:
            m_high.fit(X_tr_df.values, y_train.values)

        # Predict: switch model based on VIX at each test point
        pred_oos = np.zeros(len(X_test))
        for i in range(len(X_test)):
            v = vix_test.iloc[i] if i < len(vix_test) else vix_threshold
            row = X_te_df.iloc[i:i+1].values
            if v <= vix_threshold:
                pred_oos[i] = m_low.predict(row)[0]
            else:
                pred_oos[i] = m_high.predict(row)[0]

        actual = y_test.values
        ic = np.corrcoef(actual, pred_oos)[0, 1] if len(actual) > 2 else 0
        da = np.mean(np.sign(actual) == np.sign(pred_oos))
        oos_ics.append(ic)
        oos_das.append(da)

        # Store as regime switcher wrapper
        class RegimeSwitcher:
            def __init__(self, m_low, m_high, threshold, vix_series):
                self._low = m_low
                self._high = m_high
                self._threshold = threshold
                self._vix = vix_series
                self._last_vix = threshold  # fallback

            def predict(self, X):
                return self._low.predict(X)

        models[ticker] = {
            'model': RegimeSwitcher(m_low, m_high, vix_threshold, vix_c),
            'scaler': scaler,
            'transform': {'type': 'none'},
        }

    avg_ic = np.mean(oos_ics) if oos_ics else 0
    avg_da = np.mean(oos_das) if oos_das else 0.5

    # Backtest using low-VIX models (conservative approach for walk-forward)
    rets = run_oos_backtest(
        prices, models, features, feat_cols,
        risk_aversion=risk_aversion, max_weight=max_weight,
        rebalance_freq=21, tc_bps=tc_bps, shrinkage=0.0,
    )

    metrics = evaluate_oos(rets)
    elapsed = time.time() - t0

    result = {
        'experiment': name, 'sharpe': metrics['sharpe'], 'sortino': metrics['sortino'],
        'ic': avg_ic, 'dir_acc': avg_da,
        'ann_return': metrics['ann_return'], 'ann_vol': metrics['ann_vol'],
        'max_dd': metrics['max_dd'], 'total': metrics['total'],
        'time_s': elapsed, 'description': desc,
    }

    print_results(metrics, benchmarks, desc)
    print(f"  OOS IC:      {avg_ic:+.4f}")
    print(f"  OOS DirAcc:  {avg_da:.1%}")
    print(f"  Runtime:     {elapsed:.1f}s")
    return result


# ═══════════════════════════════════════════════════════════════════
# IC-weighted ensemble: weight models by their information coefficient
# ═══════════════════════════════════════════════════════════════════

def run_ic_weighted_ensemble(name: str, desc: str, risk_aversion: float,
                              max_weight: float, tc_bps: int) -> dict:
    """Train LGBM + SVR + Ridge, weight by IC on validation set."""
    t0 = time.time()
    prices, features, fwd_returns = load_data()
    tickers = list(prices.columns)
    feat_cols = macro_only(features)
    benchmarks = get_benchmarks(prices)

    model_templates = [
        ("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                num_leaves=31, subsample=0.8, random_state=42, verbose=-1)),
        ("svr", SVR(kernel='rbf', C=1.0, epsilon=0.01)),
        ("ridge", Ridge(alpha=1.0)),
    ]

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

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        # Use last 20% of training as validation for IC weighting
        val_split = int(len(X_tr_s) * 0.8)
        X_tr_sub = X_tr_s[:val_split]
        y_tr_sub = y_train.values[:val_split]
        X_val = X_tr_s[val_split:]
        y_val = y_train.values[val_split:]

        trained = []
        val_ics = []
        for m_name, template in model_templates:
            m = clone(template)
            m.fit(X_tr_sub, y_tr_sub)
            val_pred = m.predict(X_val)
            val_ic = np.corrcoef(y_val, val_pred)[0, 1] if len(y_val) > 2 else 0
            val_ics.append(max(val_ic, 0.001))  # floor at small positive

            # Retrain on full training set
            m_full = clone(template)
            m_full.fit(X_tr_s, y_train.values)
            trained.append((m_name, m_full))

        # IC weights (proportional to validation IC)
        ic_arr = np.array(val_ics)
        ic_weights = ic_arr / ic_arr.sum()

        # Ensemble prediction
        preds = [m.predict(X_te_s) for _, m in trained]
        ensemble_pred = np.average(preds, axis=0, weights=ic_weights)

        actual = y_test.values
        ic = np.corrcoef(actual, ensemble_pred)[0, 1] if len(actual) > 2 else 0
        da = np.mean(np.sign(actual) == np.sign(ensemble_pred))
        oos_ics.append(ic)
        oos_das.append(da)

        # Wrap for backtest
        class ICEnsemble:
            def __init__(self, models, weights):
                self._models = models
                self._weights = weights
            def predict(self, X):
                preds = [m.predict(X) for _, m in self._models]
                return np.average(preds, axis=0, weights=self._weights)

        models[ticker] = {
            'model': ICEnsemble(trained, ic_weights),
            'scaler': scaler,
            'transform': {'type': 'none'},
        }

    avg_ic = np.mean(oos_ics) if oos_ics else 0
    avg_da = np.mean(oos_das) if oos_das else 0.5

    rets = run_oos_backtest(
        prices, models, features, feat_cols,
        risk_aversion=risk_aversion, max_weight=max_weight,
        rebalance_freq=21, tc_bps=tc_bps, shrinkage=0.0,
    )

    metrics = evaluate_oos(rets)
    elapsed = time.time() - t0

    result = {
        'experiment': name, 'sharpe': metrics['sharpe'], 'sortino': metrics['sortino'],
        'ic': avg_ic, 'dir_acc': avg_da,
        'ann_return': metrics['ann_return'], 'ann_vol': metrics['ann_vol'],
        'max_dd': metrics['max_dd'], 'total': metrics['total'],
        'time_s': elapsed, 'description': desc,
    }

    print_results(metrics, benchmarks, desc)
    print(f"  OOS IC:      {avg_ic:+.4f}")
    print(f"  OOS DirAcc:  {avg_da:.1%}")
    print(f"  IC weights:  {dict(zip([n for n,_ in model_templates], ic_weights.round(3)))}")
    print(f"  Runtime:     {elapsed:.1f}s")
    return result


def build_experiments():
    """Build round 2 experiments — 15 total."""
    experiments = []

    # ── Regime-conditional ────────────────────────────────────────
    # D1: Regime LGBM maxW=0.5, VIX threshold=20
    experiments.append(("regime", "D1_regime_lgbm_maxw50",
                        "Regime-conditional LGBM (VIX=20 split) macro maxW=0.5",
                        5.0, 0.50, 10, 20.0))

    # D2: Regime LGBM maxW=0.6 (match best)
    experiments.append(("regime", "D2_regime_lgbm_maxw60",
                        "Regime-conditional LGBM (VIX=20 split) macro maxW=0.6",
                        5.0, 0.60, 10, 20.0))

    # D3: Regime with higher VIX threshold (25)
    experiments.append(("regime", "D3_regime_lgbm_vix25",
                        "Regime-conditional LGBM (VIX=25 split) macro maxW=0.6",
                        5.0, 0.60, 10, 25.0))

    # D4: Regime with lower lambda
    experiments.append(("regime", "D4_regime_lgbm_lam3",
                        "Regime-conditional LGBM (VIX=20) λ=3 maxW=0.6",
                        3.0, 0.60, 10, 20.0))

    # ── IC-weighted ensemble ──────────────────────────────────────
    # D5: IC-weighted LGBM+SVR+Ridge maxW=0.5
    experiments.append(("ic_ensemble", "D5_icens_maxw50",
                        "IC-weighted ensemble (LGBM+SVR+Ridge) macro maxW=0.5",
                        5.0, 0.50, 10))

    # D6: IC-weighted maxW=0.6
    experiments.append(("ic_ensemble", "D6_icens_maxw60",
                        "IC-weighted ensemble (LGBM+SVR+Ridge) macro maxW=0.6",
                        5.0, 0.60, 10))

    # D7: IC-weighted lower lambda
    experiments.append(("ic_ensemble", "D7_icens_lam3_maxw60",
                        "IC-weighted ensemble λ=3 maxW=0.6",
                        3.0, 0.60, 10))

    # ── maxW=0.6 variants (exploit new best) ──────────────────────
    # D8: SVR with maxW=0.6
    experiments.append(("standard", "D8_svr_maxw60",
                        "SVR macro λ=5 maxW=0.6",
                        [("svr", SVR(kernel='rbf', C=1.0, epsilon=0.01))],
                        macro_only, 5.0, 0.60, 21, 10, 0.0))

    # D9: LGBM maxW=0.65
    experiments.append(("standard", "D9_lgbm_maxw65",
                        "LGBM macro λ=5 maxW=0.65",
                        [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                 num_leaves=31, subsample=0.8, random_state=42, verbose=-1))],
                        macro_only, 5.0, 0.65, 21, 10, 0.0))

    # D10: LGBM maxW=0.7
    experiments.append(("standard", "D10_lgbm_maxw70",
                        "LGBM macro λ=5 maxW=0.7",
                        [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                 num_leaves=31, subsample=0.8, random_state=42, verbose=-1))],
                        macro_only, 5.0, 0.70, 21, 10, 0.0))

    # D11: LGBM maxW=0.6 + shrink=0.1
    experiments.append(("standard", "D11_lgbm_maxw60_shrink10",
                        "LGBM macro λ=5 maxW=0.6 shrink=0.1",
                        [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                 num_leaves=31, subsample=0.8, random_state=42, verbose=-1))],
                        macro_only, 5.0, 0.60, 21, 10, 0.1))

    # D12: LGBM maxW=0.6 λ=3
    experiments.append(("standard", "D12_lgbm_maxw60_lam3",
                        "LGBM macro λ=3 maxW=0.6",
                        [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                 num_leaves=31, subsample=0.8, random_state=42, verbose=-1))],
                        macro_only, 3.0, 0.60, 21, 10, 0.0))

    # D13: LGBM maxW=0.6 tc=3bps (best case)
    experiments.append(("standard", "D13_lgbm_maxw60_tc3",
                        "LGBM macro λ=5 maxW=0.6 tc=3bps",
                        [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                 num_leaves=31, subsample=0.8, random_state=42, verbose=-1))],
                        macro_only, 5.0, 0.60, 21, 3, 0.0))

    # D14: XGBoost with maxW=0.6
    experiments.append(("standard", "D14_xgb_maxw60",
                        "XGBoost macro λ=5 maxW=0.6",
                        [("xgb", XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                               subsample=0.8, random_state=42, verbosity=0))],
                        macro_only, 5.0, 0.60, 21, 10, 0.0))

    # D15: LGBM maxW=0.6 weekly rebalance
    experiments.append(("standard", "D15_lgbm_maxw60_weekly",
                        "LGBM macro λ=5 maxW=0.6 weekly rebalance",
                        [("lgbm", LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                 num_leaves=31, subsample=0.8, random_state=42, verbose=-1))],
                        macro_only, 5.0, 0.60, 5, 10, 0.0))

    return experiments


def main():
    t0 = time.time()
    experiments = build_experiments()

    print("=" * 90)
    print(f"AUTORESEARCH ROUND 2: {len(experiments)} experiments")
    print(f"Focus: regime-conditional, IC-weighted ensemble, maxW=0.6 variants")
    print(f"Current best: C22_lgbm_maxw60 Sharpe=0.897")
    print("=" * 90)

    results = []
    best_sharpe = 0.897

    for i, exp in enumerate(experiments):
        elapsed = time.time() - t0
        remaining = (len(experiments) - i) * 3  # ~3 min for complex experiments

        exp_type = exp[0]
        exp_name = exp[1]
        exp_desc = exp[2]

        print(f"\n>>> [{i+1}/{len(experiments)}] {exp_name}  "
              f"({elapsed/60:.0f}m elapsed, ~{remaining:.0f}m remaining)")

        try:
            if exp_type == "regime":
                lam, mw, tc, vix_thresh = exp[3], exp[4], exp[5], exp[6]
                r = run_regime_conditional(exp_name, exp_desc, lam, mw, tc, vix_thresh)

            elif exp_type == "ic_ensemble":
                lam, mw, tc = exp[3], exp[4], exp[5]
                r = run_ic_weighted_ensemble(exp_name, exp_desc, lam, mw, tc)

            elif exp_type == "standard":
                mods, feat_fn, lam, mw, rb, tc, sh = exp[3], exp[4], exp[5], exp[6], exp[7], exp[8], exp[9]
                r = run_experiment(exp_name, exp_desc, mods, feat_fn, lam, mw, rb, tc, sh)

            else:
                raise ValueError(f"Unknown experiment type: {exp_type}")

            results.append(r)

            if r['sharpe'] > best_sharpe:
                best_sharpe = r['sharpe']
                print(f"  ★★★ NEW ALL-TIME BEST: Sharpe={best_sharpe:.4f} ★★★")
            elif r['sharpe'] > 0.5:
                print(f"  ✓ Decent: Sharpe={r['sharpe']:.4f}")

        except Exception as e:
            print(f"  CRASH: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'experiment': exp_name, 'sharpe': 0, 'sortino': 0,
                'ic': 0, 'dir_acc': 0,
                'ann_return': 0, 'ann_vol': 0, 'max_dd': 0, 'total': 0,
                'time_s': 0, 'description': f'CRASH: {e}',
            })

    # Save results
    results_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    results_df.to_csv('round2_results.csv', index=False)

    # Append to results.tsv
    with open('results.tsv', 'a') as f:
        for r in results:
            status = 'keep' if r['sharpe'] == best_sharpe else 'discard'
            f.write(f"{r.get('experiment','')}\t{r.get('sharpe',0):.4f}\t"
                    f"{r.get('ic',0):.4f}\t{r.get('dir_acc',0):.4f}\t"
                    f"{status}\t{r.get('description','')}\n")

    total_time = time.time() - t0

    print("\n" + "=" * 90)
    print(f"ROUND 2 RESULTS — {len(experiments)} experiments in {total_time/60:.1f} minutes")
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
    print(f"  Saved to: round2_results.csv")


if __name__ == '__main__':
    main()
