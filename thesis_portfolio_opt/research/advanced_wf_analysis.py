#!/usr/bin/env python3
"""
Advanced Walk-Forward Analysis
==============================
Part 1: RF Walk-Forward Deep-Dive (#16) — bootstrap CI, stress test, drawdown
Part 2: Ensemble of WF Strategies (#17) — equal-weight avg of RF+LightGBM+Lasso
Part 3: Strategy Correlation (#20) — pairwise correlation heatmap
Part 4: Rolling Sharpe Comparison (#22) — 126-day rolling Sharpe for all

Figures produced:
    fig29_rf_deepdive.pdf
    fig30_ensemble_vs_single.pdf
    fig31_strategy_correlation.pdf
    fig32_rolling_sharpe_all.pdf

Author : Arhan Subasi
"""

import sys, os, time, warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp

from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor

from src.config import (
    TICKER_LIST, RESULTS_DIR, RANDOM_STATE,
    PREDICTION_HORIZON, LATEX_FONT_SETTINGS, FIGURE_DPI,
    STRESS_SCENARIOS,
)

plt.rcParams.update(LATEX_FONT_SETTINGS)
np.random.seed(RANDOM_STATE)

RES = str(RESULTS_DIR)

# ============================================================================
# Configuration
# ============================================================================
OOS_START = "2022-01-01"
RETRAIN_EVERY = 63          # ~quarterly
HORIZON = PREDICTION_HORIZON  # 21 days
TC_BPS = 10
TICKERS = TICKER_LIST
N_ASSETS = len(TICKERS)

# Three WF strategies to compare
WF_CONFIGS = {
    "WF_RF":       {"model": "random_forest",  "lam": 2.5, "max_w": 0.50},
    "WF_LightGBM": {"model": "lightgbm",       "lam": 2.5, "max_w": 0.50},
    "WF_Lasso":    {"model": "lasso",           "lam": 2.5, "max_w": 0.50},
}

DISPLAY_NAMES = {
    "WF_RF":       "WF RandomForest",
    "WF_LightGBM": "WF LightGBM",
    "WF_Lasso":    "WF Lasso",
    "SPY":         "SPY",
    "Equal_Weight": "Equal Weight",
    "60_40":       "60/40",
    "Ensemble":    "Ensemble (Equal-Wt Avg)",
}

print("=" * 72)
print("ADVANCED WALK-FORWARD ANALYSIS")
print("=" * 72)
t_global = time.time()

# ============================================================================
# Data loading
# ============================================================================
features = pd.read_csv(
    os.path.join(PROJECT_ROOT, "data", "processed", "features.csv"),
    index_col=0, parse_dates=True,
)
prices = pd.read_csv(
    os.path.join(PROJECT_ROOT, "data", "raw", "prices.csv"),
    index_col=0, parse_dates=True,
)

daily_ret = prices.pct_change().dropna()
common_idx = features.index.intersection(daily_ret.index)
features = features.loc[common_idx]
daily_ret = daily_ret.loc[common_idx]

print(f"Features shape: {features.shape}")
print(f"Date range:     {features.index[0].date()} -> {features.index[-1].date()}")

# Macro-only feature columns
FRED_PREFIXES = [
    "DFF", "DGS2", "DGS10", "T10Y2Y", "T10Y3M", "VIXCLS",
    "BAMLH0A0HYM2", "BAMLC0A4CBBB", "DTWEXBGS", "UMCSENT",
    "UNRATE", "ICSA", "CPIAUCSL", "T10YIE", "PPIACO",
    "M2SL", "USSLIND", "HOUST",
]
all_cols = list(features.columns)
macro_cols = [c for c in all_cols
              if any(c == pfx or c.startswith(pfx + "_lag") for pfx in FRED_PREFIXES)]
print(f"Macro-only features: {len(macro_cols)}")

# Forward returns (21-day)
fwd_dict = {}
for tk in TICKERS:
    fwd_dict[tk] = prices[tk].pct_change(HORIZON).shift(-HORIZON)
fwd_df = pd.DataFrame(fwd_dict, index=prices.index).reindex(features.index)

X_macro_full = features[macro_cols].values
X_macro_idx = features.index
fwd_vals = fwd_df[TICKERS].values
daily_ret_vals = daily_ret[TICKERS].values

oos_mask = X_macro_idx >= OOS_START
oos_locs = np.where(oos_mask)[0]
oos_dates = X_macro_idx[oos_mask]
n_oos = len(oos_locs)
print(f"OOS days: {n_oos}  ({oos_dates[0].date()} -> {oos_dates[-1].date()})")

rebal_oos = np.arange(0, n_oos, RETRAIN_EVERY)
print(f"Rebalance points: {len(rebal_oos)}")


# ============================================================================
# Model factory
# ============================================================================
def make_model(name):
    if name == "lightgbm":
        return LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            num_leaves=31, verbose=-1, random_state=RANDOM_STATE,
            n_jobs=1,
        )
    elif name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    elif name == "lasso":
        return Lasso(alpha=0.001, max_iter=5000, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model: {name}")


# ============================================================================
# MV optimisation
# ============================================================================
def mv_optimize(mu, cov, lam, max_w, min_w=0.0):
    n = len(mu)
    w = cp.Variable(n)
    ret = mu @ w
    risk = cp.quad_form(w, cov, assume_PSD=True)
    obj = cp.Maximize(ret - (lam / 2.0) * risk)
    cons = [cp.sum(w) == 1, w >= min_w, w <= max_w]
    prob = cp.Problem(obj, cons)
    for solver in [cp.OSQP, cp.SCS, cp.ECOS]:
        try:
            prob.solve(solver=solver, warm_start=True)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                weights = np.array(w.value).flatten()
                weights = np.clip(weights, min_w, max_w)
                weights /= weights.sum()
                return weights
        except Exception:
            continue
    return np.ones(n) / n


# ============================================================================
# Walk-forward engine
# ============================================================================
def run_walk_forward(cfg):
    model_name = cfg["model"]
    lam = cfg["lam"]
    max_w = cfg["max_w"]
    weights_arr = np.full((n_oos, N_ASSETS), 1.0 / N_ASSETS)

    for ri in range(len(rebal_oos)):
        oos_pos = rebal_oos[ri]
        global_pos = oos_locs[oos_pos]

        X_train = X_macro_full[:global_pos]
        Y_train = fwd_vals[:global_pos]

        valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(Y_train).any(axis=1)
        X_tr = X_train[valid]
        Y_tr = Y_train[valid]

        if len(X_tr) < 100:
            continue

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)

        X_now = X_macro_full[global_pos:global_pos+1].copy()
        if np.isnan(X_now).any():
            X_now = X_tr[-1:].copy()
        X_now_scaled = scaler.transform(X_now)

        mu_pred = np.zeros(N_ASSETS)
        for k in range(N_ASSETS):
            mdl = make_model(model_name)
            mdl.fit(X_tr_scaled, Y_tr[:, k])
            mu_pred[k] = mdl.predict(X_now_scaled)[0]

        lb_start = max(0, global_pos - 252)
        ret_window = daily_ret_vals[lb_start:global_pos]
        clean_mask = ~np.isnan(ret_window).any(axis=1)
        ret_clean = ret_window[clean_mask]
        if len(ret_clean) < 30:
            continue

        lw = LedoitWolf().fit(ret_clean)
        cov_mat = lw.covariance_ * 252.0
        eigvals = np.linalg.eigvalsh(cov_mat)
        if eigvals.min() < 0:
            cov_mat += (-eigvals.min() + 1e-8) * np.eye(N_ASSETS)

        mu_annual = mu_pred * (252.0 / HORIZON)

        w_opt = mv_optimize(mu_annual, cov_mat, lam, max_w)

        next_pos = rebal_oos[ri + 1] if ri + 1 < len(rebal_oos) else n_oos
        weights_arr[oos_pos:next_pos] = w_opt

    return weights_arr


def compute_strategy_returns(weights_arr):
    oos_daily = daily_ret_vals[oos_locs]
    port_ret = (weights_arr * oos_daily).sum(axis=1)
    turnover = np.abs(np.diff(weights_arr, axis=0)).sum(axis=1)
    turnover = np.concatenate([[0.0], turnover])
    tc = turnover * (TC_BPS / 10000.0)
    return pd.Series(port_ret - tc, index=oos_dates, name="returns")


# ============================================================================
# Compute all 3 WF strategies
# ============================================================================
print("\n" + "=" * 72)
print("COMPUTING WALK-FORWARD STRATEGIES (RF, LightGBM, Lasso)")
print("=" * 72)

wf_returns = {}
for name, cfg in WF_CONFIGS.items():
    t1 = time.time()
    print(f"\n--- {name} (model={cfg['model']}, lam={cfg['lam']}, max_w={cfg['max_w']}) ---", flush=True)
    w_arr = run_walk_forward(cfg)
    ret_s = compute_strategy_returns(w_arr)
    wf_returns[name] = ret_s
    elapsed = time.time() - t1
    ann_ret = ret_s.mean() * 252
    ann_vol = ret_s.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    print(f"    Done in {elapsed:.1f}s | Ann.Ret={ann_ret:.4f} | Sharpe={sharpe:.4f}", flush=True)

# Benchmarks
oos_daily_all = daily_ret_vals[oos_locs]
spy_col = TICKERS.index("SPY")
spy_ret = pd.Series(oos_daily_all[:, spy_col], index=oos_dates, name="SPY")

ew_ret_arr = oos_daily_all.mean(axis=1)
ew_ret = pd.Series(ew_ret_arr, index=oos_dates, name="Equal_Weight")

w6040 = np.zeros(N_ASSETS)
w6040[TICKERS.index("SPY")] = 0.60
w6040[TICKERS.index("AGG")] = 0.40
ret_6040_arr = (w6040 * oos_daily_all).sum(axis=1)
ret_6040 = pd.Series(ret_6040_arr, index=oos_dates, name="60_40")

benchmarks = {"SPY": spy_ret, "Equal_Weight": ew_ret, "60_40": ret_6040}


# ============================================================================
# Helper: metrics
# ============================================================================
def compute_metrics(r):
    r = r.dropna()
    ann_ret = r.mean() * 252
    ann_vol = r.std(ddof=1) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()
    total = cum.iloc[-1] - 1
    return {
        "Ann Return": ann_ret,
        "Ann Vol": ann_vol,
        "Sharpe": sharpe,
        "Max DD": max_dd,
        "Total Return": total,
    }


def compute_drawdown_series(r):
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    return dd


# ============================================================================
# PART 1: RF WALK-FORWARD DEEP-DIVE
# ============================================================================
print("\n" + "=" * 72)
print("PART 1: RF WALK-FORWARD DEEP-DIVE")
print("=" * 72)

rf_ret = wf_returns["WF_RF"]
rf_metrics = compute_metrics(rf_ret)
print(f"\nRF Walk-Forward OOS Metrics:")
for k, v in rf_metrics.items():
    print(f"  {k:15s}: {v:+.4f}")

# --- 1a. Block Bootstrap CI on Sharpe ---
print("\n--- Block Bootstrap CI on Sharpe (1000 samples, block_size=21) ---")
N_BOOT = 1000
BLOCK_SIZE = 21

rf_arr = rf_ret.values
n_rf = len(rf_arr)
n_blocks = int(np.ceil(n_rf / BLOCK_SIZE))

boot_sharpes = np.zeros(N_BOOT)
rng = np.random.RandomState(RANDOM_STATE)

for b in range(N_BOOT):
    # Circular block bootstrap
    starts = rng.randint(0, n_rf, size=n_blocks)
    idx_list = []
    for s in starts:
        block_idx = np.arange(s, s + BLOCK_SIZE) % n_rf
        idx_list.append(block_idx)
    all_idx = np.concatenate(idx_list)[:n_rf]
    boot_ret = rf_arr[all_idx]
    mu_b = boot_ret.mean() * 252
    sigma_b = boot_ret.std(ddof=1) * np.sqrt(252)
    boot_sharpes[b] = mu_b / sigma_b if sigma_b > 0 else 0.0

ci_lo = np.percentile(boot_sharpes, 2.5)
ci_hi = np.percentile(boot_sharpes, 97.5)
ci_med = np.median(boot_sharpes)
print(f"  RF Sharpe point estimate:  {rf_metrics['Sharpe']:.4f}")
print(f"  Bootstrap median:          {ci_med:.4f}")
print(f"  95% CI:                    [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  CI width:                  {ci_hi - ci_lo:.4f}")
print(f"  Sharpe > 0 in {(boot_sharpes > 0).sum()}/{N_BOOT} = {(boot_sharpes > 0).mean()*100:.1f}% of bootstraps")

# --- 1b. Bootstrap CI bands on cumulative returns ---
print("\n--- Bootstrap CI bands on cumulative returns ---")
boot_cum_paths = np.zeros((N_BOOT, n_rf))
for b in range(N_BOOT):
    starts = rng.randint(0, n_rf, size=n_blocks)
    idx_list = []
    for s in starts:
        block_idx = np.arange(s, s + BLOCK_SIZE) % n_rf
        idx_list.append(block_idx)
    all_idx = np.concatenate(idx_list)[:n_rf]
    boot_ret = rf_arr[all_idx]
    boot_cum_paths[b] = np.cumprod(1 + boot_ret)

ci_lo_cum = np.percentile(boot_cum_paths, 5, axis=0)
ci_hi_cum = np.percentile(boot_cum_paths, 95, axis=0)
actual_cum = (1 + rf_ret).cumprod().values

# --- 1c. Stress test on OOS-period stress scenarios ---
print("\n--- Stress Test: RF vs SPY on OOS-relevant scenarios ---")

# Filter scenarios that overlap with OOS period
oos_start_date = oos_dates[0]
oos_end_date = oos_dates[-1]

stress_results = []
for scenario_name, (s_start, s_end) in STRESS_SCENARIOS.items():
    s_start_dt = pd.Timestamp(s_start)
    s_end_dt = pd.Timestamp(s_end)

    # Check overlap with OOS
    if s_end_dt < oos_start_date or s_start_dt > oos_end_date:
        continue

    # Clip to OOS range
    eff_start = max(s_start_dt, oos_start_date)
    eff_end = min(s_end_dt, oos_end_date)

    mask = (rf_ret.index >= eff_start) & (rf_ret.index <= eff_end)
    if mask.sum() < 5:
        continue

    rf_stress = rf_ret[mask]
    spy_stress = spy_ret[mask]

    rf_cum_stress = (1 + rf_stress).prod() - 1
    spy_cum_stress = (1 + spy_stress).prod() - 1

    stress_results.append({
        "Scenario": scenario_name.replace("_", " ").title(),
        "Period": f"{eff_start.strftime('%Y-%m')} to {eff_end.strftime('%Y-%m')}",
        "Days": mask.sum(),
        "RF Return": rf_cum_stress,
        "SPY Return": spy_cum_stress,
        "RF - SPY": rf_cum_stress - spy_cum_stress,
    })
    print(f"  {scenario_name:25s}  RF={rf_cum_stress:+.4f}  SPY={spy_cum_stress:+.4f}  diff={rf_cum_stress-spy_cum_stress:+.4f}")

stress_df = pd.DataFrame(stress_results)
if len(stress_df) == 0:
    print("  No stress scenarios overlap with OOS period with enough data.")

# --- 1d. Drawdown comparison: RF vs SPY ---
rf_dd = compute_drawdown_series(rf_ret)
spy_dd = compute_drawdown_series(spy_ret)

print(f"\n  RF  Max Drawdown: {rf_dd.min():.4f}")
print(f"  SPY Max Drawdown: {spy_dd.min():.4f}")
print(f"  RF drawdown advantage: {spy_dd.min() - rf_dd.min():.4f} (less negative = better)")

# --- Figure 29: RF Deep Dive (3-panel) ---
print("\n--- Generating fig29_rf_deepdive.pdf ---")
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# Panel 1: Cumulative returns with CI bands
ax1 = axes[0]
ax1.fill_between(rf_ret.index, ci_lo_cum, ci_hi_cum, alpha=0.2, color="#1f77b4",
                 label="90% Bootstrap CI")
ax1.plot(rf_ret.index, actual_cum, color="#1f77b4", linewidth=2.0, label="RF Walk-Forward")
spy_cum = (1 + spy_ret).cumprod()
ax1.plot(spy_ret.index, spy_cum.values, color="#7f7f7f", linewidth=1.5, linestyle="--", label="SPY")
ax1.set_ylabel("Cumulative Return (Growth of \\$1)")
ax1.set_title("RF Walk-Forward: Cumulative Returns with Bootstrap Confidence Interval (2022--2024)")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhline(1.0, color="black", linewidth=0.5, linestyle=":")

# Panel 2: Drawdown comparison
ax2 = axes[1]
ax2.fill_between(rf_dd.index, rf_dd.values, 0, alpha=0.3, color="#1f77b4", label="RF Drawdown")
ax2.fill_between(spy_dd.index, spy_dd.values, 0, alpha=0.3, color="#d62728", label="SPY Drawdown")
ax2.plot(rf_dd.index, rf_dd.values, color="#1f77b4", linewidth=1.0)
ax2.plot(spy_dd.index, spy_dd.values, color="#d62728", linewidth=1.0)
ax2.set_ylabel("Drawdown")
ax2.set_title("Drawdown Profile: RF vs SPY")
ax2.legend(loc="lower left", fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=min(rf_dd.min(), spy_dd.min()) * 1.1)

# Panel 3: Stress test bars
ax3 = axes[2]
if len(stress_df) > 0:
    x_labels = stress_df["Scenario"].values
    x_pos = np.arange(len(x_labels))
    bar_width = 0.35

    bars_rf = ax3.bar(x_pos - bar_width/2, stress_df["RF Return"].values * 100,
                      bar_width, label="RF", color="#1f77b4", alpha=0.8)
    bars_spy = ax3.bar(x_pos + bar_width/2, stress_df["SPY Return"].values * 100,
                       bar_width, label="SPY", color="#d62728", alpha=0.8)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("Cumulative Return (%)")
    ax3.set_title("Stress Scenario Returns: RF vs SPY")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.axhline(0, color="black", linewidth=0.5)
else:
    ax3.text(0.5, 0.5, "No stress scenarios in OOS period", transform=ax3.transAxes,
             ha="center", va="center", fontsize=14)

fig.tight_layout()
fig29_path = os.path.join(RES, "fig29_rf_deepdive.pdf")
fig.savefig(fig29_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {fig29_path}")


# ============================================================================
# PART 2: ENSEMBLE OF WF STRATEGIES
# ============================================================================
print("\n" + "=" * 72)
print("PART 2: ENSEMBLE OF WF STRATEGIES")
print("=" * 72)

# Equal-weight average of the 3 strategy returns
ensemble_df = pd.DataFrame({
    "RF": wf_returns["WF_RF"],
    "LightGBM": wf_returns["WF_LightGBM"],
    "Lasso": wf_returns["WF_Lasso"],
})
ensemble_ret = ensemble_df.mean(axis=1)
ensemble_ret.name = "Ensemble"

ens_metrics = compute_metrics(ensemble_ret)
rf_metrics_full = compute_metrics(wf_returns["WF_RF"])
lgbm_metrics = compute_metrics(wf_returns["WF_LightGBM"])
lasso_metrics = compute_metrics(wf_returns["WF_Lasso"])
spy_metrics = compute_metrics(spy_ret)

print(f"\n{'Strategy':25s} {'Sharpe':>8s} {'Ann Ret':>10s} {'Ann Vol':>10s} {'Max DD':>10s} {'Total':>10s}")
print("-" * 75)
for label, m in [("WF RandomForest", rf_metrics_full),
                 ("WF LightGBM", lgbm_metrics),
                 ("WF Lasso", lasso_metrics),
                 ("Ensemble (EW Avg)", ens_metrics),
                 ("SPY", spy_metrics)]:
    print(f"{label:25s} {m['Sharpe']:8.4f} {m['Ann Return']:+10.4f} {m['Ann Vol']:10.4f} {m['Max DD']:10.4f} {m['Total Return']:+10.4f}")

# Does ensemble beat the single best?
best_single_sharpe = max(rf_metrics_full["Sharpe"], lgbm_metrics["Sharpe"], lasso_metrics["Sharpe"])
best_single_name = "RF" if rf_metrics_full["Sharpe"] == best_single_sharpe else (
    "LightGBM" if lgbm_metrics["Sharpe"] == best_single_sharpe else "Lasso"
)
print(f"\nBest single strategy:  {best_single_name} (Sharpe={best_single_sharpe:.4f})")
print(f"Ensemble Sharpe:       {ens_metrics['Sharpe']:.4f}")
if ens_metrics["Sharpe"] > best_single_sharpe:
    print(f"  => Ensemble BEATS the best single by {ens_metrics['Sharpe'] - best_single_sharpe:.4f}")
else:
    print(f"  => Ensemble does NOT beat best single (delta={ens_metrics['Sharpe'] - best_single_sharpe:.4f})")
    print(f"     However, ensemble Max DD = {ens_metrics['Max DD']:.4f} vs best single Max DD = {rf_metrics_full['Max DD']:.4f}")

# Does ensemble improve drawdown?
ens_dd = compute_drawdown_series(ensemble_ret).min()
rf_dd_val = compute_drawdown_series(wf_returns["WF_RF"]).min()
print(f"\n  Ensemble Max DD: {ens_dd:.4f}")
print(f"  RF Max DD:       {rf_dd_val:.4f}")
if ens_dd > rf_dd_val:
    print(f"  => Ensemble has BETTER (smaller) drawdown by {ens_dd - rf_dd_val:.4f}")

# --- Figure 30: Ensemble vs Single ---
print("\n--- Generating fig30_ensemble_vs_single.pdf ---")
fig, ax = plt.subplots(figsize=(14, 7))

# Plot all WF strategies
cum_rf = (1 + wf_returns["WF_RF"]).cumprod()
cum_lgbm = (1 + wf_returns["WF_LightGBM"]).cumprod()
cum_lasso = (1 + wf_returns["WF_Lasso"]).cumprod()
cum_ens = (1 + ensemble_ret).cumprod()
cum_spy = (1 + spy_ret).cumprod()

ax.plot(cum_rf.index, cum_rf.values, label="WF RandomForest", color="#1f77b4", linewidth=1.5)
ax.plot(cum_lgbm.index, cum_lgbm.values, label="WF LightGBM", color="#2ca02c", linewidth=1.5)
ax.plot(cum_lasso.index, cum_lasso.values, label="WF Lasso", color="#ff7f0e", linewidth=1.5)
ax.plot(cum_ens.index, cum_ens.values, label="Ensemble (EW Avg)", color="#d62728",
        linewidth=2.5, linestyle="-")
ax.plot(cum_spy.index, cum_spy.values, label="SPY", color="#7f7f7f", linewidth=1.5, linestyle="--")

ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return (Growth of \\$1)")
ax.set_title("Ensemble vs Individual Walk-Forward Strategies (2022--2024)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")

fig.tight_layout()
fig30_path = os.path.join(RES, "fig30_ensemble_vs_single.pdf")
fig.savefig(fig30_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {fig30_path}")


# ============================================================================
# PART 3: STRATEGY CORRELATION
# ============================================================================
print("\n" + "=" * 72)
print("PART 3: STRATEGY CORRELATION")
print("=" * 72)

all_strat_returns = pd.DataFrame({
    "WF RF": wf_returns["WF_RF"],
    "WF LightGBM": wf_returns["WF_LightGBM"],
    "WF Lasso": wf_returns["WF_Lasso"],
    "Ensemble": ensemble_ret,
    "SPY": spy_ret,
    "Equal Weight": ew_ret,
})

corr_matrix = all_strat_returns.corr()

print("\nPairwise Correlation of Daily Returns:")
print(corr_matrix.to_string(float_format="%.4f"))

# Highlight diversification benefit
ml_strats = ["WF RF", "WF LightGBM", "WF Lasso"]
print("\n--- ML Strategy Pairwise Correlations ---")
for i in range(len(ml_strats)):
    for j in range(i+1, len(ml_strats)):
        s1, s2 = ml_strats[i], ml_strats[j]
        c = corr_matrix.loc[s1, s2]
        print(f"  {s1} vs {s2}: {c:.4f}")

avg_ml_corr = np.mean([corr_matrix.loc[ml_strats[i], ml_strats[j]]
                        for i in range(len(ml_strats))
                        for j in range(i+1, len(ml_strats))])
print(f"\n  Average pairwise ML strategy correlation: {avg_ml_corr:.4f}")
if avg_ml_corr < 0.90:
    print(f"  => Correlation < 0.90: meaningful diversification benefit exists")
else:
    print(f"  => Correlation >= 0.90: strategies are highly similar; limited diversification benefit")

# --- Figure 31: Correlation Heatmap ---
print("\n--- Generating fig31_strategy_correlation.pdf ---")
fig, ax = plt.subplots(figsize=(10, 8))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="RdYlBu_r",
            center=0.5, vmin=0, vmax=1,
            square=True, linewidths=0.5,
            mask=None,  # show full matrix
            ax=ax,
            cbar_kws={"label": "Correlation", "shrink": 0.8})

ax.set_title("Strategy Return Correlations (Daily, OOS 2022--2024)")
fig.tight_layout()
fig31_path = os.path.join(RES, "fig31_strategy_correlation.pdf")
fig.savefig(fig31_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {fig31_path}")


# ============================================================================
# PART 4: ROLLING SHARPE COMPARISON
# ============================================================================
print("\n" + "=" * 72)
print("PART 4: ROLLING SHARPE COMPARISON")
print("=" * 72)

ROLLING_WINDOW = 126  # 6 months

def rolling_sharpe(ret_series, window=ROLLING_WINDOW):
    """Compute rolling annualized Sharpe ratio."""
    rolling_mean = ret_series.rolling(window=window).mean() * 252
    rolling_std = ret_series.rolling(window=window).std(ddof=1) * np.sqrt(252)
    return (rolling_mean / rolling_std).dropna()

strategies_for_rolling = {
    "WF RF": wf_returns["WF_RF"],
    "WF LightGBM": wf_returns["WF_LightGBM"],
    "WF Lasso": wf_returns["WF_Lasso"],
    "Ensemble": ensemble_ret,
    "SPY": spy_ret,
}

rolling_sharpes = {}
print(f"\n126-day Rolling Sharpe (latest values):")
for name, ret_s in strategies_for_rolling.items():
    rs = rolling_sharpe(ret_s)
    rolling_sharpes[name] = rs
    latest = rs.iloc[-1] if len(rs) > 0 else np.nan
    mean_rs = rs.mean()
    print(f"  {name:20s}  latest={latest:+.4f}  mean={mean_rs:+.4f}  std={rs.std():.4f}")

# --- Figure 32: Rolling Sharpe ---
print("\n--- Generating fig32_rolling_sharpe_all.pdf ---")
fig, ax = plt.subplots(figsize=(14, 7))

colors = {
    "WF RF": "#1f77b4",
    "WF LightGBM": "#2ca02c",
    "WF Lasso": "#ff7f0e",
    "Ensemble": "#d62728",
    "SPY": "#7f7f7f",
}
styles = {
    "WF RF": "-",
    "WF LightGBM": "-",
    "WF Lasso": "-",
    "Ensemble": "-",
    "SPY": "--",
}
linewidths = {
    "WF RF": 1.5,
    "WF LightGBM": 1.2,
    "WF Lasso": 1.2,
    "Ensemble": 2.5,
    "SPY": 1.5,
}

for name, rs in rolling_sharpes.items():
    ax.plot(rs.index, rs.values, label=name,
            color=colors.get(name, "gray"),
            linestyle=styles.get(name, "-"),
            linewidth=linewidths.get(name, 1.0))

ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
ax.set_xlabel("Date")
ax.set_ylabel("Rolling Sharpe Ratio (Annualized)")
ax.set_title(f"{ROLLING_WINDOW}-Day Rolling Sharpe Ratio: All Strategies (2022--2024)")
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig32_path = os.path.join(RES, "fig32_rolling_sharpe_all.pdf")
fig.savefig(fig32_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {fig32_path}")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 72)
print("FINAL SUMMARY")
print("=" * 72)

total_time = time.time() - t_global
print(f"\nTotal runtime: {total_time:.1f}s")

print(f"\n{'Strategy':25s} {'Sharpe':>8s} {'Ann Ret':>10s} {'Ann Vol':>10s} {'Max DD':>10s}")
print("-" * 65)
all_summary = {
    "WF RandomForest": rf_metrics_full,
    "WF LightGBM": lgbm_metrics,
    "WF Lasso": lasso_metrics,
    "Ensemble (EW Avg)": ens_metrics,
    "SPY": spy_metrics,
    "Equal Weight": compute_metrics(ew_ret),
    "60/40": compute_metrics(ret_6040),
}
for label, m in all_summary.items():
    print(f"{label:25s} {m['Sharpe']:8.4f} {m['Ann Return']:+10.4f} {m['Ann Vol']:10.4f} {m['Max DD']:10.4f}")

print(f"\nRF Sharpe 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Avg ML strategy pairwise correlation: {avg_ml_corr:.4f}")

print(f"\nFigures saved to {RES}:")
print(f"  fig29_rf_deepdive.pdf")
print(f"  fig30_ensemble_vs_single.pdf")
print(f"  fig31_strategy_correlation.pdf")
print(f"  fig32_rolling_sharpe_all.pdf")

print(f"\n{'=' * 72}")
print("DONE")
print(f"{'=' * 72}")
