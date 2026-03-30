#!/usr/bin/env python3
"""
Part 1: Monthly Returns Heatmap (fig33)
Part 2: Risk Parity Walk-Forward Comparison (fig34)

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
import matplotlib.colors as mcolors
import cvxpy as cp

from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from src.config import (
    TICKER_LIST, RESULTS_DIR, RANDOM_STATE,
    PREDICTION_HORIZON, LATEX_FONT_SETTINGS, FIGURE_DPI,
)

plt.rcParams.update(LATEX_FONT_SETTINGS)
np.random.seed(RANDOM_STATE)

# ============================================================================
# Configuration
# ============================================================================
OOS_START = "2022-01-01"
RETRAIN_EVERY = 63          # quarterly
HORIZON = PREDICTION_HORIZON  # 21 days
TC_BPS = 10
TICKERS = TICKER_LIST
N_ASSETS = len(TICKERS)

# ============================================================================
# Data loading (same as walk-forward scripts)
# ============================================================================
print("=" * 72, flush=True)
print("HEATMAP & RISK PARITY WALK-FORWARD", flush=True)
print("=" * 72, flush=True)

t0 = time.time()

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

print(f"Features shape: {features.shape}", flush=True)
print(f"Date range:     {features.index[0].date()} -> {features.index[-1].date()}", flush=True)

# ============================================================================
# Macro-only columns
# ============================================================================
FRED_PREFIXES = [
    "DFF", "DGS2", "DGS10", "T10Y2Y", "T10Y3M", "VIXCLS",
    "BAMLH0A0HYM2", "BAMLC0A4CBBB", "DTWEXBGS", "UMCSENT",
    "UNRATE", "ICSA", "CPIAUCSL", "T10YIE", "PPIACO",
    "M2SL", "USSLIND", "HOUST",
]

all_cols = list(features.columns)
macro_cols = [c for c in all_cols
              if any(c == pfx or c.startswith(pfx + "_lag") for pfx in FRED_PREFIXES)]
print(f"Macro-only features: {len(macro_cols)}", flush=True)

# ============================================================================
# Forward returns (21-day)
# ============================================================================
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
print(f"OOS days: {n_oos}  ({oos_dates[0].date()} -> {oos_dates[-1].date()})", flush=True)

rebal_oos = np.arange(0, n_oos, RETRAIN_EVERY)
print(f"Rebalance points: {len(rebal_oos)}", flush=True)


# ============================================================================
# Helper: MV optimisation
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
# Helper: Risk Parity weights
# ============================================================================
def risk_parity_weights(cov):
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(500):
        sigma_w = cov @ w
        rc = w * sigma_w
        target = rc.sum() / n
        w = w * (target / np.maximum(rc, 1e-10))
        w = np.maximum(w, 0)
        w /= w.sum()
    return w


# ============================================================================
# Helper: Inverse volatility weights
# ============================================================================
def inv_vol_weights(cov):
    vols = np.sqrt(np.diag(cov))
    vols = np.maximum(vols, 1e-12)
    inv_v = 1.0 / vols
    return inv_v / inv_v.sum()


# ============================================================================
# Shared: compute covariance at a given global position
# ============================================================================
def get_cov(global_pos, lookback=252):
    lb_start = max(0, global_pos - lookback)
    ret_window = daily_ret_vals[lb_start:global_pos]
    clean_mask = ~np.isnan(ret_window).any(axis=1)
    ret_clean = ret_window[clean_mask]
    if len(ret_clean) < 30:
        return None
    lw = LedoitWolf().fit(ret_clean)
    cov_mat = lw.covariance_ * 252.0
    eigvals = np.linalg.eigvalsh(cov_mat)
    if eigvals.min() < 0:
        cov_mat += (-eigvals.min() + 1e-8) * np.eye(N_ASSETS)
    return cov_mat


# ============================================================================
# Shared: train models and get predictions at a rebalance point
# ============================================================================
def get_predictions(global_pos, model_type="random_forest"):
    X_train = X_macro_full[:global_pos]
    Y_train = fwd_vals[:global_pos]
    valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(Y_train).any(axis=1)
    X_tr = X_train[valid]
    Y_tr = Y_train[valid]

    if len(X_tr) < 100:
        return None

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)

    X_now = X_macro_full[global_pos:global_pos+1].copy()
    if np.isnan(X_now).any():
        X_now = X_tr[-1:].copy()
    X_now_scaled = scaler.transform(X_now)

    mu_pred = np.zeros(N_ASSETS)
    for k in range(N_ASSETS):
        if model_type == "random_forest":
            mdl = RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=5,
                random_state=RANDOM_STATE, n_jobs=-1,
            )
        elif model_type == "lightgbm":
            mdl = LGBMRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                num_leaves=31, verbose=-1, random_state=RANDOM_STATE,
                n_jobs=1,
            )
        else:
            raise ValueError(f"Unknown model: {model_type}")
        mdl.fit(X_tr_scaled, Y_tr[:, k])
        mu_pred[k] = mdl.predict(X_now_scaled)[0]

    return mu_pred


# ============================================================================
# Compute portfolio returns with transaction costs
# ============================================================================
def compute_strategy_returns(weights_arr):
    oos_daily = daily_ret_vals[oos_locs]
    port_ret = (weights_arr * oos_daily).sum(axis=1)
    turnover = np.abs(np.diff(weights_arr, axis=0)).sum(axis=1)
    turnover = np.concatenate([[0.0], turnover])
    tc = turnover * (TC_BPS / 10000.0)
    return pd.Series(port_ret - tc, index=oos_dates, name="returns")


# ============================================================================
# SPY benchmark returns
# ============================================================================
spy_col = TICKERS.index("SPY")
spy_daily_oos = daily_ret_vals[oos_locs, spy_col]
spy_ret = pd.Series(spy_daily_oos, index=oos_dates, name="SPY")


# ############################################################################
# PART 1: Random Forest Walk-Forward (for heatmap)
# ############################################################################
print("\n" + "=" * 72, flush=True)
print("PART 1: RF Walk-Forward for Monthly Returns Heatmap", flush=True)
print("=" * 72, flush=True)

t1 = time.time()

rf_weights = np.full((n_oos, N_ASSETS), 1.0 / N_ASSETS)

for ri in range(len(rebal_oos)):
    oos_pos = rebal_oos[ri]
    global_pos = oos_locs[oos_pos]

    mu_pred = get_predictions(global_pos, model_type="random_forest")
    cov_mat = get_cov(global_pos)

    if mu_pred is None or cov_mat is None:
        next_pos = rebal_oos[ri + 1] if ri + 1 < len(rebal_oos) else n_oos
        continue

    mu_annual = mu_pred * (252.0 / HORIZON)
    w_opt = mv_optimize(mu_annual, cov_mat, lam=5, max_w=0.50)

    next_pos = rebal_oos[ri + 1] if ri + 1 < len(rebal_oos) else n_oos
    rf_weights[oos_pos:next_pos] = w_opt

rf_ret = compute_strategy_returns(rf_weights)
print(f"RF Walk-Forward done in {time.time()-t1:.1f}s | Ann.Ret={rf_ret.mean()*252:.4f}", flush=True)

# ============================================================================
# Monthly returns computation
# ============================================================================
def compute_monthly_returns(daily_returns):
    """Compute monthly total returns from daily returns."""
    monthly = (1 + daily_returns).resample("ME").prod() - 1
    return monthly

rf_monthly = compute_monthly_returns(rf_ret)
spy_monthly = compute_monthly_returns(spy_ret)

print(f"RF monthly returns: {len(rf_monthly)} months", flush=True)
print(f"SPY monthly returns: {len(spy_monthly)} months", flush=True)

# ============================================================================
# Figure 33: Monthly Returns Heatmap
# ============================================================================
print("\nGenerating fig33_monthly_heatmap.pdf ...", flush=True)

years = [2022, 2023, 2024]
months = list(range(1, 13))
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def build_heatmap_matrix(monthly_ret, years, months):
    """Build a (12 x len(years)) matrix of monthly returns."""
    mat = np.full((12, len(years)), np.nan)
    for j, yr in enumerate(years):
        for i, mo in enumerate(months):
            mask = (monthly_ret.index.year == yr) & (monthly_ret.index.month == mo)
            vals = monthly_ret[mask]
            if len(vals) > 0:
                mat[i, j] = vals.iloc[0]
    return mat

rf_mat = build_heatmap_matrix(rf_monthly, years, months)
spy_mat = build_heatmap_matrix(spy_monthly, years, months)

# Diverging colormap: red for negative, green for positive
vmax = max(np.nanmax(np.abs(rf_mat)), np.nanmax(np.abs(spy_mat)))
vmax = max(vmax, 0.01)  # at least 1%

# Custom green-red diverging colormap (red=negative, green=positive)
cmap_colors = [
    (0.8, 0.1, 0.1),   # dark red
    (1.0, 0.4, 0.4),   # light red
    (1.0, 1.0, 1.0),   # white (zero)
    (0.4, 0.8, 0.4),   # light green
    (0.1, 0.6, 0.1),   # dark green
]
cmap = mcolors.LinearSegmentedColormap.from_list("RedGreen", cmap_colors, N=256)

fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)

for ax, mat, title in zip(axes, [rf_mat, spy_mat],
                           ["RF Walk-Forward Strategy", "SPY Buy-and-Hold"]):
    im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    # Annotate each cell
    for i in range(12):
        for j in range(len(years)):
            val = mat[i, j]
            if np.isnan(val):
                ax.text(j, i, "--", ha="center", va="center",
                        fontsize=9, color="gray")
            else:
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val*100:.1f}\\%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years])
    ax.set_yticks(range(12))
    ax.set_yticklabels(month_names)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(axis="both", which="both", length=0)

    # Grid lines
    for i in range(13):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)
    for j in range(len(years) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.5)

fig.suptitle("Monthly Returns Heatmap (2022--2024)", fontsize=14, fontweight="bold", y=0.98)
cbar = fig.colorbar(im, ax=axes, shrink=0.6, aspect=25, pad=0.04)
cbar.set_label("Monthly Return", fontsize=10)
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}\\%"))

fig.tight_layout(rect=[0, 0, 0.92, 0.95])
fig33_path = os.path.join(str(RESULTS_DIR), "fig33_monthly_heatmap.pdf")
fig.savefig(fig33_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {fig33_path}", flush=True)


# ############################################################################
# PART 2: Risk Parity Walk-Forward Comparison
# ############################################################################
print("\n" + "=" * 72, flush=True)
print("PART 2: Risk Parity Walk-Forward Comparison", flush=True)
print("=" * 72, flush=True)

# Strategy 1: MV Walk-Forward (LightGBM, lam=5, max_w=0.50)
print("\n--- MV Walk-Forward (LightGBM) ---", flush=True)
t2 = time.time()

mv_weights = np.full((n_oos, N_ASSETS), 1.0 / N_ASSETS)
lgbm_predictions_cache = {}  # cache predictions for reuse

for ri in range(len(rebal_oos)):
    oos_pos = rebal_oos[ri]
    global_pos = oos_locs[oos_pos]

    mu_pred = get_predictions(global_pos, model_type="lightgbm")
    cov_mat = get_cov(global_pos)

    if mu_pred is None or cov_mat is None:
        next_pos = rebal_oos[ri + 1] if ri + 1 < len(rebal_oos) else n_oos
        continue

    lgbm_predictions_cache[ri] = (mu_pred, cov_mat)
    mu_annual = mu_pred * (252.0 / HORIZON)
    w_opt = mv_optimize(mu_annual, cov_mat, lam=5, max_w=0.50)

    next_pos = rebal_oos[ri + 1] if ri + 1 < len(rebal_oos) else n_oos
    mv_weights[oos_pos:next_pos] = w_opt

mv_ret = compute_strategy_returns(mv_weights)
print(f"    Done in {time.time()-t2:.1f}s | Ann.Ret={mv_ret.mean()*252:.4f}", flush=True)

# Strategy 2: Risk Parity Walk-Forward (LightGBM predictions, risk parity allocation)
print("\n--- Risk Parity Walk-Forward (LightGBM) ---", flush=True)
t3 = time.time()

rp_weights = np.full((n_oos, N_ASSETS), 1.0 / N_ASSETS)

for ri in range(len(rebal_oos)):
    oos_pos = rebal_oos[ri]
    global_pos = oos_locs[oos_pos]

    cov_mat = get_cov(global_pos)
    if cov_mat is None:
        next_pos = rebal_oos[ri + 1] if ri + 1 < len(rebal_oos) else n_oos
        continue

    # Use risk parity with ML-adjusted risk budgets
    # Assets predicted to have higher returns get larger risk budgets
    if ri in lgbm_predictions_cache:
        mu_pred = lgbm_predictions_cache[ri][0]
        # Convert predictions to risk budgets: softmax of predictions
        # so better-predicted assets get more risk budget
        mu_shifted = mu_pred - mu_pred.mean()
        risk_budgets = np.exp(mu_shifted * 10)  # temperature-scaled softmax
        risk_budgets /= risk_budgets.sum()
        # Ensure minimum budget
        risk_budgets = np.maximum(risk_budgets, 0.02)
        risk_budgets /= risk_budgets.sum()
    else:
        risk_budgets = np.ones(N_ASSETS) / N_ASSETS

    w_rp = risk_parity_weights(cov_mat)
    # Blend: 70% risk parity, 30% ML-adjusted budgets via reoptimization
    # Actually use the ML-adjusted budgets in the risk parity framework
    # Recompute with custom budgets
    n = cov_mat.shape[0]
    w = np.ones(n) / n
    for _ in range(500):
        sigma_w = cov_mat @ w
        rc = w * sigma_w
        total_rc = rc.sum()
        target = risk_budgets * total_rc
        w = w * (target / np.maximum(rc, 1e-10))
        w = np.maximum(w, 0)
        w /= w.sum()

    next_pos = rebal_oos[ri + 1] if ri + 1 < len(rebal_oos) else n_oos
    rp_weights[oos_pos:next_pos] = w

rp_ret = compute_strategy_returns(rp_weights)
print(f"    Done in {time.time()-t3:.1f}s | Ann.Ret={rp_ret.mean()*252:.4f}", flush=True)

# Strategy 3: Inverse Volatility Walk-Forward (no ML needed)
print("\n--- Inverse Volatility Walk-Forward ---", flush=True)
t4 = time.time()

iv_weights = np.full((n_oos, N_ASSETS), 1.0 / N_ASSETS)

for ri in range(len(rebal_oos)):
    oos_pos = rebal_oos[ri]
    global_pos = oos_locs[oos_pos]

    cov_mat = get_cov(global_pos)
    if cov_mat is None:
        next_pos = rebal_oos[ri + 1] if ri + 1 < len(rebal_oos) else n_oos
        continue

    w_iv = inv_vol_weights(cov_mat)

    next_pos = rebal_oos[ri + 1] if ri + 1 < len(rebal_oos) else n_oos
    iv_weights[oos_pos:next_pos] = w_iv

iv_ret = compute_strategy_returns(iv_weights)
print(f"    Done in {time.time()-t4:.1f}s | Ann.Ret={iv_ret.mean()*252:.4f}", flush=True)

# ============================================================================
# Metrics
# ============================================================================
def compute_metrics(returns, spy_returns):
    r = returns.dropna()
    if len(r) < 10:
        return {k: np.nan for k in [
            "Ann. Return", "Ann. Volatility", "Sharpe", "Sortino",
            "Max Drawdown", "Total Return"]}

    ann_ret = r.mean() * 252
    ann_vol = r.std(ddof=1) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    downside = r[r < 0]
    down_std = downside.std(ddof=1) * np.sqrt(252) if len(downside) > 1 else np.nan
    sortino = ann_ret / down_std if (not np.isnan(down_std) and down_std > 0) else np.nan

    cum = (1 + r).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    total_ret = cum.iloc[-1] - 1

    return {
        "Ann. Return": ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Total Return": total_ret,
    }


strategies = {
    "MV Walk-Forward (LightGBM)": mv_ret,
    "Risk Parity Walk-Forward (LightGBM)": rp_ret,
    "Inverse Volatility Walk-Forward": iv_ret,
    "SPY": spy_ret,
}

print(f"\n{'=' * 72}", flush=True)
print("PERFORMANCE METRICS", flush=True)
print(f"{'=' * 72}", flush=True)

metrics_rows = {}
for name, ret in strategies.items():
    metrics_rows[name] = compute_metrics(ret, spy_ret)

metrics_df = pd.DataFrame(metrics_rows).T
metrics_df.index.name = "Strategy"

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)
print(metrics_df.to_string(), flush=True)

# Save metrics CSV
csv_path = os.path.join(str(RESULTS_DIR), "riskparity_comparison.csv")
metrics_df.to_csv(csv_path)
print(f"\nSaved: {csv_path}", flush=True)

# ============================================================================
# Figure 34: Cumulative Returns Comparison
# ============================================================================
print("\nGenerating fig34_riskparity_comparison.pdf ...", flush=True)

fig, ax = plt.subplots(figsize=(14, 7))

colors = {
    "MV Walk-Forward (LightGBM)": "#1f77b4",
    "Risk Parity Walk-Forward (LightGBM)": "#d62728",
    "Inverse Volatility Walk-Forward": "#2ca02c",
    "SPY": "#7f7f7f",
}
styles = {
    "MV Walk-Forward (LightGBM)": "-",
    "Risk Parity Walk-Forward (LightGBM)": "-",
    "Inverse Volatility Walk-Forward": "-",
    "SPY": "--",
}

for name, ret in strategies.items():
    cum = (1 + ret).cumprod()
    ax.plot(cum.index, cum.values, label=name,
            color=colors[name], linestyle=styles[name],
            linewidth=2.0 if name != "SPY" else 1.5)

ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return (Growth of \\$1)")
ax.set_title("Walk-Forward Comparison: MV vs Risk Parity vs Inverse Volatility (2022--2024)")
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")

fig.tight_layout()
fig34_path = os.path.join(str(RESULTS_DIR), "fig34_riskparity_comparison.pdf")
fig.savefig(fig34_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {fig34_path}", flush=True)

# ============================================================================
# Done
# ============================================================================
total_time = time.time() - t0
print(f"\n{'=' * 72}", flush=True)
print(f"TOTAL RUNTIME: {total_time:.1f}s", flush=True)
print(f"{'=' * 72}", flush=True)
print(f"\nOutputs in {str(RESULTS_DIR)}:", flush=True)
print("  fig33_monthly_heatmap.pdf", flush=True)
print("  fig34_riskparity_comparison.pdf", flush=True)
print("  riskparity_comparison.csv", flush=True)
