#!/usr/bin/env python3
"""
Walk-forward backtest for top-5 strategy configurations.

Quarterly retraining (63-day steps), expanding training window,
macro-only features, Ledoit-Wolf covariance, MV optimization,
10 bps transaction costs.

Outputs:
    data/results/fig23_walkforward_all.pdf
    data/results/table_walkforward_all.tex
    data/results/walkforward_all_results.csv

Author : Arhan Subasi
"""

import sys, os, time, warnings
warnings.filterwarnings("ignore")

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cvxpy as cp

from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor

from src.config import (
    TICKER_LIST, RESULTS_DIR, RANDOM_STATE,
    PREDICTION_HORIZON, LATEX_FONT_SETTINGS, FIGURE_DPI,
)

# Apply thesis font settings
plt.rcParams.update(LATEX_FONT_SETTINGS)
np.random.seed(RANDOM_STATE)

# ============================================================================
# Configuration
# ============================================================================
OOS_START = "2022-01-01"
RETRAIN_EVERY = 63          # trading days (~quarterly)
HORIZON = PREDICTION_HORIZON  # 21 days
TC_BPS = 10                  # 10 bps transaction cost
TICKERS = TICKER_LIST        # 12 ETFs
N_ASSETS = len(TICKERS)

CONFIGS = {
    "LightGBM_lam5_w50": {"model": "lightgbm", "lam": 5, "max_w": 0.50},
    "LightGBM_lam5_w35": {"model": "lightgbm", "lam": 5, "max_w": 0.35},
    "LightGBM_lam3_w50": {"model": "lightgbm", "lam": 3, "max_w": 0.50},
    "GBR_lam5_w50":      {"model": "gradient_boosting", "lam": 5, "max_w": 0.50},
    "RF_lam5_w50":       {"model": "random_forest", "lam": 5, "max_w": 0.50},
}

DISPLAY_NAMES = {
    "LightGBM_lam5_w50": r"LightGBM $\lambda$=5, w$_{\max}$=0.50",
    "LightGBM_lam5_w35": r"LightGBM $\lambda$=5, w$_{\max}$=0.35",
    "LightGBM_lam3_w50": r"LightGBM $\lambda$=3, w$_{\max}$=0.50",
    "GBR_lam5_w50":      r"GBR $\lambda$=5, w$_{\max}$=0.50",
    "RF_lam5_w50":       r"RF $\lambda$=5, w$_{\max}$=0.50",
}

# ============================================================================
# Data loading
# ============================================================================
print("=" * 72, flush=True)
print("WALK-FORWARD BACKTEST — TOP 5 STRATEGIES", flush=True)
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

# Daily returns from prices
daily_ret = prices.pct_change().dropna()

# Align dates
common_idx = features.index.intersection(daily_ret.index)
features = features.loc[common_idx]
daily_ret = daily_ret.loc[common_idx]

print(f"Features shape: {features.shape}", flush=True)
print(f"Date range:     {features.index[0].date()} -> {features.index[-1].date()}", flush=True)

# ============================================================================
# Identify macro-only feature columns
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
# Forward return targets (21-day)
# ============================================================================
fwd_dict = {}
for tk in TICKERS:
    fwd_dict[tk] = prices[tk].pct_change(HORIZON).shift(-HORIZON)
fwd_df = pd.DataFrame(fwd_dict, index=prices.index).reindex(features.index)

# Pre-extract numpy arrays for speed
X_macro_full = features[macro_cols].values           # (T, F)
X_macro_idx = features.index                         # DatetimeIndex
fwd_vals = fwd_df[TICKERS].values                    # (T, N)
daily_ret_vals = daily_ret[TICKERS].values           # (T, N)

# OOS setup
oos_mask = X_macro_idx >= OOS_START
oos_locs = np.where(oos_mask)[0]                     # integer positions
oos_dates = X_macro_idx[oos_mask]
n_oos = len(oos_locs)
print(f"OOS days: {n_oos}  ({oos_dates[0].date()} -> {oos_dates[-1].date()})", flush=True)

# Rebalance integer positions within OOS
rebal_oos = np.arange(0, n_oos, RETRAIN_EVERY)       # positions in oos_locs
print(f"Rebalance points: {len(rebal_oos)}", flush=True)


# ============================================================================
# Helper: create model
# ============================================================================
def make_model(name):
    if name == "lightgbm":
        return LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            num_leaves=31, verbose=-1, random_state=RANDOM_STATE,
            n_jobs=1,
        )
    elif name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE,
        )
    elif name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


# ============================================================================
# Helper: MV optimisation via CVXPY
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
# Walk-forward engine (optimised with numpy arrays)
# ============================================================================
def run_walk_forward(cfg):
    model_name = cfg["model"]
    lam = cfg["lam"]
    max_w = cfg["max_w"]

    # Weight array: (n_oos, N_ASSETS)
    weights_arr = np.full((n_oos, N_ASSETS), 1.0 / N_ASSETS)

    for ri in range(len(rebal_oos)):
        oos_pos = rebal_oos[ri]                    # position within OOS
        global_pos = oos_locs[oos_pos]             # position in full array

        # --- Training data: everything before this date ---
        X_train = X_macro_full[:global_pos]
        Y_train = fwd_vals[:global_pos]            # (T_train, N_ASSETS)

        # Valid mask: no NaN in features or any target
        valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(Y_train).any(axis=1)
        X_tr = X_train[valid]
        Y_tr = Y_train[valid]

        # Cap training to most recent 2520 rows (10yr) for speed
        MAX_TRAIN = 2520
        if len(X_tr) > MAX_TRAIN:
            X_tr = X_tr[-MAX_TRAIN:]
            Y_tr = Y_tr[-MAX_TRAIN:]

        if len(X_tr) < 100:
            # Use equal weight (already set)
            continue

        # Scale features
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)

        # Current-date features
        X_now = X_macro_full[global_pos:global_pos+1].copy()
        if np.isnan(X_now).any():
            # Forward fill: use last valid training row
            X_now = X_tr[-1:].copy()
        X_now_scaled = scaler.transform(X_now)

        # --- Predict expected returns for each asset ---
        mu_pred = np.zeros(N_ASSETS)
        for k in range(N_ASSETS):
            mdl = make_model(model_name)
            mdl.fit(X_tr_scaled, Y_tr[:, k])
            mu_pred[k] = mdl.predict(X_now_scaled)[0]

        # --- Covariance from trailing 252 days ---
        lb_start = max(0, global_pos - 252)
        ret_window = daily_ret_vals[lb_start:global_pos]
        clean_mask = ~np.isnan(ret_window).any(axis=1)
        ret_clean = ret_window[clean_mask]
        if len(ret_clean) < 30:
            continue

        lw = LedoitWolf().fit(ret_clean)
        cov_mat = lw.covariance_ * 252.0
        # Ensure PSD
        eigvals = np.linalg.eigvalsh(cov_mat)
        if eigvals.min() < 0:
            cov_mat += (-eigvals.min() + 1e-8) * np.eye(N_ASSETS)

        # Annualise predictions
        mu_annual = mu_pred * (252.0 / HORIZON)

        # --- Optimise ---
        w_opt = mv_optimize(mu_annual, cov_mat, lam, max_w)

        # Fill from this rebalance to next (vectorized)
        next_pos = rebal_oos[ri + 1] if ri + 1 < len(rebal_oos) else n_oos
        weights_arr[oos_pos:next_pos] = w_opt

    return weights_arr


# ============================================================================
# Compute portfolio returns with transaction costs
# ============================================================================
def compute_strategy_returns(weights_arr):
    oos_daily = daily_ret_vals[oos_locs]             # (n_oos, N_ASSETS)
    port_ret = (weights_arr * oos_daily).sum(axis=1)

    # Turnover & TC
    turnover = np.abs(np.diff(weights_arr, axis=0)).sum(axis=1)
    turnover = np.concatenate([[0.0], turnover])
    tc = turnover * (TC_BPS / 10000.0)
    return pd.Series(port_ret - tc, index=oos_dates, name="returns")


# ============================================================================
# Benchmarks
# ============================================================================
def compute_benchmarks():
    oos_daily = daily_ret_vals[oos_locs]

    # SPY
    spy_col = TICKERS.index("SPY")
    spy_ret = pd.Series(oos_daily[:, spy_col], index=oos_dates, name="SPY")

    # Equal Weight
    ew = np.ones((n_oos, N_ASSETS)) / N_ASSETS
    ew_port = (ew * oos_daily).sum(axis=1)
    ew_turn = np.abs(np.diff(ew, axis=0)).sum(axis=1)
    ew_turn = np.concatenate([[0.0], ew_turn])
    ew_tc = ew_turn * (TC_BPS / 10000.0)
    ew_ret = pd.Series(ew_port - ew_tc, index=oos_dates, name="Equal Weight")

    # 60/40
    w6040 = np.zeros((n_oos, N_ASSETS))
    spy_idx = TICKERS.index("SPY")
    agg_idx = TICKERS.index("AGG")
    w6040[:, spy_idx] = 0.60
    w6040[:, agg_idx] = 0.40
    port6040 = (w6040 * oos_daily).sum(axis=1)
    t6040 = np.abs(np.diff(w6040, axis=0)).sum(axis=1)
    t6040 = np.concatenate([[0.0], t6040])
    tc6040 = t6040 * (TC_BPS / 10000.0)
    ret6040 = pd.Series(port6040 - tc6040, index=oos_dates, name="60/40")

    return {"SPY": spy_ret, "Equal Weight": ew_ret, "60/40": ret6040}


# ============================================================================
# Metrics
# ============================================================================
def compute_metrics(returns, spy_returns):
    r = returns.dropna()
    if len(r) < 10:
        return {k: np.nan for k in [
            "Ann. Return", "Ann. Volatility", "Sharpe", "Sortino",
            "Max Drawdown", "Beta vs SPY", "Alpha vs SPY"]}

    ann_ret = r.mean() * 252
    ann_vol = r.std(ddof=1) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    downside = r[r < 0]
    down_std = downside.std(ddof=1) * np.sqrt(252) if len(downside) > 1 else np.nan
    sortino = ann_ret / down_std if (not np.isnan(down_std) and down_std > 0) else np.nan

    cum = (1 + r).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    # Beta / Alpha
    s_c = spy_returns.reindex(r.index).dropna()
    common = r.index.intersection(s_c.index)
    if len(common) > 10:
        rc = r.loc[common].values
        sc = s_c.loc[common].values
        cov_m = np.cov(rc, sc)
        beta = cov_m[0, 1] / cov_m[1, 1] if cov_m[1, 1] > 0 else np.nan
        alpha = ann_ret - beta * (sc.mean() * 252)
    else:
        beta = alpha = np.nan

    return {
        "Ann. Return": ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Beta vs SPY": beta,
        "Alpha vs SPY": alpha,
    }


# ============================================================================
# MAIN
# ============================================================================
all_returns = {}

for cfg_name, cfg in CONFIGS.items():
    print(f"\n--- {cfg_name}  (model={cfg['model']}, lam={cfg['lam']}, "
          f"max_w={cfg['max_w']}) ---", flush=True)
    t1 = time.time()

    w_arr = run_walk_forward(cfg)
    ret_s = compute_strategy_returns(w_arr)
    all_returns[cfg_name] = ret_s

    elapsed = time.time() - t1
    print(f"    Done in {elapsed:.1f}s | Ann.Ret={ret_s.mean()*252:.4f}", flush=True)

# Benchmarks
print("\n--- Benchmarks ---", flush=True)
benchmarks = compute_benchmarks()
for bname, bret in benchmarks.items():
    all_returns[bname] = bret
    print(f"    {bname}: Ann.Ret={bret.mean()*252:.4f}", flush=True)

spy_ret = benchmarks["SPY"]

# ============================================================================
# Metrics table
# ============================================================================
print(f"\n{'=' * 72}", flush=True)
print("PERFORMANCE METRICS", flush=True)
print(f"{'=' * 72}", flush=True)

metrics_rows = {}
for name, ret in all_returns.items():
    metrics_rows[name] = compute_metrics(ret, spy_ret)

metrics_df = pd.DataFrame(metrics_rows).T
metrics_df.index.name = "Strategy"

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)
print(metrics_df.to_string(), flush=True)

# ============================================================================
# Save CSV
# ============================================================================
csv_path = os.path.join(str(RESULTS_DIR), "walkforward_all_results.csv")
metrics_df.to_csv(csv_path)
print(f"\nSaved: {csv_path}", flush=True)

# ============================================================================
# Figure: cumulative returns
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

colors = {
    "LightGBM_lam5_w50": "#1f77b4",
    "LightGBM_lam5_w35": "#ff7f0e",
    "LightGBM_lam3_w50": "#2ca02c",
    "GBR_lam5_w50":      "#d62728",
    "RF_lam5_w50":       "#9467bd",
    "SPY":               "#7f7f7f",
    "Equal Weight":      "#8c564b",
    "60/40":             "#bcbd22",
}
styles = {k: "-" for k in CONFIGS}
styles.update({"SPY": "--", "Equal Weight": "--", "60/40": "--"})

plot_order = list(CONFIGS.keys()) + ["SPY", "Equal Weight", "60/40"]
for name in plot_order:
    ret = all_returns[name]
    cum = (1 + ret).cumprod()
    label = DISPLAY_NAMES.get(name, name)
    ax.plot(cum.index, cum.values, label=label,
            color=colors.get(name, "gray"),
            linestyle=styles.get(name, "-"),
            linewidth=2.0 if name in CONFIGS else 1.5)

ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return (Growth of $1)")
ax.set_title("Walk-Forward Backtest: Top 5 Strategies vs Benchmarks (2022--2024)")
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
fig.tight_layout()

fig_path = os.path.join(str(RESULTS_DIR), "fig23_walkforward_all.pdf")
fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {fig_path}", flush=True)

# ============================================================================
# LaTeX table
# ============================================================================
def to_latex_table(df, caption, label):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    col_fmt = "l" + "r" * len(df.columns)
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")
    header = "Strategy"
    for col in df.columns:
        header += " & " + col
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for idx, row in df.iterrows():
        display = DISPLAY_NAMES.get(idx, idx)
        s = display
        for col in df.columns:
            val = row[col]
            if col in ("Ann. Return", "Ann. Volatility", "Max Drawdown", "Alpha vs SPY"):
                s += f" & {val*100:.2f}\\%"
            else:
                s += f" & {val:.2f}"
        s += r" \\"
        if idx == "SPY":
            lines.append(r"\midrule")
        lines.append(s)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

ordered = list(CONFIGS.keys()) + ["SPY", "Equal Weight", "60/40"]
metrics_ordered = metrics_df.reindex(ordered)

latex_str = to_latex_table(
    metrics_ordered,
    caption="Walk-forward backtest performance (quarterly retraining, 2022--2024). "
            "Transaction costs of 10\\,bps applied. Covariance estimated via Ledoit-Wolf shrinkage.",
    label="tab:walkforward_all",
)
tex_path = os.path.join(str(RESULTS_DIR), "table_walkforward_all.tex")
with open(tex_path, "w") as f:
    f.write(latex_str)
print(f"Saved: {tex_path}", flush=True)

# ============================================================================
# Done
# ============================================================================
total_time = time.time() - t0
print(f"\n{'=' * 72}", flush=True)
print(f"TOTAL RUNTIME: {total_time:.1f}s", flush=True)
print(f"{'=' * 72}", flush=True)
print("\nOutputs in", str(RESULTS_DIR) + ":", flush=True)
print("  fig23_walkforward_all.pdf", flush=True)
print("  table_walkforward_all.tex", flush=True)
print("  walkforward_all_results.csv", flush=True)
