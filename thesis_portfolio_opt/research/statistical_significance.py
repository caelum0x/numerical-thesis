"""
Statistical Significance Tests for Thesis Portfolio Optimisation Research
=========================================================================
1. Bootstrap Sharpe Ratio Confidence Intervals (block bootstrap, 10,000 samples)
2. Ledoit-Wolf (2008) Test for Equality of Two Sharpe Ratios
3. Diebold-Mariano Test on portfolio returns
4. Figure: Bootstrap distributions (fig22_statistical_significance.pdf)
5. LaTeX table (table_significance.tex)
6. CSV (significance_tests.csv)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/Users/arhansubasi/thesis/thesis_portfolio_opt")
RES  = BASE / "data" / "results"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 72)
print("LOADING DATA")
print("=" * 72)

# Benchmark: SPY buy-and-hold daily returns from raw prices
prices = pd.read_csv(BASE / "data" / "raw" / "prices.csv", parse_dates=["Date"], index_col="Date")
spy_ret = prices["SPY"].pct_change().dropna()
spy_ret.name = "SPY"

# Load strategy returns
def load_returns(path, col="return"):
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    if col in df.columns:
        return df[col]
    return df.iloc[:, 0]

ml2   = load_returns(RES / "returns_ML-Enhanced_λ2.csv")
ml5   = load_returns(RES / "returns_ML-Enhanced_λ5.csv")
ew    = load_returns(RES / "returns_Equal_Weight.csv")
bench = load_returns(RES / "returns_60_40.csv")

# Advanced strategies (OOS period)
adv_a1 = load_returns(RES / "adv_A1_lgbm_maxw50_tc5.csv")
adv_a2 = load_returns(RES / "adv_A2_lgbm_lam3_maxw50.csv")
adv_a3 = load_returns(RES / "adv_A3_lgbm_shrink10.csv")

# Walk-forward
wf = load_returns(RES / "walk_forward_returns.csv")

# For the OOS comparisons, align all series to the advanced/walk-forward period
# The advanced strategies cover 2022-02 to 2024-12 (~730 rows)
oos_start = adv_a1.index.min()
oos_end   = adv_a1.index.max()

# Build a dict of all series to test
strategies = {
    "ML-Enhanced (lam=2)":      ml2,
    "ML-Enhanced (lam=5)":      ml5,
    "A1 LGBM maxW50 tc5":      adv_a1,
    "A2 LGBM lam3 maxW50":     adv_a2,
    "A3 LGBM shrink10":        adv_a3,
    "Walk-Forward":             wf,
    "Equal Weight":             ew,
    "60/40":                    bench,
    "SPY":                      spy_ret,
}

# Print basic info
for name, s in strategies.items():
    sr = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0.0
    print(f"  {name:30s}  N={len(s):5d}  SR={sr:.4f}  "
          f"dates: {s.index.min().date()} to {s.index.max().date()}")

# ---------------------------------------------------------------------------
# Determine the "best" strategy: highest full-sample Sharpe
# ---------------------------------------------------------------------------
def annualised_sharpe(r):
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0

sharpe_all = {k: annualised_sharpe(v) for k, v in strategies.items()}
best_name = max(sharpe_all, key=sharpe_all.get)
print(f"\n>>> Best strategy overall: {best_name}  (SR={sharpe_all[best_name]:.4f})")

# For fair pairwise tests we need overlapping dates.
# The ML-Enhanced strategies span 2008-2024; advanced/WF span 2022-2024.
# We'll do the bootstrap on each strategy's own full series,
# and pairwise tests on the overlapping OOS period for advanced strategies.
# For ML-Enhanced vs benchmarks we align on the full sample.

# ---------------------------------------------------------------------------
# 2. Block Bootstrap Sharpe Ratio Confidence Intervals
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("BLOCK BOOTSTRAP SHARPE RATIO CONFIDENCE INTERVALS (10,000 samples)")
print("=" * 72)

def block_bootstrap_sharpe(returns, n_boot=10000, block_size=21, ci=0.95, seed=42):
    """Block bootstrap for annualised Sharpe ratio with 95% CI."""
    rng = np.random.RandomState(seed)
    r = returns.values
    n = len(r)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size

    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.randint(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([r[s:s + block_size] for s in starts])[:n]
        mu = sample.mean()
        sigma = sample.std(ddof=1)
        sharpes[i] = mu / sigma * np.sqrt(252) if sigma > 0 else 0.0

    alpha = (1 - ci) / 2
    lo, hi = np.percentile(sharpes, [alpha * 100, (1 - alpha) * 100])
    return sharpes, lo, hi

# Run for all strategies
boot_results = {}
for name, s in strategies.items():
    sharpes_boot, lo, hi = block_bootstrap_sharpe(s)
    point = annualised_sharpe(s)
    boot_results[name] = {
        "point": point, "ci_lo": lo, "ci_hi": hi, "distribution": sharpes_boot
    }
    print(f"  {name:30s}  SR={point:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]")

# Test: is best strategy's Sharpe > SPY's Sharpe?
best_boot = boot_results[best_name]["distribution"]
spy_boot  = boot_results["SPY"]["distribution"]
diff_boot = best_boot - spy_boot
pct_positive = np.mean(diff_boot > 0) * 100
boot_p = np.mean(diff_boot <= 0)  # one-sided p-value

print(f"\n  Bootstrap test: {best_name} SR > SPY SR")
print(f"    Diff (median)    = {np.median(diff_boot):.4f}")
print(f"    P(diff > 0)      = {pct_positive:.2f}%")
print(f"    One-sided p-val  = {boot_p:.6f}")

# ---------------------------------------------------------------------------
# 3. Ledoit-Wolf (2008) Test for Equality of Two Sharpe Ratios
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("LEDOIT-WOLF (2008) TEST FOR EQUALITY OF TWO SHARPE RATIOS")
print("=" * 72)

def ledoit_wolf_sharpe_test(r1, r2):
    """
    Ledoit & Wolf (2008) HAC-based test for H0: SR1 = SR2.
    Uses Newey-West with automatic bandwidth selection.
    Returns (test_statistic, p_value).
    """
    # Align on common dates
    df = pd.concat([r1, r2], axis=1, join="inner").dropna()
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    T = len(x)

    mu_x, mu_y = x.mean(), y.mean()
    sig_x, sig_y = x.std(ddof=1), y.std(ddof=1)
    sr_x = mu_x / sig_x
    sr_y = mu_y / sig_y

    # Gradient of the function g(mu_x, mu_y, sig2_x, sig2_y) = mu_x/sig_x - mu_y/sig_y
    # with respect to (mu_x, mu_y, var_x, var_y)
    var_x, var_y = sig_x**2, sig_y**2
    grad = np.array([
        1.0 / sig_x,
        -1.0 / sig_y,
        -0.5 * mu_x / (var_x * sig_x),
        0.5 * mu_y / (var_y * sig_y),
    ])

    # Build the moment conditions
    z = np.column_stack([
        x - mu_x,
        y - mu_y,
        (x - mu_x)**2 - var_x,
        (y - mu_y)**2 - var_y,
    ])

    # HAC variance using Newey-West with bandwidth = floor(T^(1/3))
    bw = int(np.floor(T**(1.0 / 3.0)))
    Sigma = z.T @ z / T
    for j in range(1, bw + 1):
        w = 1.0 - j / (bw + 1.0)  # Bartlett kernel
        Gamma_j = z[j:].T @ z[:-j] / T
        Sigma += w * (Gamma_j + Gamma_j.T)

    se2 = grad @ Sigma @ grad / T
    se = np.sqrt(max(se2, 1e-20))

    delta = sr_x - sr_y
    z_stat = delta / se
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    return z_stat, p_value, sr_x * np.sqrt(252), sr_y * np.sqrt(252)

# Comparisons: best strategy vs SPY, vs EW, vs 60/40
best_ret = strategies[best_name]
comparisons = [
    (best_name, "SPY",          strategies["SPY"]),
    (best_name, "Equal Weight", strategies["Equal Weight"]),
    (best_name, "60/40",        strategies["60/40"]),
]

lw_results = []
for name_a, name_b, ret_b in comparisons:
    z_stat, p_val, sr_a, sr_b = ledoit_wolf_sharpe_test(best_ret, ret_b)
    lw_results.append({
        "comparison": f"{name_a} vs {name_b}",
        "SR_A": sr_a, "SR_B": sr_b,
        "z_stat": z_stat, "p_value": p_val,
    })
    sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
    print(f"  {name_a} vs {name_b}:")
    print(f"    SR_A={sr_a:.4f}  SR_B={sr_b:.4f}  z={z_stat:.4f}  p={p_val:.6f} {sig}")

# Also test top advanced strategies vs SPY (OOS period only)
adv_comparisons = [
    ("A1 LGBM maxW50 tc5", strategies["A1 LGBM maxW50 tc5"]),
    ("A2 LGBM lam3 maxW50", strategies["A2 LGBM lam3 maxW50"]),
    ("Walk-Forward",         strategies["Walk-Forward"]),
]
print("\n  --- OOS Advanced strategies vs SPY ---")
for adv_name, adv_ret in adv_comparisons:
    z_stat, p_val, sr_a, sr_b = ledoit_wolf_sharpe_test(adv_ret, spy_ret)
    lw_results.append({
        "comparison": f"{adv_name} vs SPY",
        "SR_A": sr_a, "SR_B": sr_b,
        "z_stat": z_stat, "p_value": p_val,
    })
    sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
    print(f"  {adv_name} vs SPY:")
    print(f"    SR_A={sr_a:.4f}  SR_B={sr_b:.4f}  z={z_stat:.4f}  p={p_val:.6f} {sig}")

# ---------------------------------------------------------------------------
# 4. Diebold-Mariano Test
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("DIEBOLD-MARIANO TEST")
print("=" * 72)

def diebold_mariano_test(r1, r2, h=1):
    """
    Diebold-Mariano test comparing squared return losses.
    H0: equal predictive accuracy (same expected squared returns).
    Uses HAC variance estimator.
    """
    df = pd.concat([r1, r2], axis=1, join="inner").dropna()
    e1 = df.iloc[:, 0].values
    e2 = df.iloc[:, 1].values

    # Loss differential: squared returns (lower = better)
    d = e1**2 - e2**2
    T = len(d)
    d_bar = d.mean()

    # Newey-West HAC variance
    bw = int(np.floor(T**(1.0 / 3.0)))
    gamma_0 = np.sum((d - d_bar)**2) / T
    V = gamma_0
    for j in range(1, bw + 1):
        w = 1.0 - j / (bw + 1.0)
        gamma_j = np.sum((d[j:] - d_bar) * (d[:-j] - d_bar)) / T
        V += 2.0 * w * gamma_j

    se = np.sqrt(max(V / T, 1e-20))
    dm_stat = d_bar / se
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))

    return dm_stat, p_value

dm_results = []

# Best strategy vs SPY
dm_stat, dm_p = diebold_mariano_test(best_ret, spy_ret)
dm_results.append({
    "comparison": f"{best_name} vs SPY",
    "DM_stat": dm_stat, "p_value": dm_p
})
sig = "***" if dm_p < 0.01 else ("**" if dm_p < 0.05 else ("*" if dm_p < 0.1 else ""))
print(f"  {best_name} vs SPY:")
print(f"    DM stat = {dm_stat:.4f}   p-value = {dm_p:.6f} {sig}")

# Best vs EW
dm_stat, dm_p = diebold_mariano_test(best_ret, ew)
dm_results.append({
    "comparison": f"{best_name} vs Equal Weight",
    "DM_stat": dm_stat, "p_value": dm_p
})
sig = "***" if dm_p < 0.01 else ("**" if dm_p < 0.05 else ("*" if dm_p < 0.1 else ""))
print(f"  {best_name} vs Equal Weight:")
print(f"    DM stat = {dm_stat:.4f}   p-value = {dm_p:.6f} {sig}")

# Best vs 60/40
dm_stat, dm_p = diebold_mariano_test(best_ret, bench)
dm_results.append({
    "comparison": f"{best_name} vs 60/40",
    "DM_stat": dm_stat, "p_value": dm_p
})
sig = "***" if dm_p < 0.01 else ("**" if dm_p < 0.05 else ("*" if dm_p < 0.1 else ""))
print(f"  {best_name} vs 60/40:")
print(f"    DM stat = {dm_stat:.4f}   p-value = {dm_p:.6f} {sig}")

# Advanced OOS strategies vs SPY
print("\n  --- OOS Advanced strategies vs SPY ---")
for adv_name, adv_ret in adv_comparisons:
    dm_stat, dm_p = diebold_mariano_test(adv_ret, spy_ret)
    dm_results.append({
        "comparison": f"{adv_name} vs SPY",
        "DM_stat": dm_stat, "p_value": dm_p
    })
    sig = "***" if dm_p < 0.01 else ("**" if dm_p < 0.05 else ("*" if dm_p < 0.1 else ""))
    print(f"  {adv_name} vs SPY:")
    print(f"    DM stat = {dm_stat:.4f}   p-value = {dm_p:.6f} {sig}")

# ---------------------------------------------------------------------------
# 5. Figure: Bootstrap distributions (fig22_statistical_significance.pdf)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("GENERATING FIGURE")
print("=" * 72)

# Top 3 by Sharpe + SPY
ranked = sorted(sharpe_all.items(), key=lambda x: x[1], reverse=True)
top3_names = [r[0] for r in ranked[:3]]
plot_names = top3_names + ["SPY"]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for i, name in enumerate(plot_names):
    dist = boot_results[name]["distribution"]
    point = boot_results[name]["point"]
    ax.hist(dist, bins=80, alpha=0.35, color=colors[i], label=name, density=True)
    ax.axvline(point, color=colors[i], linestyle="--", linewidth=2,
               label=f"{name} point est. = {point:.3f}")

ax.set_xlabel("Annualised Sharpe Ratio", fontsize=13)
ax.set_ylabel("Density", fontsize=13)
ax.set_title("Bootstrap Distributions of Sharpe Ratios (10,000 block-bootstrap samples, block=21)",
             fontsize=13)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()

fig_path = RES / "fig22_statistical_significance.pdf"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"  Saved: {fig_path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. LaTeX table (table_significance.tex)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("GENERATING LATEX TABLE")
print("=" * 72)

def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Statistical Significance Tests}")
lines.append(r"\label{tab:significance}")
lines.append(r"\small")

# Panel A: Bootstrap CIs
lines.append(r"\begin{tabular}{lcccc}")
lines.append(r"\toprule")
lines.append(r"\multicolumn{5}{l}{\textbf{Panel A: Block Bootstrap Sharpe Ratio 95\% Confidence Intervals}} \\")
lines.append(r"\midrule")
lines.append(r"Strategy & Sharpe (point) & 95\% CI Lower & 95\% CI Upper & N \\")
lines.append(r"\midrule")
for name in plot_names + [n for n in strategies if n not in plot_names]:
    br = boot_results[name]
    n_obs = len(strategies[name])
    lines.append(f"{name} & {br['point']:.4f} & {br['ci_lo']:.4f} & {br['ci_hi']:.4f} & {n_obs} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

lines.append(r"\\[12pt]")

# Panel B: Ledoit-Wolf tests
lines.append(r"\begin{tabular}{lccccc}")
lines.append(r"\toprule")
lines.append(r"\multicolumn{6}{l}{\textbf{Panel B: Ledoit--Wolf (2008) Test for Equality of Sharpe Ratios}} \\")
lines.append(r"\midrule")
lines.append(r"Comparison & SR$_A$ & SR$_B$ & $z$-stat & $p$-value & \\")
lines.append(r"\midrule")
for row in lw_results:
    s = stars(row["p_value"])
    lines.append(f"{row['comparison']} & {row['SR_A']:.4f} & {row['SR_B']:.4f} & "
                 f"{row['z_stat']:.4f} & {row['p_value']:.4f}{s} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

lines.append(r"\\[12pt]")

# Panel C: Diebold-Mariano
lines.append(r"\begin{tabular}{lccc}")
lines.append(r"\toprule")
lines.append(r"\multicolumn{4}{l}{\textbf{Panel C: Diebold--Mariano Test}} \\")
lines.append(r"\midrule")
lines.append(r"Comparison & DM stat & $p$-value & \\")
lines.append(r"\midrule")
for row in dm_results:
    s = stars(row["p_value"])
    lines.append(f"{row['comparison']} & {row['DM_stat']:.4f} & {row['p_value']:.4f}{s} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

lines.append(r"\\[6pt]")
lines.append(r"\begin{minipage}{0.9\textwidth}")
lines.append(r"\footnotesize")
lines.append(r"\textit{Notes:} $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. "
             r"Panel~A reports block-bootstrap (block size = 21 trading days) 95\% confidence intervals "
             r"for annualised Sharpe ratios (10,000 replications). "
             r"Panel~B reports the Ledoit--Wolf (2008) HAC-based test for the null hypothesis that two "
             r"Sharpe ratios are equal, using Newey--West standard errors with Bartlett kernel. "
             r"Panel~C reports the Diebold--Mariano (1995) test comparing squared return losses "
             r"under the null of equal predictive accuracy, with HAC standard errors.")
lines.append(r"\end{minipage}")

lines.append(r"\end{table}")

tex_path = RES / "table_significance.tex"
with open(tex_path, "w") as f:
    f.write("\n".join(lines))
print(f"  Saved: {tex_path}")

# ---------------------------------------------------------------------------
# 7. CSV with all results
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SAVING CSV")
print("=" * 72)

rows = []

# Bootstrap CIs
for name in strategies:
    br = boot_results[name]
    rows.append({
        "test": "Bootstrap Sharpe CI",
        "strategy_A": name,
        "strategy_B": "",
        "point_estimate": br["point"],
        "ci_lo_95": br["ci_lo"],
        "ci_hi_95": br["ci_hi"],
        "test_statistic": np.nan,
        "p_value": np.nan,
        "N": len(strategies[name]),
    })

# Bootstrap difference test
rows.append({
    "test": "Bootstrap SR Difference (A > SPY)",
    "strategy_A": best_name,
    "strategy_B": "SPY",
    "point_estimate": np.median(diff_boot),
    "ci_lo_95": np.percentile(diff_boot, 2.5),
    "ci_hi_95": np.percentile(diff_boot, 97.5),
    "test_statistic": np.nan,
    "p_value": boot_p,
    "N": len(best_ret),
})

# Ledoit-Wolf
for row in lw_results:
    rows.append({
        "test": "Ledoit-Wolf SR Equality",
        "strategy_A": row["comparison"].split(" vs ")[0],
        "strategy_B": row["comparison"].split(" vs ")[1],
        "point_estimate": row["SR_A"] - row["SR_B"],
        "ci_lo_95": np.nan,
        "ci_hi_95": np.nan,
        "test_statistic": row["z_stat"],
        "p_value": row["p_value"],
        "N": np.nan,
    })

# Diebold-Mariano
for row in dm_results:
    rows.append({
        "test": "Diebold-Mariano",
        "strategy_A": row["comparison"].split(" vs ")[0],
        "strategy_B": row["comparison"].split(" vs ")[1],
        "point_estimate": np.nan,
        "ci_lo_95": np.nan,
        "ci_hi_95": np.nan,
        "test_statistic": row["DM_stat"],
        "p_value": row["p_value"],
        "N": np.nan,
    })

csv_df = pd.DataFrame(rows)
csv_path = RES / "significance_tests.csv"
csv_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"\nBest strategy: {best_name} (Sharpe = {sharpe_all[best_name]:.4f})")
print(f"SPY Sharpe:    {sharpe_all['SPY']:.4f}")
print(f"\nBootstrap: {best_name} SR > SPY SR  =>  p = {boot_p:.6f}")
for row in lw_results[:3]:
    s = stars(row["p_value"])
    print(f"Ledoit-Wolf: {row['comparison']:40s}  z={row['z_stat']:+.4f}  p={row['p_value']:.6f} {s}")
for row in dm_results[:3]:
    s = stars(row["p_value"])
    print(f"Diebold-Mariano: {row['comparison']:40s}  DM={row['DM_stat']:+.4f}  p={row['p_value']:.6f} {s}")

print("\nOutput files:")
print(f"  Figure: {fig_path}")
print(f"  LaTeX:  {tex_path}")
print(f"  CSV:    {csv_path}")
print("\nDone.")
