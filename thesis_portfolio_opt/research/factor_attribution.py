"""
Fama-French 5-Factor Attribution for Thesis Portfolio
=====================================================
Runs OLS regression of portfolio daily returns on FF5 factors:
  R_p - R_f = alpha + b1*(Mkt-RF) + b2*SMB + b3*HML + b4*RMW + b5*CMA + eps

Uses both:
  1. Walk-forward OOS returns
  2. Best fixed-split strategy returns (ML-Enhanced lambda=2)
  3. Best advanced strategy returns (A1 LightGBM)
"""

import os
import io
import zipfile
import warnings
import urllib.request

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
BASE = "/Users/arhansubasi/thesis/thesis_portfolio_opt"
RES  = os.path.join(BASE, "data", "results")
RAW  = os.path.join(BASE, "data", "raw")
OUT  = RES  # put outputs alongside other results

# ── 1. Load portfolio returns ──────────────────────────────────────────────
def load_returns(path, col=None):
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    if col is None:
        col = df.columns[0]
    s = df[col].dropna()
    s.index.name = "Date"
    s.name = "port_ret"
    return s

wf_ret = load_returns(os.path.join(RES, "walk_forward_returns.csv"))
ml_ret = load_returns(os.path.join(RES, "returns_ML-Enhanced_λ2.csv"), "return")
adv_ret = load_returns(os.path.join(RES, "adv_A1_lgbm_maxw50_tc5.csv"))

print(f"Walk-forward returns : {wf_ret.index[0].date()} → {wf_ret.index[-1].date()}  ({len(wf_ret)} obs)")
print(f"ML-Enhanced λ=2      : {ml_ret.index[0].date()} → {ml_ret.index[-1].date()}  ({len(ml_ret)} obs)")
print(f"Best adv (A1 LightGBM): {adv_ret.index[0].date()} → {adv_ret.index[-1].date()}  ({len(adv_ret)} obs)")

# ── 2. Download / construct Fama-French 5 factors ─────────────────────────
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

def download_ff5():
    """Download FF5 daily factors from Ken French's website."""
    print("\nDownloading Fama-French 5 factors from Ken French data library...")
    try:
        req = urllib.request.Request(FF_URL, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=30)
        zdata = resp.read()
        zf = zipfile.ZipFile(io.BytesIO(zdata))
        csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        raw_text = zf.read(csv_name).decode("utf-8")

        # Find the header row (contains Mkt-RF)
        lines = raw_text.split("\n")
        start_idx = None
        for i, line in enumerate(lines):
            if "Mkt-RF" in line:
                start_idx = i
                break
        if start_idx is None:
            raise ValueError("Could not find header row in FF CSV")

        # Read from that row onward
        df = pd.read_csv(io.StringIO("\n".join(lines[start_idx:])),
                         skipinitialspace=True)
        # First column is date (YYYYMMDD)
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.set_index("Date")
        # Convert from percent to decimal
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
        df = df.dropna()
        print(f"  ✓ Downloaded {len(df)} daily factor observations")
        print(f"    Date range: {df.index[0].date()} → {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return None


def construct_synthetic_ff5():
    """Construct synthetic FF5 proxies from our ETF price data."""
    print("\nConstructing synthetic FF5 factor proxies from ETF data...")
    prices = pd.read_csv(os.path.join(RAW, "prices.csv"), parse_dates=["Date"], index_col="Date")
    rets = prices.pct_change().dropna()

    ff = pd.DataFrame(index=rets.index)
    # Risk-free rate proxy: ~0 for daily, use 0.0001/252 ≈ 0
    ff["RF"] = 0.0001 / 252
    # Mkt-RF: SPY excess return
    ff["Mkt-RF"] = rets["SPY"] - ff["RF"]
    # SMB: IWM - SPY (small minus large)
    ff["SMB"] = rets["IWM"] - rets["SPY"] if "IWM" in rets.columns else 0.0
    # HML: VNQ - SPY (value proxy minus growth proxy)
    ff["HML"] = rets["VNQ"] - rets["SPY"] if "VNQ" in rets.columns else 0.0
    # RMW: proxy with quality tilt — low-vol (AGG) minus high-vol (HYG)
    if "AGG" in rets.columns and "HYG" in rets.columns:
        ff["RMW"] = rets["AGG"] - rets["HYG"]
    else:
        ff["RMW"] = 0.0
    # CMA: conservative minus aggressive — TLT minus EEM
    if "TLT" in rets.columns and "EEM" in rets.columns:
        ff["CMA"] = rets["TLT"] - rets["EEM"]
    else:
        ff["CMA"] = 0.0
    ff = ff.dropna()
    print(f"  ✓ Constructed {len(ff)} daily synthetic factor observations")
    return ff


# Try download first, fall back to synthetic
ff5 = download_ff5()
if ff5 is None:
    ff5 = construct_synthetic_ff5()
    factor_source = "Synthetic (ETF-based proxies)"
else:
    factor_source = "Ken French Data Library"

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
# Ensure we have the right columns
for c in FACTOR_COLS:
    assert c in ff5.columns, f"Missing factor column: {c}"
if "RF" not in ff5.columns:
    ff5["RF"] = 0.0

# ── 3. Run factor regressions ─────────────────────────────────────────────
def run_factor_regression(port_returns, ff_data, label="Portfolio"):
    """Run FF5 factor regression and return results dict."""
    # Align dates
    combined = pd.DataFrame({"port_ret": port_returns}).join(ff_data[FACTOR_COLS + ["RF"]], how="inner")
    combined = combined.dropna()
    n = len(combined)
    if n < 30:
        print(f"  ⚠ Only {n} overlapping observations for {label} — skipping")
        return None

    # Excess return
    y = combined["port_ret"] - combined["RF"]
    X = combined[FACTOR_COLS]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HC1")  # heteroscedasticity-robust SEs

    # Annualize alpha (daily → annual)
    alpha_daily = model.params["const"]
    alpha_annual = alpha_daily * 252
    alpha_se_annual = model.bse["const"] * np.sqrt(252)
    alpha_tstat = model.tvalues["const"]
    alpha_pval = model.pvalues["const"]

    print(f"\n{'='*65}")
    print(f"  Fama-French 5-Factor Attribution: {label}")
    print(f"  Factor source: {factor_source}")
    print(f"  Observations: {n}  |  Date range: {combined.index[0].date()} → {combined.index[-1].date()}")
    print(f"{'='*65}")
    print(f"  {'Factor':<10} {'Loading':>10} {'Std Err':>10} {'t-stat':>10} {'p-value':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Alpha':.<10} {alpha_annual:>10.4f} {alpha_se_annual:>10.4f} {alpha_tstat:>10.2f} {alpha_pval:>10.4f}  (annualized)")
    for f in FACTOR_COLS:
        print(f"  {f:<10} {model.params[f]:>10.4f} {model.bse[f]:>10.4f} {model.tvalues[f]:>10.2f} {model.pvalues[f]:>10.4f}")
    print(f"  {'-'*50}")
    print(f"  R²       = {model.rsquared:.4f}")
    print(f"  Adj. R²  = {model.rsquared_adj:.4f}")
    print(f"  F-stat   = {model.fvalue:.2f}  (p={model.f_pvalue:.2e})")

    # Identify dominant factors
    factor_contribs = {}
    for f in FACTOR_COLS:
        # Variance contribution ≈ beta² * var(factor)
        factor_contribs[f] = model.params[f]**2 * combined[f].var()
    total_explained = sum(factor_contribs.values())
    print(f"\n  Variance contribution by factor:")
    for f in sorted(factor_contribs, key=factor_contribs.get, reverse=True):
        pct = factor_contribs[f] / total_explained * 100 if total_explained > 0 else 0
        print(f"    {f:<10} {pct:6.1f}%")

    return {
        "label": label,
        "n_obs": n,
        "date_start": combined.index[0],
        "date_end": combined.index[-1],
        "alpha_daily": alpha_daily,
        "alpha_annual": alpha_annual,
        "alpha_se_annual": alpha_se_annual,
        "alpha_tstat": alpha_tstat,
        "alpha_pval": alpha_pval,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_stat": model.fvalue,
        "f_pval": model.f_pvalue,
        "model": model,
        "factor_contribs": factor_contribs,
        "betas": {f: model.params[f] for f in FACTOR_COLS},
        "betas_se": {f: model.bse[f] for f in FACTOR_COLS},
        "betas_tstat": {f: model.tvalues[f] for f in FACTOR_COLS},
        "betas_pval": {f: model.pvalues[f] for f in FACTOR_COLS},
    }


results = {}
for label, ret_series in [
    ("Walk-Forward OOS", wf_ret),
    ("ML-Enhanced λ=2 (fixed split)", ml_ret),
    ("Best Adv: A1 LightGBM", adv_ret),
]:
    r = run_factor_regression(ret_series, ff5, label)
    if r is not None:
        results[label] = r

# ── 4. Save factor_attribution.csv ────────────────────────────────────────
rows = []
for label, r in results.items():
    row = {
        "Strategy": label,
        "N_obs": r["n_obs"],
        "Date_start": r["date_start"].date(),
        "Date_end": r["date_end"].date(),
        "Alpha_annual": r["alpha_annual"],
        "Alpha_tstat": r["alpha_tstat"],
        "Alpha_pval": r["alpha_pval"],
        "R_squared": r["r_squared"],
        "Adj_R_squared": r["adj_r_squared"],
    }
    for f in FACTOR_COLS:
        row[f"Beta_{f}"] = r["betas"][f]
        row[f"tstat_{f}"] = r["betas_tstat"][f]
        row[f"pval_{f}"] = r["betas_pval"][f]
    rows.append(row)

df_out = pd.DataFrame(rows)
csv_path = os.path.join(OUT, "factor_attribution.csv")
df_out.to_csv(csv_path, index=False)
print(f"\n✓ Saved {csv_path}")

# ── 5. Save table_factor_attribution.tex ──────────────────────────────────
def make_latex_table(results_dict):
    """Create a LaTeX table of factor attribution results."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Fama-French Five-Factor Attribution}")
    lines.append(r"\label{tab:factor_attribution}")
    lines.append(r"\small")

    ncols = 1 + len(results_dict)
    col_spec = "l" + "r" * len(results_dict)
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    # Header
    headers = [""] + [r.get("label", "Portfolio") for r in results_dict.values()]
    # Shorten labels for table
    short_labels = {
        "Walk-Forward OOS": "Walk-Forward",
        "ML-Enhanced λ=2 (fixed split)": r"ML-Enh.\ $\lambda{=}2$",
        "Best Adv: A1 LightGBM": r"Best Adv.\ (A1)",
    }
    headers = [""] + [short_labels.get(r["label"], r["label"]) for r in results_dict.values()]
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    # Observations
    row_n = ["Observations"]
    for r in results_dict.values():
        row_n.append(str(r["n_obs"]))
    lines.append(" & ".join(row_n) + r" \\")
    lines.append(r"\midrule")

    # Alpha (annualized)
    row_alpha = [r"$\alpha$ (ann.)"]
    for r in results_dict.values():
        stars = ""
        if r["alpha_pval"] < 0.01:
            stars = "^{***}"
        elif r["alpha_pval"] < 0.05:
            stars = "^{**}"
        elif r["alpha_pval"] < 0.10:
            stars = "^{*}"
        row_alpha.append(f"${r['alpha_annual']:.4f}{stars}$")
    lines.append(" & ".join(row_alpha) + r" \\")

    # Alpha t-stat
    row_at = [""]
    for r in results_dict.values():
        row_at.append(f"$({r['alpha_tstat']:.2f})$")
    lines.append(" & ".join(row_at) + r" \\")
    lines.append(r"\midrule")

    # Factor betas
    for f in FACTOR_COLS:
        row_b = [f.replace("-", "--")]
        for r in results_dict.values():
            stars = ""
            if r["betas_pval"][f] < 0.01:
                stars = "^{***}"
            elif r["betas_pval"][f] < 0.05:
                stars = "^{**}"
            elif r["betas_pval"][f] < 0.10:
                stars = "^{*}"
            row_b.append(f"${r['betas'][f]:.4f}{stars}$")
        lines.append(" & ".join(row_b) + r" \\")

        row_t = [""]
        for r in results_dict.values():
            row_t.append(f"$({r['betas_tstat'][f]:.2f})$")
        lines.append(" & ".join(row_t) + r" \\")

    lines.append(r"\midrule")

    # R-squared
    row_r2 = [r"$R^2$"]
    for r in results_dict.values():
        row_r2.append(f"${r['r_squared']:.4f}$")
    lines.append(" & ".join(row_r2) + r" \\")

    # Adj R-squared
    row_ar2 = [r"Adj.\ $R^2$"]
    for r in results_dict.values():
        row_ar2.append(f"${r['adj_r_squared']:.4f}$")
    lines.append(" & ".join(row_ar2) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\par\smallskip\footnotesize")
    lines.append(r"Heteroscedasticity-robust (HC1) standard errors. $t$-statistics in parentheses.")
    lines.append(r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.")
    lines.append(f"Factor data source: {factor_source}.")
    lines.append(r"\end{table}")
    return "\n".join(lines)


tex = make_latex_table(results)
tex_path = os.path.join(OUT, "table_factor_attribution.tex")
with open(tex_path, "w") as f:
    f.write(tex)
print(f"✓ Saved {tex_path}")

# Also copy to deliverables
tex_deliv = os.path.join(BASE, "deliverables", "table_factor_attribution.tex")
with open(tex_deliv, "w") as f:
    f.write(tex)
print(f"✓ Saved {tex_deliv}")

# ── 6. Generate fig27_factor_attribution.pdf ──────────────────────────────
def plot_factor_attribution(results_dict, out_path):
    """Bar chart of factor loadings with 95% CI error bars."""
    n_strategies = len(results_dict)

    fig, axes = plt.subplots(1, n_strategies, figsize=(5.5 * n_strategies, 5.5),
                              sharey=True, squeeze=False)
    axes = axes[0]

    colors_map = {
        "Mkt-RF": "#1f77b4",
        "SMB":    "#ff7f0e",
        "HML":    "#2ca02c",
        "RMW":    "#d62728",
        "CMA":    "#9467bd",
    }

    all_factors = ["Alpha\n(ann.)"] + FACTOR_COLS

    for idx, (label, r) in enumerate(results_dict.items()):
        ax = axes[idx]
        model = r["model"]

        # Values: alpha (annualized) + factor betas
        vals = [r["alpha_annual"]] + [r["betas"][f] for f in FACTOR_COLS]
        # 95% CI half-widths
        errs = [1.96 * r["alpha_se_annual"]] + [1.96 * r["betas_se"][f] for f in FACTOR_COLS]
        # p-values for significance markers
        pvals = [r["alpha_pval"]] + [r["betas_pval"][f] for f in FACTOR_COLS]

        colors = ["#333333"] + [colors_map[f] for f in FACTOR_COLS]

        x = np.arange(len(all_factors))
        bars = ax.bar(x, vals, yerr=errs, capsize=4,
                      color=colors, edgecolor="white", linewidth=0.5,
                      alpha=0.85, zorder=3)

        # Add significance stars
        for i, (v, p, e) in enumerate(zip(vals, pvals, errs)):
            star = ""
            if p < 0.01:
                star = "***"
            elif p < 0.05:
                star = "**"
            elif p < 0.10:
                star = "*"
            if star:
                y_pos = v + e + 0.005 if v >= 0 else v - e - 0.015
                ax.text(i, y_pos, star, ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=10, fontweight="bold", color="#333")

        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="-", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(all_factors, fontsize=9, rotation=0)
        ax.set_title(label, fontsize=11, fontweight="bold", pad=10)

        if idx == 0:
            ax.set_ylabel("Loading / Annualized Alpha", fontsize=10)

        # Annotate R² and alpha
        textstr = (f"$\\alpha_{{ann}}$ = {r['alpha_annual']:.4f}\n"
                   f"$t$-stat = {r['alpha_tstat']:.2f}\n"
                   f"$R^2$ = {r['r_squared']:.4f}")
        props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7)
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
                fontsize=8.5, verticalalignment="top", horizontalalignment="right",
                bbox=props)

        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Fama-French 5-Factor Attribution", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"✓ Saved {out_path}")


fig_path = os.path.join(OUT, "fig27_factor_attribution.pdf")
plot_factor_attribution(results, fig_path)

# ── 7. Summary ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SUMMARY")
print("="*65)
for label, r in results.items():
    sig = "YES" if r["alpha_pval"] < 0.05 else ("marginal" if r["alpha_pval"] < 0.10 else "NO")
    print(f"\n  {label}:")
    print(f"    Alpha (annualized): {r['alpha_annual']:.4f}  ({r['alpha_annual']*100:.2f}%)")
    print(f"    Alpha significant at 5%? {sig}  (p={r['alpha_pval']:.4f})")
    print(f"    R² = {r['r_squared']:.4f}  →  {(1-r['r_squared'])*100:.1f}% of variance unexplained by FF5")
    # Top factors
    sorted_f = sorted(r["factor_contribs"].items(), key=lambda x: x[1], reverse=True)
    print(f"    Dominant factor: {sorted_f[0][0]} (beta={r['betas'][sorted_f[0][0]]:.4f})")
    print(f"    Market beta: {r['betas']['Mkt-RF']:.4f}")

print("\n✓ All outputs generated successfully.")
print(f"  - {csv_path}")
print(f"  - {tex_path}")
print(f"  - {tex_deliv}")
print(f"  - {fig_path}")
