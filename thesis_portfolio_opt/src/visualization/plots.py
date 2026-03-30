"""
Publication-quality visualizations for the thesis portfolio optimization project.

Every figure is rendered at 300 DPI and saved as PDF, suitable for direct
inclusion in a LaTeX document.  The module is organised into nine thematic
sections that mirror the chapters of the thesis:

  1. Setup & Utilities
  2. Price & Return Plots
  3. Risk Plots
  4. Portfolio Plots
  5. Optimization Plots
  6. Model Plots
  7. Macro & Factor Plots
  8. Stress Test & Regime Plots
  9. Summary Tables & Tear Sheet

Author : Arhan Subasi
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats as sp_stats

from src.config import (
    FIGURE_DPI,
    FIGURE_FORMAT,
    COLOR_PALETTE,
    FIGURE_SIZE,
    FIGURE_SIZE_WIDE,
    FIGURE_SIZE_TALL,
    RESULTS_DIR,
    STRATEGY_COLORS,
    ASSET_CLASS_COLORS,
    STRESS_SCENARIOS,
    LATEX_FONT_SETTINGS,
)

# ============================================================================
# SECTION 1 : Setup & Utilities
# ============================================================================

# ---------------------------------------------------------------------------
# Global thesis style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette=COLOR_PALETTE)
plt.rcParams.update(LATEX_FONT_SETTINGS)
plt.rcParams.update({
    "figure.figsize": FIGURE_SIZE,
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "savefig.format": FIGURE_FORMAT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
    "lines.antialiased": True,
    "patch.antialiased": True,
    "figure.autolayout": False,
    "figure.constrained_layout.use": False,
})


def save_figure(fig: plt.Figure, name: str) -> None:
    """Persist *fig* to ``RESULTS_DIR/<name>.<FIGURE_FORMAT>`` and close it."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    print(f"[plots] Saved figure -> {path}")
    plt.close(fig)


def get_strategy_color(name: str) -> str:
    """Return the hex colour for *name* from ``STRATEGY_COLORS``, with fallback."""
    if name in STRATEGY_COLORS:
        return STRATEGY_COLORS[name]
    palette = sns.color_palette(COLOR_PALETTE, 12)
    idx = hash(name) % len(palette)
    return mcolors.to_hex(palette[idx])


def format_pct(x: float, decimals: int = 2) -> str:
    """Format *x* (in decimal, e.g. 0.052) as a percentage string."""
    return f"{x * 100:.{decimals}f}%"


def _default_colors(n: int) -> list:
    """Return *n* colours from the project palette."""
    return [mcolors.to_hex(c) for c in sns.color_palette(COLOR_PALETTE, n)]


def _year_month_index(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Extract year and month arrays from a datetime-indexed Series."""
    idx = pd.DatetimeIndex(series.index)
    return idx.year.values, idx.month.values


# ============================================================================
# SECTION 2 : Price & Return Plots
# ============================================================================

def plot_cumulative_returns(
    returns_dict: Dict[str, pd.Series],
    title: str = "Cumulative Returns",
    name: str = "cumulative_returns",
) -> None:
    """Plot cumulative growth of $1 for every strategy in *returns_dict*."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    for label, rets in returns_dict.items():
        cum = (1 + rets).cumprod()
        color = get_strategy_color(label)
        ax.plot(cum.index, cum.values, label=label, color=color, linewidth=1.6)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="0.8")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    save_figure(fig, name)


def plot_normalized_prices(
    prices: pd.DataFrame,
    title: str = "Normalized Prices (Base 100)",
    name: str = "normalized_prices",
) -> None:
    """Normalize each column to 100 at the first valid observation and plot
    on a log scale."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    normed = prices.div(prices.bfill().iloc[0]) * 100
    colors = _default_colors(len(normed.columns))

    for col, clr in zip(normed.columns, colors):
        ax.plot(normed.index, normed[col], label=col, color=clr, linewidth=1.2)

    ax.set_yscale("log")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (log scale, base 100)")
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=True,
              framealpha=0.9, edgecolor="0.8")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    save_figure(fig, name)


def plot_return_distribution(
    returns_dict: Dict[str, pd.Series],
    title: str = "Return Distributions",
    name: str = "return_distributions",
) -> None:
    """Kernel density estimate overlay for each strategy's returns."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for label, rets in returns_dict.items():
        color = get_strategy_color(label)
        rets.dropna().plot.kde(ax=ax, label=label, color=color, linewidth=1.5)

    ax.axvline(x=0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")
    save_figure(fig, name)


def plot_return_heatmap(
    returns: pd.Series,
    title: str = "Monthly Returns Heatmap",
    name: str = "return_heatmap",
) -> None:
    """Calendar heatmap: rows = months (Jan-Dec), columns = years."""
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    idx = pd.DatetimeIndex(monthly.index)
    years = idx.year
    months = idx.month
    pivot = pd.DataFrame({"year": years, "month": months, "ret": monthly.values})
    pivot = pivot.pivot(index="month", columns="year", values="ret")
    pivot.index = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][: len(pivot.index)]
    # Re-index to ensure all 12 months even if data is partial
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(month_labels)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1%",
        center=0,
        cmap="RdYlGn",
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"format": mticker.PercentFormatter(1.0, 0), "shrink": 0.8},
    )
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    save_figure(fig, name)


def plot_qq_plots(
    returns: Union[pd.Series, pd.DataFrame],
    title: str = "QQ Plots — Normality Assessment",
    name: str = "qq_plots",
) -> None:
    """Grid of QQ plots against a normal distribution."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    cols = returns.columns.tolist()
    n = len(cols)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    for i, col in enumerate(cols):
        row_idx, col_idx = divmod(i, ncols)
        ax = axes[row_idx, col_idx]
        data = returns[col].dropna().values
        sp_stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(col, fontsize=10)
        ax.get_lines()[0].set(marker="o", markersize=2, alpha=0.5, color="#1f77b4")
        ax.get_lines()[1].set(color="#d62728", linewidth=1.2)

    # Turn off unused axes
    for j in range(n, nrows * ncols):
        row_idx, col_idx = divmod(j, ncols)
        axes[row_idx, col_idx].set_visible(False)

    fig.suptitle(title, fontweight="bold", fontsize=13, y=1.01)
    fig.tight_layout()
    save_figure(fig, name)


# ============================================================================
# SECTION 3 : Risk Plots
# ============================================================================

def plot_drawdowns(
    returns: pd.Series,
    title: str = "Portfolio Drawdowns",
    name: str = "drawdowns",
) -> None:
    """Filled drawdown chart for a single strategy."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.45)
    ax.plot(dd.index, dd.values, color="#d62728", linewidth=0.6, alpha=0.7)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    save_figure(fig, name)


def plot_drawdown_underwater(
    returns_dict: Dict[str, pd.Series],
    title: str = "Underwater Plot",
    name: str = "drawdown_underwater",
) -> None:
    """Multiple strategies' drawdowns overlaid (underwater chart)."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    for label, rets in returns_dict.items():
        cum = (1 + rets).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        color = get_strategy_color(label)
        ax.plot(dd.index, dd.values, label=label, color=color, linewidth=1.0,
                alpha=0.85)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="0.8")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    save_figure(fig, name)


def plot_rolling_volatility(
    returns_dict: Dict[str, pd.Series],
    windows: Optional[List[int]] = None,
    title: str = "Rolling Annualized Volatility",
    name: str = "rolling_volatility",
) -> None:
    """Rolling annualized volatility for multiple strategies / windows.

    If *windows* is provided, each strategy is plotted once per window.
    Otherwise a single default window of 63 days is used.
    """
    if windows is None:
        windows = [63]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    linestyles = ["-", "--", "-.", ":"]

    for label, rets in returns_dict.items():
        color = get_strategy_color(label)
        for j, w in enumerate(windows):
            vol = rets.rolling(w).std() * np.sqrt(252)
            ls = linestyles[j % len(linestyles)]
            lbl = f"{label} ({w}d)" if len(windows) > 1 else label
            ax.plot(vol.index, vol.values, label=lbl, color=color,
                    linestyle=ls, linewidth=1.2)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=True,
              framealpha=0.9, edgecolor="0.8")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    save_figure(fig, name)


def plot_rolling_sharpe(
    sharpe_dict: Dict[str, pd.Series],
    title: str = "Rolling 1-Year Sharpe Ratio",
    name: str = "rolling_sharpe",
) -> None:
    """Rolling Sharpe for multiple strategies."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    for label, sharpe in sharpe_dict.items():
        color = get_strategy_color(label)
        ax.plot(sharpe.index, sharpe.values, label=label, color=color,
                linewidth=1.2)

    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="0.8")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    save_figure(fig, name)


def plot_var_cvar(
    returns: pd.Series,
    confidence_levels: Optional[List[float]] = None,
    title: str = "Value-at-Risk & Conditional VaR",
    name: str = "var_cvar",
) -> None:
    """Grouped bar chart of historical VaR and CVaR at multiple confidence
    levels."""
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99]

    data = returns.dropna().values
    var_vals, cvar_vals, labels = [], [], []

    for cl in confidence_levels:
        q = np.percentile(data, (1 - cl) * 100)
        var_vals.append(-q)
        cvar_vals.append(-data[data <= q].mean())
        labels.append(f"{cl:.0%}")

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.bar(x - width / 2, np.array(var_vals) * 100, width, label="VaR",
           color="#1f77b4", edgecolor="white")
    ax.bar(x + width / 2, np.array(cvar_vals) * 100, width, label="CVaR",
           color="#d62728", edgecolor="white")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("Loss (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")

    # Annotate values
    for i, (v, c) in enumerate(zip(var_vals, cvar_vals)):
        ax.text(i - width / 2, v * 100 + 0.05, f"{v:.2%}", ha="center",
                va="bottom", fontsize=8)
        ax.text(i + width / 2, c * 100 + 0.05, f"{c:.2%}", ha="center",
                va="bottom", fontsize=8)

    save_figure(fig, name)


# ============================================================================
# SECTION 4 : Portfolio Plots
# ============================================================================

def plot_weights_over_time(
    weights: pd.DataFrame,
    title: str = "Portfolio Weights Over Time",
    name: str = "weights_over_time",
) -> None:
    """Stacked area chart of portfolio weights."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    colors = _default_colors(len(weights.columns))
    ax.stackplot(weights.index, weights.T.values, labels=weights.columns,
                 colors=colors, alpha=0.85)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    ax.legend(loc="upper left", fontsize=7, ncol=3, frameon=True,
              framealpha=0.9, edgecolor="0.8")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    save_figure(fig, name)


def plot_weight_bars(
    weights: np.ndarray,
    labels: List[str],
    title: str = "Optimal Portfolio Weights",
    name: str = "weight_bars",
) -> None:
    """Single-period bar chart of portfolio weights with value annotations."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    colors = _default_colors(len(labels))
    bars = ax.bar(labels, np.asarray(weights) * 100, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Weight (%)")
    ax.tick_params(axis="x", rotation=45)

    for bar, w in zip(bars, weights):
        if w > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{w:.1%}", ha="center", va="bottom", fontsize=8)
    save_figure(fig, name)


def plot_weight_transition(
    weights: pd.DataFrame,
    title: str = "Weight Transition at Rebalances",
    name: str = "weight_transition",
) -> None:
    """Line chart showing how each asset's weight changes at each
    rebalance date."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    colors = _default_colors(len(weights.columns))

    for col, clr in zip(weights.columns, colors):
        ax.plot(weights.index, weights[col], marker="o", markersize=3,
                label=col, color=clr, linewidth=1.0, alpha=0.85)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Rebalance Date")
    ax.set_ylabel("Weight")
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    ax.legend(loc="upper right", fontsize=7, ncol=3, frameon=True,
              framealpha=0.9, edgecolor="0.8")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    save_figure(fig, name)


def plot_turnover(
    turnover_df: pd.DataFrame,
    title: str = "Portfolio Turnover Over Time",
    name: str = "turnover",
) -> None:
    """Bar chart of turnover at each rebalance.

    *turnover_df* is expected to have a DatetimeIndex and a ``'turnover'``
    column (or be a single-column DataFrame / Series).
    """
    if isinstance(turnover_df, pd.Series):
        turnover_df = turnover_df.to_frame("turnover")
    col = turnover_df.columns[0]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    ax.bar(turnover_df.index, turnover_df[col] * 100, width=15,
           color="#1f77b4", alpha=0.75, edgecolor="white", linewidth=0.3)
    avg = turnover_df[col].mean() * 100
    ax.axhline(y=avg, color="#d62728", linewidth=1.2, linestyle="--",
               label=f"Mean = {avg:.2f}%")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover (%)")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    save_figure(fig, name)


def plot_risk_contribution(
    weights: np.ndarray,
    cov: np.ndarray,
    labels: List[str],
    title: str = "Risk Contribution by Asset",
    name: str = "risk_contribution",
) -> None:
    """Pie chart of marginal risk contributions (percentage of total
    portfolio volatility attributable to each asset)."""
    w = np.asarray(weights).flatten()
    sigma = np.asarray(cov)
    port_vol = np.sqrt(w @ sigma @ w)
    marginal = sigma @ w
    rc = w * marginal / port_vol  # risk contribution
    rc_pct = rc / rc.sum()

    # Filter out negligible contributions for cleaner chart
    threshold = 0.01
    mask = rc_pct >= threshold
    plot_labels = [l for l, m in zip(labels, mask) if m]
    plot_vals = rc_pct[mask]
    if (~mask).any():
        plot_labels.append("Other")
        plot_vals = np.append(plot_vals, rc_pct[~mask].sum())

    colors = _default_colors(len(plot_labels))
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        plot_vals, labels=plot_labels, autopct="%1.1f%%", colors=colors,
        startangle=140, pctdistance=0.8, textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(title, fontweight="bold", pad=20)
    save_figure(fig, name)


# ============================================================================
# SECTION 5 : Optimization Plots
# ============================================================================

def plot_efficient_frontier(
    frontier: pd.DataFrame,
    portfolios: Optional[Dict[str, Tuple[float, float]]] = None,
    title: str = "Efficient Frontier",
    name: str = "efficient_frontier",
    individual_assets: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """Plot efficient frontier curve with optional named portfolios and
    individual asset risk/return markers.

    Parameters
    ----------
    frontier : DataFrame with ``'volatility'`` and ``'return'`` columns.
    portfolios : dict  ``{name: (vol, ret)}``.
    individual_assets : dict  ``{ticker: (vol, ret)}``.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Frontier curve
    ax.plot(frontier["volatility"], frontier["return"], color="#1f77b4",
            linewidth=2.2, label="Efficient Frontier", zorder=3)

    # Named portfolios
    if portfolios:
        markers = ["*", "D", "s", "^", "v", "P", "X", "h"]
        for i, (label, (vol, ret)) in enumerate(portfolios.items()):
            color = get_strategy_color(label)
            m = markers[i % len(markers)]
            ax.scatter(vol, ret, s=160, marker=m, color=color, zorder=5,
                       edgecolors="black", linewidths=0.5, label=label)

    # Individual assets
    if individual_assets:
        for ticker, (vol, ret) in individual_assets.items():
            ax.scatter(vol, ret, s=50, marker="o", color="grey", alpha=0.7,
                       zorder=4, edgecolors="black", linewidths=0.4)
            ax.annotate(ticker, (vol, ret), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, alpha=0.8)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9,
              edgecolor="0.8")
    save_figure(fig, name)


def plot_sensitivity_analysis(
    lambdas: np.ndarray,
    weight_paths: np.ndarray,
    labels: List[str],
    title: str = "Sensitivity Analysis: Risk Aversion vs. Weights",
    name: str = "sensitivity_analysis",
) -> None:
    """Plot how optimal weights change as the risk-aversion parameter varies.

    Parameters
    ----------
    lambdas : 1-D array of risk-aversion values (x-axis).
    weight_paths : 2-D array (n_lambdas x n_assets).
    labels : asset names.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    colors = _default_colors(len(labels))

    for j, (lbl, clr) in enumerate(zip(labels, colors)):
        ax.plot(lambdas, weight_paths[:, j] * 100, label=lbl, color=clr,
                linewidth=1.4)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Risk Aversion ($\\lambda$)")
    ax.set_ylabel("Weight (%)")
    ax.legend(loc="upper right", fontsize=7, ncol=2, frameon=True,
              framealpha=0.9, edgecolor="0.8")
    ax.set_xlim(lambdas.min(), lambdas.max())
    save_figure(fig, name)


def plot_asset_class_allocation(
    lambdas: np.ndarray,
    weight_paths: np.ndarray,
    asset_classes: Dict[str, List[int]],
    title: str = "Asset Class Allocation vs. Risk Aversion",
    name: str = "asset_class_allocation",
) -> None:
    """Stacked area chart of asset-class weights across risk-aversion levels.

    Parameters
    ----------
    lambdas : 1-D array of risk-aversion values.
    weight_paths : 2-D array (n_lambdas x n_assets).
    asset_classes : ``{class_name: [column_indices]}`` mapping.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    class_weights = {}
    for cls_name, indices in asset_classes.items():
        class_weights[cls_name] = weight_paths[:, indices].sum(axis=1)

    class_names = list(class_weights.keys())
    values = np.column_stack([class_weights[c] for c in class_names])
    colors = [ASSET_CLASS_COLORS.get(c, "#999999") for c in class_names]

    ax.stackplot(lambdas, values.T, labels=class_names, colors=colors,
                 alpha=0.85)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Risk Aversion ($\\lambda$)")
    ax.set_ylabel("Allocation")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.8")
    save_figure(fig, name)


# ============================================================================
# SECTION 6 : Model Plots
# ============================================================================

def plot_model_comparison(
    metrics_df: pd.DataFrame,
    metric: str = "rmse",
    title: str = "Model Comparison",
    name: str = "model_comparison",
) -> None:
    """Horizontal bar chart comparing model performance on a given metric."""
    df = metrics_df.sort_values(metric, ascending=True).copy()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    colors = _default_colors(len(df))

    ax.barh(df["model"], df[metric], color=colors, edgecolor="white",
            linewidth=0.5)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("")

    # Annotate
    for i, v in enumerate(df[metric]):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8)

    ax.invert_yaxis()
    save_figure(fig, name)


def plot_feature_importance(
    fi_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Top Feature Importances",
    name: str = "feature_importance",
) -> None:
    """Horizontal bar chart of top-*top_n* feature importances."""
    top = fi_df.head(top_n).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.38)))
    colors = _default_colors(len(top))

    ax.barh(top["feature"], top["importance_pct"], color=colors,
            edgecolor="white", linewidth=0.4)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Importance (%)")

    for i, v in enumerate(top["importance_pct"]):
        ax.text(v + 0.15, i, f"{v:.1f}%", va="center", fontsize=7)

    save_figure(fig, name)


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    title: str = "Learning Curve",
    name: str = "learning_curve",
) -> None:
    """Plot learning curve with training and validation scores.

    *train_scores* and *test_scores* may be 2-D (n_sizes x n_folds) or 1-D.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    if train_scores.ndim == 2:
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)
        ax.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.15, color="#1f77b4")
        ax.fill_between(train_sizes, test_mean - test_std,
                        test_mean + test_std, alpha=0.15, color="#d62728")
    else:
        train_mean = train_scores
        test_mean = test_scores

    ax.plot(train_sizes, train_mean, "o-", color="#1f77b4", label="Training",
            markersize=4, linewidth=1.4)
    ax.plot(train_sizes, test_mean, "o-", color="#d62728", label="Validation",
            markersize=4, linewidth=1.4)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Score")
    ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="0.8")
    save_figure(fig, name)


def plot_residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
    name: str = "residual_analysis",
) -> None:
    """Four-panel residual diagnostic:
    1. Residuals vs. fitted
    2. Histogram of residuals
    3. QQ plot of residuals
    4. Autocorrelation of residuals
    """
    residuals = np.asarray(y_true).flatten() - np.asarray(y_pred).flatten()
    fitted = np.asarray(y_pred).flatten()

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_TALL)
    fig.suptitle(title, fontweight="bold", fontsize=13, y=1.01)

    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(fitted, residuals, alpha=0.4, s=12, color="#1f77b4",
               edgecolors="none")
    ax.axhline(0, color="#d62728", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs. Fitted", fontsize=10)

    # 2. Histogram
    ax = axes[0, 1]
    ax.hist(residuals, bins=50, density=True, color="#1f77b4", alpha=0.7,
            edgecolor="white", linewidth=0.3)
    xr = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(xr, sp_stats.norm.pdf(xr, residuals.mean(), residuals.std()),
            color="#d62728", linewidth=1.2, label="Normal fit")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution", fontsize=10)
    ax.legend(fontsize=8)

    # 3. QQ
    ax = axes[1, 0]
    sp_stats.probplot(residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set(marker="o", markersize=2, alpha=0.5, color="#1f77b4")
    ax.get_lines()[1].set(color="#d62728", linewidth=1.2)
    ax.set_title("Normal QQ Plot", fontsize=10)

    # 4. Autocorrelation
    ax = axes[1, 1]
    n_lags = min(40, len(residuals) // 2)
    acf_vals = [1.0]
    for lag in range(1, n_lags + 1):
        c = np.corrcoef(residuals[lag:], residuals[:-lag])[0, 1]
        acf_vals.append(c)
    ax.bar(range(n_lags + 1), acf_vals, color="#1f77b4", alpha=0.7, width=0.6)
    ci = 1.96 / np.sqrt(len(residuals))
    ax.axhline(ci, color="#d62728", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(-ci, color="#d62728", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title("Autocorrelation", fontsize=10)

    fig.tight_layout()
    save_figure(fig, name)


def plot_prediction_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs. Actual",
    name: str = "prediction_vs_actual",
) -> None:
    """Scatter plot with 45-degree line and R-squared annotation."""
    y_t = np.asarray(y_true).flatten()
    y_p = np.asarray(y_pred).flatten()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_t, y_p, alpha=0.35, s=14, color="#1f77b4", edgecolors="none")

    # 45-degree line
    lims = [
        min(y_t.min(), y_p.min()),
        max(y_t.max(), y_p.max()),
    ]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, "--", color="#d62728", linewidth=1.2, label="Perfect")

    # R-squared
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - y_t.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))

    ax.text(0.05, 0.92, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.6f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="0.8", alpha=0.9))

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="0.8")
    save_figure(fig, name)


# ============================================================================
# SECTION 7 : Macro & Factor Plots
# ============================================================================

def plot_macro_series(
    macro: pd.DataFrame,
    series_names: Optional[Dict[str, str]] = None,
    title: str = "Macroeconomic Indicators",
    name: str = "macro_series",
) -> None:
    """Multi-panel subplot of macroeconomic time series."""
    cols = macro.columns.tolist()
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        label = series_names.get(col, col) if series_names else col
        ax.plot(macro.index, macro[col], linewidth=1.0, color="#1f77b4")
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.margins(x=0.01)
        # Light shading for recessions could be added here

    axes[0].set_title(title, fontweight="bold")
    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save_figure(fig, name)


def plot_macro_asset_heatmap(
    cross_corr: pd.DataFrame,
    title: str = "Macro-Return Cross-Correlations",
    name: str = "macro_asset_heatmap",
) -> None:
    """Heatmap of correlations between macro indicators (rows) and asset
    returns (columns)."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    vmax = max(abs(cross_corr.values.min()), abs(cross_corr.values.max()))
    sns.heatmap(
        cross_corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    save_figure(fig, name)


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    title: str = "Asset Correlation Matrix",
    name: str = "correlation_heatmap",
) -> None:
    """Lower-triangular correlation heatmap of asset returns."""
    corr = returns.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"shrink": 0.75, "label": "Correlation"},
    )
    ax.set_title(title, fontweight="bold")
    save_figure(fig, name)


def plot_rolling_correlation(
    returns: pd.DataFrame,
    base_asset: str,
    window: int = 63,
    title: str = "Rolling Correlation",
    name: str = "rolling_correlation",
) -> None:
    """Rolling correlation of every column in *returns* with *base_asset*."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    other = [c for c in returns.columns if c != base_asset]
    colors = _default_colors(len(other))

    for col, clr in zip(other, colors):
        rc = returns[base_asset].rolling(window).corr(returns[col])
        ax.plot(rc.index, rc.values, label=col, color=clr, linewidth=1.0,
                alpha=0.85)

    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title(f"{title} (with {base_asset}, {window}-day window)",
                 fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="lower left", fontsize=7, ncol=3, frameon=True,
              framealpha=0.9, edgecolor="0.8")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    save_figure(fig, name)


# ============================================================================
# SECTION 8 : Stress Test & Regime Plots
# ============================================================================

def plot_stress_test_results(
    stress_df: pd.DataFrame,
    title: str = "Stress Test: Total Returns by Scenario",
    name: str = "stress_test",
) -> None:
    """Grouped bar chart of portfolio (and optional benchmark) returns during
    stress periods."""
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(stress_df))
    has_bench = "benchmark_return" in stress_df.columns
    width = 0.35 if has_bench else 0.5
    offset = width / 2 if has_bench else 0

    ax.bar(x - offset, stress_df["total_return"] * 100, width,
           label="Portfolio", color="#1f77b4", edgecolor="white", linewidth=0.5)
    if has_bench:
        ax.bar(x + offset, stress_df["benchmark_return"] * 100, width,
               label="Benchmark", color="#d62728", edgecolor="white",
               linewidth=0.5)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Total Return (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(stress_df.index, rotation=45, ha="right", fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")

    # Value annotations
    for i, v in enumerate(stress_df["total_return"]):
        ax.text(i - offset, v * 100 + (0.3 if v >= 0 else -0.8),
                f"{v:.1%}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=7)

    save_figure(fig, name)


def plot_regime_returns(
    returns: pd.Series,
    regime_labels: pd.Series,
    title: str = "Returns by Regime",
    name: str = "regime_returns",
) -> None:
    """Plot returns with background colour indicating the market regime.

    *regime_labels* should be an integer Series aligned with *returns*
    (e.g. 0 = bull, 1 = bear).
    """
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE_WIDE, sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    # Top panel: cumulative returns
    cum = (1 + returns).cumprod()
    axes[0].plot(cum.index, cum.values, color="#1f77b4", linewidth=1.2)

    regime_colors = {0: "#2ca02c", 1: "#d62728", 2: "#ff7f0e"}
    regime_names = {0: "Bull / Low Vol", 1: "Bear / High Vol", 2: "Transition"}
    unique = sorted(regime_labels.dropna().unique())
    for reg in unique:
        mask = regime_labels == reg
        spans = _contiguous_spans(mask)
        color = regime_colors.get(reg, "#cccccc")
        for s, e in spans:
            axes[0].axvspan(returns.index[s], returns.index[e],
                            alpha=0.12, color=color)
            axes[1].axvspan(returns.index[s], returns.index[e],
                            alpha=0.25, color=color)

    axes[0].set_ylabel("Cumulative Growth")
    axes[0].set_title(title, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))

    # Bottom panel: regime indicator
    axes[1].plot(returns.index, regime_labels.reindex(returns.index),
                 drawstyle="steps-post", color="black", linewidth=0.8)
    axes[1].set_ylabel("Regime")
    axes[1].set_xlabel("Date")
    axes[1].set_yticks(unique)
    axes[1].set_yticklabels([regime_names.get(r, str(r)) for r in unique],
                            fontsize=8)

    # Legend patches
    handles = [mpatches.Patch(color=regime_colors.get(r, "#ccc"), alpha=0.3,
               label=regime_names.get(r, str(r))) for r in unique]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8, frameon=True,
                   framealpha=0.9, edgecolor="0.8")

    fig.tight_layout()
    save_figure(fig, name)


def _contiguous_spans(mask: pd.Series) -> List[Tuple[int, int]]:
    """Return list of (start_idx, end_idx) for contiguous True regions."""
    arr = mask.values.astype(int)
    diffs = np.diff(arr, prepend=0, append=0)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    return list(zip(starts, ends))


def plot_crisis_deep_dive(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    crisis_start: str,
    crisis_end: str,
    title: str = "Crisis Deep Dive",
    name: str = "crisis_deep_dive",
) -> None:
    """Detailed crisis chart: cumulative returns, drawdowns, and rolling
    volatility for the portfolio vs. benchmark over a crisis window."""
    start = pd.Timestamp(crisis_start)
    end = pd.Timestamp(crisis_end)
    port = portfolio_returns.loc[start:end]
    bench = benchmark_returns.loc[start:end]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontweight="bold", fontsize=14, y=1.01)

    # Panel 1: Cumulative
    cum_p = (1 + port).cumprod()
    cum_b = (1 + bench).cumprod()
    axes[0].plot(cum_p.index, cum_p.values, label="Portfolio", color="#1f77b4",
                 linewidth=1.5)
    axes[0].plot(cum_b.index, cum_b.values, label="Benchmark", color="#d62728",
                 linewidth=1.5, linestyle="--")
    axes[0].set_ylabel("Growth of $1")
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    axes[0].legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor="0.8")
    axes[0].set_title("Cumulative Returns", fontsize=11)

    # Panel 2: Drawdown
    for series, label, clr, ls in [
        (port, "Portfolio", "#1f77b4", "-"),
        (bench, "Benchmark", "#d62728", "--"),
    ]:
        cum = (1 + series).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        axes[1].fill_between(dd.index, dd.values, 0, color=clr, alpha=0.25)
        axes[1].plot(dd.index, dd.values, color=clr, linewidth=1.0,
                     linestyle=ls, label=label)
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    axes[1].legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor="0.8")
    axes[1].set_title("Drawdowns", fontsize=11)

    # Panel 3: Rolling 21-day volatility
    vol_window = min(21, len(port) // 2) if len(port) > 4 else len(port)
    vol_p = port.rolling(vol_window).std() * np.sqrt(252)
    vol_b = bench.rolling(vol_window).std() * np.sqrt(252)
    axes[2].plot(vol_p.index, vol_p.values, label="Portfolio", color="#1f77b4",
                 linewidth=1.2)
    axes[2].plot(vol_b.index, vol_b.values, label="Benchmark", color="#d62728",
                 linewidth=1.2, linestyle="--")
    axes[2].set_ylabel("Ann. Volatility")
    axes[2].yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    axes[2].set_xlabel("Date")
    axes[2].legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor="0.8")
    axes[2].set_title("Rolling Volatility", fontsize=11)

    fig.tight_layout()
    save_figure(fig, name)


# ============================================================================
# SECTION 9 : Summary Tables & Tear Sheet
# ============================================================================

def plot_metrics_table(
    metrics_df: pd.DataFrame,
    title: str = "Strategy Performance Comparison",
    name: str = "metrics_table",
) -> None:
    """Render a performance-metrics DataFrame as a publication-quality table
    figure."""
    display_df = metrics_df.copy()

    # Auto-format columns based on name heuristics
    for col in display_df.columns:
        cl = col.lower()
        if any(k in cl for k in ("return", "vol", "drawdown", "var", "cvar",
                                  "tracking")):
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        elif any(k in cl for k in ("ratio", "sharpe", "sortino", "calmar",
                                    "information")):
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        elif "turnover" in cl:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        elif "duration" in cl:
            display_df[col] = display_df[col].apply(
                lambda x: f"{int(x)} days" if pd.notna(x) else "N/A")

    n_rows = len(display_df)
    n_cols = len(display_df.columns)
    fig_h = max(3.0, n_rows * 0.55 + 2.0)
    fig_w = max(12.0, n_cols * 1.6 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        rowLabels=display_df.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2171b5")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f0f0")
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    save_figure(fig, name)


def plot_tear_sheet(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    name: str = "tear_sheet",
) -> None:
    """Six-panel performance tear sheet.

    Panels
    ------
    1. Cumulative returns (portfolio + benchmark)
    2. Drawdowns
    3. Monthly returns heatmap
    4. Return distribution (KDE + histogram)
    5. Rolling 12-month Sharpe and volatility
    6. Summary statistics table
    """
    title = "Performance Tear Sheet"
    fig = plt.figure(figsize=(18, 24))
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.28,
                           left=0.06, right=0.96, top=0.94, bottom=0.04)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.97)

    # ---- Panel 1: Cumulative Returns ----
    ax1 = fig.add_subplot(gs[0, :])
    cum = (1 + returns).cumprod()
    ax1.plot(cum.index, cum.values, label="Portfolio", color="#1f77b4",
             linewidth=1.5)
    if benchmark_returns is not None:
        cum_b = (1 + benchmark_returns).cumprod()
        ax1.plot(cum_b.index, cum_b.values, label="Benchmark",
                 color="#d62728", linewidth=1.3, linestyle="--")
    ax1.set_ylabel("Growth of $1")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax1.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor="0.8")
    ax1.set_title("Cumulative Returns", fontsize=12, fontweight="bold")
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ---- Panel 2: Drawdowns ----
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    peak = cum.cummax()
    dd = (cum - peak) / peak
    ax2.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.4)
    ax2.plot(dd.index, dd.values, color="#d62728", linewidth=0.6)
    if benchmark_returns is not None:
        peak_b = cum_b.cummax()
        dd_b = (cum_b - peak_b) / peak_b
        ax2.plot(dd_b.index, dd_b.values, color="#7f7f7f", linewidth=0.8,
                 linestyle="--", alpha=0.7, label="Benchmark")
        ax2.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor="0.8")
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    ax2.set_title("Drawdowns", fontsize=12, fontweight="bold")

    # ---- Panel 3: Monthly Returns Heatmap ----
    ax3 = fig.add_subplot(gs[2, 0])
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    m_idx = pd.DatetimeIndex(monthly.index)
    pivot = pd.DataFrame({
        "year": m_idx.year, "month": m_idx.month, "ret": monthly.values
    }).pivot(index="month", columns="year", values="ret")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.index = month_labels[:len(pivot.index)]
    pivot = pivot.reindex(month_labels)
    vabs = max(abs(np.nanmin(pivot.values)), abs(np.nanmax(pivot.values)))
    sns.heatmap(pivot, annot=True, fmt=".1%", center=0, cmap="RdYlGn",
                vmin=-vabs, vmax=vabs, linewidths=0.5, linecolor="white",
                ax=ax3, cbar_kws={"format": mticker.PercentFormatter(1.0, 0),
                                  "shrink": 0.8})
    ax3.set_title("Monthly Returns", fontsize=12, fontweight="bold")
    ax3.set_ylabel("")
    ax3.tick_params(labelsize=7)

    # ---- Panel 4: Return Distribution ----
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(returns.dropna(), bins=80, density=True, alpha=0.6,
             color="#1f77b4", edgecolor="white", linewidth=0.3,
             label="Portfolio")
    returns.dropna().plot.kde(ax=ax4, color="#1f77b4", linewidth=1.4)
    if benchmark_returns is not None:
        benchmark_returns.dropna().plot.kde(ax=ax4, color="#d62728",
                                           linewidth=1.2, linestyle="--",
                                           label="Benchmark")
    ax4.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax4.set_title("Return Distribution", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Daily Return")
    ax4.set_ylabel("Density")
    ax4.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor="0.8")

    # ---- Panel 5: Rolling Sharpe & Vol ----
    ax5 = fig.add_subplot(gs[3, 0])
    roll_window = 252
    roll_mean = returns.rolling(roll_window).mean() * 252
    roll_vol = returns.rolling(roll_window).std() * np.sqrt(252)
    roll_sharpe = roll_mean / roll_vol

    color_sharpe = "#1f77b4"
    color_vol = "#d62728"
    ln1 = ax5.plot(roll_sharpe.index, roll_sharpe.values, color=color_sharpe,
                   linewidth=1.2, label="Sharpe (LHS)")
    ax5.set_ylabel("Sharpe Ratio", color=color_sharpe)
    ax5.tick_params(axis="y", labelcolor=color_sharpe)
    ax5.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.4)

    ax5b = ax5.twinx()
    ln2 = ax5b.plot(roll_vol.index, roll_vol.values, color=color_vol,
                    linewidth=1.0, linestyle="--", label="Volatility (RHS)")
    ax5b.set_ylabel("Annualized Volatility", color=color_vol)
    ax5b.tick_params(axis="y", labelcolor=color_vol)
    ax5b.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax5.legend(lns, labs, loc="upper left", fontsize=8, frameon=True,
               framealpha=0.9, edgecolor="0.8")
    ax5.set_title("Rolling 12-Month Metrics", fontsize=12, fontweight="bold")
    ax5.xaxis.set_major_locator(mdates.YearLocator(2))
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ---- Panel 6: Summary Stats Table ----
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis("off")

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    neg_vol = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_ret / neg_vol if neg_vol > 0 else 0.0
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    total_ret = cum.iloc[-1] - 1
    skew_val = returns.skew()
    kurt_val = returns.kurtosis()
    var95 = -np.percentile(returns.dropna(), 5)
    cvar95 = -returns.dropna()[returns.dropna() <= -var95].mean() if len(
        returns.dropna()[returns.dropna() <= -var95]) > 0 else var95

    stats_data = [
        ["Total Return", f"{total_ret:.2%}"],
        ["Ann. Return", f"{ann_ret:.2%}"],
        ["Ann. Volatility", f"{ann_vol:.2%}"],
        ["Sharpe Ratio", f"{sharpe:.3f}"],
        ["Sortino Ratio", f"{sortino:.3f}"],
        ["Max Drawdown", f"{max_dd:.2%}"],
        ["Calmar Ratio", f"{calmar:.3f}"],
        ["Skewness", f"{skew_val:.3f}"],
        ["Kurtosis", f"{kurt_val:.3f}"],
        ["Daily VaR (95%)", f"{var95:.2%}"],
        ["Daily CVaR (95%)", f"{cvar95:.2%}"],
    ]

    table = ax6.table(
        cellText=stats_data,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2171b5")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f0f0")
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)

    ax6.set_title("Summary Statistics", fontsize=12, fontweight="bold",
                  pad=15)

    save_figure(fig, name)
