"""
Rolling-window backtesting framework for portfolio strategies.

Provides a comprehensive, research-grade backtesting infrastructure for an
Industrial Engineering thesis on portfolio optimization. Includes portfolio
performance metrics, transaction cost modelling, a configurable backtesting
engine, benchmark strategies, stress testing utilities, strategy comparison
tools, and performance-attribution analytics.

Author : Arhan Subasi
Module : thesis_portfolio_opt.src.optimization.backtester
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.config import (
    BACKTEST_REBALANCE_FREQ,
    BACKTEST_LOOKBACK,
    BACKTEST_START_OFFSET,
    TRANSACTION_COST_BPS,
    TC_FIXED_BPS,
    TC_SPREAD_BPS,
    TC_IMPACT_COEFFICIENT,
    STRESS_SCENARIOS,
    TICKER_LIST,
    REBALANCE_FREQUENCIES,
    SIXTY_FORTY_WEIGHTS,
    BENCHMARK_TICKER,
    get_stress_dates,
)
from src.optimization.optimizer import (
    mean_variance_optimize,
    estimate_covariance,
    minimum_variance_optimize,
    risk_parity_optimize,
    inverse_volatility_weights,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TRADING_DAYS_PER_YEAR = 252
_SQRT_252 = np.sqrt(_TRADING_DAYS_PER_YEAR)


# ============================================================================
# Section 1 — Portfolio Metrics
# ============================================================================

def compute_portfolio_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict:
    """Compute a comprehensive set of portfolio performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns (not log returns).
    risk_free_rate : float
        Annualised risk-free rate used for Sharpe / Sortino computation.

    Returns
    -------
    dict
        Dictionary with the following keys:
        annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio,
        max_drawdown, max_drawdown_duration, calmar_ratio, total_return,
        var_95, cvar_95, skewness, kurtosis, winning_days_pct.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return {k: np.nan for k in [
            "annualized_return", "annualized_volatility", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "max_drawdown_duration",
            "calmar_ratio", "total_return", "var_95", "cvar_95",
            "skewness", "kurtosis", "winning_days_pct",
        ]}

    # --- Basic annualised statistics ---
    ann_return = returns.mean() * _TRADING_DAYS_PER_YEAR
    ann_vol = returns.std(ddof=1) * _SQRT_252
    sharpe = (
        (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
    )

    # --- Sortino ratio (downside deviation) ---
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1) * _SQRT_252 if len(downside) > 1 else np.nan
    sortino = (
        (ann_return - risk_free_rate) / downside_std
        if downside_std and downside_std > 0 else 0.0
    )

    # --- Drawdown analysis ---
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown_series = (cumulative - running_max) / running_max
    max_drawdown = drawdown_series.min()

    # Maximum drawdown duration (in trading days)
    is_underwater = drawdown_series < 0
    if is_underwater.any():
        groups = (~is_underwater).cumsum()
        underwater_groups = groups[is_underwater]
        if len(underwater_groups) > 0:
            max_dd_duration = int(underwater_groups.value_counts().max())
        else:
            max_dd_duration = 0
    else:
        max_dd_duration = 0

    # --- Calmar ratio ---
    calmar = (
        ann_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    )

    # --- Total compounded return ---
    total_return = cumulative.iloc[-1] - 1.0

    # --- Value at Risk and Conditional VaR (95 %) ---
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95

    # --- Higher moments ---
    skewness = float(sp_stats.skew(returns, nan_policy="omit"))
    kurtosis = float(sp_stats.kurtosis(returns, nan_policy="omit"))

    # --- Win rate ---
    winning_days_pct = (returns > 0).sum() / len(returns) * 100.0

    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": max_dd_duration,
        "calmar_ratio": calmar,
        "total_return": total_return,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "winning_days_pct": winning_days_pct,
    }


def compute_relative_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    """Compute relative / benchmark-aware performance metrics.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.
    benchmark_returns : pd.Series
        Daily benchmark returns (must share index with *portfolio_returns*).

    Returns
    -------
    dict
        tracking_error, information_ratio, beta, alpha, up_capture,
        down_capture, active_return.
    """
    # Align the two series on their common dates
    aligned = pd.concat(
        [portfolio_returns.rename("port"), benchmark_returns.rename("bench")],
        axis=1,
    ).dropna()
    port = aligned["port"]
    bench = aligned["bench"]

    if len(aligned) < 2:
        return {k: np.nan for k in [
            "tracking_error", "information_ratio", "beta", "alpha",
            "up_capture", "down_capture", "active_return",
        ]}

    active = port - bench
    active_return = active.mean() * _TRADING_DAYS_PER_YEAR
    tracking_error = active.std(ddof=1) * _SQRT_252
    information_ratio = (
        active_return / tracking_error if tracking_error > 0 else 0.0
    )

    # CAPM beta and alpha (annualised)
    cov_matrix = np.cov(port.values, bench.values)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else np.nan
    alpha = (
        port.mean() * _TRADING_DAYS_PER_YEAR
        - beta * bench.mean() * _TRADING_DAYS_PER_YEAR
    )

    # Up / down capture ratios
    up_days = bench > 0
    down_days = bench < 0

    if up_days.sum() > 0:
        up_capture = (
            port[up_days].mean() / bench[up_days].mean()
        ) * 100.0
    else:
        up_capture = np.nan

    if down_days.sum() > 0:
        down_capture = (
            port[down_days].mean() / bench[down_days].mean()
        ) * 100.0
    else:
        down_capture = np.nan

    return {
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "beta": beta,
        "alpha": alpha,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "active_return": active_return,
    }


def compute_drawdown_series(
    returns: pd.Series,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute the full drawdown time-series and recovery information.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.

    Returns
    -------
    dd_series : pd.Series
        Daily drawdown values (non-positive, 0 at peaks).
    recovery_df : pd.DataFrame
        DataFrame with columns ``start``, ``trough``, ``end``, ``depth``,
        ``duration``, ``recovery`` for each drawdown episode.
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    dd_series = (cumulative - running_max) / running_max

    # Identify drawdown episodes
    is_dd = dd_series < 0
    episodes: list[dict] = []
    in_dd = False
    start = None
    trough_date = None
    trough_val = 0.0

    for date, val in dd_series.items():
        if val < 0 and not in_dd:
            in_dd = True
            start = date
            trough_date = date
            trough_val = val
        elif val < 0 and in_dd:
            if val < trough_val:
                trough_val = val
                trough_date = date
        elif val >= 0 and in_dd:
            episodes.append({
                "start": start,
                "trough": trough_date,
                "end": date,
                "depth": trough_val,
                "duration": (date - start).days if hasattr(date, "days") or hasattr(start, "days") else np.nan,
                "recovery": (date - trough_date).days if hasattr(date, "days") or hasattr(trough_date, "days") else np.nan,
            })
            in_dd = False
            trough_val = 0.0

    # Handle ongoing drawdown at end of series
    if in_dd:
        last_date = dd_series.index[-1]
        episodes.append({
            "start": start,
            "trough": trough_date,
            "end": pd.NaT,
            "depth": trough_val,
            "duration": np.nan,
            "recovery": np.nan,
        })

    # Compute durations using positional index for robustness
    idx_list = list(dd_series.index)
    for ep in episodes:
        try:
            s_idx = idx_list.index(ep["start"])
            if ep["end"] is not pd.NaT and ep["end"] in idx_list:
                e_idx = idx_list.index(ep["end"])
                t_idx = idx_list.index(ep["trough"])
                ep["duration"] = e_idx - s_idx
                ep["recovery"] = e_idx - t_idx
            else:
                ep["duration"] = len(idx_list) - 1 - s_idx
                ep["recovery"] = np.nan
        except (ValueError, TypeError):
            pass

    recovery_df = pd.DataFrame(episodes) if episodes else pd.DataFrame(
        columns=["start", "trough", "end", "depth", "duration", "recovery"]
    )
    return dd_series, recovery_df


def compute_monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """Build a calendar-year monthly returns matrix.

    Parameters
    ----------
    returns : pd.Series
        Daily returns with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Index = years, columns = month names (Jan–Dec) + YTD.
    """
    monthly = (1 + returns).resample("ME").prod() - 1
    table = monthly.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.month

    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    table["month_name"] = table["month"].map(month_names)

    pivot = table.pivot(index="year", columns="month_name", values="return")
    # Order columns Jan-Dec
    col_order = [month_names[m] for m in range(1, 13) if month_names[m] in pivot.columns]
    pivot = pivot[col_order]

    # Compute YTD
    annual = (1 + returns).resample("YE").prod() - 1
    annual.index = annual.index.year
    pivot["YTD"] = annual

    return pivot


# ============================================================================
# Section 2 — Transaction Cost Models
# ============================================================================

class TransactionCostModel(ABC):
    """Abstract base class for transaction cost models."""

    @abstractmethod
    def compute_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float = 1.0,
        **kwargs: Any,
    ) -> float:
        """Return the total cost of rebalancing (as a fraction of portfolio)."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class FixedCostModel(TransactionCostModel):
    """Fixed cost in basis points applied to total absolute turnover.

    Parameters
    ----------
    bps : float
        Cost per unit of turnover in basis points.
    """

    def __init__(self, bps: float = TC_FIXED_BPS):
        self.bps = bps

    def compute_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float = 1.0,
        **kwargs: Any,
    ) -> float:
        turnover = np.sum(np.abs(new_weights - old_weights))
        return turnover * self.bps / 10_000

    def __repr__(self) -> str:
        return f"FixedCostModel(bps={self.bps})"


class ProportionalCostModel(TransactionCostModel):
    """Proportional cost model — cost proportional to traded dollar amount.

    Parameters
    ----------
    bps : float
        Cost per unit of turnover in basis points.
    """

    def __init__(self, bps: float = TRANSACTION_COST_BPS):
        self.bps = bps

    def compute_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float = 1.0,
        **kwargs: Any,
    ) -> float:
        trade_amounts = np.abs(new_weights - old_weights) * portfolio_value
        return np.sum(trade_amounts) * self.bps / 10_000

    def __repr__(self) -> str:
        return f"ProportionalCostModel(bps={self.bps})"


class SquareRootImpactModel(TransactionCostModel):
    """Square-root market impact model (Almgren et al., 2005).

    cost_i = coefficient * sigma_i * sqrt(|trade_i| / adv_i)

    Parameters
    ----------
    coefficient : float
        Scaling coefficient for the impact model.
    adv : np.ndarray or float
        Average daily volume per asset (dollar terms) or a scalar applied
        uniformly.
    daily_vol : np.ndarray or None
        Daily return volatility per asset; if *None*, uses 1 % for all.
    """

    def __init__(
        self,
        coefficient: float = TC_IMPACT_COEFFICIENT,
        adv: Union[np.ndarray, float] = 1e8,
        daily_vol: Optional[np.ndarray] = None,
    ):
        self.coefficient = coefficient
        self.adv = adv
        self.daily_vol = daily_vol

    def compute_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float = 1.0,
        **kwargs: Any,
    ) -> float:
        trade_weights = np.abs(new_weights - old_weights)
        trade_dollars = trade_weights * portfolio_value

        adv = self.adv if isinstance(self.adv, np.ndarray) else np.full(
            len(old_weights), self.adv
        )
        dvol = (
            self.daily_vol
            if self.daily_vol is not None
            else np.full(len(old_weights), 0.01)
        )

        # Avoid division by zero
        participation = np.where(adv > 0, trade_dollars / adv, 0.0)
        impact = self.coefficient * dvol * np.sqrt(participation)
        cost = np.sum(impact * trade_dollars) / portfolio_value
        return cost

    def __repr__(self) -> str:
        return (
            f"SquareRootImpactModel(coefficient={self.coefficient}, "
            f"adv={self.adv})"
        )


class CompositeCostModel(TransactionCostModel):
    """Composite transaction cost model combining fixed, spread, and impact.

    total_cost = fixed + spread + market_impact

    Parameters
    ----------
    fixed_bps : float
        Fixed commission in basis points.
    spread_bps : float
        Bid-ask spread cost in basis points.
    impact_coeff : float
        Market impact coefficient for the sqrt model.
    adv : float or np.ndarray
        Average daily volume (dollar terms).
    daily_vol : np.ndarray or None
        Daily volatility per asset.
    """

    def __init__(
        self,
        fixed_bps: float = TC_FIXED_BPS,
        spread_bps: float = TC_SPREAD_BPS,
        impact_coeff: float = TC_IMPACT_COEFFICIENT,
        adv: Union[np.ndarray, float] = 1e8,
        daily_vol: Optional[np.ndarray] = None,
    ):
        self.fixed_model = FixedCostModel(bps=fixed_bps)
        self.spread_model = ProportionalCostModel(bps=spread_bps)
        self.impact_model = SquareRootImpactModel(
            coefficient=impact_coeff, adv=adv, daily_vol=daily_vol,
        )

    def compute_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float = 1.0,
        **kwargs: Any,
    ) -> float:
        fixed = self.fixed_model.compute_cost(
            old_weights, new_weights, portfolio_value, **kwargs,
        )
        spread = self.spread_model.compute_cost(
            old_weights, new_weights, portfolio_value, **kwargs,
        )
        impact = self.impact_model.compute_cost(
            old_weights, new_weights, portfolio_value, **kwargs,
        )
        return fixed + spread + impact

    def __repr__(self) -> str:
        return (
            f"CompositeCostModel(fixed={self.fixed_model.bps}bps, "
            f"spread={self.spread_model.bps}bps, "
            f"impact_coeff={self.impact_model.coefficient})"
        )


# ============================================================================
# Section 3 — Core Backtester (functional API + class-based engine)
# ============================================================================

def backtest_strategy(
    prices: pd.DataFrame,
    expected_returns_fn: Optional[Callable] = None,
    risk_aversion: float = 2.0,
    rebalance_freq: int = BACKTEST_REBALANCE_FREQ,
    lookback: int = BACKTEST_LOOKBACK,
    start_offset: int = BACKTEST_START_OFFSET,
    transaction_cost_bps: int = TRANSACTION_COST_BPS,
    cost_model: Optional[TransactionCostModel] = None,
) -> dict:
    """Run a rolling-window back-test with detailed tracking.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices (columns = tickers, index = dates).
    expected_returns_fn : callable or None
        ``fn(returns_window) -> expected_returns_vector``.  If *None*,
        annualised historical mean is used.
    risk_aversion : float
        Risk-aversion parameter for the mean-variance optimiser.
    rebalance_freq : int
        Number of trading days between rebalance events.
    lookback : int
        Rolling estimation window length (trading days).
    start_offset : int
        Number of initial days to skip (warm-up period).
    transaction_cost_bps : int
        Simple proportional cost in basis points (used when *cost_model* is
        ``None``).
    cost_model : TransactionCostModel or None
        An explicit cost model overriding *transaction_cost_bps*.

    Returns
    -------
    dict
        ``returns`` (pd.Series), ``weights`` (pd.DataFrame),
        ``turnover`` (pd.DataFrame), ``metrics`` (dict),
        ``cumulative`` (pd.Series), ``rebalance_dates`` (list).
    """
    daily_returns = prices.pct_change().dropna()
    n_assets = daily_returns.shape[1]
    dates = daily_returns.index

    weights_history: list[dict] = []
    portfolio_returns: list[dict] = []
    turnover_history: list[dict] = []
    rebalance_dates: list = []

    current_weights = np.ones(n_assets) / n_assets
    portfolio_value = 1.0

    for t in range(start_offset, len(dates)):
        date = dates[t]
        day_returns = daily_returns.iloc[t].values

        tc = 0.0
        # --- Rebalance decision ---
        if (t - start_offset) % rebalance_freq == 0:
            window = daily_returns.iloc[max(0, t - lookback):t]
            if len(window) < 20:
                # Not enough data; skip optimisation
                pass
            else:
                if expected_returns_fn is not None:
                    mu = expected_returns_fn(window)
                else:
                    mu = window.mean().values * _TRADING_DAYS_PER_YEAR

                cov = estimate_covariance(window)
                new_weights = mean_variance_optimize(
                    mu, cov, risk_aversion=risk_aversion,
                )

                if new_weights is not None:
                    turnover = np.sum(np.abs(new_weights - current_weights))
                    turnover_history.append(
                        {"date": date, "turnover": turnover}
                    )

                    # Transaction cost
                    if cost_model is not None:
                        tc = cost_model.compute_cost(
                            current_weights, new_weights, portfolio_value,
                        )
                    else:
                        tc = turnover * transaction_cost_bps / 10_000

                    current_weights = new_weights
                    rebalance_dates.append(date)

        # --- Daily portfolio return ---
        port_ret = np.dot(current_weights, day_returns) - tc
        portfolio_returns.append({"date": date, "return": port_ret})
        portfolio_value *= (1 + port_ret)

        # --- Drift weights forward ---
        current_weights = current_weights * (1 + day_returns)
        weight_sum = current_weights.sum()
        if weight_sum > 0:
            current_weights /= weight_sum
        else:
            current_weights = np.ones(n_assets) / n_assets

        weights_history.append(
            {"date": date, **dict(zip(daily_returns.columns, current_weights))}
        )

    # Assemble output
    returns_series = pd.DataFrame(portfolio_returns).set_index("date")["return"]
    weights_df = pd.DataFrame(weights_history).set_index("date")
    turnover_df = (
        pd.DataFrame(turnover_history).set_index("date")
        if turnover_history
        else pd.DataFrame(columns=["turnover"])
    )
    cumulative_series = (1 + returns_series).cumprod()
    metrics = compute_portfolio_metrics(returns_series)

    return {
        "returns": returns_series,
        "weights": weights_df,
        "turnover": turnover_df,
        "metrics": metrics,
        "cumulative": cumulative_series,
        "rebalance_dates": rebalance_dates,
    }


# ---------------------------------------------------------------------------
# BacktestResult — rich result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Container for back-test outputs with convenience methods.

    Attributes
    ----------
    returns : pd.Series
    weights : pd.DataFrame
    turnover : pd.DataFrame
    metrics : dict
    cumulative : pd.Series
    rebalance_dates : list
    strategy_name : str
    """

    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.DataFrame
    metrics: dict
    cumulative: pd.Series
    rebalance_dates: list = field(default_factory=list)
    strategy_name: str = "Strategy"

    # --- convenience ---------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"=== {self.strategy_name} — Backtest Summary ===",
            f"Period         : {self.returns.index[0].date()} -> "
            f"{self.returns.index[-1].date()}",
            f"Trading days   : {len(self.returns)}",
        ]
        for key, val in self.metrics.items():
            if isinstance(val, float):
                lines.append(f"  {key:30s}: {val:>12.4f}")
            else:
                lines.append(f"  {key:30s}: {val}")
        if len(self.turnover) > 0 and "turnover" in self.turnover.columns:
            avg_to = self.turnover["turnover"].mean()
            lines.append(f"  {'avg_turnover':30s}: {avg_to:>12.4f}")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Export all daily series into a single DataFrame."""
        df = pd.DataFrame({"return": self.returns, "cumulative": self.cumulative})
        df = df.join(self.weights, rsuffix="_w")
        if len(self.turnover) > 0 and "turnover" in self.turnover.columns:
            df = df.join(self.turnover)
        return df

    def plot(
        self,
        figsize: Tuple[int, int] = (14, 8),
        title: Optional[str] = None,
    ) -> Any:
        """Quick four-panel visualisation: cumulative, drawdown, weights, turnover."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not available — cannot plot.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title or f"{self.strategy_name} — Backtest", fontsize=14)

        # 1 — Cumulative return
        ax = axes[0, 0]
        self.cumulative.plot(ax=ax, linewidth=1)
        ax.set_title("Cumulative Return")
        ax.set_ylabel("Growth of $1")

        # 2 — Drawdown
        ax = axes[0, 1]
        dd, _ = compute_drawdown_series(self.returns)
        dd.plot(ax=ax, linewidth=1, color="crimson")
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown (%)")
        ax.fill_between(dd.index, dd, 0, color="crimson", alpha=0.25)

        # 3 — Portfolio weights (stacked area)
        ax = axes[1, 0]
        self.weights.plot.area(ax=ax, linewidth=0, legend=False)
        ax.set_title("Weight Allocation")
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)

        # 4 — Turnover at rebalances
        ax = axes[1, 1]
        if len(self.turnover) > 0 and "turnover" in self.turnover.columns:
            self.turnover["turnover"].plot.bar(ax=ax, width=1.0)
        ax.set_title("Turnover at Rebalance")
        ax.set_ylabel("Two-way Turnover")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def compare(self, other: "BacktestResult") -> pd.DataFrame:
        """Side-by-side metric comparison with another BacktestResult."""
        rows = []
        all_keys = sorted(
            set(self.metrics.keys()) | set(other.metrics.keys())
        )
        for key in all_keys:
            rows.append({
                "metric": key,
                self.strategy_name: self.metrics.get(key, np.nan),
                other.strategy_name: other.metrics.get(key, np.nan),
            })
        return pd.DataFrame(rows).set_index("metric")


# ---------------------------------------------------------------------------
# BacktestEngine — class-based configurable engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Configurable backtesting engine.

    Usage
    -----
    >>> engine = BacktestEngine()
    >>> engine.set_strategy(my_opt_fn)
    >>> engine.set_rebalance_schedule("monthly")
    >>> engine.set_cost_model(CompositeCostModel())
    >>> result = engine.run(prices)
    """

    def __init__(
        self,
        lookback: int = BACKTEST_LOOKBACK,
        start_offset: int = BACKTEST_START_OFFSET,
        risk_aversion: float = 2.0,
        strategy_name: str = "Strategy",
    ):
        self.lookback = lookback
        self.start_offset = start_offset
        self.risk_aversion = risk_aversion
        self.strategy_name = strategy_name

        # Defaults
        self._optimization_fn: Optional[Callable] = None
        self._rebalance_freq: int = BACKTEST_REBALANCE_FREQ
        self._custom_rebalance_dates: Optional[List] = None
        self._cost_model: Optional[TransactionCostModel] = None
        self._cost_bps: int = TRANSACTION_COST_BPS

    # --- configuration ------------------------------------------------

    def set_strategy(self, optimization_fn: Callable) -> "BacktestEngine":
        """Set the portfolio optimisation function.

        ``optimization_fn(returns_window, cov_matrix, **kwargs) -> weights``
        """
        self._optimization_fn = optimization_fn
        return self

    def set_rebalance_schedule(
        self,
        freq: Union[int, str, None] = None,
        custom_dates: Optional[List] = None,
    ) -> "BacktestEngine":
        """Set rebalance frequency.

        Parameters
        ----------
        freq : int or str
            Integer (trading days) or a key from ``REBALANCE_FREQUENCIES``
            (e.g. ``"monthly"``).
        custom_dates : list of datetime-like, optional
            Explicit list of rebalance dates.
        """
        if custom_dates is not None:
            self._custom_rebalance_dates = [pd.Timestamp(d) for d in custom_dates]
        elif isinstance(freq, str):
            if freq in REBALANCE_FREQUENCIES:
                self._rebalance_freq = REBALANCE_FREQUENCIES[freq]
            else:
                raise ValueError(
                    f"Unknown frequency '{freq}'. "
                    f"Choose from {list(REBALANCE_FREQUENCIES.keys())}."
                )
        elif isinstance(freq, (int, float)):
            self._rebalance_freq = int(freq)
        return self

    def set_cost_model(self, model: TransactionCostModel) -> "BacktestEngine":
        """Attach a transaction cost model."""
        self._cost_model = model
        return self

    # --- execution ----------------------------------------------------

    def run(self, prices: pd.DataFrame) -> BacktestResult:
        """Execute the back-test on *prices* and return a ``BacktestResult``."""
        daily_returns = prices.pct_change().dropna()
        n_assets = daily_returns.shape[1]
        dates = daily_returns.index

        weights_history: list[dict] = []
        portfolio_returns: list[dict] = []
        turnover_history: list[dict] = []
        rebalance_dates_out: list = []

        current_weights = np.ones(n_assets) / n_assets
        portfolio_value = 1.0

        custom_dates_set = (
            set(self._custom_rebalance_dates)
            if self._custom_rebalance_dates
            else None
        )

        for t in range(self.start_offset, len(dates)):
            date = dates[t]
            day_returns = daily_returns.iloc[t].values
            tc = 0.0

            # Determine whether to rebalance
            should_rebalance = False
            if custom_dates_set is not None:
                should_rebalance = date in custom_dates_set
            else:
                should_rebalance = (
                    (t - self.start_offset) % self._rebalance_freq == 0
                )

            if should_rebalance:
                window = daily_returns.iloc[max(0, t - self.lookback):t]
                if len(window) >= 20:
                    cov = estimate_covariance(window)

                    if self._optimization_fn is not None:
                        new_weights = self._optimization_fn(
                            window, cov,
                            risk_aversion=self.risk_aversion,
                        )
                    else:
                        mu = window.mean().values * _TRADING_DAYS_PER_YEAR
                        new_weights = mean_variance_optimize(
                            mu, cov, risk_aversion=self.risk_aversion,
                        )

                    if new_weights is not None:
                        turnover = np.sum(np.abs(new_weights - current_weights))
                        turnover_history.append(
                            {"date": date, "turnover": turnover}
                        )
                        if self._cost_model is not None:
                            tc = self._cost_model.compute_cost(
                                current_weights, new_weights, portfolio_value,
                            )
                        else:
                            tc = turnover * self._cost_bps / 10_000

                        current_weights = new_weights
                        rebalance_dates_out.append(date)

            port_ret = np.dot(current_weights, day_returns) - tc
            portfolio_returns.append({"date": date, "return": port_ret})
            portfolio_value *= (1 + port_ret)

            current_weights = current_weights * (1 + day_returns)
            wsum = current_weights.sum()
            if wsum > 0:
                current_weights /= wsum
            else:
                current_weights = np.ones(n_assets) / n_assets

            weights_history.append(
                {"date": date, **dict(zip(daily_returns.columns, current_weights))}
            )

        returns_series = pd.DataFrame(portfolio_returns).set_index("date")["return"]
        weights_df = pd.DataFrame(weights_history).set_index("date")
        turnover_df = (
            pd.DataFrame(turnover_history).set_index("date")
            if turnover_history
            else pd.DataFrame(columns=["turnover"])
        )
        cumulative_series = (1 + returns_series).cumprod()
        metrics = compute_portfolio_metrics(returns_series)

        return BacktestResult(
            returns=returns_series,
            weights=weights_df,
            turnover=turnover_df,
            metrics=metrics,
            cumulative=cumulative_series,
            rebalance_dates=rebalance_dates_out,
            strategy_name=self.strategy_name,
        )

    def run_walk_forward(
        self,
        prices: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        train_fn: Optional[Callable] = None,
        train_window: int = 252 * 3,
        test_window: int = 21,
        step: int = 21,
    ) -> BacktestResult:
        """Walk-forward optimisation back-test.

        At each step the model is re-trained on the most recent
        *train_window* days, then the resulting expected-returns function
        is used for the next *test_window* days.

        Parameters
        ----------
        prices : pd.DataFrame
        features : pd.DataFrame or None
            Feature matrix aligned to the same dates as *prices*.
        train_fn : callable
            ``train_fn(returns_train, features_train)``
            returning a callable ``predict_fn(returns_window) -> mu``.
        train_window : int
        test_window : int
        step : int

        Returns
        -------
        BacktestResult
        """
        daily_returns = prices.pct_change().dropna()
        n_assets = daily_returns.shape[1]
        dates = daily_returns.index

        all_port_returns: list[dict] = []
        all_weights: list[dict] = []
        all_turnover: list[dict] = []
        rebalance_dates_out: list = []

        current_weights = np.ones(n_assets) / n_assets
        portfolio_value = 1.0

        t = train_window
        while t + test_window <= len(dates):
            # Train
            train_ret = daily_returns.iloc[t - train_window:t]
            train_feat = (
                features.iloc[t - train_window:t]
                if features is not None else None
            )

            if train_fn is not None:
                predict_fn = train_fn(train_ret, train_feat)
            else:
                predict_fn = None

            # Test period
            for j in range(test_window):
                idx = t + j
                if idx >= len(dates):
                    break
                date = dates[idx]
                day_returns = daily_returns.iloc[idx].values
                tc = 0.0

                # Rebalance at start of each test block
                if j == 0:
                    window = daily_returns.iloc[max(0, idx - self.lookback):idx]
                    if len(window) >= 20:
                        cov = estimate_covariance(window)
                        if predict_fn is not None:
                            mu = predict_fn(window)
                        else:
                            mu = window.mean().values * _TRADING_DAYS_PER_YEAR

                        new_weights = mean_variance_optimize(
                            mu, cov, risk_aversion=self.risk_aversion,
                        )
                        if new_weights is not None:
                            turnover = np.sum(
                                np.abs(new_weights - current_weights)
                            )
                            all_turnover.append(
                                {"date": date, "turnover": turnover}
                            )
                            if self._cost_model is not None:
                                tc = self._cost_model.compute_cost(
                                    current_weights, new_weights,
                                    portfolio_value,
                                )
                            else:
                                tc = turnover * self._cost_bps / 10_000
                            current_weights = new_weights
                            rebalance_dates_out.append(date)

                port_ret = np.dot(current_weights, day_returns) - tc
                all_port_returns.append({"date": date, "return": port_ret})
                portfolio_value *= (1 + port_ret)

                current_weights = current_weights * (1 + day_returns)
                wsum = current_weights.sum()
                if wsum > 0:
                    current_weights /= wsum
                else:
                    current_weights = np.ones(n_assets) / n_assets

                all_weights.append(
                    {"date": date,
                     **dict(zip(daily_returns.columns, current_weights))}
                )

            t += step

        returns_series = pd.DataFrame(all_port_returns).set_index("date")["return"]
        weights_df = pd.DataFrame(all_weights).set_index("date")
        turnover_df = (
            pd.DataFrame(all_turnover).set_index("date")
            if all_turnover
            else pd.DataFrame(columns=["turnover"])
        )
        cumulative_series = (1 + returns_series).cumprod()
        metrics = compute_portfolio_metrics(returns_series)

        return BacktestResult(
            returns=returns_series,
            weights=weights_df,
            turnover=turnover_df,
            metrics=metrics,
            cumulative=cumulative_series,
            rebalance_dates=rebalance_dates_out,
            strategy_name=self.strategy_name + " (Walk-Forward)",
        )


# ============================================================================
# Section 5 — Benchmark Strategies
# ============================================================================

def benchmark_equal_weight(
    prices: pd.DataFrame,
    rebalance_freq: int = BACKTEST_REBALANCE_FREQ,
    start_offset: int = BACKTEST_START_OFFSET,
    transaction_cost_bps: int = TRANSACTION_COST_BPS,
) -> dict:
    """Equal-weight (1/N) benchmark, rebalanced periodically.

    Every rebalance date the portfolio is reset to equal weights.
    """
    daily_returns = prices.pct_change().dropna()
    n_assets = daily_returns.shape[1]
    dates = daily_returns.index

    weights_history: list[dict] = []
    portfolio_returns: list[dict] = []
    turnover_history: list[dict] = []

    current_weights = np.ones(n_assets) / n_assets

    for t in range(start_offset, len(dates)):
        date = dates[t]
        day_returns = daily_returns.iloc[t].values
        tc = 0.0

        if (t - start_offset) % rebalance_freq == 0:
            target = np.ones(n_assets) / n_assets
            turnover = np.sum(np.abs(target - current_weights))
            tc = turnover * transaction_cost_bps / 10_000
            turnover_history.append({"date": date, "turnover": turnover})
            current_weights = target

        port_ret = np.dot(current_weights, day_returns) - tc
        portfolio_returns.append({"date": date, "return": port_ret})

        current_weights = current_weights * (1 + day_returns)
        wsum = current_weights.sum()
        if wsum > 0:
            current_weights /= wsum

        weights_history.append(
            {"date": date, **dict(zip(daily_returns.columns, current_weights))}
        )

    ret_s = pd.DataFrame(portfolio_returns).set_index("date")["return"]
    return {
        "returns": ret_s,
        "weights": pd.DataFrame(weights_history).set_index("date"),
        "turnover": pd.DataFrame(turnover_history).set_index("date") if turnover_history else pd.DataFrame(columns=["turnover"]),
        "metrics": compute_portfolio_metrics(ret_s),
        "cumulative": (1 + ret_s).cumprod(),
    }


def benchmark_inverse_vol(
    prices: pd.DataFrame,
    lookback: int = BACKTEST_LOOKBACK,
    rebalance_freq: int = BACKTEST_REBALANCE_FREQ,
    start_offset: int = BACKTEST_START_OFFSET,
    transaction_cost_bps: int = TRANSACTION_COST_BPS,
) -> dict:
    """Inverse-volatility weighted benchmark."""
    daily_returns = prices.pct_change().dropna()
    n_assets = daily_returns.shape[1]
    dates = daily_returns.index

    weights_history: list[dict] = []
    portfolio_returns: list[dict] = []
    turnover_history: list[dict] = []
    current_weights = np.ones(n_assets) / n_assets

    for t in range(start_offset, len(dates)):
        date = dates[t]
        day_returns = daily_returns.iloc[t].values
        tc = 0.0

        if (t - start_offset) % rebalance_freq == 0:
            window = daily_returns.iloc[max(0, t - lookback):t]
            if len(window) >= 20:
                cov = estimate_covariance(window)
                target = inverse_volatility_weights(cov)
                turnover = np.sum(np.abs(target - current_weights))
                tc = turnover * transaction_cost_bps / 10_000
                turnover_history.append({"date": date, "turnover": turnover})
                current_weights = target

        port_ret = np.dot(current_weights, day_returns) - tc
        portfolio_returns.append({"date": date, "return": port_ret})

        current_weights = current_weights * (1 + day_returns)
        wsum = current_weights.sum()
        if wsum > 0:
            current_weights /= wsum

        weights_history.append(
            {"date": date, **dict(zip(daily_returns.columns, current_weights))}
        )

    ret_s = pd.DataFrame(portfolio_returns).set_index("date")["return"]
    return {
        "returns": ret_s,
        "weights": pd.DataFrame(weights_history).set_index("date"),
        "turnover": pd.DataFrame(turnover_history).set_index("date") if turnover_history else pd.DataFrame(columns=["turnover"]),
        "metrics": compute_portfolio_metrics(ret_s),
        "cumulative": (1 + ret_s).cumprod(),
    }


def benchmark_sixty_forty(
    prices: pd.DataFrame,
    weights_map: Optional[Dict[str, float]] = None,
    rebalance_freq: int = BACKTEST_REBALANCE_FREQ,
    start_offset: int = BACKTEST_START_OFFSET,
    transaction_cost_bps: int = TRANSACTION_COST_BPS,
) -> dict:
    """60/40 (or custom static allocation) benchmark.

    Parameters
    ----------
    prices : pd.DataFrame
        Must contain columns for the tickers referenced in *weights_map*.
    weights_map : dict
        Mapping of ticker -> target weight.  Defaults to
        ``SIXTY_FORTY_WEIGHTS`` from config (``SPY: 0.60, AGG: 0.40``).
    """
    if weights_map is None:
        weights_map = SIXTY_FORTY_WEIGHTS

    # Subset prices to relevant tickers
    tickers = list(weights_map.keys())
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        warnings.warn(
            f"60/40 benchmark: tickers {missing} not found in prices; "
            f"falling back to equal weight across available tickers."
        )
        tickers = [t for t in tickers if t in prices.columns]
        if not tickers:
            raise ValueError("No valid tickers for 60/40 benchmark.")
        weights_map = {t: 1.0 / len(tickers) for t in tickers}

    sub_prices = prices[tickers]
    daily_returns = sub_prices.pct_change().dropna()
    n_assets = len(tickers)
    dates = daily_returns.index

    target_weights = np.array([weights_map[t] for t in tickers])
    target_weights /= target_weights.sum()

    weights_history: list[dict] = []
    portfolio_returns: list[dict] = []
    turnover_history: list[dict] = []
    current_weights = target_weights.copy()

    for t in range(start_offset, len(dates)):
        date = dates[t]
        day_returns = daily_returns.iloc[t].values
        tc = 0.0

        if (t - start_offset) % rebalance_freq == 0:
            turnover = np.sum(np.abs(target_weights - current_weights))
            tc = turnover * transaction_cost_bps / 10_000
            turnover_history.append({"date": date, "turnover": turnover})
            current_weights = target_weights.copy()

        port_ret = np.dot(current_weights, day_returns) - tc
        portfolio_returns.append({"date": date, "return": port_ret})

        current_weights = current_weights * (1 + day_returns)
        wsum = current_weights.sum()
        if wsum > 0:
            current_weights /= wsum

        weights_history.append(
            {"date": date, **dict(zip(tickers, current_weights))}
        )

    ret_s = pd.DataFrame(portfolio_returns).set_index("date")["return"]
    return {
        "returns": ret_s,
        "weights": pd.DataFrame(weights_history).set_index("date"),
        "turnover": pd.DataFrame(turnover_history).set_index("date") if turnover_history else pd.DataFrame(columns=["turnover"]),
        "metrics": compute_portfolio_metrics(ret_s),
        "cumulative": (1 + ret_s).cumprod(),
    }


def benchmark_buy_and_hold(
    prices: pd.DataFrame,
    initial_weights: Optional[np.ndarray] = None,
    start_offset: int = BACKTEST_START_OFFSET,
) -> dict:
    """Buy-and-hold benchmark — initial allocation drifts with prices.

    No rebalancing is performed after the initial allocation.
    """
    daily_returns = prices.pct_change().dropna()
    n_assets = daily_returns.shape[1]
    dates = daily_returns.index

    if initial_weights is None:
        initial_weights = np.ones(n_assets) / n_assets

    current_weights = initial_weights.copy()
    weights_history: list[dict] = []
    portfolio_returns: list[dict] = []

    for t in range(start_offset, len(dates)):
        date = dates[t]
        day_returns = daily_returns.iloc[t].values

        port_ret = np.dot(current_weights, day_returns)
        portfolio_returns.append({"date": date, "return": port_ret})

        current_weights = current_weights * (1 + day_returns)
        wsum = current_weights.sum()
        if wsum > 0:
            current_weights /= wsum

        weights_history.append(
            {"date": date, **dict(zip(daily_returns.columns, current_weights))}
        )

    ret_s = pd.DataFrame(portfolio_returns).set_index("date")["return"]
    return {
        "returns": ret_s,
        "weights": pd.DataFrame(weights_history).set_index("date"),
        "turnover": pd.DataFrame(columns=["turnover"]),
        "metrics": compute_portfolio_metrics(ret_s),
        "cumulative": (1 + ret_s).cumprod(),
    }


# ============================================================================
# Section 6 — Stress Testing
# ============================================================================

def stress_test(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    scenarios: Optional[dict] = None,
) -> pd.DataFrame:
    """Evaluate portfolio performance during predefined historical crises.

    Parameters
    ----------
    portfolio_returns : pd.Series
    benchmark_returns : pd.Series or None
    scenarios : dict or None
        Dictionary of ``{name: {start, end, ...}}`` or ``{name: (start, end)}``.
        If *None*, uses ``STRESS_SCENARIOS`` from config.

    Returns
    -------
    pd.DataFrame
        One row per scenario with metrics.
    """
    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    results: list[dict] = []
    for name, info in scenarios.items():
        if isinstance(info, dict):
            start_str = info["start"]
            end_str = info["end"]
            description = info.get("description", name)
        elif isinstance(info, (list, tuple)) and len(info) >= 2:
            start_str, end_str = info[0], info[1]
            description = name
        else:
            continue

        start_dt = pd.Timestamp(start_str)
        end_dt = pd.Timestamp(end_str)
        mask = (portfolio_returns.index >= start_dt) & (
            portfolio_returns.index <= end_dt
        )
        period = portfolio_returns[mask]

        if len(period) == 0:
            continue

        pm = compute_portfolio_metrics(period)
        row: dict = {
            "scenario": name,
            "description": description,
            "start": start_str,
            "end": end_str,
            "n_days": len(period),
            "total_return": pm["total_return"],
            "annualized_vol": pm["annualized_volatility"],
            "max_drawdown": pm["max_drawdown"],
            "sharpe": pm["sharpe_ratio"],
            "var_95": pm["var_95"],
            "cvar_95": pm["cvar_95"],
        }

        if benchmark_returns is not None:
            bench_period = benchmark_returns[mask]
            if len(bench_period) > 0:
                bm = compute_portfolio_metrics(bench_period)
                row["benchmark_return"] = bm["total_return"]
                row["excess_return"] = pm["total_return"] - bm["total_return"]
                row["benchmark_max_dd"] = bm["max_drawdown"]

        results.append(row)

    return pd.DataFrame(results).set_index("scenario") if results else pd.DataFrame()


def monte_carlo_stress(
    returns: pd.Series,
    n_simulations: int = 1000,
    block_size: int = 21,
    horizon: int = 252,
    confidence_levels: Optional[List[float]] = None,
    random_state: int = 42,
) -> dict:
    """Block-bootstrap Monte Carlo stress simulation.

    Resamples blocks of historical returns to build simulated paths.

    Parameters
    ----------
    returns : pd.Series
        Historical daily returns.
    n_simulations : int
        Number of simulation paths.
    block_size : int
        Length of each resampled block (trading days).
    horizon : int
        Length of each simulated path (trading days).
    confidence_levels : list of float
        Quantile levels for reporting (e.g. [0.01, 0.05, 0.10]).
    random_state : int

    Returns
    -------
    dict
        ``simulated_returns`` (np.ndarray of shape (n_simulations, horizon)),
        ``terminal_values``, ``statistics`` DataFrame.
    """
    rng = np.random.RandomState(random_state)
    n = len(returns)
    vals = returns.values

    if confidence_levels is None:
        confidence_levels = [0.01, 0.05, 0.10, 0.25, 0.50]

    simulated = np.empty((n_simulations, horizon))
    n_blocks = int(np.ceil(horizon / block_size))

    for i in range(n_simulations):
        path: list[float] = []
        for _ in range(n_blocks):
            start_idx = rng.randint(0, max(1, n - block_size))
            block = vals[start_idx:start_idx + block_size]
            path.extend(block.tolist())
        simulated[i, :] = np.array(path[:horizon])

    # Terminal cumulative values
    terminal_values = np.prod(1 + simulated, axis=1)

    # Statistics table
    stats_rows: list[dict] = []
    ann_returns_sim = simulated.mean(axis=1) * _TRADING_DAYS_PER_YEAR
    ann_vols_sim = simulated.std(axis=1) * _SQRT_252

    for cl in confidence_levels:
        stats_rows.append({
            "quantile": cl,
            "terminal_value": np.percentile(terminal_values, cl * 100),
            "annualized_return": np.percentile(ann_returns_sim, cl * 100),
            "annualized_vol": np.percentile(ann_vols_sim, cl * 100),
            "max_drawdown": np.percentile(
                [_max_dd_array(simulated[i]) for i in range(n_simulations)],
                cl * 100,
            ),
        })

    # Add mean row
    stats_rows.append({
        "quantile": "mean",
        "terminal_value": terminal_values.mean(),
        "annualized_return": ann_returns_sim.mean(),
        "annualized_vol": ann_vols_sim.mean(),
        "max_drawdown": np.mean(
            [_max_dd_array(simulated[i]) for i in range(n_simulations)]
        ),
    })

    stats_df = pd.DataFrame(stats_rows).set_index("quantile")

    return {
        "simulated_returns": simulated,
        "terminal_values": terminal_values,
        "statistics": stats_df,
    }


def _max_dd_array(returns_arr: np.ndarray) -> float:
    """Compute maximum drawdown from a 1-D numpy array of returns."""
    cumulative = np.cumprod(1 + returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return float(drawdowns.min()) if len(drawdowns) > 0 else 0.0


def historical_scenario_analysis(
    returns: pd.Series,
    scenario_returns: pd.DataFrame,
) -> pd.DataFrame:
    """What-if analysis: overlay portfolio returns onto historical scenarios.

    Parameters
    ----------
    returns : pd.Series
        Portfolio daily returns (the actual track record).
    scenario_returns : pd.DataFrame
        Columns are different scenarios, rows are daily returns for those
        hypothetical periods.

    Returns
    -------
    pd.DataFrame
        Summary metrics for the portfolio under each scenario.
    """
    results: list[dict] = []
    for scenario_name in scenario_returns.columns:
        sr = scenario_returns[scenario_name].dropna()
        if len(sr) == 0:
            continue
        metrics = compute_portfolio_metrics(sr)
        metrics["scenario"] = scenario_name
        metrics["n_days"] = len(sr)
        results.append(metrics)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).set_index("scenario")


def tail_risk_analysis(
    returns: pd.Series,
    confidence_levels: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Compute VaR and CVaR at multiple confidence levels.

    Parameters
    ----------
    returns : pd.Series
    confidence_levels : list of float
        E.g. ``[0.90, 0.95, 0.99]``.

    Returns
    -------
    pd.DataFrame
        Columns: ``confidence``, ``var``, ``cvar``, ``var_ann``, ``cvar_ann``.
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.975, 0.99]

    rows: list[dict] = []
    vals = returns.dropna().values

    for cl in confidence_levels:
        alpha = 1.0 - cl
        var = np.percentile(vals, alpha * 100)
        tail = vals[vals <= var]
        cvar = tail.mean() if len(tail) > 0 else var

        rows.append({
            "confidence": cl,
            "var": var,
            "cvar": cvar,
            "var_annualized": var * _SQRT_252,
            "cvar_annualized": cvar * _SQRT_252,
        })

    return pd.DataFrame(rows).set_index("confidence")


# ============================================================================
# Section 7 — Strategy Comparison
# ============================================================================

def compare_strategies(
    prices: pd.DataFrame,
    strategies: Optional[Dict[str, dict]] = None,
    start_offset: int = BACKTEST_START_OFFSET,
    include_benchmarks: bool = True,
    cost_model: Optional[TransactionCostModel] = None,
) -> dict:
    """Run multiple strategies and produce a comparison table.

    Parameters
    ----------
    prices : pd.DataFrame
    strategies : dict
        ``{name: {kwarg_key: value, ...}}`` passed to ``backtest_strategy``.
    start_offset : int
    include_benchmarks : bool
        If *True* add equal-weight and inverse-vol benchmarks.
    cost_model : TransactionCostModel or None

    Returns
    -------
    dict
        ``metrics`` (pd.DataFrame), ``results`` (dict of backtest dicts).
    """
    if strategies is None:
        strategies = {
            "Mean-Variance (lambda=2)": {"risk_aversion": 2.0},
            "Mean-Variance (lambda=5)": {"risk_aversion": 5.0},
            "Mean-Variance (lambda=10)": {"risk_aversion": 10.0},
        }

    all_results: Dict[str, dict] = {}
    comparison: list[dict] = []

    for name, params in strategies.items():
        print(f"Running: {name} ...")
        result = backtest_strategy(
            prices,
            start_offset=start_offset,
            cost_model=cost_model,
            **params,
        )
        all_results[name] = result

        row: dict = {"strategy": name, **result["metrics"]}
        if (
            result["turnover"] is not None
            and len(result["turnover"]) > 0
            and "turnover" in result["turnover"].columns
        ):
            row["avg_turnover"] = result["turnover"]["turnover"].mean()
        comparison.append(row)

    if include_benchmarks:
        print("Running: Equal Weight ...")
        eq = benchmark_equal_weight(prices, start_offset=start_offset)
        all_results["Equal Weight"] = eq
        comparison.append({"strategy": "Equal Weight", **eq["metrics"]})

        print("Running: Inverse Volatility ...")
        iv = benchmark_inverse_vol(prices, start_offset=start_offset)
        all_results["Inverse Volatility"] = iv
        comparison.append({"strategy": "Inverse Volatility", **iv["metrics"]})

        # Attempt 60/40 if possible
        try:
            print("Running: 60/40 ...")
            sf = benchmark_sixty_forty(prices, start_offset=start_offset)
            all_results["60/40"] = sf
            comparison.append({"strategy": "60/40", **sf["metrics"]})
        except (KeyError, ValueError) as exc:
            warnings.warn(f"60/40 benchmark skipped: {exc}")

    comparison_df = pd.DataFrame(comparison).set_index("strategy")

    return {"metrics": comparison_df, "results": all_results}


def statistical_comparison(
    returns_dict: Dict[str, pd.Series],
    test: str = "both",
) -> pd.DataFrame:
    """Test statistical significance of performance differences.

    For each pair of strategies, performs:
    - Paired *t*-test on daily returns (``H0: mean_diff = 0``)
    - Wilcoxon signed-rank test (non-parametric)

    Parameters
    ----------
    returns_dict : dict
        ``{strategy_name: pd.Series of daily returns}``.
    test : str
        ``"ttest"``, ``"wilcoxon"``, or ``"both"``.

    Returns
    -------
    pd.DataFrame
        Rows are strategy pairs; columns are test statistics, p-values, and
        mean difference.
    """
    names = list(returns_dict.keys())
    rows: list[dict] = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a = returns_dict[a_name]
            b = returns_dict[b_name]

            # Align on common dates
            aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
            if len(aligned) < 10:
                continue
            diff = aligned["a"] - aligned["b"]

            row: dict = {
                "strategy_a": a_name,
                "strategy_b": b_name,
                "mean_diff_annual": diff.mean() * _TRADING_DAYS_PER_YEAR,
                "n_obs": len(aligned),
            }

            if test in ("ttest", "both"):
                t_stat, t_pval = sp_stats.ttest_rel(
                    aligned["a"].values, aligned["b"].values,
                )
                row["t_statistic"] = t_stat
                row["t_pvalue"] = t_pval

            if test in ("wilcoxon", "both"):
                try:
                    w_stat, w_pval = sp_stats.wilcoxon(
                        aligned["a"].values, aligned["b"].values,
                    )
                    row["wilcoxon_statistic"] = w_stat
                    row["wilcoxon_pvalue"] = w_pval
                except ValueError:
                    row["wilcoxon_statistic"] = np.nan
                    row["wilcoxon_pvalue"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_rolling_metrics(
    returns: pd.Series,
    window: int = _TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Compute rolling annualised Sharpe, volatility, and drawdown.

    Parameters
    ----------
    returns : pd.Series
    window : int
        Rolling window in trading days.

    Returns
    -------
    pd.DataFrame
        Columns: ``rolling_sharpe``, ``rolling_vol``, ``rolling_return``,
        ``rolling_drawdown``.
    """
    rolling_mean = returns.rolling(window).mean() * _TRADING_DAYS_PER_YEAR
    rolling_std = returns.rolling(window).std() * _SQRT_252
    rolling_sharpe = rolling_mean / rolling_std

    # Rolling drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.rolling(window, min_periods=1).max()
    rolling_dd = (cumulative - rolling_max) / rolling_max

    df = pd.DataFrame({
        "rolling_sharpe": rolling_sharpe,
        "rolling_vol": rolling_std,
        "rolling_return": rolling_mean,
        "rolling_drawdown": rolling_dd,
    }).dropna()
    return df


# ============================================================================
# Section 8 — Performance Attribution
# ============================================================================

def attribute_returns(
    portfolio_returns: pd.Series,
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    benchmark_weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Brinson-Fachler style attribution (allocation + selection + interaction).

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.
    weights : pd.DataFrame
        Daily portfolio weights (columns = assets, index aligned with returns).
    asset_returns : pd.DataFrame
        Daily asset returns (columns = assets).
    benchmark_weights : np.ndarray or None
        Static benchmark weights; defaults to equal weight.

    Returns
    -------
    pd.DataFrame
        Columns: ``allocation``, ``selection``, ``interaction``, ``total_active``
        per asset, plus a summary row.
    """
    common_idx = weights.index.intersection(asset_returns.index)
    weights_aligned = weights.loc[common_idx]
    asset_ret_aligned = asset_returns.loc[common_idx]
    n_assets = asset_ret_aligned.shape[1]

    if benchmark_weights is None:
        benchmark_weights = np.ones(n_assets) / n_assets

    bw = benchmark_weights
    # Average portfolio weights over the period
    pw = weights_aligned.values.mean(axis=0)
    # Average asset returns over the period (annualised)
    ar = asset_ret_aligned.mean().values * _TRADING_DAYS_PER_YEAR
    # Benchmark return per asset
    br_total = np.dot(bw, ar)

    allocation = (pw - bw) * (ar - br_total)
    selection = bw * (ar - ar)  # placeholder — needs actual benchmark return per asset
    # More precise: selection_i = bw_i * (r_portfolio_asset_i - r_benchmark_asset_i)
    # Since portfolio and benchmark hold same assets, selection = bw * (ar_p - ar_b)
    # For simplicity with same asset universe:
    selection = bw * (ar - ar)  # zero when same assets
    interaction = (pw - bw) * (ar - ar)

    # Simplified: total active = (pw - bw) * ar
    total_active = (pw - bw) * ar

    tickers = list(asset_ret_aligned.columns)
    attribution_df = pd.DataFrame({
        "asset": tickers,
        "portfolio_weight": pw,
        "benchmark_weight": bw,
        "asset_return_ann": ar,
        "allocation_effect": allocation,
        "selection_effect": selection,
        "interaction_effect": interaction,
        "total_active": total_active,
    }).set_index("asset")

    # Add a summary row
    summary = pd.DataFrame({
        "portfolio_weight": [pw.sum()],
        "benchmark_weight": [bw.sum()],
        "asset_return_ann": [np.dot(pw, ar)],
        "allocation_effect": [allocation.sum()],
        "selection_effect": [selection.sum()],
        "interaction_effect": [interaction.sum()],
        "total_active": [total_active.sum()],
    }, index=["TOTAL"])
    attribution_df = pd.concat([attribution_df, summary])

    return attribution_df


def factor_attribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    annualise: bool = True,
) -> dict:
    """Regression-based factor attribution (multi-factor model).

    Regresses portfolio excess returns on supplied factor returns:
        R_p = alpha + sum(beta_k * F_k) + epsilon

    Parameters
    ----------
    portfolio_returns : pd.Series
    factor_returns : pd.DataFrame
        Columns are factor return series (e.g. market, size, value, momentum).
    annualise : bool
        If *True*, alpha is reported on an annualised basis.

    Returns
    -------
    dict
        ``alpha``, ``betas`` (pd.Series), ``r_squared``, ``adj_r_squared``,
        ``residual_vol``, ``factor_contributions`` (pd.Series),
        ``regression_summary`` (dict).
    """
    aligned = pd.concat(
        [portfolio_returns.rename("portfolio"), factor_returns],
        axis=1,
    ).dropna()

    if len(aligned) < 30:
        warnings.warn(
            "factor_attribution: fewer than 30 common observations; "
            "results may be unreliable."
        )

    y = aligned["portfolio"].values
    X = aligned.drop(columns=["portfolio"]).values
    factor_names = list(factor_returns.columns)

    # OLS with intercept
    X_with_const = np.column_stack([np.ones(len(y)), X])
    try:
        beta_hat, residuals, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "alpha": np.nan,
            "betas": pd.Series(dtype=float),
            "r_squared": np.nan,
            "adj_r_squared": np.nan,
            "residual_vol": np.nan,
            "factor_contributions": pd.Series(dtype=float),
            "regression_summary": {},
        }

    alpha_daily = beta_hat[0]
    betas = beta_hat[1:]
    y_hat = X_with_const @ beta_hat
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n_obs = len(y)
    k = len(betas)
    adj_r_squared = (
        1 - (1 - r_squared) * (n_obs - 1) / (n_obs - k - 1)
        if n_obs > k + 1 else np.nan
    )
    residual_vol = np.sqrt(ss_res / max(1, n_obs - k - 1)) * _SQRT_252

    alpha = alpha_daily * _TRADING_DAYS_PER_YEAR if annualise else alpha_daily

    # Factor contributions: beta * mean(factor)
    factor_means = aligned.drop(columns=["portfolio"]).mean().values
    factor_contrib_daily = betas * factor_means
    factor_contrib_ann = factor_contrib_daily * _TRADING_DAYS_PER_YEAR

    # T-statistics for betas
    mse = ss_res / max(1, n_obs - k - 1)
    try:
        var_beta = mse * np.diag(np.linalg.inv(X_with_const.T @ X_with_const))
        se_beta = np.sqrt(np.maximum(var_beta, 0))
        t_stats = beta_hat / se_beta
        p_values = 2 * (1 - sp_stats.t.cdf(np.abs(t_stats), df=n_obs - k - 1))
    except np.linalg.LinAlgError:
        t_stats = np.full(len(beta_hat), np.nan)
        p_values = np.full(len(beta_hat), np.nan)

    return {
        "alpha": alpha,
        "betas": pd.Series(betas, index=factor_names),
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "residual_vol": residual_vol,
        "factor_contributions": pd.Series(
            factor_contrib_ann, index=factor_names,
        ),
        "regression_summary": {
            "alpha_t": t_stats[0] if len(t_stats) > 0 else np.nan,
            "alpha_p": p_values[0] if len(p_values) > 0 else np.nan,
            "beta_t": dict(zip(factor_names, t_stats[1:])),
            "beta_p": dict(zip(factor_names, p_values[1:])),
            "n_obs": n_obs,
        },
    }


def risk_contribution(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> dict:
    """Compute marginal and component risk contributions.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weight vector (sums to 1).
    cov_matrix : np.ndarray
        Annualised covariance matrix.

    Returns
    -------
    dict
        ``portfolio_vol`` : float,
        ``marginal_risk`` : np.ndarray,
        ``component_risk`` : np.ndarray,
        ``pct_risk_contrib`` : np.ndarray  (sums to 1).
    """
    sigma_w = cov_matrix @ weights
    portfolio_var = weights @ sigma_w
    portfolio_vol = np.sqrt(portfolio_var) if portfolio_var > 0 else 0.0

    if portfolio_vol == 0:
        n = len(weights)
        return {
            "portfolio_vol": 0.0,
            "marginal_risk": np.zeros(n),
            "component_risk": np.zeros(n),
            "pct_risk_contrib": np.zeros(n),
        }

    marginal_risk = sigma_w / portfolio_vol
    component_risk = weights * marginal_risk
    pct_risk_contrib = component_risk / portfolio_vol

    return {
        "portfolio_vol": portfolio_vol,
        "marginal_risk": marginal_risk,
        "component_risk": component_risk,
        "pct_risk_contrib": pct_risk_contrib,
    }


# ============================================================================
# Legacy helper — retained for backward compatibility
# ============================================================================

def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """Compute rolling annualised Sharpe ratio (convenience alias)."""
    rolling_mean = returns.rolling(window).mean() * _TRADING_DAYS_PER_YEAR
    rolling_std = returns.rolling(window).std() * _SQRT_252
    return (rolling_mean / rolling_std).dropna()
