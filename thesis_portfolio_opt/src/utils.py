"""
Shared utility functions for the thesis portfolio optimization project.

Provides logging, timing/profiling, data validation, financial metrics,
portfolio analytics, statistical tests, and export utilities used throughout
the research pipeline.

Author: Arhan Subasi
"""

import time
import logging
import functools
import tracemalloc
from typing import Any, Callable, Optional, Union
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.linalg import eigvalsh

from src.config import (
    PROJECT_ROOT,
    RESULTS_DIR,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    SUMMARY_METRICS,
)


# ============================================================================
# SECTION 1: LOGGING
# ============================================================================

def setup_logger(
    name: str = "thesis",
    level: Union[int, str] = LOG_LEVEL,
) -> logging.Logger:
    """
    Configure a project logger with console and file output.

    Parameters
    ----------
    name : str
        Logger name. Using the same name returns the existing logger.
    level : int or str
        Logging level (e.g. logging.INFO, "DEBUG", "WARNING").

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger_inst = logging.getLogger(name)

    # Avoid duplicate handlers when called multiple times
    if logger_inst.handlers:
        return logger_inst

    # Resolve string levels
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger_inst.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger_inst.addHandler(console_handler)

    # File handler — write into RESULTS_DIR
    log_file_path = RESULTS_DIR / "pipeline.log"
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger_inst.addHandler(file_handler)

    # Prevent log propagation to root logger
    logger_inst.propagate = False

    return logger_inst


# Pre-configured global logger instance
logger = setup_logger()


# ============================================================================
# SECTION 2: TIMING & PROFILING
# ============================================================================

def timer(func: Callable) -> Callable:
    """
    Decorator that logs execution time of a function.

    Logs start, completion, and elapsed wall-clock time at INFO level.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"[TIMER] Starting {func.__module__}.{func.__name__}")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if elapsed < 60:
            logger.info(f"[TIMER] {func.__name__} completed in {elapsed:.2f}s")
        else:
            minutes, seconds = divmod(elapsed, 60)
            logger.info(
                f"[TIMER] {func.__name__} completed in "
                f"{int(minutes)}m {seconds:.1f}s"
            )
        return result
    return wrapper


def memory_usage(func: Callable) -> Callable:
    """
    Decorator that logs peak memory usage of a function.

    Uses tracemalloc to measure memory allocated during execution.
    Reports the peak memory in human-readable units (KB, MB, GB).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        try:
            result = func(*args, **kwargs)
        finally:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        # Format peak memory into readable units
        if peak < 1024:
            mem_str = f"{peak} B"
        elif peak < 1024 ** 2:
            mem_str = f"{peak / 1024:.1f} KB"
        elif peak < 1024 ** 3:
            mem_str = f"{peak / (1024 ** 2):.1f} MB"
        else:
            mem_str = f"{peak / (1024 ** 3):.2f} GB"

        logger.info(f"[MEMORY] {func.__name__} peak memory: {mem_str}")
        return result
    return wrapper


class ProgressTracker:
    """
    Context manager that tracks and reports progress of long operations.

    Usage
    -----
    >>> with ProgressTracker("Backtesting", total=100) as pt:
    ...     for i in range(100):
    ...         do_work(i)
    ...         pt.update(1)

    Parameters
    ----------
    description : str
        Human-readable description of the operation.
    total : int
        Total number of steps expected.
    log_every_pct : float
        Log progress every N percent (default 10%).
    """

    def __init__(
        self,
        description: str,
        total: int = 100,
        log_every_pct: float = 10.0,
    ):
        self.description = description
        self.total = total
        self.log_every_pct = log_every_pct
        self.current = 0
        self._last_logged_pct = 0.0
        self._start_time: Optional[float] = None

    def __enter__(self):
        self._start_time = time.perf_counter()
        logger.info(f"[PROGRESS] {self.description} — started (total={self.total})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._start_time
        if exc_type is not None:
            logger.warning(
                f"[PROGRESS] {self.description} — FAILED after {elapsed:.1f}s "
                f"({self.current}/{self.total} steps)"
            )
        else:
            logger.info(
                f"[PROGRESS] {self.description} — completed in {elapsed:.1f}s "
                f"({self.current}/{self.total} steps)"
            )
        return False  # Do not suppress exceptions

    def update(self, n: int = 1) -> None:
        """Advance the tracker by n steps and log if a threshold is crossed."""
        self.current += n
        if self.total <= 0:
            return
        pct = (self.current / self.total) * 100.0
        if pct - self._last_logged_pct >= self.log_every_pct or self.current >= self.total:
            elapsed = time.perf_counter() - self._start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            logger.info(
                f"[PROGRESS] {self.description}: {pct:.0f}% "
                f"({self.current}/{self.total}) "
                f"[{elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining]"
            )
            self._last_logged_pct = pct


# ============================================================================
# SECTION 3: DATA VALIDATION
# ============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    name: str = "DataFrame",
    min_rows: int = 10,
    max_null_pct: float = 0.5,
    check_inf: bool = True,
) -> bool:
    """
    Comprehensive validation of a DataFrame for data quality issues.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate.
    name : str
        Name for logging purposes.
    min_rows : int
        Minimum acceptable number of rows.
    max_null_pct : float
        Maximum acceptable fraction of nulls per column (0 to 1).
    check_inf : bool
        Whether to check for infinite values.

    Returns
    -------
    bool
        True if all checks pass, False otherwise.
    """
    issues = []

    if len(df) < min_rows:
        issues.append(f"Only {len(df)} rows (minimum required: {min_rows})")

    null_pct = df.isnull().mean()
    bad_cols = null_pct[null_pct > max_null_pct]
    if len(bad_cols) > 0:
        issues.append(
            f"{len(bad_cols)} columns with >{max_null_pct:.0%} nulls: "
            f"{list(bad_cols.index[:5])}"
        )

    if check_inf:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] > 0:
            inf_mask = np.isinf(numeric)
            inf_cols = numeric.columns[inf_mask.any()]
            if len(inf_cols) > 0:
                issues.append(f"Inf values found in columns: {list(inf_cols[:5])}")

    if hasattr(df.index, "is_monotonic_increasing"):
        if not df.index.is_monotonic_increasing:
            issues.append("Index is not monotonically increasing (unsorted dates?)")

    if df.index.duplicated().any():
        n_dup = df.index.duplicated().sum()
        issues.append(f"{n_dup} duplicate index values found")

    # Check for completely empty columns
    empty_cols = df.columns[df.isnull().all()]
    if len(empty_cols) > 0:
        issues.append(f"{len(empty_cols)} completely empty columns: {list(empty_cols[:5])}")

    if issues:
        for issue in issues:
            logger.warning(f"[{name}] {issue}")
        return False

    logger.info(
        f"[{name}] Validation passed — "
        f"{df.shape[0]:,} rows, {df.shape[1]} cols, "
        f"date range: {df.index.min()} to {df.index.max()}"
    )
    return True


def validate_weights(
    weights: np.ndarray,
    n_assets: int,
    tolerance: float = 1e-4,
) -> bool:
    """
    Validate portfolio weights for feasibility.

    Parameters
    ----------
    weights : np.ndarray
        Array of portfolio weights.
    n_assets : int
        Expected number of assets.
    tolerance : float
        Numerical tolerance for sum-to-one constraint.

    Returns
    -------
    bool
        True if weights are valid.

    Raises
    ------
    ValueError
        If weights fail validation with a descriptive message.
    """
    weights = np.asarray(weights, dtype=np.float64)

    if weights.ndim != 1:
        raise ValueError(f"Weights must be 1-D, got shape {weights.shape}")

    if len(weights) != n_assets:
        raise ValueError(
            f"Expected {n_assets} weights, got {len(weights)}"
        )

    if not np.all(np.isfinite(weights)):
        n_bad = np.sum(~np.isfinite(weights))
        raise ValueError(f"Weights contain {n_bad} non-finite values (NaN/Inf)")

    weight_sum = np.sum(weights)
    if abs(weight_sum - 1.0) > tolerance:
        raise ValueError(
            f"Weights sum to {weight_sum:.6f}, expected 1.0 "
            f"(tolerance={tolerance})"
        )

    if np.any(weights < -tolerance):
        min_w = np.min(weights)
        raise ValueError(
            f"Negative weight detected: min={min_w:.6f}. "
            f"Short selling not permitted (tolerance={tolerance})"
        )

    return True


def validate_covariance(cov_matrix: np.ndarray) -> bool:
    """
    Validate a covariance matrix: symmetric, finite, positive semi-definite.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Square covariance matrix.

    Returns
    -------
    bool
        True if all checks pass.

    Raises
    ------
    ValueError
        If the matrix fails validation.
    """
    cov = np.asarray(cov_matrix, dtype=np.float64)

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance must be square, got shape {cov.shape}")

    if not np.all(np.isfinite(cov)):
        n_bad = np.sum(~np.isfinite(cov))
        raise ValueError(f"Covariance contains {n_bad} non-finite values")

    # Symmetry check
    max_asym = np.max(np.abs(cov - cov.T))
    if max_asym > 1e-8:
        raise ValueError(
            f"Covariance is not symmetric. Max asymmetry: {max_asym:.2e}"
        )

    # Positive semi-definiteness via eigenvalues
    eigenvalues = eigvalsh(cov)
    min_eig = np.min(eigenvalues)
    if min_eig < -1e-8:
        raise ValueError(
            f"Covariance is not positive semi-definite. "
            f"Min eigenvalue: {min_eig:.2e}"
        )

    # Check diagonal is positive
    diag = np.diag(cov)
    if np.any(diag <= 0):
        raise ValueError("Covariance has non-positive diagonal entries (variances)")

    return True


def validate_returns(returns: pd.DataFrame) -> bool:
    """
    Validate a returns DataFrame for suspicious values.

    Checks for:
    - All-zero columns (dead assets)
    - Extreme daily returns (>100% gain or >50% loss)
    - Excessive NaNs
    - Constant columns (zero variance)

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns.

    Returns
    -------
    bool
        True if returns pass all checks.
    """
    issues = []

    # All-zero columns
    zero_cols = returns.columns[(returns == 0).all()]
    if len(zero_cols) > 0:
        issues.append(f"All-zero columns (dead assets): {list(zero_cols)}")

    # Constant columns (zero variance)
    near_zero_std = returns.columns[returns.std() < 1e-12]
    constant_cols = [c for c in near_zero_std if c not in zero_cols]
    if len(constant_cols) > 0:
        issues.append(f"Near-constant columns: {list(constant_cols)}")

    # Extreme returns
    extreme_high = (returns > 1.0).any()  # >100% daily return
    extreme_low = (returns < -0.5).any()  # >50% daily loss
    suspect_high = list(extreme_high[extreme_high].index)
    suspect_low = list(extreme_low[extreme_low].index)
    if suspect_high:
        issues.append(f"Columns with >100% daily returns: {suspect_high}")
    if suspect_low:
        issues.append(f"Columns with >50% daily losses: {suspect_low}")

    # NaN check
    nan_pct = returns.isnull().mean()
    high_nan = nan_pct[nan_pct > 0.1]
    if len(high_nan) > 0:
        issues.append(
            f"{len(high_nan)} columns with >10% NaNs: "
            f"{list(high_nan.index[:5])}"
        )

    if issues:
        for issue in issues:
            logger.warning(f"[Returns Validation] {issue}")
        return False

    logger.info(
        f"[Returns Validation] Passed — {returns.shape[1]} assets, "
        f"{returns.shape[0]:,} observations"
    )
    return True


# ============================================================================
# SECTION 4: FINANCIAL METRICS (comprehensive)
# ============================================================================

def annualize_return(
    daily_returns: Union[pd.Series, np.ndarray],
    periods: int = 252,
) -> float:
    """Annualize a series of daily returns using geometric compounding."""
    r = np.asarray(daily_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    total = np.prod(1.0 + r)
    n_days = len(r)
    return float(total ** (periods / n_days) - 1.0)


def annualize_volatility(
    daily_returns: Union[pd.Series, np.ndarray],
    periods: int = 252,
) -> float:
    """Annualize daily volatility by sqrt(periods) scaling."""
    r = np.asarray(daily_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(periods))


def sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free: float = 0.0,
    periods: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio.

    Parameters
    ----------
    returns : array-like
        Daily returns.
    risk_free : float
        Annualized risk-free rate.
    periods : int
        Trading days per year.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    excess = r - risk_free / periods
    mean_excess = np.mean(excess)
    std_excess = np.std(excess, ddof=1)
    if std_excess < 1e-12:
        return 0.0
    return float(mean_excess / std_excess * np.sqrt(periods))


def sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free: float = 0.0,
    periods: int = 252,
) -> float:
    """
    Compute annualized Sortino ratio (return / downside deviation).

    Only negative excess returns contribute to the downside deviation.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    daily_rf = risk_free / periods
    excess = r - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    downside_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods)
    if downside_dev < 1e-12:
        return 0.0
    ann_excess_return = np.mean(excess) * periods
    return float(ann_excess_return / downside_dev)


def information_ratio(
    portfolio_returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute information ratio: annualized excess return / tracking error.
    """
    p = np.asarray(portfolio_returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)
    min_len = min(len(p), len(b))
    excess = p[:min_len] - b[:min_len]
    excess = excess[np.isfinite(excess)]
    if len(excess) < 2:
        return 0.0
    te = np.std(excess, ddof=1) * np.sqrt(252)
    if te < 1e-12:
        return 0.0
    ann_excess = np.mean(excess) * 252
    return float(ann_excess / te)


def calmar_ratio(
    returns: Union[pd.Series, np.ndarray],
    periods: int = 252,
) -> float:
    """
    Compute Calmar ratio: annualized return / absolute max drawdown.
    """
    ann_ret = annualize_return(returns, periods)
    mdd = max_drawdown(returns)
    if abs(mdd) < 1e-12:
        return 0.0
    return float(ann_ret / abs(mdd))


def omega_ratio(
    returns: Union[pd.Series, np.ndarray],
    threshold: float = 0.0,
    periods: int = 252,
) -> float:
    """
    Compute Omega ratio: probability-weighted gain/loss ratio above threshold.

    Omega = E[max(R - threshold, 0)] / E[max(threshold - R, 0)]
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    daily_threshold = threshold / periods
    gains = np.sum(np.maximum(r - daily_threshold, 0.0))
    losses = np.sum(np.maximum(daily_threshold - r, 0.0))
    if losses < 1e-12:
        return np.inf if gains > 0 else 0.0
    return float(gains / losses)


def max_drawdown(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Compute maximum peak-to-trough decline.

    Returns a negative number (e.g. -0.25 means a 25% drawdown).
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    cumulative = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1.0
    return float(np.min(drawdowns))


def max_drawdown_duration(returns: Union[pd.Series, np.ndarray]) -> int:
    """
    Compute the longest drawdown duration in trading days.

    This is the number of consecutive days spent below a previous peak.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0
    cumulative = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(cumulative)
    in_drawdown = cumulative < running_max

    if not np.any(in_drawdown):
        return 0

    # Count consecutive True values (drawdown streaks)
    max_streak = 0
    current_streak = 0
    for dd in in_drawdown:
        if dd:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0
    return max_streak


def value_at_risk(
    returns: Union[pd.Series, np.ndarray],
    confidence: float = 0.95,
) -> float:
    """
    Compute historical Value at Risk at a given confidence level.

    Returns a negative number representing the loss threshold.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    return float(np.percentile(r, (1.0 - confidence) * 100.0))


def conditional_var(
    returns: Union[pd.Series, np.ndarray],
    confidence: float = 0.95,
) -> float:
    """
    Compute Conditional VaR (Expected Shortfall / CVaR).

    The average loss in the worst (1 - confidence) fraction of days.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    var_threshold = value_at_risk(r, confidence)
    tail = r[r <= var_threshold]
    if len(tail) == 0:
        return var_threshold
    return float(np.mean(tail))


def parametric_var(
    returns: Union[pd.Series, np.ndarray],
    confidence: float = 0.95,
) -> float:
    """
    Compute parametric (Gaussian) Value at Risk.

    Assumes returns are normally distributed.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    mu = np.mean(r)
    sigma = np.std(r, ddof=1)
    z = sp_stats.norm.ppf(1.0 - confidence)
    return float(mu + z * sigma)


def cornish_fisher_var(
    returns: Union[pd.Series, np.ndarray],
    confidence: float = 0.95,
) -> float:
    """
    Compute Cornish-Fisher VaR, adjusting for skewness and kurtosis.

    Uses the Cornish-Fisher expansion to correct the Gaussian quantile
    for non-normal higher moments.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 4:
        return parametric_var(r, confidence)

    mu = np.mean(r)
    sigma = np.std(r, ddof=1)
    s = sp_stats.skew(r)
    k = sp_stats.kurtosis(r)  # excess kurtosis

    z = sp_stats.norm.ppf(1.0 - confidence)

    # Cornish-Fisher expansion
    z_cf = (
        z
        + (z ** 2 - 1) * s / 6.0
        + (z ** 3 - 3 * z) * k / 24.0
        - (2 * z ** 3 - 5 * z) * (s ** 2) / 36.0
    )

    return float(mu + z_cf * sigma)


def tail_ratio(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Compute the tail ratio: abs(95th percentile) / abs(5th percentile).

    A ratio > 1 indicates a fatter right tail (more upside).
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    p95 = np.percentile(r, 95)
    p5 = np.percentile(r, 5)
    if abs(p5) < 1e-12:
        return 0.0
    return float(abs(p95) / abs(p5))


def stability_of_return(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Compute stability of returns as R-squared of cumulative log returns vs time.

    A value close to 1 indicates highly stable, trend-like growth.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return 0.0
    # Guard against log of non-positive values
    cum_log = np.cumsum(np.log1p(np.clip(r, -0.9999, None)))
    t = np.arange(len(cum_log))

    # Linear regression: cum_log = a * t + b
    slope, intercept, r_value, _, _ = sp_stats.linregress(t, cum_log)
    return float(r_value ** 2)


def capture_ratio(
    portfolio_returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
) -> dict:
    """
    Compute up-capture and down-capture ratios.

    Parameters
    ----------
    portfolio_returns, benchmark_returns : array-like
        Aligned daily returns.

    Returns
    -------
    dict
        Keys: 'up_capture', 'down_capture', 'capture_ratio'.
    """
    p = np.asarray(portfolio_returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)
    min_len = min(len(p), len(b))
    p, b = p[:min_len], b[:min_len]

    mask = np.isfinite(p) & np.isfinite(b)
    p, b = p[mask], b[mask]

    up_mask = b > 0
    down_mask = b < 0

    # Up capture
    if np.sum(up_mask) > 0:
        up_p = np.mean(p[up_mask])
        up_b = np.mean(b[up_mask])
        up_capture = up_p / up_b if abs(up_b) > 1e-12 else 0.0
    else:
        up_capture = 0.0

    # Down capture
    if np.sum(down_mask) > 0:
        down_p = np.mean(p[down_mask])
        down_b = np.mean(b[down_mask])
        down_capture = down_p / down_b if abs(down_b) > 1e-12 else 0.0
    else:
        down_capture = 0.0

    overall = up_capture / down_capture if abs(down_capture) > 1e-12 else 0.0

    return {
        "up_capture": float(up_capture),
        "down_capture": float(down_capture),
        "capture_ratio": float(overall),
    }


def tracking_error(
    portfolio_returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
    periods: int = 252,
) -> float:
    """
    Compute annualized tracking error (standard deviation of excess returns).
    """
    p = np.asarray(portfolio_returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)
    min_len = min(len(p), len(b))
    excess = p[:min_len] - b[:min_len]
    excess = excess[np.isfinite(excess)]
    if len(excess) < 2:
        return 0.0
    return float(np.std(excess, ddof=1) * np.sqrt(periods))


def beta(
    portfolio_returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute CAPM beta of portfolio relative to benchmark.

    beta = Cov(Rp, Rb) / Var(Rb)
    """
    p = np.asarray(portfolio_returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)
    min_len = min(len(p), len(b))
    p, b = p[:min_len], b[:min_len]
    mask = np.isfinite(p) & np.isfinite(b)
    p, b = p[mask], b[mask]
    if len(p) < 2:
        return 0.0
    var_b = np.var(b, ddof=1)
    if var_b < 1e-16:
        return 0.0
    cov_pb = np.cov(p, b, ddof=1)[0, 1]
    return float(cov_pb / var_b)


def alpha_jensen(
    portfolio_returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
    risk_free: float = 0.0,
    periods: int = 252,
) -> float:
    """
    Compute Jensen's alpha: annualized excess return beyond CAPM prediction.

    alpha = Rp - [Rf + beta * (Rb - Rf)]
    """
    p = np.asarray(portfolio_returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)
    min_len = min(len(p), len(b))
    p, b = p[:min_len], b[:min_len]
    mask = np.isfinite(p) & np.isfinite(b)
    p, b = p[mask], b[mask]

    if len(p) < 2:
        return 0.0

    daily_rf = risk_free / periods
    beta_val = beta(p, b)

    ann_port = annualize_return(p, periods)
    ann_bench = annualize_return(b, periods)

    return float(ann_port - (risk_free + beta_val * (ann_bench - risk_free)))


def treynor_ratio(
    portfolio_returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
    risk_free: float = 0.0,
    periods: int = 252,
) -> float:
    """
    Compute Treynor ratio: excess return per unit of systematic risk (beta).

    Treynor = (Rp - Rf) / beta
    """
    p = np.asarray(portfolio_returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)

    beta_val = beta(p, b)
    if abs(beta_val) < 1e-12:
        return 0.0

    ann_port = annualize_return(p, periods)
    return float((ann_port - risk_free) / beta_val)


# ============================================================================
# SECTION 5: PORTFOLIO ANALYTICS
# ============================================================================

def portfolio_return(
    weights: np.ndarray,
    returns: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute expected portfolio return as the weighted sum of asset returns.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (length n_assets).
    returns : array-like
        Expected returns per asset (length n_assets).

    Returns
    -------
    float
        Expected portfolio return.
    """
    w = np.asarray(weights, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    return float(w @ r)


def portfolio_volatility(
    weights: np.ndarray,
    cov: np.ndarray,
) -> float:
    """
    Compute portfolio standard deviation: sqrt(w' * Sigma * w).
    """
    w = np.asarray(weights, dtype=np.float64)
    c = np.asarray(cov, dtype=np.float64)
    variance = w @ c @ w
    if variance < 0:
        logger.warning(
            f"Negative portfolio variance ({variance:.2e}), "
            "likely numerical issue. Returning 0."
        )
        return 0.0
    return float(np.sqrt(variance))


def marginal_risk_contribution(
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """
    Compute marginal risk contribution of each asset.

    MRC_i = (Sigma * w)_i / sigma_p

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    cov : np.ndarray
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Marginal risk contribution per asset.
    """
    w = np.asarray(weights, dtype=np.float64)
    c = np.asarray(cov, dtype=np.float64)
    sigma_p = portfolio_volatility(w, c)
    if sigma_p < 1e-12:
        return np.zeros_like(w)
    return (c @ w) / sigma_p


def component_risk_contribution(
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """
    Compute component risk contribution (CRC) of each asset.

    CRC_i = w_i * MRC_i
    The sum of all CRCs equals portfolio volatility.
    """
    w = np.asarray(weights, dtype=np.float64)
    mrc = marginal_risk_contribution(w, cov)
    return w * mrc


def diversification_ratio(
    weights: np.ndarray,
    cov: np.ndarray,
) -> float:
    """
    Compute diversification ratio: weighted average volatility / portfolio vol.

    DR = (w' * sigma) / sigma_p

    A DR > 1 indicates diversification benefit. Higher is better.
    """
    w = np.asarray(weights, dtype=np.float64)
    c = np.asarray(cov, dtype=np.float64)
    asset_vols = np.sqrt(np.diag(c))
    weighted_avg_vol = w @ asset_vols
    port_vol = portfolio_volatility(w, c)
    if port_vol < 1e-12:
        return 0.0
    return float(weighted_avg_vol / port_vol)


def effective_number_of_bets(
    weights: np.ndarray,
    cov: np.ndarray,
) -> float:
    """
    Compute the Effective Number of Bets (ENB) using risk contributions.

    ENB = exp(-sum(p_i * log(p_i))) where p_i are the fractional risk
    contributions. This is the exponential of Shannon entropy of risk
    contributions.

    A higher ENB indicates a more diversified portfolio from a risk
    perspective.
    """
    w = np.asarray(weights, dtype=np.float64)
    crc = component_risk_contribution(w, cov)
    total_risk = np.sum(crc)
    if total_risk < 1e-12:
        return 0.0
    # Fractional risk contributions
    p = crc / total_risk
    # Remove zero or negative entries for log
    p = p[p > 1e-12]
    if len(p) == 0:
        return 0.0
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def herfindahl_index(weights: np.ndarray) -> float:
    """
    Compute the Herfindahl-Hirschman Index of portfolio concentration.

    HHI = sum(w_i^2)

    Ranges from 1/n (equal weight) to 1 (fully concentrated).
    """
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sum(w ** 2))


# ============================================================================
# SECTION 6: STATISTICAL TESTS
# ============================================================================

def jarque_bera_test(
    returns: Union[pd.Series, np.ndarray],
) -> dict:
    """
    Jarque-Bera test for normality of returns.

    Tests whether the sample skewness and kurtosis match a normal
    distribution. Rejection (p < 0.05) implies non-normality.

    Returns
    -------
    dict
        Keys: 'statistic', 'p_value', 'skewness', 'kurtosis', 'is_normal'.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    stat, p_value = sp_stats.jarque_bera(r)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "skewness": float(sp_stats.skew(r)),
        "kurtosis": float(sp_stats.kurtosis(r)),
        "is_normal": bool(p_value > 0.05),
    }


def ljung_box_test(
    returns: Union[pd.Series, np.ndarray],
    lags: int = 10,
) -> dict:
    """
    Ljung-Box test for autocorrelation in returns.

    Tests whether the first `lags` autocorrelations are jointly zero.
    Rejection indicates serial dependence.

    Returns
    -------
    dict
        Keys: 'statistic', 'p_value', 'lags', 'has_autocorrelation'.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < lags + 1:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "lags": lags,
            "has_autocorrelation": False,
        }

    # Compute autocorrelations manually
    r_centered = r - np.mean(r)
    var_r = np.sum(r_centered ** 2) / n
    if var_r < 1e-16:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "lags": lags,
            "has_autocorrelation": False,
        }

    acf_values = []
    for k in range(1, lags + 1):
        c_k = np.sum(r_centered[k:] * r_centered[:-k]) / n
        acf_values.append(c_k / var_r)

    # Ljung-Box Q statistic
    q_stat = 0.0
    for k, rho_k in enumerate(acf_values, start=1):
        q_stat += (rho_k ** 2) / (n - k)
    q_stat *= n * (n + 2)

    # Under H0, Q ~ chi2(lags)
    p_value = 1.0 - sp_stats.chi2.cdf(q_stat, df=lags)

    return {
        "statistic": float(q_stat),
        "p_value": float(p_value),
        "lags": lags,
        "has_autocorrelation": bool(p_value < 0.05),
    }


def runs_test(returns: Union[pd.Series, np.ndarray]) -> dict:
    """
    Wald-Wolfowitz runs test for randomness of returns.

    Converts returns to a binary sequence (positive/negative) and tests
    whether the number of runs is consistent with randomness.

    Returns
    -------
    dict
        Keys: 'statistic', 'p_value', 'n_runs', 'is_random'.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    # Remove zeros for a clean binary split
    r = r[r != 0]
    if len(r) < 10:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n_runs": 0,
            "is_random": True,
        }

    binary = (r > 0).astype(int)
    n_pos = np.sum(binary)
    n_neg = len(binary) - n_pos

    if n_pos == 0 or n_neg == 0:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n_runs": 1,
            "is_random": False,
        }

    # Count runs
    n_runs = 1 + np.sum(binary[1:] != binary[:-1])

    # Expected number of runs and variance under H0
    n = len(binary)
    expected_runs = 1.0 + (2.0 * n_pos * n_neg) / n
    var_runs = (
        (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n))
        / (n ** 2 * (n - 1.0))
    )

    if var_runs < 1e-12:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_runs": int(n_runs),
            "is_random": True,
        }

    z_stat = (n_runs - expected_runs) / np.sqrt(var_runs)
    p_value = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z_stat)))

    return {
        "statistic": float(z_stat),
        "p_value": float(p_value),
        "n_runs": int(n_runs),
        "is_random": bool(p_value > 0.05),
    }


def kolmogorov_smirnov_test(
    returns: Union[pd.Series, np.ndarray],
) -> dict:
    """
    Kolmogorov-Smirnov test against a normal distribution.

    Tests whether the empirical distribution of returns differs from a
    Gaussian with the same mean and standard deviation.

    Returns
    -------
    dict
        Keys: 'statistic', 'p_value', 'is_normal'.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "is_normal": False,
        }
    mu = np.mean(r)
    sigma = np.std(r, ddof=1)
    stat, p_value = sp_stats.kstest(r, "norm", args=(mu, sigma))
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_normal": bool(p_value > 0.05),
    }


def bootstrap_confidence_interval(
    statistic_fn: Callable,
    data: Union[pd.Series, np.ndarray],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Compute a bootstrap confidence interval for any statistic.

    Parameters
    ----------
    statistic_fn : callable
        Function that takes an array and returns a scalar statistic.
    data : array-like
        Input data to resample.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g. 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: 'point_estimate', 'ci_lower', 'ci_upper', 'std_error',
              'n_bootstrap', 'ci_level'.
    """
    d = np.asarray(data, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return {
            "point_estimate": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "std_error": np.nan,
            "n_bootstrap": n_bootstrap,
            "ci_level": ci,
        }

    rng = np.random.RandomState(seed)
    point_estimate = statistic_fn(d)

    bootstrap_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(d, size=n, replace=True)
        bootstrap_stats[i] = statistic_fn(sample)

    alpha = (1.0 - ci) / 2.0
    ci_lower = np.percentile(bootstrap_stats, alpha * 100)
    ci_upper = np.percentile(bootstrap_stats, (1.0 - alpha) * 100)
    std_error = np.std(bootstrap_stats, ddof=1)

    return {
        "point_estimate": float(point_estimate),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "std_error": float(std_error),
        "n_bootstrap": n_bootstrap,
        "ci_level": ci,
    }


# ============================================================================
# SECTION 7: EXPORT UTILITIES
# ============================================================================

def results_to_latex(
    df: pd.DataFrame,
    filename: str,
    caption: str = "",
    label: str = "",
    float_format: str = "%.4f",
) -> Path:
    """
    Export a DataFrame to a LaTeX table file in RESULTS_DIR.

    Parameters
    ----------
    df : pd.DataFrame
        Data to export.
    filename : str
        Filename (without extension).
    caption : str
        LaTeX table caption.
    label : str
        LaTeX table label for cross-referencing.
    float_format : str
        C-style format string for floats.

    Returns
    -------
    Path
        Path to the created file.
    """
    path = RESULTS_DIR / f"{filename}.tex"
    latex_str = df.to_latex(
        float_format=float_format,
        caption=caption,
        label=label,
        bold_rows=True,
        escape=True,
    )
    path.write_text(latex_str, encoding="utf-8")
    logger.info(f"Saved LaTeX table to {path}")
    return path


def results_to_excel(
    sheets: dict,
    filename: str = "thesis_results",
) -> Path:
    """
    Export multiple DataFrames to a multi-sheet Excel workbook.

    Parameters
    ----------
    sheets : dict[str, pd.DataFrame]
        Mapping of sheet names to DataFrames.
    filename : str
        Filename (without extension).

    Returns
    -------
    Path
        Path to the created file.
    """
    path = RESULTS_DIR / f"{filename}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            # Excel sheet names are max 31 characters
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name)
    logger.info(f"Saved Excel workbook with {len(sheets)} sheets to {path}")
    return path


def export_summary_report(
    metrics_dict: dict,
    filename: str = "summary_report",
) -> Path:
    """
    Export a comprehensive text-based summary report of portfolio metrics.

    Parameters
    ----------
    metrics_dict : dict
        Mapping of strategy names to dicts of metric values.
        Example: {"Max Sharpe": {"sharpe_ratio": 1.2, ...}, ...}
    filename : str
        Filename (without extension).

    Returns
    -------
    Path
        Path to the created file.
    """
    path = RESULTS_DIR / f"{filename}.txt"
    lines = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("PORTFOLIO OPTIMIZATION — SUMMARY REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append("")

    for strategy_name, metrics in metrics_dict.items():
        lines.append(f"  Strategy: {strategy_name}")
        lines.append("-" * 48)
        formatted = format_metrics_for_display(metrics)
        for metric_name, display_val in formatted.items():
            lines.append(f"    {metric_name:<30s} {display_val:>15s}")
        lines.append("")

    lines.append(sep)
    lines.append("END OF REPORT")
    lines.append(sep)

    report_text = "\n".join(lines)
    path.write_text(report_text, encoding="utf-8")
    logger.info(f"Saved summary report to {path}")
    return path


def format_metrics_for_display(metrics: dict) -> dict:
    """
    Format a metrics dictionary with proper percentage signs and decimal places.

    Conventions:
    - Returns, volatility, drawdowns, alphas: displayed as percentages (x100, %)
    - Ratios (Sharpe, Sortino, etc.): displayed with 3 decimal places
    - Durations: displayed as integers with 'days' suffix
    - Counts: displayed as integers

    Parameters
    ----------
    metrics : dict
        Raw metric name -> numeric value mapping.

    Returns
    -------
    dict
        Metric name -> formatted string mapping.
    """
    pct_keywords = [
        "return", "volatility", "drawdown", "var", "cvar", "alpha",
        "tracking_error", "capture",
    ]
    ratio_keywords = [
        "sharpe", "sortino", "calmar", "treynor", "information",
        "omega", "tail_ratio", "diversification", "beta",
    ]
    duration_keywords = ["duration", "days"]
    count_keywords = ["n_", "count", "number", "enb", "herfindahl"]

    formatted = {}
    for key, value in metrics.items():
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            formatted[key] = "N/A"
            continue

        key_lower = key.lower()

        if any(kw in key_lower for kw in duration_keywords):
            formatted[key] = f"{int(value)} days"
        elif any(kw in key_lower for kw in pct_keywords):
            formatted[key] = f"{value * 100:.2f}%"
        elif any(kw in key_lower for kw in ratio_keywords):
            formatted[key] = f"{value:.3f}"
        elif any(kw in key_lower for kw in count_keywords):
            if isinstance(value, float) and value == int(value):
                formatted[key] = f"{int(value)}"
            else:
                formatted[key] = f"{value:.3f}"
        else:
            # Default: 4 decimal places
            if isinstance(value, (int, np.integer)):
                formatted[key] = str(int(value))
            else:
                formatted[key] = f"{value:.4f}"

    return formatted
