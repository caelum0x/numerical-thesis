"""
Data cleaning, feature engineering, and stationarity testing for
macro-based multi-asset portfolio optimization.

This module implements a comprehensive preprocessing pipeline that transforms
raw price and macroeconomic data into a feature matrix suitable for machine
learning models. The pipeline covers:

    1. Return features (multi-horizon, log, excess)
    2. Volatility features (rolling, realized, Garman-Klass, EWMA)
    3. Momentum and trend indicators (RSI, MACD, Bollinger, crossovers)
    4. Cross-asset features (rolling beta, correlation, dispersion, relative strength)
    5. Macro feature engineering (lags, changes, regimes, surprises)
    6. Dimensionality reduction (PCA, mutual-information feature selection)
    7. Statistical tests (ADF, KPSS, Granger causality, cointegration, VIF)
    8. Data cleaning (winsorization, outlier detection, missing data handling)
    9. Master builders for full feature matrix construction

References:
    - Garman, M. B. & Klass, M. J. (1980). On the estimation of security
      price volatilities from historical data. Journal of Business, 53(1), 67-78.
    - Bollerslev, T. (1986). Generalized autoregressive conditional
      heteroskedasticity. Journal of Econometrics, 31(3), 307-327.
    - Engle, R. F. & Granger, C. W. J. (1987). Co-integration and error
      correction: Representation, estimation, and testing. Econometrica, 55(2).

Author : Arhan Subasi
Project: Industrial Engineering Thesis -- Macro-Based Portfolio Optimization
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, coint
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import (
    RETURN_HORIZONS,
    VOLATILITY_WINDOWS,
    MOMENTUM_WINDOWS,
    MACRO_LAGS,
    PROCESSED_DIR,
    RSI_WINDOW,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    BOLLINGER_WINDOW,
    BOLLINGER_STD,
    ROLLING_BETA_WINDOW,
    ROLLING_CORR_WINDOW,
    DISPERSION_WINDOW,
    PCA_VARIANCE_THRESHOLD,
    MAX_PCA_COMPONENTS,
    VIF_THRESHOLD,
    PREDICTION_HORIZON,
    TARGET_COLUMN_TEMPLATE,
    TICKER_LIST,
    REGIME_VIX_THRESHOLD,
    REGIME_VIX_EXTREME,
    EWMA_HALFLIFE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants used internally
# ---------------------------------------------------------------------------
_ANNUALIZATION_FACTOR = 252
_MIN_OBS_FOR_STAT_TEST = 20
_DEFAULT_GRANGER_MAX_LAG = 5


# ============================================================================
# SECTION 1: Return Features
# ============================================================================


def compute_returns(
    prices: pd.DataFrame,
    horizons: list[int] = RETURN_HORIZONS,
) -> pd.DataFrame:
    """
    Compute simple percentage returns at multiple horizons.

    For each horizon *h* and each asset column in *prices*, computes
    ``(P_t - P_{t-h}) / P_{t-h}`` and labels the resulting column as
    ``{ticker}_ret_{h}d``.

    Parameters
    ----------
    prices : pd.DataFrame
        Price panel with ``DatetimeIndex`` and one column per asset.
    horizons : list[int]
        Look-back periods in trading days.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``{ticker}_ret_{h}d`` for every combination
        of ticker and horizon.
    """
    frames: list[pd.DataFrame] = []
    for h in horizons:
        ret = prices.pct_change(h)
        ret.columns = [f"{col}_ret_{h}d" for col in ret.columns]
        frames.append(ret)
    result = pd.concat(frames, axis=1)
    logger.debug("compute_returns: generated %d columns for %d horizons",
                 result.shape[1], len(horizons))
    return result


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 1-day log returns: ``ln(P_t / P_{t-1})``.

    Log returns are additive over time and more appropriate for
    statistical modelling assuming normally-distributed innovations.

    Parameters
    ----------
    prices : pd.DataFrame
        Price panel.

    Returns
    -------
    pd.DataFrame
        Log returns with columns named ``{ticker}_logret``.
    """
    log_ret = np.log(prices / prices.shift(1))
    log_ret.columns = [f"{col}_logret" for col in prices.columns]
    logger.debug("compute_log_returns: %d columns", log_ret.shape[1])
    return log_ret


def compute_excess_returns(
    prices: pd.DataFrame,
    risk_free_series: pd.Series,
) -> pd.DataFrame:
    """
    Compute daily excess returns over a risk-free rate.

    The risk-free series is assumed to be in annualized percentage terms
    (e.g., 4.5 for 4.5 %).  It is converted to a daily rate before
    subtraction.

    Parameters
    ----------
    prices : pd.DataFrame
        Price panel.
    risk_free_series : pd.Series
        Annualized risk-free rate (in percentage points), aligned on the
        same date index.

    Returns
    -------
    pd.DataFrame
        Excess returns with columns named ``{ticker}_exret``.
    """
    daily_ret = prices.pct_change()
    # Convert annualized pct rate to daily decimal
    rf_daily = (risk_free_series / 100.0) / _ANNUALIZATION_FACTOR
    rf_daily = rf_daily.reindex(daily_ret.index).ffill().bfill()

    excess = daily_ret.subtract(rf_daily, axis=0)
    excess.columns = [f"{col}_exret" for col in prices.columns]
    logger.debug("compute_excess_returns: %d columns", excess.shape[1])
    return excess


# ============================================================================
# SECTION 2: Volatility Features
# ============================================================================


def compute_volatility(
    prices: pd.DataFrame,
    windows: list[int] = VOLATILITY_WINDOWS,
) -> pd.DataFrame:
    """
    Compute rolling annualized volatility for multiple window sizes.

    Volatility is the rolling standard deviation of daily returns,
    annualized by multiplying with ``sqrt(252)``.

    Parameters
    ----------
    prices : pd.DataFrame
        Price panel.
    windows : list[int]
        Rolling-window lengths in trading days.

    Returns
    -------
    pd.DataFrame
        Columns named ``{ticker}_vol_{w}d``.
    """
    daily_ret = prices.pct_change()
    frames: list[pd.DataFrame] = []
    for w in windows:
        vol = daily_ret.rolling(w, min_periods=max(w // 2, 1)).std() * np.sqrt(
            _ANNUALIZATION_FACTOR
        )
        vol.columns = [f"{col}_vol_{w}d" for col in vol.columns]
        frames.append(vol)
    result = pd.concat(frames, axis=1)
    logger.debug("compute_volatility: %d columns for %d windows",
                 result.shape[1], len(windows))
    return result


def compute_realized_variance(
    prices: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """
    Compute rolling realized variance (sum of squared returns).

    Realized variance is a non-parametric estimator of integrated variance
    commonly used in the financial econometrics literature (Andersen &
    Bollerslev, 1998).

    Parameters
    ----------
    prices : pd.DataFrame
        Price panel.
    window : int
        Rolling window in trading days.

    Returns
    -------
    pd.DataFrame
        Columns named ``{ticker}_rvar_{window}d``.
    """
    daily_ret = prices.pct_change()
    sq_ret = daily_ret ** 2
    rv = sq_ret.rolling(window, min_periods=max(window // 2, 1)).sum()
    rv = rv * (_ANNUALIZATION_FACTOR / window)  # annualize
    rv.columns = [f"{col}_rvar_{window}d" for col in prices.columns]
    logger.debug("compute_realized_variance: %d columns, window=%d",
                 rv.shape[1], window)
    return rv


def compute_garman_klass_vol(
    high: Optional[pd.DataFrame],
    low: Optional[pd.DataFrame],
    open_: Optional[pd.DataFrame],
    close: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """
    Compute Garman-Klass volatility estimator.

    The Garman-Klass estimator uses OHLC data to provide a more
    efficient estimate of volatility than the close-to-close estimator.
    If high/low/open data are not available, falls back to a close-to-close
    rolling volatility estimator.

    GK variance per day:
        ``0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2``

    Parameters
    ----------
    high, low, open_ : pd.DataFrame or None
        OHLC price panels.  If any is None, falls back to close-to-close.
    close : pd.DataFrame
        Close prices (always required).
    window : int
        Rolling window for averaging.

    Returns
    -------
    pd.DataFrame
        Columns named ``{ticker}_gkvol_{window}d``.
    """
    if high is not None and low is not None and open_ is not None:
        # Garman-Klass estimator
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        gk_var = 0.5 * log_hl ** 2 - (2.0 * np.log(2) - 1.0) * log_co ** 2
        # Rolling mean and annualize
        gk_vol = np.sqrt(
            gk_var.rolling(window, min_periods=max(window // 2, 1)).mean()
            * _ANNUALIZATION_FACTOR
        )
        gk_vol.columns = [f"{col}_gkvol_{window}d" for col in close.columns]
        logger.info("compute_garman_klass_vol: using full OHLC estimator")
    else:
        # Fallback: close-to-close volatility
        logger.warning(
            "compute_garman_klass_vol: OHLC data unavailable, "
            "falling back to close-to-close volatility"
        )
        daily_ret = close.pct_change()
        gk_vol = (
            daily_ret.rolling(window, min_periods=max(window // 2, 1)).std()
            * np.sqrt(_ANNUALIZATION_FACTOR)
        )
        gk_vol.columns = [f"{col}_gkvol_{window}d" for col in close.columns]
    return gk_vol


def compute_ewma_volatility(
    prices: pd.DataFrame,
    halflife: int = EWMA_HALFLIFE,
) -> pd.DataFrame:
    """
    Compute exponentially-weighted moving-average volatility.

    Uses pandas ``ewm`` with the specified half-life to place more
    weight on recent observations, following the RiskMetrics approach.

    Parameters
    ----------
    prices : pd.DataFrame
        Price panel.
    halflife : int
        EWMA half-life in trading days.

    Returns
    -------
    pd.DataFrame
        Columns named ``{ticker}_ewmavol_{halflife}d``.
    """
    daily_ret = prices.pct_change()
    ewma_var = daily_ret.ewm(halflife=halflife, min_periods=max(halflife, 1)).var()
    ewma_vol = np.sqrt(ewma_var * _ANNUALIZATION_FACTOR)
    ewma_vol.columns = [f"{col}_ewmavol_{halflife}d" for col in prices.columns]
    logger.debug("compute_ewma_volatility: halflife=%d, %d columns",
                 halflife, ewma_vol.shape[1])
    return ewma_vol


# ============================================================================
# SECTION 3: Momentum & Trend Features
# ============================================================================


def compute_momentum(
    prices: pd.DataFrame,
    windows: list[int] = MOMENTUM_WINDOWS,
) -> pd.DataFrame:
    """
    Compute price momentum (cumulative return over rolling windows).

    Momentum is defined as the percentage change over ``w`` trading days,
    i.e. ``(P_t - P_{t-w}) / P_{t-w}``.

    Parameters
    ----------
    prices : pd.DataFrame
        Price panel.
    windows : list[int]
        Look-back periods.

    Returns
    -------
    pd.DataFrame
        Columns named ``{ticker}_mom_{w}d``.
    """
    frames: list[pd.DataFrame] = []
    for w in windows:
        mom = prices.pct_change(w)
        mom.columns = [f"{col}_mom_{w}d" for col in mom.columns]
        frames.append(mom)
    result = pd.concat(frames, axis=1)
    logger.debug("compute_momentum: %d columns for %d windows",
                 result.shape[1], len(windows))
    return result


def compute_rsi(
    prices: pd.DataFrame,
    window: int = RSI_WINDOW,
) -> pd.DataFrame:
    """
    Compute the Relative Strength Index (RSI) for each asset.

    RSI = 100 - 100 / (1 + RS)   where RS = avg_gain / avg_loss

    Uses the exponential moving average (Wilder smoothing) variant,
    consistent with the original J. Welles Wilder (1978) definition.

    Parameters
    ----------
    prices : pd.DataFrame
        Price panel.
    window : int
        Look-back window for RSI.

    Returns
    -------
    pd.DataFrame
        Columns named ``{ticker}_rsi_{window}``.
    """
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    # Wilder smoothing (equivalent to EMA with alpha = 1/window)
    avg_gain = gains.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.columns = [f"{col}_rsi_{window}" for col in prices.columns]
    logger.debug("compute_rsi: window=%d, %d columns", window, rsi.shape[1])
    return rsi


def compute_macd(
    prices: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """
    Compute Moving Average Convergence/Divergence (MACD).

    Returns three sets of columns per asset:
        - MACD line    (fast EMA - slow EMA)
        - Signal line  (EMA of MACD line)
        - Histogram    (MACD - Signal)

    Parameters
    ----------
    prices : pd.DataFrame
        Price panel.
    fast, slow, signal : int
        Window parameters for MACD calculation.

    Returns
    -------
    pd.DataFrame
        Columns named ``{ticker}_macd``, ``{ticker}_macd_signal``,
        ``{ticker}_macd_hist``.
    """
    frames: list[pd.DataFrame] = []
    for col in prices.columns:
        s = prices[col]
        ema_fast = s.ewm(span=fast, min_periods=fast, adjust=False).mean()
        ema_slow = s.ewm(span=slow, min_periods=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        df_tmp = pd.DataFrame(
            {
                f"{col}_macd": macd_line,
                f"{col}_macd_signal": signal_line,
                f"{col}_macd_hist": histogram,
            },
            index=prices.index,
        )
        frames.append(df_tmp)
    result = pd.concat(frames, axis=1)
    logger.debug("compute_macd: %d columns (fast=%d, slow=%d, signal=%d)",
                 result.shape[1], fast, slow, signal)
    return result


def compute_bollinger_bands(
    prices: pd.DataFrame,
    window: int = BOLLINGER_WINDOW,
    num_std: float = BOLLINGER_STD,
) -> pd.DataFrame:
    """
    Compute Bollinger Bands and the %B indicator.

    Bollinger Bands consist of:
        - Middle band = SMA(window)
        - Upper band  = SMA + num_std * rolling_std
        - Lower band  = SMA - num_std * rolling_std
        - %B          = (Price - Lower) / (Upper - Lower)

    %B is bounded [0, 1] when price is within the bands; values outside
    indicate overbought (>1) or oversold (<0) conditions.

    Parameters
    ----------
    prices : pd.DataFrame
        Price panel.
    window : int
        Rolling-window length for the simple moving average.
    num_std : float
        Number of standard deviations for the band width.

    Returns
    -------
    pd.DataFrame
        Columns: ``{ticker}_bb_upper``, ``{ticker}_bb_lower``,
        ``{ticker}_bb_pctb``, ``{ticker}_bb_width``.
    """
    frames: list[pd.DataFrame] = []
    for col in prices.columns:
        s = prices[col]
        sma = s.rolling(window, min_periods=window).mean()
        std = s.rolling(window, min_periods=window).std()

        upper = sma + num_std * std
        lower = sma - num_std * std
        band_width = (upper - lower) / sma
        pct_b = (s - lower) / (upper - lower)

        df_tmp = pd.DataFrame(
            {
                f"{col}_bb_upper": upper,
                f"{col}_bb_lower": lower,
                f"{col}_bb_pctb": pct_b,
                f"{col}_bb_width": band_width,
            },
            index=prices.index,
        )
        frames.append(df_tmp)
    result = pd.concat(frames, axis=1)
    logger.debug("compute_bollinger_bands: %d columns (window=%d, std=%.1f)",
                 result.shape[1], window, num_std)
    return result


def compute_moving_average_crossovers(
    prices: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
) -> pd.DataFrame:
    """
    Compute SMA crossover signals (Golden Cross / Death Cross).

    Generates three features per asset:
        - ``{ticker}_sma_ratio``: SMA(short) / SMA(long)  (continuous signal)
        - ``{ticker}_golden_cross``: binary 1 when SMA(short) > SMA(long)
        - ``{ticker}_sma_spread``: SMA(short) - SMA(long) normalised by price

    The 50/200-day crossover is a widely-followed trend indicator used by
    institutional investors and is included as a proxy for regime shifts.

    Parameters
    ----------
    prices : pd.DataFrame
        Price panel.
    short_window : int
        Short-term SMA window (default: 50).
    long_window : int
        Long-term SMA window (default: 200).

    Returns
    -------
    pd.DataFrame
        Crossover features.
    """
    frames: list[pd.DataFrame] = []
    for col in prices.columns:
        s = prices[col]
        sma_short = s.rolling(short_window, min_periods=short_window).mean()
        sma_long = s.rolling(long_window, min_periods=long_window).mean()

        sma_ratio = sma_short / sma_long
        golden = (sma_short > sma_long).astype(float)
        spread = (sma_short - sma_long) / s

        df_tmp = pd.DataFrame(
            {
                f"{col}_sma_ratio": sma_ratio,
                f"{col}_golden_cross": golden,
                f"{col}_sma_spread": spread,
            },
            index=prices.index,
        )
        frames.append(df_tmp)
    result = pd.concat(frames, axis=1)
    logger.debug("compute_moving_average_crossovers: %d columns", result.shape[1])
    return result


# ============================================================================
# SECTION 4: Cross-Asset Features
# ============================================================================


def compute_rolling_beta(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int = ROLLING_BETA_WINDOW,
) -> pd.DataFrame:
    """
    Compute rolling CAPM beta of each asset relative to the market.

    Uses a rolling OLS regression of asset excess returns on market
    excess returns.  Beta_i = Cov(R_i, R_m) / Var(R_m).

    Parameters
    ----------
    returns : pd.DataFrame
        Asset daily returns.
    market_returns : pd.Series
        Market (benchmark) daily returns, e.g. SPY.
    window : int
        Rolling-window length.

    Returns
    -------
    pd.DataFrame
        Columns named ``{ticker}_beta_{window}d``.
    """
    frames: list[pd.Series] = []
    market_var = market_returns.rolling(window, min_periods=max(window // 2, 1)).var()

    for col in returns.columns:
        cov = (
            returns[col]
            .rolling(window, min_periods=max(window // 2, 1))
            .cov(market_returns)
        )
        beta = cov / market_var.replace(0, np.nan)
        beta.name = f"{col}_beta_{window}d"
        frames.append(beta)
    result = pd.concat(frames, axis=1)
    logger.debug("compute_rolling_beta: %d columns, window=%d",
                 result.shape[1], window)
    return result


def compute_rolling_correlation(
    returns: pd.DataFrame,
    window: int = ROLLING_CORR_WINDOW,
) -> pd.DataFrame:
    """
    Compute rolling average pairwise correlation among assets.

    At each date, estimates the full correlation matrix from the trailing
    ``window`` observations and reports the mean off-diagonal element.
    This serves as a diversification indicator: high average correlation
    implies reduced diversification benefits.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset daily returns.
    window : int
        Rolling-window length.

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame ``avg_correlation`` indexed by date.
    """
    n_assets = returns.shape[1]
    if n_assets < 2:
        logger.warning("compute_rolling_correlation: need >= 2 assets, got %d",
                       n_assets)
        return pd.DataFrame(index=returns.index, columns=["avg_correlation"])

    # Vectorised approach using rolling covariance
    rolling_corr_data = returns.rolling(window, min_periods=max(window // 2, 1)).corr()

    avg_corr: list[dict] = []
    valid_dates = returns.index[window - 1:]
    for date in valid_dates:
        try:
            corr_matrix = rolling_corr_data.loc[date]
            if isinstance(corr_matrix, pd.DataFrame):
                n = len(corr_matrix)
                if n < 2:
                    continue
                mask = np.ones((n, n), dtype=bool)
                np.fill_diagonal(mask, False)
                mean_corr = corr_matrix.values[mask].mean()
                avg_corr.append({"date": date, "avg_correlation": mean_corr})
        except (KeyError, ValueError, IndexError):
            continue

    if not avg_corr:
        return pd.DataFrame(index=returns.index, columns=["avg_correlation"])

    result = pd.DataFrame(avg_corr).set_index("date")
    logger.debug("compute_rolling_correlation: %d observations, window=%d",
                 len(result), window)
    return result


def compute_cross_sectional_dispersion(
    returns: pd.DataFrame,
    window: int = DISPERSION_WINDOW,
) -> pd.DataFrame:
    """
    Compute cross-sectional return dispersion.

    At each date, computes the cross-sectional standard deviation of
    asset returns.  High dispersion signals differentiated performance
    across assets (opportunity for active allocation), while low
    dispersion implies herd behaviour.

    Uses a rolling mean to smooth the daily dispersion series.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset daily returns.
    window : int
        Smoothing window applied to daily dispersion.

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame ``cross_dispersion``.
    """
    # Daily cross-sectional standard deviation
    daily_disp = returns.std(axis=1)
    # Smooth with rolling mean
    smoothed = daily_disp.rolling(window, min_periods=max(window // 2, 1)).mean()
    result = smoothed.to_frame("cross_dispersion")
    logger.debug("compute_cross_sectional_dispersion: window=%d", window)
    return result


def compute_relative_strength(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute relative strength of each asset vs the equal-weight portfolio.

    Relative strength is defined as ``P_i / P_eq``, where ``P_eq`` is the
    equal-weighted average of all asset price indices.  This is then
    normalised to start at 1.0 and expressed as a ratio.

    Parameters
    ----------
    prices : pd.DataFrame
        Price panel.

    Returns
    -------
    pd.DataFrame
        Columns named ``{ticker}_relstr``.
    """
    # Normalise prices to start at 1.0 for fair comparison
    normalized = prices / prices.iloc[0]
    eq_weight = normalized.mean(axis=1)

    rel_str = normalized.divide(eq_weight, axis=0)
    rel_str.columns = [f"{col}_relstr" for col in prices.columns]
    logger.debug("compute_relative_strength: %d columns", rel_str.shape[1])
    return rel_str


# ============================================================================
# SECTION 5: Macro Feature Engineering
# ============================================================================


def lag_macro_features(
    macro: pd.DataFrame,
    lags: list[int] = MACRO_LAGS,
) -> pd.DataFrame:
    """
    Create lagged versions of macro features to avoid look-ahead bias.

    For each lag ``l`` and each column in *macro*, produces a new column
    shifted forward by ``l`` periods, named ``{col}_lag{l}``.  The
    original (unlagged) columns are also included to allow the model to
    determine which lag structure is most predictive.

    Parameters
    ----------
    macro : pd.DataFrame
        Macro indicator panel.
    lags : list[int]
        Lag periods in trading days.

    Returns
    -------
    pd.DataFrame
        Original columns plus all lagged variants.
    """
    frames = [macro]
    for lag in lags:
        lagged = macro.shift(lag)
        lagged.columns = [f"{col}_lag{lag}" for col in lagged.columns]
        frames.append(lagged)
    result = pd.concat(frames, axis=1)
    logger.debug("lag_macro_features: %d lags, %d total columns",
                 len(lags), result.shape[1])
    return result


def compute_macro_changes(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Compute first differences, percentage changes, and z-scores for
    macro series.

    For each macro column, produces:
        - ``{col}_diff``   : first difference (level change)
        - ``{col}_pctchg`` : percentage change
        - ``{col}_zscore`` : rolling z-score (63-day window)

    These transformations help achieve stationarity and normalise the
    scale of macro indicators.

    Parameters
    ----------
    macro : pd.DataFrame
        Macro indicator panel.

    Returns
    -------
    pd.DataFrame
        All derived change features.
    """
    z_window = 63  # quarterly rolling window for z-score
    frames: list[pd.DataFrame] = []

    for col in macro.columns:
        s = macro[col]

        # First difference
        diff = s.diff()
        diff.name = f"{col}_diff"
        frames.append(diff)

        # Percentage change (handle zero values)
        pctchg = s.pct_change().replace([np.inf, -np.inf], np.nan)
        pctchg.name = f"{col}_pctchg"
        frames.append(pctchg)

        # Rolling z-score
        rolling_mean = s.rolling(z_window, min_periods=max(z_window // 2, 1)).mean()
        rolling_std = s.rolling(z_window, min_periods=max(z_window // 2, 1)).std()
        zscore = (s - rolling_mean) / rolling_std.replace(0, np.nan)
        zscore.name = f"{col}_zscore"
        frames.append(zscore)

    result = pd.concat(frames, axis=1)
    logger.debug("compute_macro_changes: %d derived columns from %d inputs",
                 result.shape[1], macro.shape[1])
    return result


def compute_macro_regime_indicators(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Compute binary and categorical regime indicators from macro data.

    Produces the following indicators (if the relevant series are present):

        - ``vix_high``          : 1 if VIXCLS > 20
        - ``vix_extreme``       : 1 if VIXCLS > 30
        - ``yield_curve_inv``   : 1 if T10Y2Y < 0 (inverted yield curve)
        - ``yield_curve_flat``  : 1 if T10Y2Y < 0.5
        - ``tight_spreads``     : 1 if HY OAS < 25th percentile
        - ``wide_spreads``      : 1 if HY OAS > 75th percentile
        - ``rising_rates``      : 1 if DFF diff(21) > 0
        - ``high_inflation``    : 1 if T10YIE > 2.5%

    Missing series are silently skipped.

    Parameters
    ----------
    macro : pd.DataFrame
        Macro indicator panel.

    Returns
    -------
    pd.DataFrame
        Binary regime indicators.
    """
    indicators: dict[str, pd.Series] = {}

    # VIX regimes
    if "VIXCLS" in macro.columns:
        vix = macro["VIXCLS"]
        indicators["vix_high"] = (vix > REGIME_VIX_THRESHOLD).astype(float)
        indicators["vix_extreme"] = (vix > REGIME_VIX_EXTREME).astype(float)
        # Normalised VIX level (percentile rank)
        indicators["vix_percentile"] = vix.rank(pct=True)

    # Yield curve regimes
    if "T10Y2Y" in macro.columns:
        spread = macro["T10Y2Y"]
        indicators["yield_curve_inv"] = (spread < 0).astype(float)
        indicators["yield_curve_flat"] = (spread < 0.5).astype(float)
        indicators["yield_curve_steep"] = (spread > 2.0).astype(float)

    # Credit spread regimes
    if "BAMLH0A0HYM2" in macro.columns:
        hy_oas = macro["BAMLH0A0HYM2"]
        expanding_q25 = hy_oas.expanding(min_periods=63).quantile(0.25)
        expanding_q75 = hy_oas.expanding(min_periods=63).quantile(0.75)
        indicators["tight_spreads"] = (hy_oas < expanding_q25).astype(float)
        indicators["wide_spreads"] = (hy_oas > expanding_q75).astype(float)

    # Monetary policy direction
    if "DFF" in macro.columns:
        dff = macro["DFF"]
        dff_change = dff.diff(21)
        indicators["rising_rates"] = (dff_change > 0).astype(float)
        indicators["falling_rates"] = (dff_change < 0).astype(float)

    # Inflation regime
    if "T10YIE" in macro.columns:
        bei = macro["T10YIE"]
        indicators["high_inflation"] = (bei > 2.5).astype(float)
        indicators["low_inflation"] = (bei < 1.5).astype(float)

    # Consumer sentiment regime
    if "UMCSENT" in macro.columns:
        sent = macro["UMCSENT"]
        rolling_median = sent.rolling(252, min_periods=63).median()
        indicators["sentiment_above_median"] = (sent > rolling_median).astype(float)

    # Labor market regime
    if "UNRATE" in macro.columns:
        unrate = macro["UNRATE"]
        unrate_diff = unrate.diff(63)  # quarterly change
        indicators["rising_unemployment"] = (unrate_diff > 0.5).astype(float)

    if not indicators:
        logger.warning("compute_macro_regime_indicators: no recognised series found")
        return pd.DataFrame(index=macro.index)

    result = pd.DataFrame(indicators, index=macro.index)
    logger.debug("compute_macro_regime_indicators: %d indicators", result.shape[1])
    return result


def compute_macro_surprise(
    macro: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """
    Compute macro surprise factors as deviations from rolling means.

    The surprise is defined as the z-score of the current value relative
    to the trailing ``window``-period distribution:

        surprise_t = (x_t - mean_{t-window:t}) / std_{t-window:t}

    Positive surprises indicate readings above recent trend; negative
    surprises indicate below-trend readings.  This captures the market's
    "surprise" reaction to macro releases.

    Parameters
    ----------
    macro : pd.DataFrame
        Macro indicator panel.
    window : int
        Look-back window for the rolling mean/std.

    Returns
    -------
    pd.DataFrame
        Columns named ``{col}_surprise``.
    """
    rolling_mean = macro.rolling(window, min_periods=max(window // 2, 1)).mean()
    rolling_std = macro.rolling(window, min_periods=max(window // 2, 1)).std()

    surprise = (macro - rolling_mean) / rolling_std.replace(0, np.nan)
    surprise.columns = [f"{col}_surprise" for col in macro.columns]

    # Clip extreme values to avoid numerical issues
    surprise = surprise.clip(-5.0, 5.0)

    logger.debug("compute_macro_surprise: %d columns, window=%d",
                 surprise.shape[1], window)
    return surprise


# ============================================================================
# SECTION 6: Dimensionality Reduction
# ============================================================================


def apply_pca(
    features: pd.DataFrame,
    n_components: Optional[int] = None,
    variance_threshold: float = PCA_VARIANCE_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply Principal Component Analysis to the feature matrix.

    If ``n_components`` is not specified, determines the number of
    components needed to explain at least ``variance_threshold`` of total
    variance (capped at ``MAX_PCA_COMPONENTS``).

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix (observations x features).  Must not contain NaN.
    n_components : int or None
        Fixed number of components.  If None, determined by
        ``variance_threshold``.
    variance_threshold : float
        Minimum cumulative explained variance ratio (default 0.95).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - DataFrame of principal components (``PC_1``, ``PC_2``, ...).
        - Dictionary with keys: ``explained_variance_ratio``,
          ``cumulative_variance``, ``n_components``, ``pca_object``.

    Raises
    ------
    ValueError
        If features contain NaN values.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if features.isnull().any().any():
        logger.warning("apply_pca: input contains NaN; dropping rows with NaN")
        features = features.dropna()

    if features.empty:
        raise ValueError("apply_pca: no valid rows after dropping NaN")

    # Standardise before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    if n_components is None:
        # Fit full PCA to determine component count
        pca_full = PCA()
        pca_full.fit(X_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
        n_components = min(n_components, MAX_PCA_COMPONENTS, features.shape[1])
        logger.info(
            "apply_pca: auto-selected %d components (%.1f%% variance)",
            n_components,
            cumvar[n_components - 1] * 100,
        )

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)

    col_names = [f"PC_{i + 1}" for i in range(n_components)]
    df_pca = pd.DataFrame(components, index=features.index, columns=col_names)

    info = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "n_components": n_components,
        "pca_object": pca,
        "scaler_object": scaler,
        "feature_names": list(features.columns),
    }

    logger.info(
        "apply_pca: reduced %d features to %d components (%.1f%% variance)",
        features.shape[1],
        n_components,
        info["cumulative_variance"][-1] * 100,
    )
    return df_pca, info


def select_features_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "mutual_info",
    top_k: int = 50,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select top features based on importance scores.

    Supports two methods:
        - ``mutual_info`` : mutual information regression (non-linear)
        - ``f_regression`` : univariate F-test (linear)

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    method : str
        Selection method (``'mutual_info'`` or ``'f_regression'``).
    top_k : int
        Number of top features to retain.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        - Filtered feature matrix with only the top-k features.
        - Series of importance scores for all features.

    Raises
    ------
    ValueError
        If method is not recognised.
    """
    from sklearn.feature_selection import mutual_info_regression, f_regression
    from sklearn.preprocessing import StandardScaler

    # Align and drop NaN
    combined = pd.concat([X, y.rename("__target__")], axis=1).dropna()
    X_clean = combined.drop(columns=["__target__"])
    y_clean = combined["__target__"]

    if X_clean.empty:
        raise ValueError("select_features_by_importance: no valid rows")

    # Standardise for numerical stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    if method == "mutual_info":
        scores = mutual_info_regression(X_scaled, y_clean, random_state=42)
    elif method == "f_regression":
        scores, _ = f_regression(X_scaled, y_clean)
        # Replace NaN F-scores with 0
        scores = np.nan_to_num(scores, nan=0.0)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'mutual_info' or 'f_regression'."
        )

    importance = pd.Series(scores, index=X_clean.columns, name="importance")
    importance = importance.sort_values(ascending=False)

    top_k = min(top_k, len(importance))
    selected = importance.head(top_k).index.tolist()
    X_selected = X_clean[selected]

    logger.info(
        "select_features_by_importance: selected %d/%d features (method=%s)",
        top_k,
        X_clean.shape[1],
        method,
    )
    return X_selected, importance


# ============================================================================
# SECTION 7: Statistical Tests
# ============================================================================


def test_stationarity(
    series: pd.Series,
    significance: float = 0.05,
) -> dict:
    """
    Run the Augmented Dickey-Fuller (ADF) test for a unit root.

    Null hypothesis: the series has a unit root (non-stationary).
    Rejection (p < significance) implies stationarity.

    Parameters
    ----------
    series : pd.Series
        Time series to test.
    significance : float
        Significance level for the test.

    Returns
    -------
    dict
        Keys: ``stationary``, ``p_value``, ``test_statistic``,
        ``critical_values``, ``n_obs``, ``reason`` (if insufficient data).
    """
    clean = series.dropna()
    if len(clean) < _MIN_OBS_FOR_STAT_TEST:
        return {
            "stationary": None,
            "p_value": None,
            "test_statistic": None,
            "critical_values": None,
            "n_obs": len(clean),
            "reason": "insufficient data",
        }

    try:
        result = adfuller(clean, autolag="AIC")
        return {
            "stationary": result[1] < significance,
            "p_value": round(result[1], 6),
            "test_statistic": round(result[0], 4),
            "critical_values": result[4],
            "n_obs": result[3],
            "reason": None,
        }
    except Exception as e:
        logger.error("test_stationarity failed for %s: %s", series.name, e)
        return {
            "stationary": None,
            "p_value": None,
            "test_statistic": None,
            "critical_values": None,
            "n_obs": len(clean),
            "reason": str(e),
        }


def test_stationarity_kpss(
    series: pd.Series,
    significance: float = 0.05,
    regression: str = "c",
) -> dict:
    """
    Run the KPSS test for stationarity.

    Null hypothesis: the series is stationary.
    Rejection (p < significance) implies non-stationarity.

    This test complements the ADF test.  A series is confidently
    stationary when ADF rejects and KPSS does not.

    Parameters
    ----------
    series : pd.Series
        Time series to test.
    significance : float
        Significance level.
    regression : str
        ``'c'`` for level stationarity, ``'ct'`` for trend stationarity.

    Returns
    -------
    dict
        Keys: ``stationary``, ``p_value``, ``test_statistic``,
        ``critical_values``, ``n_obs``.
    """
    clean = series.dropna()
    if len(clean) < _MIN_OBS_FOR_STAT_TEST:
        return {
            "stationary": None,
            "p_value": None,
            "test_statistic": None,
            "critical_values": None,
            "n_obs": len(clean),
            "reason": "insufficient data",
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p_value, n_lags, crit = kpss(clean, regression=regression, nlags="auto")
        return {
            "stationary": p_value > significance,  # KPSS: high p = stationary
            "p_value": round(p_value, 6),
            "test_statistic": round(stat, 4),
            "critical_values": crit,
            "n_lags": n_lags,
            "n_obs": len(clean),
            "reason": None,
        }
    except Exception as e:
        logger.error("test_stationarity_kpss failed for %s: %s", series.name, e)
        return {
            "stationary": None,
            "p_value": None,
            "test_statistic": None,
            "critical_values": None,
            "n_obs": len(clean),
            "reason": str(e),
        }


def stationarity_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run ADF stationarity test on all numeric columns and produce a report.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with numeric columns.

    Returns
    -------
    pd.DataFrame
        Report with columns: ``stationary``, ``p_value``,
        ``test_statistic``, indexed by feature name.
    """
    results: list[dict] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        result = test_stationarity(df[col])
        result["feature"] = col
        results.append(result)

    report = pd.DataFrame(results)
    if "feature" in report.columns:
        report = report.set_index("feature")

    n_stationary = report["stationary"].sum() if "stationary" in report.columns else 0
    n_total = len(report)
    logger.info(
        "Stationarity report: %d/%d features are stationary (ADF, p < 0.05)",
        n_stationary,
        n_total,
    )
    print(f"Stationarity: {n_stationary}/{n_total} features are stationary (p < 0.05)")
    return report


def test_granger_causality(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = _DEFAULT_GRANGER_MAX_LAG,
) -> dict:
    """
    Run a Granger causality test: does *x* Granger-cause *y*?

    Uses the F-test variant from ``statsmodels.tsa.stattools``.

    Parameters
    ----------
    x : pd.Series
        Potential causal variable.
    y : pd.Series
        Dependent variable.
    max_lag : int
        Maximum lag order to test.

    Returns
    -------
    dict
        Keys: ``min_p_value`` (best lag), ``best_lag``,
        ``all_results`` (p-value per lag), ``granger_causes`` (bool).
    """
    combined = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()

    if len(combined) < max_lag + _MIN_OBS_FOR_STAT_TEST:
        return {
            "granger_causes": None,
            "min_p_value": None,
            "best_lag": None,
            "all_results": {},
            "reason": "insufficient data",
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc_result = grangercausalitytests(
                combined[["y", "x"]], maxlag=max_lag, verbose=False
            )

        p_values = {}
        for lag in range(1, max_lag + 1):
            f_test = gc_result[lag][0]["ssr_ftest"]
            p_values[lag] = f_test[1]

        best_lag = min(p_values, key=p_values.get)
        min_p = p_values[best_lag]

        return {
            "granger_causes": min_p < 0.05,
            "min_p_value": round(min_p, 6),
            "best_lag": best_lag,
            "all_results": {k: round(v, 6) for k, v in p_values.items()},
            "reason": None,
        }
    except Exception as e:
        logger.error("test_granger_causality failed: %s", e)
        return {
            "granger_causes": None,
            "min_p_value": None,
            "best_lag": None,
            "all_results": {},
            "reason": str(e),
        }


def test_cointegration(
    x: pd.Series,
    y: pd.Series,
    significance: float = 0.05,
) -> dict:
    """
    Run the Engle-Granger two-step cointegration test.

    Tests whether *x* and *y* share a common stochastic trend,
    implying a long-run equilibrium relationship.

    Parameters
    ----------
    x, y : pd.Series
        Two time series of the same length.
    significance : float
        Significance level.

    Returns
    -------
    dict
        Keys: ``cointegrated``, ``p_value``, ``test_statistic``,
        ``critical_values``.
    """
    combined = pd.concat([x, y], axis=1).dropna()
    if len(combined) < _MIN_OBS_FOR_STAT_TEST:
        return {
            "cointegrated": None,
            "p_value": None,
            "test_statistic": None,
            "critical_values": None,
            "reason": "insufficient data",
        }

    try:
        stat, p_value, crit = coint(combined.iloc[:, 0], combined.iloc[:, 1])
        return {
            "cointegrated": p_value < significance,
            "p_value": round(p_value, 6),
            "test_statistic": round(stat, 4),
            "critical_values": dict(zip(["1%", "5%", "10%"], crit)),
            "reason": None,
        }
    except Exception as e:
        logger.error("test_cointegration failed: %s", e)
        return {
            "cointegrated": None,
            "p_value": None,
            "test_statistic": None,
            "critical_values": None,
            "reason": str(e),
        }


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Variance Inflation Factor for each feature.

    VIF > 10 indicates severe multicollinearity that may destabilise
    regression estimates.  VIF > 5 warrants attention.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (must be numeric, no NaN).

    Returns
    -------
    pd.DataFrame
        Two columns: ``feature`` and ``vif``, sorted descending by VIF.
    """
    X_clean = X.dropna().select_dtypes(include=[np.number])

    if X_clean.shape[1] < 2:
        logger.warning("compute_vif: need at least 2 features, got %d",
                       X_clean.shape[1])
        return pd.DataFrame(columns=["feature", "vif"])

    # Add constant for VIF calculation
    X_arr = X_clean.values
    vif_data: list[dict] = []
    for i in range(X_arr.shape[1]):
        try:
            vif_val = variance_inflation_factor(X_arr, i)
        except Exception:
            vif_val = np.nan
        vif_data.append({
            "feature": X_clean.columns[i],
            "vif": vif_val,
        })

    vif_df = pd.DataFrame(vif_data).sort_values("vif", ascending=False)
    logger.debug("compute_vif: max VIF = %.2f (%s)",
                 vif_df["vif"].max(), vif_df.iloc[0]["feature"])
    return vif_df


def remove_high_vif(
    X: pd.DataFrame,
    threshold: float = VIF_THRESHOLD,
    max_iterations: int = 100,
) -> pd.DataFrame:
    """
    Iteratively remove features with VIF above the threshold.

    At each iteration, the feature with the highest VIF is dropped.
    Iteration continues until all remaining features have VIF below
    ``threshold`` or until fewer than 2 features remain.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    threshold : float
        VIF threshold (default: 10.0).
    max_iterations : int
        Safety limit on iterations.

    Returns
    -------
    pd.DataFrame
        Feature matrix with multicollinear features removed.
    """
    X_clean = X.dropna().select_dtypes(include=[np.number])
    dropped: list[str] = []
    iteration = 0

    while iteration < max_iterations:
        if X_clean.shape[1] <= 2:
            break

        vif_df = compute_vif(X_clean)
        max_vif = vif_df["vif"].max()

        if np.isnan(max_vif) or max_vif <= threshold:
            break

        worst = vif_df.iloc[0]["feature"]
        X_clean = X_clean.drop(columns=[worst])
        dropped.append(worst)
        iteration += 1

    if dropped:
        logger.info(
            "remove_high_vif: dropped %d features (threshold=%.1f): %s",
            len(dropped),
            threshold,
            dropped,
        )
        print(f"Dropped {len(dropped)} high-VIF features: {dropped}")
    else:
        logger.info("remove_high_vif: no features exceeded VIF threshold %.1f",
                     threshold)
    return X_clean


# ============================================================================
# SECTION 8: Data Cleaning
# ============================================================================


def winsorize_returns(
    returns: pd.DataFrame,
    quantile: float = 0.01,
) -> pd.DataFrame:
    """
    Winsorize extreme return values by clipping at specified quantiles.

    Clips each column independently at the ``quantile`` and
    ``1 - quantile`` percentiles.  This reduces the influence of
    extreme observations (fat tails) without removing them entirely.

    Parameters
    ----------
    returns : pd.DataFrame
        Return series (may contain NaN).
    quantile : float
        Lower tail quantile for clipping (default: 0.01 = 1%).

    Returns
    -------
    pd.DataFrame
        Winsorized returns.
    """
    result = returns.copy()
    for col in result.columns:
        series = result[col].dropna()
        if len(series) == 0:
            continue
        lower = series.quantile(quantile)
        upper = series.quantile(1.0 - quantile)
        result[col] = result[col].clip(lower=lower, upper=upper)
        n_clipped = ((returns[col] < lower) | (returns[col] > upper)).sum()
        if n_clipped > 0:
            logger.debug("winsorize_returns: %s clipped %d values", col, n_clipped)

    total_clipped = (result != returns).sum().sum()
    logger.info("winsorize_returns: clipped %d values total (quantile=%.3f)",
                total_clipped, quantile)
    return result


def detect_outliers_zscore(
    df: pd.DataFrame,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Detect outliers using the z-score method.

    A value is flagged as an outlier if its absolute z-score exceeds
    ``threshold``.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric data frame.
    threshold : float
        Z-score threshold (default: 3.0).

    Returns
    -------
    pd.DataFrame
        Boolean mask where True indicates an outlier.
    """
    numeric = df.select_dtypes(include=[np.number])
    mean = numeric.mean()
    std = numeric.std()
    z_scores = (numeric - mean) / std.replace(0, np.nan)
    outlier_mask = z_scores.abs() > threshold

    n_outliers = outlier_mask.sum().sum()
    n_total = outlier_mask.size
    pct = (n_outliers / n_total * 100) if n_total > 0 else 0
    logger.info(
        "detect_outliers_zscore: %d outliers (%.2f%%) at threshold=%.1f",
        n_outliers,
        pct,
        threshold,
    )
    return outlier_mask


def detect_outliers_iqr(
    df: pd.DataFrame,
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Detect outliers using the Interquartile Range (IQR) method.

    A value is an outlier if it falls below ``Q1 - factor * IQR`` or
    above ``Q3 + factor * IQR``, where ``IQR = Q3 - Q1``.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric data frame.
    factor : float
        IQR multiplier (default: 1.5 for standard, 3.0 for extreme).

    Returns
    -------
    pd.DataFrame
        Boolean mask where True indicates an outlier.
    """
    numeric = df.select_dtypes(include=[np.number])
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    outlier_mask = (numeric < lower) | (numeric > upper)

    n_outliers = outlier_mask.sum().sum()
    n_total = outlier_mask.size
    pct = (n_outliers / n_total * 100) if n_total > 0 else 0
    logger.info(
        "detect_outliers_iqr: %d outliers (%.2f%%) at factor=%.1f",
        n_outliers,
        pct,
        factor,
    )
    return outlier_mask


def handle_missing_data(
    df: pd.DataFrame,
    method: str = "ffill",
    max_gap: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Handle missing data with configurable strategies and gap limits.

    Supports three filling methods:
        - ``'ffill'``: forward-fill (last observation carried forward)
        - ``'bfill'``: backward-fill
        - ``'interpolate'``: linear interpolation

    Any gap longer than ``max_gap`` consecutive NaN values is left as NaN
    to avoid filling across structural data breaks (e.g., weekends,
    holidays, or genuine data gaps).

    After filling, any remaining NaN at the boundaries is handled by
    forward-fill followed by backward-fill.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame possibly containing NaN.
    method : str
        Fill method.
    max_gap : int
        Maximum consecutive NaN gap to fill (default: 5).
    verbose : bool
        If True, print a summary of missing data handling.

    Returns
    -------
    pd.DataFrame
        Data frame with missing data handled.
    """
    original_nan_count = df.isnull().sum().sum()

    if original_nan_count == 0:
        if verbose:
            logger.info("handle_missing_data: no missing values found")
        return df

    result = df.copy()

    if method == "ffill":
        result = result.ffill(limit=max_gap)
    elif method == "bfill":
        result = result.bfill(limit=max_gap)
    elif method == "interpolate":
        result = result.interpolate(method="linear", limit=max_gap, limit_direction="both")
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'ffill', 'bfill', or 'interpolate'.")

    # Handle remaining edge NaN
    result = result.ffill(limit=1).bfill(limit=1)

    remaining_nan = result.isnull().sum().sum()

    if verbose:
        filled = original_nan_count - remaining_nan
        logger.info(
            "handle_missing_data: filled %d/%d NaN (method=%s, max_gap=%d); "
            "%d remaining",
            filled,
            original_nan_count,
            method,
            max_gap,
            remaining_nan,
        )
        print(
            f"Missing data: filled {filled}/{original_nan_count} NaN "
            f"(method={method}, max_gap={max_gap}); {remaining_nan} remaining"
        )

    return result


# ============================================================================
# SECTION 9: Master Builders
# ============================================================================


def build_features(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    risk_free_series: Optional[pd.Series] = None,
    high: Optional[pd.DataFrame] = None,
    low: Optional[pd.DataFrame] = None,
    open_: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Orchestrate all feature computation and produce the master feature matrix.

    This is the main entry point for preprocessing.  It sequentially
    computes all feature categories, aligns them on a common date index,
    handles missing data, and optionally saves the result to disk.

    Pipeline steps:
        1. Clean prices and macro (handle missing, winsorize)
        2. Compute return features (multi-horizon, log, excess)
        3. Compute volatility features (rolling, realised, EWMA, GK)
        4. Compute momentum and trend features (RSI, MACD, Bollinger, crossovers)
        5. Compute cross-asset features (beta, correlation, dispersion, relative strength)
        6. Compute macro features (lags, changes, regimes, surprises)
        7. Concatenate, align, and clean the master matrix
        8. Run stationarity report
        9. Save to ``PROCESSED_DIR / "features.csv"``

    Parameters
    ----------
    prices : pd.DataFrame
        Asset price panel (adjusted close).
    macro : pd.DataFrame
        Macroeconomic indicator panel.
    risk_free_series : pd.Series or None
        Annualized risk-free rate for excess returns.
    high, low, open_ : pd.DataFrame or None
        OHLC data for Garman-Klass volatility.
    save : bool
        Whether to save the feature matrix to disk.

    Returns
    -------
    pd.DataFrame
        Complete feature matrix.
    """
    print("=" * 70)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    # Step 1: Handle missing data in inputs
    print("\n[1/9] Cleaning input data...")
    prices = handle_missing_data(prices, method="ffill", max_gap=5, verbose=True)
    macro = handle_missing_data(macro, method="ffill", max_gap=10, verbose=True)

    all_features: list[pd.DataFrame] = []

    # Step 2: Return features
    print("\n[2/9] Computing return features...")
    returns_multi = compute_returns(prices)
    all_features.append(returns_multi)

    log_returns = compute_log_returns(prices)
    all_features.append(log_returns)

    if risk_free_series is not None:
        excess_ret = compute_excess_returns(prices, risk_free_series)
        all_features.append(excess_ret)
        print(f"  - Excess returns: {excess_ret.shape[1]} columns")

    print(f"  - Multi-horizon returns: {returns_multi.shape[1]} columns")
    print(f"  - Log returns: {log_returns.shape[1]} columns")

    # Step 3: Volatility features
    print("\n[3/9] Computing volatility features...")
    volatility = compute_volatility(prices)
    all_features.append(volatility)
    print(f"  - Rolling volatility: {volatility.shape[1]} columns")

    rv = compute_realized_variance(prices, window=21)
    all_features.append(rv)
    print(f"  - Realized variance: {rv.shape[1]} columns")

    gk_vol = compute_garman_klass_vol(high, low, open_, prices, window=21)
    all_features.append(gk_vol)
    print(f"  - Garman-Klass vol: {gk_vol.shape[1]} columns")

    ewma_vol = compute_ewma_volatility(prices)
    all_features.append(ewma_vol)
    print(f"  - EWMA volatility: {ewma_vol.shape[1]} columns")

    # Step 4: Momentum and trend features
    print("\n[4/9] Computing momentum & trend features...")
    momentum = compute_momentum(prices)
    all_features.append(momentum)
    print(f"  - Momentum: {momentum.shape[1]} columns")

    rsi = compute_rsi(prices)
    all_features.append(rsi)
    print(f"  - RSI: {rsi.shape[1]} columns")

    macd = compute_macd(prices)
    all_features.append(macd)
    print(f"  - MACD: {macd.shape[1]} columns")

    bollinger = compute_bollinger_bands(prices)
    all_features.append(bollinger)
    print(f"  - Bollinger Bands: {bollinger.shape[1]} columns")

    crossovers = compute_moving_average_crossovers(prices)
    all_features.append(crossovers)
    print(f"  - MA Crossovers: {crossovers.shape[1]} columns")

    # Step 5: Cross-asset features
    print("\n[5/9] Computing cross-asset features...")
    daily_returns = prices.pct_change()

    # Rolling beta vs market (first column assumed to be market proxy)
    market_col = "SPY" if "SPY" in daily_returns.columns else daily_returns.columns[0]
    market_ret = daily_returns[market_col]
    other_returns = daily_returns.drop(columns=[market_col], errors="ignore")
    if not other_returns.empty:
        rolling_beta = compute_rolling_beta(other_returns, market_ret)
        all_features.append(rolling_beta)
        print(f"  - Rolling beta: {rolling_beta.shape[1]} columns")

    rolling_corr = compute_rolling_correlation(daily_returns)
    all_features.append(rolling_corr)
    print(f"  - Avg pairwise correlation: {rolling_corr.shape[1]} columns")

    dispersion = compute_cross_sectional_dispersion(daily_returns)
    all_features.append(dispersion)
    print(f"  - Cross-sectional dispersion: {dispersion.shape[1]} columns")

    rel_str = compute_relative_strength(prices)
    all_features.append(rel_str)
    print(f"  - Relative strength: {rel_str.shape[1]} columns")

    # Step 6: Macro features
    print("\n[6/9] Computing macro features...")
    macro_lagged = lag_macro_features(macro)
    macro_lagged = macro_lagged.reindex(prices.index).ffill()
    all_features.append(macro_lagged)
    print(f"  - Lagged macro: {macro_lagged.shape[1]} columns")

    macro_changes = compute_macro_changes(macro)
    macro_changes = macro_changes.reindex(prices.index).ffill()
    all_features.append(macro_changes)
    print(f"  - Macro changes: {macro_changes.shape[1]} columns")

    macro_regimes = compute_macro_regime_indicators(macro)
    if not macro_regimes.empty:
        macro_regimes = macro_regimes.reindex(prices.index).ffill()
        all_features.append(macro_regimes)
        print(f"  - Macro regimes: {macro_regimes.shape[1]} columns")

    macro_surprise = compute_macro_surprise(macro)
    macro_surprise = macro_surprise.reindex(prices.index).ffill()
    all_features.append(macro_surprise)
    print(f"  - Macro surprise: {macro_surprise.shape[1]} columns")

    # Step 7: Concatenate and clean
    print("\n[7/9] Assembling master feature matrix...")
    features = pd.concat(all_features, axis=1)

    # Remove duplicate columns (can arise from overlapping computations)
    features = features.loc[:, ~features.columns.duplicated()]

    # Replace infinities with NaN
    features = features.replace([np.inf, -np.inf], np.nan)

    # Drop rows where all values are NaN
    features = features.dropna(how="all")

    # Forward-fill remaining sparse NaN (with limit)
    features = features.ffill(limit=3)

    # Drop rows that still have too many NaN
    max_nan_pct = 0.20
    nan_pct = features.isnull().mean(axis=1)
    features = features[nan_pct <= max_nan_pct]

    # Final drop of any remaining NaN rows
    features = features.dropna()

    print(f"  - Final shape: {features.shape}")
    print(f"  - Date range: {features.index[0]} to {features.index[-1]}")
    print(f"  - Total features: {features.shape[1]}")

    # Step 8: Stationarity check
    print("\n[8/9] Running stationarity diagnostics...")
    # Sample a subset to keep this fast
    sample_cols = features.columns[:min(30, features.shape[1])]
    _ = stationarity_report(features[sample_cols])

    # Step 9: Save
    if save:
        print("\n[9/9] Saving features...")
        path = PROCESSED_DIR / "features.csv"
        features.to_csv(path)
        print(f"  - Saved to {path}")
        logger.info("build_features: saved %s (shape=%s)", path, features.shape)
    else:
        print("\n[9/9] Skipping save (save=False)")

    print("\n" + "=" * 70)
    print(f"FEATURE ENGINEERING COMPLETE: {features.shape[1]} features, "
          f"{features.shape[0]} observations")
    print("=" * 70)

    return features


def build_features_for_asset(
    features: pd.DataFrame,
    ticker: str,
    horizon: int = PREDICTION_HORIZON,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract the feature matrix X and target y for a single asset.

    The target variable is the forward return at the given horizon.
    Features are lagged appropriately to prevent look-ahead bias.

    Parameters
    ----------
    features : pd.DataFrame
        Master feature matrix from ``build_features``.
    ticker : str
        Asset ticker (e.g. ``'SPY'``).
    horizon : int
        Forward return horizon in trading days (default from config).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        - X: feature matrix (all features except forward returns)
        - y: forward return target

    Raises
    ------
    KeyError
        If the target column is not found in the feature matrix.
    """
    target_col = TARGET_COLUMN_TEMPLATE.format(ticker=ticker, horizon=horizon)

    if target_col not in features.columns:
        available_targets = [c for c in features.columns if c.endswith(f"_ret_{horizon}d")]
        raise KeyError(
            f"Target column '{target_col}' not found. "
            f"Available {horizon}d targets: {available_targets}"
        )

    # Target: forward return (shift back by horizon so features at time t
    # predict returns from t to t+horizon)
    y = features[target_col].shift(-horizon)
    y.name = target_col

    # Features: exclude all forward-looking return columns for this horizon
    # to prevent data leakage.  Keep returns at shorter horizons as features.
    exclude_patterns = [f"_ret_{horizon}d"]
    feature_cols = [
        c for c in features.columns
        if not any(pat in c for pat in exclude_patterns)
    ]

    X = features[feature_cols]

    # Align and drop NaN
    mask = X.notnull().all(axis=1) & y.notnull()
    X = X[mask]
    y = y[mask]

    logger.info(
        "build_features_for_asset: ticker=%s, horizon=%d, "
        "X shape=%s, y shape=%s",
        ticker,
        horizon,
        X.shape,
        y.shape,
    )
    print(
        f"Asset {ticker} (horizon={horizon}d): "
        f"X={X.shape}, y={y.shape}, "
        f"date range: {X.index[0]} to {X.index[-1]}"
    )

    return X, y


# ============================================================================
# UTILITY HELPERS
# ============================================================================


def _validate_dataframe(df: pd.DataFrame, name: str = "input") -> None:
    """Validate that a DataFrame is non-empty and has a DatetimeIndex."""
    if df.empty:
        raise ValueError(f"{name}: DataFrame is empty")
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("%s: index is not DatetimeIndex (type=%s)",
                       name, type(df.index).__name__)


def summarize_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary statistics table for the feature matrix.

    Includes count, mean, std, min, max, skewness, kurtosis, and
    percentage of missing values for each column.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.

    Returns
    -------
    pd.DataFrame
        Summary statistics indexed by feature name.
    """
    desc = features.describe().T
    desc["skewness"] = features.skew()
    desc["kurtosis"] = features.kurtosis()
    desc["pct_missing"] = features.isnull().mean() * 100
    desc["pct_zero"] = (features == 0).mean() * 100

    logger.info("summarize_features: %d features summarized", len(desc))
    return desc


def align_dataframes(
    *dfs: pd.DataFrame,
    method: str = "inner",
) -> list[pd.DataFrame]:
    """
    Align multiple DataFrames on a common date index.

    Parameters
    ----------
    *dfs : pd.DataFrame
        DataFrames to align.
    method : str
        Join method: ``'inner'`` (intersection) or ``'outer'`` (union).

    Returns
    -------
    list[pd.DataFrame]
        Aligned DataFrames.
    """
    if not dfs:
        return []

    if method == "inner":
        common_idx = dfs[0].index
        for df in dfs[1:]:
            common_idx = common_idx.intersection(df.index)
        return [df.loc[common_idx] for df in dfs]
    elif method == "outer":
        all_idx = dfs[0].index
        for df in dfs[1:]:
            all_idx = all_idx.union(df.index)
        return [df.reindex(all_idx) for df in dfs]
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'inner' or 'outer'.")


def check_data_quality(
    df: pd.DataFrame,
    name: str = "data",
) -> dict:
    """
    Run a comprehensive data quality check and return a report.

    Checks for:
        - Missing values (total and per column)
        - Infinite values
        - Constant columns (zero variance)
        - Duplicate rows
        - Duplicate columns

    Parameters
    ----------
    df : pd.DataFrame
        Data to check.
    name : str
        Label for logging.

    Returns
    -------
    dict
        Quality metrics.
    """
    numeric = df.select_dtypes(include=[np.number])

    n_missing = df.isnull().sum().sum()
    pct_missing = (n_missing / df.size * 100) if df.size > 0 else 0
    n_inf = np.isinf(numeric).sum().sum() if not numeric.empty else 0

    # Constant columns
    constant_cols = [c for c in numeric.columns if numeric[c].std() == 0]

    # Duplicate rows / columns
    n_dup_rows = df.duplicated().sum()
    n_dup_cols = df.columns.duplicated().sum()

    # Columns with > 50% missing
    high_missing_cols = [
        c for c in df.columns if df[c].isnull().mean() > 0.5
    ]

    report = {
        "name": name,
        "shape": df.shape,
        "n_missing": int(n_missing),
        "pct_missing": round(pct_missing, 2),
        "n_infinite": int(n_inf),
        "n_constant_cols": len(constant_cols),
        "constant_cols": constant_cols,
        "n_duplicate_rows": int(n_dup_rows),
        "n_duplicate_columns": int(n_dup_cols),
        "high_missing_cols": high_missing_cols,
    }

    logger.info(
        "check_data_quality [%s]: shape=%s, missing=%.1f%%, "
        "infinite=%d, constant=%d, dup_rows=%d",
        name,
        df.shape,
        pct_missing,
        n_inf,
        len(constant_cols),
        n_dup_rows,
    )

    # Print human-readable summary
    print(f"\n--- Data Quality Report: {name} ---")
    print(f"  Shape           : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Missing values  : {n_missing} ({pct_missing:.1f}%)")
    print(f"  Infinite values : {n_inf}")
    print(f"  Constant columns: {len(constant_cols)}")
    print(f"  Duplicate rows  : {n_dup_rows}")
    print(f"  Duplicate cols  : {n_dup_cols}")
    if high_missing_cols:
        print(f"  High-missing (>50%): {high_missing_cols}")
    print("---")

    return report
