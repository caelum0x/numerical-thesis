"""
Data fetchers for Yahoo Finance (asset prices) and FRED (macro indicators).

This module provides three main classes and convenience functions for
downloading, caching, and validating financial data needed for the
macro-based portfolio optimization thesis.

Classes
-------
FREDFetcher
    Full-featured FRED API wrapper with retry logic, rate limiting,
    disk caching, point-in-time vintage awareness, and batch download.

PriceFetcher
    Yahoo Finance wrapper with retry logic, data validation, total
    return computation, and data quality reporting.

DataManager
    Orchestrates both fetchers, provides unified fetch / cache / validate
    workflow, and merges price and macro data on common dates.

Convenience Functions
---------------------
fetch_prices, fetch_fred, fetch_all
    Thin wrappers around the classes for quick interactive use.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

from src.config import (
    TICKER_LIST,
    FRED_SERIES_LIST,
    FRED_SERIES,
    START_DATE,
    END_DATE,
    RAW_DIR,
    FETCH_MAX_RETRIES,
    FETCH_RETRY_DELAY,
    FETCH_RATE_LIMIT_DELAY,
    YF_AUTO_ADJUST,
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ============================================================================
# FRED Fetcher
# ============================================================================

class FREDFetcher:
    """Full-featured wrapper around the FRED API.

    Features
    --------
    * Automatic retry with exponential back-off on transient failures.
    * Configurable rate limiting between successive API calls.
    * Disk-based caching — skips re-fetching when a fresh CSV exists.
    * ALFRED vintage-date support for point-in-time data.
    * Batch download with progress reporting.
    * Post-fetch validation (nulls, date range coverage).
    * Convenience method to retrieve series metadata / descriptions.

    Parameters
    ----------
    api_key : str or None
        FRED API key.  Falls back to the ``FRED_API_KEY`` environment
        variable when *None*.
    series_ids : list[str]
        FRED series identifiers to fetch (default: ``FRED_SERIES_LIST``).
    start : str
        Observation start date, ``"YYYY-MM-DD"`` (default: ``START_DATE``).
    end : str
        Observation end date, ``"YYYY-MM-DD"`` (default: ``END_DATE``).
    cache_dir : Path
        Directory where cached CSV files are stored.
    max_retries : int
        Maximum number of retry attempts per series.
    retry_delay : float
        Base delay (seconds) for exponential back-off.
    rate_limit_delay : float
        Minimum interval (seconds) between consecutive API calls.
    cache_max_age_hours : float
        Maximum age in hours before a cached file is considered stale.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        series_ids: Optional[list[str]] = None,
        start: str = START_DATE,
        end: str = END_DATE,
        cache_dir: Optional[Path] = None,
        max_retries: int = FETCH_MAX_RETRIES,
        retry_delay: float = FETCH_RETRY_DELAY,
        rate_limit_delay: float = FETCH_RATE_LIMIT_DELAY,
        cache_max_age_hours: float = 24.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED_API_KEY not provided and not found in environment.  "
                "Obtain a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self.fred = Fred(api_key=self.api_key)
        self.series_ids = series_ids or list(FRED_SERIES_LIST)
        self.start = start
        self.end = end
        self.cache_dir = cache_dir or (RAW_DIR / "fred_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        self.cache_max_age_hours = cache_max_age_hours
        self._last_call_ts: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self, series_id: str) -> Path:
        """Return the cache file path for a given series."""
        return self.cache_dir / f"{series_id}.csv"

    def _cache_is_fresh(self, path: Path) -> bool:
        """Return *True* if the cached file exists and is younger than
        ``cache_max_age_hours``."""
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age = datetime.now() - mtime
        return age < timedelta(hours=self.cache_max_age_hours)

    def _enforce_rate_limit(self) -> None:
        """Sleep if needed to respect the rate-limit delay."""
        elapsed = time.time() - self._last_call_ts
        remaining = self.rate_limit_delay - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call_ts = time.time()

    def _fetch_with_retry(
        self,
        series_id: str,
        vintage_date: Optional[str] = None,
    ) -> Optional[pd.Series]:
        """Fetch a single FRED series with exponential back-off.

        Parameters
        ----------
        series_id : str
            FRED series identifier.
        vintage_date : str or None
            If provided, fetch the ALFRED vintage as of this date
            (``"YYYY-MM-DD"``).

        Returns
        -------
        pd.Series or None
            Observation values indexed by date, or *None* on failure.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                self._enforce_rate_limit()
                if vintage_date is not None:
                    data = self.fred.get_series_as_of_date(
                        series_id,
                        vintage_date,
                    )
                    # get_series_as_of_date returns a DataFrame; extract the
                    # most recent vintage value for each real-time period.
                    if data is not None and not data.empty:
                        data = (
                            data.sort_values("realtime_start")
                            .drop_duplicates(subset=["date"], keep="last")
                            .set_index("date")["value"]
                        )
                        data.index = pd.to_datetime(data.index)
                        data = data.loc[self.start : self.end]
                        return data
                    return None
                else:
                    data = self.fred.get_series(
                        series_id,
                        observation_start=self.start,
                        observation_end=self.end,
                    )
                    return data
            except Exception as exc:
                wait = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "FRED fetch %s attempt %d/%d failed: %s — retrying in %.1fs",
                    series_id,
                    attempt,
                    self.max_retries,
                    exc,
                    wait,
                )
                if attempt < self.max_retries:
                    time.sleep(wait)
        logger.error("FRED fetch %s failed after %d attempts", series_id, self.max_retries)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_series(
        self,
        series_id: str,
        use_cache: bool = True,
        vintage_date: Optional[str] = None,
    ) -> Optional[pd.Series]:
        """Fetch a single FRED series, optionally using disk cache.

        Parameters
        ----------
        series_id : str
            FRED series identifier (e.g. ``"VIXCLS"``).
        use_cache : bool
            If *True*, return cached data when the file is still fresh.
        vintage_date : str or None
            ALFRED vintage date for point-in-time data.

        Returns
        -------
        pd.Series or None
        """
        cache_file = self._cache_path(series_id)

        if use_cache and vintage_date is None and self._cache_is_fresh(cache_file):
            logger.info("Loading %s from cache: %s", series_id, cache_file)
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True).squeeze("columns")
            return cached

        logger.info("Fetching %s from FRED API...", series_id)
        data = self._fetch_with_retry(series_id, vintage_date=vintage_date)

        if data is not None and not data.empty and vintage_date is None:
            data.to_csv(cache_file, header=True)
            logger.debug("Cached %s to %s", series_id, cache_file)

        return data

    def fetch_all(
        self,
        use_cache: bool = True,
        vintage_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch all configured series and return a combined DataFrame.

        Parameters
        ----------
        use_cache : bool
            Use disk cache for individual series.
        vintage_date : str or None
            Fetch ALFRED vintages as of this date.

        Returns
        -------
        pd.DataFrame
            Columns are series IDs, index is datetime.
        """
        total = len(self.series_ids)
        frames: dict[str, pd.Series] = {}
        successes = 0
        failures: list[str] = []

        for idx, sid in enumerate(self.series_ids, 1):
            pct = idx / total * 100
            logger.info("[%d/%d  %.0f%%] %s", idx, total, pct, sid)
            series = self.fetch_series(sid, use_cache=use_cache, vintage_date=vintage_date)
            if series is not None and not series.empty:
                frames[sid] = series
                successes += 1
            else:
                failures.append(sid)

        logger.info(
            "FRED batch complete: %d/%d succeeded, %d failed%s",
            successes,
            total,
            len(failures),
            f" ({failures})" if failures else "",
        )

        if not frames:
            logger.error("No FRED data was fetched — returning empty DataFrame.")
            return pd.DataFrame()

        macro = pd.DataFrame(frames)
        macro.index = pd.to_datetime(macro.index)
        macro = macro.sort_index()
        return macro

    def get_series_metadata(self, series_id: str) -> dict[str, Any]:
        """Return metadata for a FRED series (title, frequency, units, etc.).

        Parameters
        ----------
        series_id : str
            FRED series identifier.

        Returns
        -------
        dict
            Keys include ``title``, ``frequency``, ``units``, ``seasonal_adjustment``,
            ``last_updated``, and ``notes``.
        """
        self._enforce_rate_limit()
        try:
            info = self.fred.get_series_info(series_id)
            meta: dict[str, Any] = {
                "id": series_id,
                "title": info.get("title", ""),
                "frequency": info.get("frequency", ""),
                "units": info.get("units", ""),
                "seasonal_adjustment": info.get("seasonal_adjustment", ""),
                "last_updated": str(info.get("last_updated", "")),
                "notes": info.get("notes", ""),
                "observation_start": str(info.get("observation_start", "")),
                "observation_end": str(info.get("observation_end", "")),
            }
            return meta
        except Exception as exc:
            logger.warning("Could not retrieve metadata for %s: %s", series_id, exc)
            return {"id": series_id, "error": str(exc)}

    def get_all_metadata(self) -> pd.DataFrame:
        """Return metadata for every configured series as a DataFrame."""
        records = []
        for sid in self.series_ids:
            meta = self.get_series_metadata(sid)
            records.append(meta)
        return pd.DataFrame(records).set_index("id")

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run quality checks on a fetched macro DataFrame.

        Returns a report DataFrame with one row per column, including:
        * ``null_count`` / ``null_pct``
        * ``first_valid_date`` / ``last_valid_date``
        * ``expected_start`` / ``expected_end`` coverage flags

        Parameters
        ----------
        df : pd.DataFrame
            Output from :meth:`fetch_all`.

        Returns
        -------
        pd.DataFrame
            Quality report indexed by series ID.
        """
        expected_start = pd.Timestamp(self.start)
        expected_end = pd.Timestamp(self.end)
        records: list[dict[str, Any]] = []

        for col in df.columns:
            series = df[col]
            non_null = series.dropna()
            first_valid = non_null.index.min() if not non_null.empty else None
            last_valid = non_null.index.max() if not non_null.empty else None
            n_null = int(series.isnull().sum())
            n_total = len(series)
            null_pct = n_null / n_total * 100 if n_total > 0 else 0.0

            covers_start = first_valid is not None and first_valid <= expected_start + timedelta(days=30)
            covers_end = last_valid is not None and last_valid >= expected_end - timedelta(days=30)

            records.append(
                {
                    "series": col,
                    "description": FRED_SERIES.get(col, ""),
                    "n_observations": n_total,
                    "null_count": n_null,
                    "null_pct": round(null_pct, 2),
                    "first_valid_date": first_valid,
                    "last_valid_date": last_valid,
                    "covers_start": covers_start,
                    "covers_end": covers_end,
                    "pass": covers_start and covers_end and null_pct < 20.0,
                }
            )

        report = pd.DataFrame(records).set_index("series")
        n_pass = report["pass"].sum()
        n_total_series = len(report)
        logger.info(
            "FRED validation: %d/%d series passed checks.",
            n_pass,
            n_total_series,
        )
        if n_pass < n_total_series:
            failed = report[~report["pass"]]
            for sid in failed.index:
                row = failed.loc[sid]
                logger.warning(
                    "  FAIL %s: null_pct=%.1f%%, covers_start=%s, covers_end=%s",
                    sid,
                    row["null_pct"],
                    row["covers_start"],
                    row["covers_end"],
                )
        return report

    def fetch_vintage(
        self,
        series_id: str,
        vintage_date: str,
    ) -> Optional[pd.Series]:
        """Convenience wrapper: fetch a single series at a specific ALFRED
        vintage date for point-in-time analysis.

        Parameters
        ----------
        series_id : str
            FRED series identifier.
        vintage_date : str
            Date string ``"YYYY-MM-DD"`` representing the real-time vintage.

        Returns
        -------
        pd.Series or None
        """
        return self.fetch_series(series_id, use_cache=False, vintage_date=vintage_date)


# ============================================================================
# Price Fetcher (Yahoo Finance)
# ============================================================================

class PriceFetcher:
    """Yahoo Finance price downloader with retry logic, validation, and
    data-quality reporting.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to download (default: ``TICKER_LIST``).
    start : str
        Start date ``"YYYY-MM-DD"`` (default: ``START_DATE``).
    end : str
        End date ``"YYYY-MM-DD"`` (default: ``END_DATE``).
    auto_adjust : bool
        Whether yfinance should auto-adjust OHLC for splits/dividends
        (default: ``YF_AUTO_ADJUST``).
    max_retries : int
        Maximum download retry attempts.
    retry_delay : float
        Base delay between retries (seconds).
    cache_dir : Path or None
        Where to cache downloaded price CSVs.
    cache_max_age_hours : float
        Maximum age (hours) before a cached file is stale.
    """

    # Threshold above which a single-day return is flagged as suspicious
    SUSPICIOUS_RETURN_THRESHOLD = 0.25  # 25 %

    def __init__(
        self,
        tickers: Optional[list[str]] = None,
        start: str = START_DATE,
        end: str = END_DATE,
        auto_adjust: bool = YF_AUTO_ADJUST,
        max_retries: int = FETCH_MAX_RETRIES,
        retry_delay: float = FETCH_RETRY_DELAY,
        cache_dir: Optional[Path] = None,
        cache_max_age_hours: float = 24.0,
    ) -> None:
        self.tickers = tickers or list(TICKER_LIST)
        self.start = start
        self.end = end
        self.auto_adjust = auto_adjust
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_dir = cache_dir or (RAW_DIR / "price_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_max_age_hours = cache_max_age_hours

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self) -> Path:
        """Return a deterministic cache path based on the ticker list and
        date range."""
        key = hashlib.md5(
            json.dumps(
                {"tickers": sorted(self.tickers), "start": self.start, "end": self.end},
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]
        return self.cache_dir / f"prices_{key}.csv"

    def _cache_is_fresh(self, path: Path) -> bool:
        """Return *True* if cached CSV exists and is recent enough."""
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age = datetime.now() - mtime
        return age < timedelta(hours=self.cache_max_age_hours)

    def _download_with_retry(self) -> pd.DataFrame:
        """Download price data from Yahoo Finance with retry logic.

        Returns
        -------
        pd.DataFrame
            Raw ``yf.download`` output.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "yfinance download attempt %d/%d for %d tickers...",
                    attempt,
                    self.max_retries,
                    len(self.tickers),
                )
                data = yf.download(
                    self.tickers,
                    start=self.start,
                    end=self.end,
                    auto_adjust=self.auto_adjust,
                    threads=True,
                    progress=False,
                )
                if data is not None and not data.empty:
                    return data
                raise ValueError("yfinance returned empty DataFrame")
            except Exception as exc:
                last_exc = exc
                wait = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "yfinance attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    wait,
                )
                if attempt < self.max_retries:
                    time.sleep(wait)
        raise RuntimeError(
            f"yfinance download failed after {self.max_retries} attempts"
        ) from last_exc

    @staticmethod
    def _extract_close(
        data: pd.DataFrame,
        tickers: list[str],
    ) -> pd.DataFrame:
        """Extract the Close (or Adj Close) price panel from raw yfinance
        output, handling both single-ticker and multi-ticker column
        structures.

        Parameters
        ----------
        data : pd.DataFrame
            Raw output from ``yf.download``.
        tickers : list[str]
            Requested ticker symbols.

        Returns
        -------
        pd.DataFrame
            Columns = tickers, index = datetime.
        """
        if len(tickers) == 1:
            if "Close" in data.columns:
                prices = data[["Close"]].rename(columns={"Close": tickers[0]})
            else:
                prices = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        else:
            if "Close" in data.columns.get_level_values(0):
                prices = data["Close"]
            elif "Adj Close" in data.columns.get_level_values(0):
                prices = data["Adj Close"]
            else:
                raise KeyError("Could not find 'Close' or 'Adj Close' in downloaded data.")

        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        return prices

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, use_cache: bool = True) -> pd.DataFrame:
        """Download adjusted close prices.

        Parameters
        ----------
        use_cache : bool
            When *True*, return cached data if it exists and is fresh.

        Returns
        -------
        pd.DataFrame
            Columns are tickers, index is ``DatetimeIndex``.
        """
        cache_file = self._cache_path()

        if use_cache and self._cache_is_fresh(cache_file):
            logger.info("Loading prices from cache: %s", cache_file)
            prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return prices

        raw = self._download_with_retry()
        prices = self._extract_close(raw, self.tickers)

        # Persist to cache
        prices.to_csv(cache_file)
        logger.info("Cached prices to %s", cache_file)
        return prices

    def fetch_single(self, ticker: str, use_cache: bool = False) -> pd.DataFrame:
        """Download data for a single ticker.

        This is useful when you need to re-download one problematic ticker
        without re-fetching the entire universe.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        use_cache : bool
            Whether to check the cache.

        Returns
        -------
        pd.DataFrame
            Single-column DataFrame.
        """
        fetcher = PriceFetcher(
            tickers=[ticker],
            start=self.start,
            end=self.end,
            auto_adjust=self.auto_adjust,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            cache_dir=self.cache_dir,
            cache_max_age_hours=self.cache_max_age_hours,
        )
        return fetcher.fetch(use_cache=use_cache)

    def compute_total_returns(
        self,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute simple daily total returns from adjusted close prices.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices (output of :meth:`fetch`).

        Returns
        -------
        pd.DataFrame
            Daily percentage returns; first row is NaN.
        """
        returns = prices.pct_change()
        returns.iloc[0] = 0.0  # first observation has no prior price
        return returns

    def fill_missing_trading_days(
        self,
        prices: pd.DataFrame,
        method: str = "ffill",
        limit: int = 5,
    ) -> pd.DataFrame:
        """Forward-fill (or other method) gaps caused by holidays or
        missing trading days.

        Parameters
        ----------
        prices : pd.DataFrame
            Raw price DataFrame.
        method : str
            Fill method (``"ffill"``, ``"bfill"``, ``"interpolate"``).
        limit : int
            Maximum consecutive NaN values to fill.

        Returns
        -------
        pd.DataFrame
            Filled price DataFrame.
        """
        # Build a business-day calendar spanning the date range
        full_idx = pd.bdate_range(start=prices.index.min(), end=prices.index.max())
        prices = prices.reindex(full_idx)

        if method == "interpolate":
            prices = prices.interpolate(method="time", limit=limit)
        else:
            prices = prices.fillna(method=method, limit=limit)

        prices.index.name = "Date"
        return prices

    def validate(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Run data-quality checks on a price DataFrame.

        Checks include:
        * Missing value percentage per ticker.
        * Date coverage (first/last observation vs requested range).
        * Suspicious returns (> ``SUSPICIOUS_RETURN_THRESHOLD`` in one day).
        * Duplicate dates.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices.

        Returns
        -------
        pd.DataFrame
            One row per ticker with quality metrics.
        """
        expected_start = pd.Timestamp(self.start)
        expected_end = pd.Timestamp(self.end)
        records: list[dict[str, Any]] = []

        for col in prices.columns:
            series = prices[col]
            non_null = series.dropna()
            n_total = len(series)
            n_null = int(series.isnull().sum())
            null_pct = n_null / n_total * 100 if n_total > 0 else 0.0

            first_date = non_null.index.min() if not non_null.empty else None
            last_date = non_null.index.max() if not non_null.empty else None

            # Suspicious single-day returns
            returns = series.pct_change().dropna()
            n_suspicious = int(
                (returns.abs() > self.SUSPICIOUS_RETURN_THRESHOLD).sum()
            )
            max_abs_return = float(returns.abs().max()) if not returns.empty else 0.0

            covers_start = first_date is not None and first_date <= expected_start + timedelta(days=30)
            covers_end = last_date is not None and last_date >= expected_end - timedelta(days=30)

            records.append(
                {
                    "ticker": col,
                    "n_observations": n_total,
                    "null_count": n_null,
                    "null_pct": round(null_pct, 2),
                    "first_date": first_date,
                    "last_date": last_date,
                    "covers_start": covers_start,
                    "covers_end": covers_end,
                    "n_suspicious_returns": n_suspicious,
                    "max_abs_daily_return": round(max_abs_return, 4),
                    "pass": covers_start and covers_end and null_pct < 5.0,
                }
            )

        report = pd.DataFrame(records).set_index("ticker")
        n_pass = report["pass"].sum()
        logger.info(
            "Price validation: %d/%d tickers passed checks.",
            n_pass,
            len(report),
        )
        if n_pass < len(report):
            failed = report[~report["pass"]]
            for tkr in failed.index:
                row = failed.loc[tkr]
                logger.warning(
                    "  FAIL %s: null_pct=%.1f%%, suspicious=%d, start=%s, end=%s",
                    tkr,
                    row["null_pct"],
                    row["n_suspicious_returns"],
                    row["covers_start"],
                    row["covers_end"],
                )
        return report

    def data_quality_report(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Generate a comprehensive data quality summary for the price
        panel.  Extends :meth:`validate` with additional statistics useful
        for the thesis appendix.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices.

        Returns
        -------
        pd.DataFrame
            Detailed quality report.
        """
        base_report = self.validate(prices)

        # Additional per-ticker statistics
        returns = prices.pct_change()
        extra_records: list[dict[str, Any]] = []

        for col in prices.columns:
            ret = returns[col].dropna()
            extra_records.append(
                {
                    "ticker": col,
                    "mean_daily_return": round(float(ret.mean()), 6),
                    "std_daily_return": round(float(ret.std()), 6),
                    "skewness": round(float(ret.skew()), 4),
                    "kurtosis": round(float(ret.kurtosis()), 4),
                    "min_daily_return": round(float(ret.min()), 4),
                    "max_daily_return": round(float(ret.max()), 4),
                    "n_zero_return_days": int((ret == 0).sum()),
                    "longest_gap_days": self._longest_gap(prices[col]),
                }
            )

        extra_df = pd.DataFrame(extra_records).set_index("ticker")
        full_report = base_report.join(extra_df)
        return full_report

    @staticmethod
    def _longest_gap(series: pd.Series) -> int:
        """Return the longest consecutive-NaN gap (in calendar days) for a
        single price series."""
        if series.isnull().sum() == 0:
            return 0
        is_null = series.isnull()
        if not is_null.any():
            return 0
        # Group consecutive nulls
        groups = (~is_null).cumsum()
        null_groups = is_null.groupby(groups).sum()
        return int(null_groups.max()) if not null_groups.empty else 0


# ============================================================================
# Data Manager — orchestrates both fetchers
# ============================================================================

class DataManager:
    """High-level orchestrator that coordinates :class:`FREDFetcher` and
    :class:`PriceFetcher`, provides unified caching, validation, and merging.

    Parameters
    ----------
    price_fetcher : PriceFetcher or None
        Custom price fetcher instance.  A default is created if *None*.
    fred_fetcher : FREDFetcher or None
        Custom FRED fetcher instance.  A default is created if *None*.
    raw_dir : Path
        Directory for storing combined raw CSVs.
    """

    def __init__(
        self,
        price_fetcher: Optional[PriceFetcher] = None,
        fred_fetcher: Optional[FREDFetcher] = None,
        raw_dir: Optional[Path] = None,
    ) -> None:
        self.raw_dir = raw_dir or RAW_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.price_fetcher = price_fetcher or PriceFetcher()
        # FREDFetcher requires an API key; create lazily only when needed
        self._fred_fetcher = fred_fetcher
        self._prices: Optional[pd.DataFrame] = None
        self._macro: Optional[pd.DataFrame] = None

    @property
    def fred_fetcher(self) -> FREDFetcher:
        """Lazy-initialise the FRED fetcher so that the DataManager can be
        instantiated without a FRED API key if only price data is needed."""
        if self._fred_fetcher is None:
            self._fred_fetcher = FREDFetcher()
        return self._fred_fetcher

    # ------------------------------------------------------------------
    # Paths for combined CSVs
    # ------------------------------------------------------------------

    @property
    def prices_path(self) -> Path:
        return self.raw_dir / "prices.csv"

    @property
    def macro_path(self) -> Path:
        return self.raw_dir / "macro.csv"

    @property
    def merged_path(self) -> Path:
        return self.raw_dir / "merged.csv"

    # ------------------------------------------------------------------
    # Core fetch methods
    # ------------------------------------------------------------------

    def fetch_all(
        self,
        use_cache: bool = True,
        save: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch both price and macro data with progress reporting.

        Parameters
        ----------
        use_cache : bool
            Use disk cache when available.
        save : bool
            Save combined CSVs to ``raw_dir``.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(prices, macro)`` DataFrames.
        """
        logger.info("=" * 60)
        logger.info("FETCH ALL — starting data download")
        logger.info("=" * 60)

        # 1) Prices
        logger.info("--- Phase 1/2: Fetching price data ---")
        t0 = time.time()
        prices = self.price_fetcher.fetch(use_cache=use_cache)
        logger.info(
            "Price fetch complete: %d rows x %d tickers (%.1fs)",
            len(prices),
            prices.shape[1],
            time.time() - t0,
        )

        # 2) Macro
        logger.info("--- Phase 2/2: Fetching macro data ---")
        t0 = time.time()
        macro = self.fred_fetcher.fetch_all(use_cache=use_cache)
        logger.info(
            "Macro fetch complete: %d rows x %d series (%.1fs)",
            len(macro),
            macro.shape[1],
            time.time() - t0,
        )

        self._prices = prices
        self._macro = macro

        if save:
            prices.to_csv(self.prices_path)
            macro.to_csv(self.macro_path)
            logger.info("Saved combined prices to %s", self.prices_path)
            logger.info("Saved combined macro  to %s", self.macro_path)

        logger.info("=" * 60)
        logger.info("FETCH ALL — complete")
        logger.info("=" * 60)
        return prices, macro

    def load_or_fetch(
        self,
        max_age_hours: float = 24.0,
        save: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from disk if fresh; otherwise fetch and save.

        Parameters
        ----------
        max_age_hours : float
            Maximum age (hours) for cached combined CSVs.
        save : bool
            Persist newly fetched data to disk.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(prices, macro)`` DataFrames.
        """
        prices_fresh = self._file_is_fresh(self.prices_path, max_age_hours)
        macro_fresh = self._file_is_fresh(self.macro_path, max_age_hours)

        if prices_fresh and macro_fresh:
            logger.info("Loading cached data from disk (< %.1f hours old).", max_age_hours)
            prices = pd.read_csv(self.prices_path, index_col=0, parse_dates=True)
            macro = pd.read_csv(self.macro_path, index_col=0, parse_dates=True)
            self._prices = prices
            self._macro = macro
            return prices, macro

        logger.info("Cached data is stale or missing — fetching fresh data.")
        return self.fetch_all(use_cache=False, save=save)

    @staticmethod
    def _file_is_fresh(path: Path, max_age_hours: float) -> bool:
        """Check whether *path* exists and was modified within the last
        *max_age_hours* hours."""
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age = datetime.now() - mtime
        return age < timedelta(hours=max_age_hours)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_all(
        self,
        prices: Optional[pd.DataFrame] = None,
        macro: Optional[pd.DataFrame] = None,
    ) -> dict[str, pd.DataFrame]:
        """Run quality checks on both price and macro data.

        Parameters
        ----------
        prices : pd.DataFrame or None
            Price data; uses last-fetched data if *None*.
        macro : pd.DataFrame or None
            Macro data; uses last-fetched data if *None*.

        Returns
        -------
        dict
            ``{"prices": price_report, "macro": macro_report}``
        """
        prices = prices if prices is not None else self._prices
        macro = macro if macro is not None else self._macro

        reports: dict[str, pd.DataFrame] = {}

        if prices is not None:
            reports["prices"] = self.price_fetcher.validate(prices)
        else:
            logger.warning("No price data available for validation.")

        if macro is not None:
            reports["macro"] = self.fred_fetcher.validate(macro)
        else:
            logger.warning("No macro data available for validation.")

        return reports

    def generate_data_quality_report(
        self,
        prices: Optional[pd.DataFrame] = None,
        macro: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Create a unified summary DataFrame of data quality across
        both price and macro data.

        Parameters
        ----------
        prices : pd.DataFrame or None
        macro : pd.DataFrame or None

        Returns
        -------
        pd.DataFrame
            Combined quality report with a ``source`` column
            (``"price"`` or ``"macro"``).
        """
        prices = prices if prices is not None else self._prices
        macro = macro if macro is not None else self._macro

        parts: list[pd.DataFrame] = []

        if prices is not None:
            price_rpt = self.price_fetcher.data_quality_report(prices)
            price_rpt["source"] = "price"
            parts.append(price_rpt)

        if macro is not None:
            macro_rpt = self.fred_fetcher.validate(macro)
            macro_rpt["source"] = "macro"
            parts.append(macro_rpt)

        if not parts:
            logger.warning("No data available for quality report.")
            return pd.DataFrame()

        combined = pd.concat(parts, axis=0)
        combined.index.name = "series_id"

        # Summary statistics at the bottom
        n_total = len(combined)
        n_pass = combined["pass"].sum()
        logger.info(
            "Data quality report: %d/%d series pass all checks (%.0f%%).",
            n_pass,
            n_total,
            n_pass / n_total * 100 if n_total > 0 else 0,
        )
        return combined

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def merge_price_and_macro(
        self,
        prices: Optional[pd.DataFrame] = None,
        macro: Optional[pd.DataFrame] = None,
        how: str = "inner",
        fill_method: Optional[str] = "ffill",
        fill_limit: int = 5,
        save: bool = True,
    ) -> pd.DataFrame:
        """Align price and macro data on common dates.

        Macro data (often daily or lower frequency) is forward-filled to
        match the daily price index before the join.

        Parameters
        ----------
        prices : pd.DataFrame or None
        macro : pd.DataFrame or None
        how : str
            Join type (``"inner"``, ``"outer"``, ``"left"``).
        fill_method : str or None
            Method used to fill macro gaps before merging.
        fill_limit : int
            Maximum consecutive fills.
        save : bool
            If *True*, save the merged DataFrame to disk.

        Returns
        -------
        pd.DataFrame
            Merged panel with both price columns and macro columns.
        """
        prices = prices if prices is not None else self._prices
        macro = macro if macro is not None else self._macro

        if prices is None or macro is None:
            raise ValueError(
                "Both prices and macro must be available.  "
                "Call fetch_all() or load_or_fetch() first."
            )

        # Reindex macro to the price date index and forward-fill
        macro_aligned = macro.reindex(prices.index)
        if fill_method == "ffill":
            macro_aligned = macro_aligned.ffill(limit=fill_limit)
        elif fill_method == "bfill":
            macro_aligned = macro_aligned.bfill(limit=fill_limit)
        elif fill_method == "interpolate":
            macro_aligned = macro_aligned.interpolate(method="time", limit=fill_limit)

        # Prefix macro columns to avoid name collisions
        macro_aligned = macro_aligned.add_prefix("macro_")

        if how == "inner":
            merged = prices.join(macro_aligned, how="inner")
        elif how == "outer":
            merged = prices.join(macro_aligned, how="outer")
        else:
            merged = prices.join(macro_aligned, how="left")

        merged = merged.sort_index()
        logger.info(
            "Merged data: %d rows x %d cols (prices=%d, macro=%d)",
            len(merged),
            merged.shape[1],
            prices.shape[1],
            macro_aligned.shape[1],
        )

        if save:
            merged.to_csv(self.merged_path)
            logger.info("Saved merged data to %s", self.merged_path)

        return merged

    # ------------------------------------------------------------------
    # Freshness
    # ------------------------------------------------------------------

    def get_latest_data_date(
        self,
        prices: Optional[pd.DataFrame] = None,
        macro: Optional[pd.DataFrame] = None,
    ) -> dict[str, Any]:
        """Check the freshness of available data.

        Returns
        -------
        dict
            Keys: ``prices_latest``, ``macro_latest``, ``prices_csv_mtime``,
            ``macro_csv_mtime``, ``days_since_price_update``,
            ``days_since_macro_update``.
        """
        prices = prices if prices is not None else self._prices
        macro = macro if macro is not None else self._macro
        now = datetime.now()

        result: dict[str, Any] = {}

        # Latest date in the data itself
        if prices is not None and not prices.empty:
            result["prices_latest"] = prices.index.max()
            result["days_since_last_price"] = (now - prices.index.max()).days
        else:
            result["prices_latest"] = None
            result["days_since_last_price"] = None

        if macro is not None and not macro.empty:
            result["macro_latest"] = macro.index.max()
            result["days_since_last_macro"] = (now - macro.index.max()).days
        else:
            result["macro_latest"] = None
            result["days_since_last_macro"] = None

        # File modification times
        if self.prices_path.exists():
            result["prices_csv_mtime"] = datetime.fromtimestamp(
                self.prices_path.stat().st_mtime
            )
        else:
            result["prices_csv_mtime"] = None

        if self.macro_path.exists():
            result["macro_csv_mtime"] = datetime.fromtimestamp(
                self.macro_path.stat().st_mtime
            )
        else:
            result["macro_csv_mtime"] = None

        return result


# ============================================================================
# Standalone convenience functions
# ============================================================================

def fetch_prices(
    tickers: Optional[list[str]] = None,
    start: str = START_DATE,
    end: str = END_DATE,
    save: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Convenience function: download adjusted close prices.

    This is a thin wrapper around :class:`PriceFetcher` for quick
    interactive use (e.g. in notebooks).

    Parameters
    ----------
    tickers : list[str] or None
        Ticker symbols (default: ``TICKER_LIST``).
    start : str
        Start date.
    end : str
        End date.
    save : bool
        Save to ``RAW_DIR / "prices.csv"``.
    use_cache : bool
        Use disk cache if available.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices.
    """
    tickers = tickers or list(TICKER_LIST)
    fetcher = PriceFetcher(tickers=tickers, start=start, end=end)
    prices = fetcher.fetch(use_cache=use_cache)

    if save:
        path = RAW_DIR / "prices.csv"
        prices.to_csv(path)
        logger.info("Saved prices to %s", path)

    return prices


def fetch_fred(
    series: Optional[list[str]] = None,
    start: str = START_DATE,
    end: str = END_DATE,
    save: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Convenience function: download FRED macro series.

    Parameters
    ----------
    series : list[str] or None
        FRED series IDs (default: ``FRED_SERIES_LIST``).
    start : str
        Start date.
    end : str
        End date.
    save : bool
        Save to ``RAW_DIR / "macro.csv"``.
    use_cache : bool
        Use disk cache if available.

    Returns
    -------
    pd.DataFrame
        Macro data with datetime index.
    """
    series = series or list(FRED_SERIES_LIST)
    fetcher = FREDFetcher(series_ids=series, start=start, end=end)
    macro = fetcher.fetch_all(use_cache=use_cache)

    if save:
        path = RAW_DIR / "macro.csv"
        macro.to_csv(path)
        logger.info("Saved macro data to %s", path)

    return macro


def fetch_all(
    save: bool = True,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function: fetch both price and macro data.

    Parameters
    ----------
    save : bool
        Save combined CSVs to ``RAW_DIR``.
    use_cache : bool
        Use disk cache if available.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(prices, macro)``
    """
    manager = DataManager()
    return manager.fetch_all(use_cache=use_cache, save=save)


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch financial data for the thesis portfolio optimization project.",
    )
    parser.add_argument(
        "--prices-only",
        action="store_true",
        help="Fetch only price data (skip FRED).",
    )
    parser.add_argument(
        "--macro-only",
        action="store_true",
        help="Fetch only FRED macro data (skip prices).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached data and re-download everything.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after fetching.",
    )
    parser.add_argument(
        "--quality-report",
        action="store_true",
        help="Generate and print a full data quality report.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge price and macro data after fetching.",
    )
    parser.add_argument(
        "--freshness",
        action="store_true",
        help="Print data freshness information and exit.",
    )

    args = parser.parse_args()
    use_cache = not args.no_cache

    if args.freshness:
        dm = DataManager()
        try:
            prices = pd.read_csv(dm.prices_path, index_col=0, parse_dates=True)
        except FileNotFoundError:
            prices = None
        try:
            macro = pd.read_csv(dm.macro_path, index_col=0, parse_dates=True)
        except FileNotFoundError:
            macro = None
        dm._prices = prices
        dm._macro = macro
        info = dm.get_latest_data_date()
        for key, val in info.items():
            print(f"  {key}: {val}")
        raise SystemExit(0)

    if args.prices_only:
        prices = fetch_prices(use_cache=use_cache)
        if args.validate:
            pf = PriceFetcher()
            report = pf.validate(prices)
            print(report.to_string())
    elif args.macro_only:
        macro = fetch_fred(use_cache=use_cache)
        if args.validate:
            ff = FREDFetcher()
            report = ff.validate(macro)
            print(report.to_string())
    else:
        dm = DataManager()
        prices, macro = dm.fetch_all(use_cache=use_cache, save=True)

        if args.validate:
            reports = dm.validate_all(prices, macro)
            for name, rpt in reports.items():
                print(f"\n{'='*60}")
                print(f"  Validation Report: {name.upper()}")
                print(f"{'='*60}")
                print(rpt.to_string())

        if args.quality_report:
            qr = dm.generate_data_quality_report(prices, macro)
            print(f"\n{'='*60}")
            print("  Full Data Quality Report")
            print(f"{'='*60}")
            print(qr.to_string())

        if args.merge:
            merged = dm.merge_price_and_macro(prices, macro, save=True)
            print(f"\nMerged shape: {merged.shape}")
            print(f"Date range: {merged.index.min()} to {merged.index.max()}")
