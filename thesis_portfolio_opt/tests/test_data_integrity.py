"""
Unit tests for data integrity and pipeline validation.
"""

import pytest
import numpy as np
import pandas as pd
from src.config import TICKER_LIST, FRED_SERIES_LIST, RAW_DIR, PROCESSED_DIR


class TestPriceData:
    """Tests for raw price data."""

    @pytest.fixture
    def prices(self):
        path = RAW_DIR / "prices.csv"
        if not path.exists():
            pytest.skip("Price data not available")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def test_no_all_null_columns(self, prices):
        """No ticker should be entirely null."""
        all_null = prices.columns[prices.isnull().all()]
        assert len(all_null) == 0, f"All-null columns: {list(all_null)}"

    def test_prices_are_positive(self, prices):
        """Prices should be positive."""
        assert (prices.dropna() > 0).all().all(), "Found non-positive prices"

    def test_index_is_sorted(self, prices):
        """Index should be chronologically sorted."""
        assert prices.index.is_monotonic_increasing

    def test_no_duplicate_dates(self, prices):
        """No duplicate dates in index."""
        assert not prices.index.duplicated().any()

    def test_has_expected_tickers(self, prices):
        """All configured tickers should be present."""
        for ticker in TICKER_LIST:
            assert ticker in prices.columns, f"Missing ticker: {ticker}"

    def test_sufficient_history(self, prices):
        """Should have at least 5 years of data."""
        date_range = (prices.index[-1] - prices.index[0]).days
        assert date_range >= 365 * 5, f"Only {date_range / 365:.1f} years of data"

    def test_returns_are_reasonable(self, prices):
        """Daily returns should not exceed 50% (catches data errors)."""
        returns = prices.pct_change().dropna()
        max_ret = returns.abs().max().max()
        assert max_ret < 0.5, f"Suspicious daily return: {max_ret:.2%}"


class TestMacroData:
    """Tests for FRED macro data."""

    @pytest.fixture
    def macro(self):
        path = RAW_DIR / "macro.csv"
        if not path.exists():
            pytest.skip("Macro data not available")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def test_index_is_sorted(self, macro):
        """Index should be chronologically sorted."""
        assert macro.index.is_monotonic_increasing

    def test_reasonable_missing_rate(self, macro):
        """No column should have more than 99% missing values.

        Note: monthly FRED series (UNRATE, CPI, etc.) are stored on a daily
        index, so up to ~96 % of rows can be NaN by design.  The threshold
        is therefore set to 99 % to catch truly broken downloads.
        """
        missing_rate = macro.isnull().mean()
        bad = missing_rate[missing_rate > 0.99]
        assert len(bad) == 0, f"High missing rate: {bad.to_dict()}"

    def test_has_expected_series(self, macro):
        """All configured FRED series should be present."""
        for series in FRED_SERIES_LIST:
            assert series in macro.columns, f"Missing series: {series}"

    def test_no_all_zero_columns(self, macro):
        """No series should be all zeros."""
        for col in macro.columns:
            non_null = macro[col].dropna()
            if len(non_null) > 0:
                assert not (non_null == 0).all(), f"All-zero column: {col}"


class TestFeatureData:
    """Tests for processed feature data."""

    @pytest.fixture
    def features(self):
        path = PROCESSED_DIR / "features.csv"
        if not path.exists():
            pytest.skip("Feature data not available")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def test_no_inf_values(self, features):
        """Features should not contain infinity."""
        assert not np.isinf(features.select_dtypes(include=[np.number])).any().any()

    def test_no_null_values(self, features):
        """Processed features should have no nulls."""
        assert not features.isnull().any().any()

    def test_minimum_rows(self, features):
        """Should have a reasonable number of data points."""
        assert len(features) >= 100, f"Only {len(features)} rows"

    def test_has_return_columns(self, features):
        """Should contain return features."""
        ret_cols = [c for c in features.columns if "_ret_" in c]
        assert len(ret_cols) > 0, "No return features found"

    def test_has_volatility_columns(self, features):
        """Should contain volatility features."""
        vol_cols = [c for c in features.columns if "_vol_" in c]
        assert len(vol_cols) > 0, "No volatility features found"

    def test_has_momentum_columns(self, features):
        """Should contain momentum features."""
        mom_cols = [c for c in features.columns if "_mom_" in c]
        assert len(mom_cols) > 0, "No momentum features found"

    def test_index_is_datetime(self, features):
        """Index should be datetime type."""
        assert pd.api.types.is_datetime64_any_dtype(features.index)
