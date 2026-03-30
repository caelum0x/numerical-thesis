"""
Shared test fixtures and configuration.
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Ensure src is importable from tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================================
# SYNTHETIC DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Generate synthetic price data for 5 assets over 2000 trading days."""
    np.random.seed(42)
    dates = pd.bdate_range("2010-01-01", periods=2000)
    n_assets = 5
    tickers = ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D", "ASSET_E"]

    # Simulate correlated returns
    mean_returns = np.array([0.0003, 0.0002, 0.0004, 0.0001, 0.00025])
    vol = np.array([0.015, 0.012, 0.018, 0.008, 0.010])

    returns = np.random.randn(len(dates), n_assets) * vol + mean_returns
    prices = pd.DataFrame(
        100 * np.cumprod(1 + returns, axis=0),
        index=dates,
        columns=tickers,
    )
    return prices


@pytest.fixture
def sample_returns(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """Daily returns from synthetic prices."""
    return sample_prices.pct_change().dropna()


@pytest.fixture
def sample_macro() -> pd.DataFrame:
    """Generate synthetic macro data."""
    np.random.seed(42)
    dates = pd.bdate_range("2010-01-01", periods=2000)
    macro = pd.DataFrame({
        "RATE": np.cumsum(np.random.randn(len(dates)) * 0.01) + 2.0,
        "SPREAD": np.cumsum(np.random.randn(len(dates)) * 0.005) + 1.5,
        "VIX": np.abs(np.cumsum(np.random.randn(len(dates)) * 0.5) + 20),
        "SENTIMENT": np.cumsum(np.random.randn(len(dates)) * 0.3) + 70,
    }, index=dates)
    return macro


@pytest.fixture
def sample_cov() -> np.ndarray:
    """Generate a valid positive semi-definite covariance matrix."""
    np.random.seed(42)
    n = 5
    A = np.random.randn(n, n) * 0.1
    return A @ A.T + np.eye(n) * 0.01


@pytest.fixture
def sample_mu() -> np.ndarray:
    """Expected returns vector for 5 assets."""
    return np.array([0.08, 0.06, 0.10, 0.04, 0.07])


@pytest.fixture
def sample_features(sample_prices: pd.DataFrame, sample_macro: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from synthetic data."""
    from src.data.preprocessor import build_features
    return build_features(sample_prices, sample_macro, save=False)
