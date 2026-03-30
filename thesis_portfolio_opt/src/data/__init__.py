"""Data fetching and preprocessing pipeline."""

from src.data.fetcher import fetch_prices, fetch_fred, fetch_all
from src.data.preprocessor import (
    compute_returns,
    compute_volatility,
    compute_momentum,
    lag_macro_features,
    test_stationarity,
    stationarity_report,
    compute_vif,
    remove_high_vif,
    compute_rolling_correlation,
    build_features,
)

__all__ = [
    "fetch_prices",
    "fetch_fred",
    "fetch_all",
    "compute_returns",
    "compute_volatility",
    "compute_momentum",
    "lag_macro_features",
    "test_stationarity",
    "stationarity_report",
    "compute_vif",
    "remove_high_vif",
    "compute_rolling_correlation",
    "build_features",
]
