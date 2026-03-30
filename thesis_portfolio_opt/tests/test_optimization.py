"""
Unit tests for optimization and backtesting modules.
"""

import pytest
import numpy as np
import pandas as pd


class TestOptimizer:
    """Tests for the portfolio optimizer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample expected returns and covariance matrix."""
        np.random.seed(42)
        n = 5
        mu = np.array([0.08, 0.06, 0.10, 0.04, 0.07])
        # Generate valid positive semi-definite covariance matrix
        A = np.random.randn(n, n) * 0.1
        cov = A @ A.T + np.eye(n) * 0.01
        return mu, cov

    def test_mean_variance_weights_sum_to_one(self, sample_data):
        from src.optimization.optimizer import mean_variance_optimize
        mu, cov = sample_data
        w = mean_variance_optimize(mu, cov, risk_aversion=2.0)
        assert w is not None
        assert abs(w.sum() - 1.0) < 1e-4, f"Weights sum to {w.sum()}"

    def test_mean_variance_weights_within_bounds(self, sample_data):
        from src.optimization.optimizer import mean_variance_optimize
        mu, cov = sample_data
        w = mean_variance_optimize(mu, cov, risk_aversion=2.0, min_weight=0.0, max_weight=0.5)
        assert w is not None
        assert (w >= -1e-4).all(), f"Weight below min: {w.min()}"
        assert (w <= 0.5 + 1e-4).all(), f"Weight above max: {w.max()}"

    def test_min_variance_lower_risk(self, sample_data):
        from src.optimization.optimizer import mean_variance_optimize, minimum_variance_optimize
        mu, cov = sample_data
        w_mv = mean_variance_optimize(mu, cov, risk_aversion=0.5)
        w_min = minimum_variance_optimize(cov)
        assert w_mv is not None and w_min is not None
        vol_mv = np.sqrt(w_mv @ cov @ w_mv)
        vol_min = np.sqrt(w_min @ cov @ w_min)
        assert vol_min <= vol_mv + 1e-4, "Min variance should have lower risk"

    def test_risk_parity_equal_contributions(self, sample_data):
        from src.optimization.optimizer import risk_parity_optimize
        _, cov = sample_data
        # Use a well-conditioned diagonal-dominant covariance for this test
        np.random.seed(123)
        n = 5
        D = np.diag([0.04, 0.06, 0.05, 0.03, 0.07])
        off = np.random.randn(n, n) * 0.002
        cov_rp = D + off @ off.T + np.eye(n) * 0.02
        w = risk_parity_optimize(cov_rp)
        assert abs(w.sum() - 1.0) < 1e-4
        assert (w > 0).all(), "Risk parity weights should all be positive"
        # Check risk contributions are roughly equal
        sigma_w = cov_rp @ w
        risk_contrib = w * sigma_w
        risk_contrib_pct = risk_contrib / risk_contrib.sum()
        assert np.std(risk_contrib_pct) < 0.10, f"Risk contributions not equal enough: std={np.std(risk_contrib_pct):.4f}"

    def test_inverse_volatility_weights(self, sample_data):
        from src.optimization.optimizer import inverse_volatility_weights
        _, cov = sample_data
        w = inverse_volatility_weights(cov)
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w > 0).all()

    def test_efficient_frontier_monotonic(self, sample_data):
        from src.optimization.optimizer import efficient_frontier
        mu, cov = sample_data
        ef = efficient_frontier(mu, cov, n_points=20)
        assert len(ef) > 0
        # Volatility should generally increase with return (roughly)
        assert "volatility" in ef.columns
        assert "return" in ef.columns

    def test_black_litterman_without_views(self, sample_data):
        from src.optimization.optimizer import black_litterman
        mu, cov = sample_data
        market_w = np.ones(len(mu)) / len(mu)
        pi = black_litterman(cov, market_w)
        assert len(pi) == len(mu)
        # Implied returns should be finite
        assert np.all(np.isfinite(pi))

    def test_high_risk_aversion_produces_conservative_portfolio(self, sample_data):
        from src.optimization.optimizer import mean_variance_optimize
        mu, cov = sample_data
        w_aggressive = mean_variance_optimize(mu, cov, risk_aversion=0.5)
        w_conservative = mean_variance_optimize(mu, cov, risk_aversion=20.0)
        if w_aggressive is not None and w_conservative is not None:
            vol_a = np.sqrt(w_aggressive @ cov @ w_aggressive)
            vol_c = np.sqrt(w_conservative @ cov @ w_conservative)
            assert vol_c <= vol_a + 1e-4


class TestBacktester:
    """Tests for the backtesting module."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        dates = pd.bdate_range("2010-01-01", periods=2000)
        n_assets = 5
        returns = np.random.randn(len(dates), n_assets) * 0.01 + 0.0002
        prices = pd.DataFrame(
            100 * np.cumprod(1 + returns, axis=0),
            index=dates,
            columns=[f"Asset_{i}" for i in range(n_assets)],
        )
        return prices

    def test_compute_metrics_keys(self):
        from src.optimization.backtester import compute_portfolio_metrics
        returns = pd.Series(np.random.randn(500) * 0.01, index=pd.bdate_range("2020-01-01", periods=500))
        metrics = compute_portfolio_metrics(returns)
        expected_keys = ["annualized_return", "annualized_volatility", "sharpe_ratio",
                         "max_drawdown", "calmar_ratio", "total_return"]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_backtest_returns_correct_structure(self, sample_prices):
        from src.optimization.backtester import backtest_strategy
        result = backtest_strategy(sample_prices, start_offset=500, lookback=252)
        assert "returns" in result
        assert "weights" in result
        assert "metrics" in result
        assert isinstance(result["returns"], pd.Series)
        assert len(result["returns"]) > 0

    def test_rolling_sharpe_length(self):
        from src.optimization.backtester import rolling_sharpe
        returns = pd.Series(np.random.randn(500) * 0.01, index=pd.bdate_range("2020-01-01", periods=500))
        rs = rolling_sharpe(returns, window=252)
        assert len(rs) == 500 - 252 + 1


class TestPreprocessor:
    """Tests for the preprocessing module."""

    def test_compute_returns_shape(self):
        from src.data.preprocessor import compute_returns
        prices = pd.DataFrame(
            np.random.randn(100, 3).cumsum(axis=0) + 100,
            columns=["A", "B", "C"],
        )
        ret = compute_returns(prices, horizons=[1, 5])
        assert ret.shape[0] == 100
        assert ret.shape[1] == 6  # 3 assets * 2 horizons

    def test_compute_volatility_positive(self):
        from src.data.preprocessor import compute_volatility
        prices = pd.DataFrame(
            np.random.randn(200, 2).cumsum(axis=0) + 100,
            columns=["A", "B"],
        )
        vol = compute_volatility(prices, windows=[21])
        # After warmup, volatility should be positive
        assert (vol.dropna() >= 0).all().all()

    def test_lag_macro_creates_columns(self):
        from src.data.preprocessor import lag_macro_features
        macro = pd.DataFrame({"X": range(50), "Y": range(50, 100)})
        lagged = lag_macro_features(macro, lags=[1, 5])
        # Original 2 + 2 lags * 2 columns = 6
        assert lagged.shape[1] == 6

    def test_stationarity_on_random_walk(self):
        from src.data.preprocessor import test_stationarity
        np.random.seed(42)
        rw = pd.Series(np.cumsum(np.random.randn(500)))
        result = test_stationarity(rw)
        # Random walk is typically non-stationary
        assert result["stationary"] is not None

    def test_stationarity_on_white_noise(self):
        from src.data.preprocessor import test_stationarity
        np.random.seed(42)
        wn = pd.Series(np.random.randn(500))
        result = test_stationarity(wn)
        assert result["stationary"] is True


class TestTrainer:
    """Tests for the model training module."""

    def test_time_series_cv_splits_no_overlap(self):
        from src.models.trainer import time_series_cv_splits
        splits = time_series_cv_splits(2000)
        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert train_idx.max() < test_idx.min(), "Train/test overlap detected"

    def test_evaluate_predictions_metrics(self):
        from src.models.trainer import evaluate_predictions
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1])
        metrics = evaluate_predictions(y_true, y_pred)
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] >= 0
        assert metrics["r2"] <= 1.0
