"""Portfolio optimization and backtesting engine."""

from src.optimization.optimizer import (
    estimate_covariance,
    mean_variance_optimize,
    max_sharpe_optimize,
    minimum_variance_optimize,
    risk_parity_optimize,
    black_litterman,
    inverse_volatility_weights,
    efficient_frontier,
)
from src.optimization.backtester import (
    compute_portfolio_metrics,
    backtest_strategy,
    benchmark_equal_weight,
    benchmark_inverse_vol,
    stress_test,
    compare_strategies,
    rolling_sharpe,
)

__all__ = [
    "estimate_covariance",
    "mean_variance_optimize",
    "max_sharpe_optimize",
    "minimum_variance_optimize",
    "risk_parity_optimize",
    "black_litterman",
    "inverse_volatility_weights",
    "efficient_frontier",
    "compute_portfolio_metrics",
    "backtest_strategy",
    "benchmark_equal_weight",
    "benchmark_inverse_vol",
    "stress_test",
    "compare_strategies",
    "rolling_sharpe",
]
