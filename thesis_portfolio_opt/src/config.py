"""
Central configuration for the thesis portfolio optimization project.

This is the single source of truth for all parameters. Every configurable
value — tickers, dates, model hyperparameters, optimization constraints,
visualization defaults — lives here. No magic numbers in the codebase.

Usage:
    from src.config import TICKER_LIST, START_DATE, RANDOM_STATE
"""

from pathlib import Path
from datetime import datetime

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"

# Ensure directories exist at import time
for _dir in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATE RANGE
# ============================================================================
START_DATE = "2005-01-01"
END_DATE = "2024-12-31"
TRAIN_END_DATE = "2021-12-31"       # cutoff for in-sample training
OOS_START_DATE = "2022-01-01"       # out-of-sample evaluation begins

# ============================================================================
# ASSET UNIVERSE — PRIMARY (Multi-Asset ETFs)
# ============================================================================
TICKERS = {
    "SPY": "US Large Cap Equity",
    "IWM": "US Small Cap Equity",
    "EFA": "International Developed Equity",
    "EEM": "Emerging Markets Equity",
    "AGG": "US Aggregate Bonds",
    "TLT": "US Long-Term Treasuries",
    "LQD": "US Investment Grade Corporate Bonds",
    "HYG": "US High Yield Bonds",
    "GLD": "Gold",
    "VNQ": "US REITs",
    "DBC": "Commodities",
    "TIP": "TIPS (Inflation-Protected)",
}

TICKER_LIST = list(TICKERS.keys())
N_ASSETS = len(TICKER_LIST)

# ============================================================================
# ASSET UNIVERSE — ALTERNATIVES (for robustness checks)
# ============================================================================
TICKERS_SECTOR = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communications",
}

TICKERS_INTERNATIONAL = {
    "EWJ": "Japan",
    "EWG": "Germany",
    "EWU": "United Kingdom",
    "FXI": "China",
    "EWZ": "Brazil",
    "INDA": "India",
    "EWY": "South Korea",
    "EWA": "Australia",
}

# ============================================================================
# ASSET CLASS GROUPINGS
# ============================================================================
ASSET_CLASSES = {
    "Equity": ["SPY", "IWM", "EFA", "EEM"],
    "Fixed Income": ["AGG", "TLT", "LQD", "HYG", "TIP"],
    "Alternatives": ["GLD", "VNQ", "DBC"],
}

ASSET_CLASS_COLORS = {
    "Equity": "#2171b5",
    "Fixed Income": "#238b45",
    "Alternatives": "#d94801",
}

def get_asset_class(ticker: str) -> str:
    """Return the asset class for a given ticker."""
    for cls, tickers in ASSET_CLASSES.items():
        if ticker in tickers:
            return cls
    return "Unknown"

# Max allocation per asset class
MAX_CLASS_WEIGHT = 0.70
MIN_CLASS_WEIGHT = 0.05

# ============================================================================
# FRED MACRO INDICATORS
# ============================================================================
FRED_SERIES = {
    # Interest Rates & Yield Curve
    "DFF": "Federal Funds Rate",
    "DGS2": "2-Year Treasury Yield",
    "DGS10": "10-Year Treasury Yield",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "T10Y3M": "10Y-3M Treasury Spread",
    # Volatility & Risk
    "VIXCLS": "VIX",
    "BAMLH0A0HYM2": "High Yield OAS",
    "BAMLC0A4CBBB": "BBB Corporate Spread",
    # Currency
    "DTWEXBGS": "Trade-Weighted USD Index",
    # Consumer & Sentiment
    "UMCSENT": "Consumer Sentiment (UMich)",
    # Labor Market
    "UNRATE": "Unemployment Rate",
    "ICSA": "Initial Jobless Claims",
    # Inflation
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "T10YIE": "10Y Breakeven Inflation",
    "PPIACO": "PPI (All Commodities)",
    # Money Supply & Credit
    "M2SL": "M2 Money Supply",
    # Leading Indicators
    "USSLIND": "Leading Economic Index",
    # Housing
    "HOUST": "Housing Starts",
}

FRED_SERIES_LIST = list(FRED_SERIES.keys())
N_MACRO_FEATURES = len(FRED_SERIES_LIST)

# Categorize macro series for analysis
MACRO_CATEGORIES = {
    "Interest Rates": ["DFF", "DGS2", "DGS10", "T10Y2Y", "T10Y3M"],
    "Risk Spreads": ["VIXCLS", "BAMLH0A0HYM2", "BAMLC0A4CBBB"],
    "Currency": ["DTWEXBGS"],
    "Sentiment & Labor": ["UMCSENT", "UNRATE", "ICSA"],
    "Inflation": ["CPIAUCSL", "T10YIE", "PPIACO"],
    "Liquidity": ["M2SL"],
    "Leading": ["USSLIND"],
    "Housing": ["HOUST"],
}

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
RETURN_HORIZONS = [1, 5, 21, 63]           # daily, weekly, monthly, quarterly
VOLATILITY_WINDOWS = [21, 63, 126, 252]    # 1m, 3m, 6m, 12m
MOMENTUM_WINDOWS = [21, 63, 126, 252]      # 1m, 3m, 6m, 12m
MACRO_LAGS = [1, 5, 21]                    # lag macro features by these periods

# Technical indicators
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2.0
ATR_WINDOW = 14

# Cross-asset features
ROLLING_BETA_WINDOW = 63        # rolling beta vs market (SPY)
ROLLING_CORR_WINDOW = 63        # rolling pairwise correlation
DISPERSION_WINDOW = 21          # cross-sectional return dispersion

# PCA / Dimensionality reduction
PCA_VARIANCE_THRESHOLD = 0.95   # keep components explaining 95% variance
MAX_PCA_COMPONENTS = 20

# VIF threshold for multicollinearity removal
VIF_THRESHOLD = 10.0

# ============================================================================
# PREDICTION TARGET
# ============================================================================
PREDICTION_HORIZON = 21  # predict 21-day (monthly) forward returns
TARGET_COLUMN_TEMPLATE = "{ticker}_ret_{horizon}d"

# Alternative horizons for multi-horizon prediction
PREDICTION_HORIZONS = [5, 21, 63]   # weekly, monthly, quarterly

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
# Random seed (used everywhere for reproducibility)
RANDOM_STATE = 42

# Time-series cross-validation — expanding window
CV_N_SPLITS = 5
CV_TRAIN_WINDOW = 252 * 3          # 3 years of trading days
CV_TEST_WINDOW = 63                # ~1 quarter
CV_GAP = 5                         # gap between train and test to avoid look-ahead

# Walk-forward configuration
WF_TRAIN_WINDOW = 252 * 3          # 3 years
WF_TEST_WINDOW = 21                # 1 month
WF_STEP = 21                       # move 1 month forward each step

# ============================================================================
# HYPERPARAMETER GRIDS (for grid search / tuning)
# ============================================================================
PARAM_GRIDS = {
    "ridge": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    },
    "lasso": {
        "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1],
    },
    "elastic_net": {
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "l1_ratio": [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9],
    },
    "random_forest": {
        "n_estimators": [100, 200, 500, 1000],
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200, 500, 1000],
        "max_depth": [2, 3, 5, 7],
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "min_samples_leaf": [1, 5, 10],
    },
    "xgboost": {
        "n_estimators": [100, 200, 500, 1000],
        "max_depth": [2, 3, 5, 7],
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
        "gamma": [0, 0.1, 0.5],
    },
    "lightgbm": {
        "n_estimators": [100, 200, 500, 1000],
        "max_depth": [3, 5, 8, -1],
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.5, 0.7, 0.9],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [0, 1.0, 10.0],
        "min_child_samples": [5, 10, 20, 50],
    },
    "svr": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "epsilon": [0.001, 0.01, 0.05, 0.1, 0.5],
        "kernel": ["rbf", "linear", "poly"],
        "gamma": ["scale", "auto"],
    },
    "mlp": {
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
        "learning_rate_init": [0.001, 0.01],
        "alpha": [0.0001, 0.001, 0.01],
        "batch_size": [32, 64, 128],
        "max_iter": [500, 1000],
    },
    "adaboost": {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "loss": ["linear", "square", "exponential"],
    },
}

# Default model parameters (used when not tuning)
DEFAULT_MODEL_PARAMS = {
    "ridge": {"alpha": 1.0},
    "lasso": {"alpha": 0.001},
    "elastic_net": {"alpha": 0.001, "l1_ratio": 0.5},
    "random_forest": {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 5},
    "gradient_boosting": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
    "xgboost": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    "lightgbm": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "num_leaves": 31, "verbose": -1},
    "svr": {"C": 1.0, "epsilon": 0.1, "kernel": "rbf"},
    "mlp": {"hidden_layer_sizes": (128, 64), "max_iter": 1000, "early_stopping": True},
    "adaboost": {"n_estimators": 200, "learning_rate": 0.1},
}

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================
RISK_FREE_RATE = 0.0                # annualized, adjust as needed
MIN_WEIGHT = 0.0                    # no short selling
MAX_WEIGHT = 0.40                   # max 40% in any single asset
RISK_AVERSION_RANGE = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# Turnover constraint
MAX_TURNOVER = 0.50                 # max 50% portfolio turnover per rebalance

# CVaR optimization
CVAR_CONFIDENCE = 0.95              # 95% CVaR
CVAR_N_SCENARIOS = 10000            # Monte Carlo scenarios for CVaR

# Robust optimization
ROBUST_EPSILON_MU = 0.05            # uncertainty radius for expected returns
ROBUST_EPSILON_COV = 0.10           # uncertainty radius for covariance

# Black-Litterman defaults
BL_TAU = 0.05                       # prior uncertainty scalar
BL_RISK_AVERSION = 2.5              # market risk aversion

# Covariance estimation
COV_METHODS = ["sample", "shrinkage", "ewma", "min_cov_det"]
COV_DEFAULT_METHOD = "shrinkage"
EWMA_HALFLIFE = 63                  # exponentially weighted half-life (3 months)
EWMA_MIN_PERIODS = 63

# ============================================================================
# BACKTESTING
# ============================================================================
BACKTEST_REBALANCE_FREQ = 21        # trading days (~monthly)
BACKTEST_LOOKBACK = 252             # 1 year of data for estimation
BACKTEST_START_OFFSET = 252 * 3     # start backtest after 3 years of data
TRANSACTION_COST_BPS = 10           # 10 basis points per trade

# Alternative transaction cost models
TC_FIXED_BPS = 5                    # fixed cost component
TC_SPREAD_BPS = 3                   # bid-ask spread component
TC_IMPACT_COEFFICIENT = 0.1         # market impact coefficient (square root model)

# Rebalancing frequencies to compare
REBALANCE_FREQUENCIES = {
    "daily": 1,
    "weekly": 5,
    "biweekly": 10,
    "monthly": 21,
    "quarterly": 63,
    "semiannual": 126,
    "annual": 252,
}

# ============================================================================
# STRESS TEST SCENARIOS — (start_date, end_date) tuples
# ============================================================================
STRESS_SCENARIOS = {
    "gfc_2008": ("2007-10-01", "2009-03-31"),
    "euro_crisis_2011": ("2011-07-01", "2012-01-31"),
    "taper_tantrum_2013": ("2013-05-01", "2013-09-30"),
    "china_deval_2015": ("2015-08-01", "2016-02-29"),
    "volmageddon_2018": ("2018-01-26", "2018-04-06"),
    "q4_selloff_2018": ("2018-10-01", "2018-12-31"),
    "covid_crash_2020": ("2020-02-19", "2020-03-23"),
    "covid_recovery_2020": ("2020-03-23", "2020-08-31"),
    "rate_hikes_2022": ("2022-01-01", "2022-10-12"),
    "svb_crisis_2023": ("2023-03-08", "2023-03-31"),
}

def get_stress_dates(scenario: str) -> tuple[str, str]:
    """Return (start_date, end_date) for a named stress scenario."""
    if scenario not in STRESS_SCENARIOS:
        raise KeyError(f"Unknown stress scenario: {scenario}")
    return STRESS_SCENARIOS[scenario]

# Extended metadata for each scenario
STRESS_SCENARIO_META = {
    "gfc_2008": {"description": "Global Financial Crisis", "category": "systemic"},
    "euro_crisis_2011": {"description": "European Sovereign Debt Crisis", "category": "regional"},
    "taper_tantrum_2013": {"description": "Fed Taper Tantrum", "category": "monetary_policy"},
    "china_deval_2015": {"description": "China Devaluation & Global Selloff", "category": "regional"},
    "volmageddon_2018": {"description": "Volmageddon — VIX Spike", "category": "volatility"},
    "q4_selloff_2018": {"description": "Q4 2018 Rate Hike Selloff", "category": "monetary_policy"},
    "covid_crash_2020": {"description": "COVID-19 Market Crash", "category": "systemic"},
    "covid_recovery_2020": {"description": "COVID-19 Recovery Rally", "category": "recovery"},
    "rate_hikes_2022": {"description": "Fed Aggressive Rate Hikes", "category": "monetary_policy"},
    "svb_crisis_2023": {"description": "SVB & Regional Banking Crisis", "category": "banking"},
}


# ============================================================================
# REGIME DETECTION
# ============================================================================
REGIME_N_STATES = 2                 # 2-state HMM (bull/bear) or 3-state
REGIME_VIX_THRESHOLD = 20.0         # VIX above this = high-volatility regime
REGIME_VIX_EXTREME = 30.0           # VIX above this = crisis regime
REGIME_LOOKBACK = 252               # lookback for regime classification
REGIME_SMOOTHING_WINDOW = 21        # smooth regime signals

# Volatility regime thresholds (percentiles of realized vol)
VOL_REGIME_LOW = 0.25               # below 25th percentile = low vol
VOL_REGIME_HIGH = 0.75              # above 75th percentile = high vol

# ============================================================================
# VISUALIZATION
# ============================================================================
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"
COLOR_PALETTE = "deep"
FIGURE_SIZE = (12, 6)
FIGURE_SIZE_WIDE = (16, 6)
FIGURE_SIZE_TALL = (12, 10)
FIGURE_SIZE_SQUARE = (10, 10)

# Color scheme for strategies
STRATEGY_COLORS = {
    "Mean-Variance": "#1f77b4",
    "Max Sharpe": "#ff7f0e",
    "Min Variance": "#2ca02c",
    "Risk Parity": "#d62728",
    "Black-Litterman": "#9467bd",
    "Inv. Volatility": "#8c564b",
    "Equal Weight": "#7f7f7f",
    "ML-Enhanced": "#e377c2",
    "CVaR": "#bcbd22",
    "HRP": "#17becf",
}

# LaTeX-friendly font settings
LATEX_FONT_SETTINGS = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": False,           # set True if LaTeX is installed
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}

# ============================================================================
# REPORTING & EXPORT
# ============================================================================
REPORT_FLOAT_FORMAT = "%.4f"
REPORT_PCT_FORMAT = "%.2f%%"
EXCEL_ENGINE = "openpyxl"

# Metrics to include in summary tables
SUMMARY_METRICS = [
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "max_drawdown_duration",
    "calmar_ratio",
    "information_ratio",
    "total_return",
    "var_95",
    "cvar_95",
    "avg_turnover",
    "tracking_error",
]

# ============================================================================
# BENCHMARKS
# ============================================================================
BENCHMARK_TICKER = "SPY"            # primary benchmark
BENCHMARK_RISK_FREE = "^IRX"        # 3-month T-bill for Sharpe

BENCHMARK_STRATEGIES = [
    "equal_weight",
    "inverse_volatility",
    "market_cap",                    # requires market cap data
    "sixty_forty",                   # 60% SPY, 40% AGG
]

# 60/40 static allocation
SIXTY_FORTY_WEIGHTS = {
    "SPY": 0.60,
    "AGG": 0.40,
}

# ============================================================================
# DATA FETCHING
# ============================================================================
FETCH_MAX_RETRIES = 3
FETCH_RETRY_DELAY = 5               # seconds between retries
FETCH_TIMEOUT = 30                  # request timeout in seconds
FETCH_RATE_LIMIT_DELAY = 0.5        # seconds between FRED API calls

# YFinance specific
YF_AUTO_ADJUST = True
YF_THREADS = True

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
NUMPY_SEED = RANDOM_STATE
PYTHON_HASH_SEED = str(RANDOM_STATE)

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
