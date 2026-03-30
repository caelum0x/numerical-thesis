"""
ML model training, evaluation, and diagnostics for portfolio return prediction.

This module implements the full machine-learning pipeline for an Industrial
Engineering thesis on ML-enhanced portfolio optimisation.  It covers:

    1. Model Registry        -- unified creation of 12 regression models
    2. Cross-Validation      -- expanding-window, walk-forward, purged CV
    3. Evaluation Metrics    -- MSE, RMSE, MAE, R2, MAPE, directional accuracy,
                                information coefficient, Diebold-Mariano test
    4. Feature Importance    -- native, permutation, SHAP
    5. Training Functions    -- single model, hyper-parameter tuning, stacking
    6. Model Persistence     -- save / load / list / registry
    7. Model Diagnostics     -- residuals, learning curves, calibration

All tuneable constants are imported from ``src.config`` so that there are no
magic numbers in this file.

Author  : Arhan Subasi
Created : 2025
"""

from __future__ import annotations

import json
import logging
import pickle
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import BaseCrossValidator, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.config import (
    CV_N_SPLITS,
    CV_TRAIN_WINDOW,
    CV_TEST_WINDOW,
    CV_GAP,
    WF_TRAIN_WINDOW,
    WF_TEST_WINDOW,
    WF_STEP,
    RANDOM_STATE,
    RESULTS_DIR,
    MODELS_DIR,
    DEFAULT_MODEL_PARAMS,
    PARAM_GRIDS,
)

logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 1 -- MODEL REGISTRY
# ============================================================================

# ---------------------------------------------------------------------------
# Model factory dictionary
# ---------------------------------------------------------------------------
# Each value is a *callable* that returns an unfitted sklearn-compatible
# estimator.  The default hyper-parameters come from ``config.DEFAULT_MODEL_PARAMS``
# so they can be changed in one place without touching this file.
# ---------------------------------------------------------------------------

MODELS: Dict[str, type] = {
    "ridge": Ridge,
    "lasso": Lasso,
    "elastic_net": ElasticNet,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "xgboost": XGBRegressor,
    "lightgbm": LGBMRegressor,
    "svr": SVR,
    "mlp": MLPRegressor,
    "adaboost": AdaBoostRegressor,
    "bagging": BaggingRegressor,
    "stacking": StackingRegressor,
}


def _default_params(name: str) -> dict:
    """Return default parameters for *name*, injecting RANDOM_STATE where
    applicable and adding sensible defaults for models not in
    ``DEFAULT_MODEL_PARAMS``."""

    params = deepcopy(DEFAULT_MODEL_PARAMS.get(name, {}))

    # Inject random_state for models that accept it
    rs_models = {
        "random_forest", "gradient_boosting", "xgboost", "lightgbm",
        "mlp", "adaboost", "bagging",
    }
    if name in rs_models and "random_state" not in params:
        params["random_state"] = RANDOM_STATE

    # Model-specific defaults not in config ---------------------------------
    if name == "bagging" and not params:
        params.update({
            "estimator": DecisionTreeRegressor(max_depth=5),
            "n_estimators": 200,
            "max_samples": 0.8,
            "max_features": 0.8,
            "random_state": RANDOM_STATE,
        })

    if name == "stacking" and not params:
        # A minimal stacking config; callers should override via
        # train_stacking_ensemble.
        params.update({
            "estimators": [
                ("ridge", Ridge(alpha=1.0)),
                ("rf", RandomForestRegressor(
                    n_estimators=200, max_depth=5, random_state=RANDOM_STATE)),
            ],
            "final_estimator": Ridge(alpha=1.0),
            "cv": 3,
        })

    if name == "xgboost":
        params.setdefault("verbosity", 0)
    if name == "lightgbm":
        params.setdefault("verbose", -1)

    return params


def create_model(
    name: str,
    params: Optional[Dict[str, Any]] = None,
) -> BaseEstimator:
    """Instantiate a model by *name* with optional custom parameters.

    Parameters
    ----------
    name : str
        Key in ``MODELS`` (e.g. ``"xgboost"``).
    params : dict, optional
        If given, these are passed directly to the constructor and override
        every default.  If ``None``, ``DEFAULT_MODEL_PARAMS`` from
        ``src.config`` is used (plus sensible extras such as random_state).

    Returns
    -------
    sklearn-compatible estimator (unfitted).
    """
    if name not in MODELS:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(MODELS.keys())}"
        )

    cls = MODELS[name]
    if params is not None:
        used_params = deepcopy(params)
    else:
        used_params = _default_params(name)

    try:
        model = cls(**used_params)
    except TypeError as exc:
        # Graceful fallback: if unexpected kwargs, log and try without them
        logger.warning("Could not create %s with params %s: %s", name, used_params, exc)
        model = cls()

    return model


# ============================================================================
# SECTION 2 -- CROSS-VALIDATION SCHEMES
# ============================================================================

def time_series_cv_splits(
    n_samples: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window time-series cross-validation.

    The training window grows by ``CV_TEST_WINDOW`` observations each fold
    while the test window is always ``CV_TEST_WINDOW`` observations, with a
    gap of ``CV_GAP`` between train and test to prevent look-ahead bias.

    Parameters
    ----------
    n_samples : int
        Total number of observations in the dataset.

    Returns
    -------
    list of (train_indices, test_indices) tuples.
    """
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(CV_N_SPLITS):
        train_end = CV_TRAIN_WINDOW + i * CV_TEST_WINDOW
        test_start = train_end + CV_GAP
        test_end = test_start + CV_TEST_WINDOW

        if test_end > n_samples:
            break

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    if len(splits) == 0:
        logger.warning(
            "No valid expanding-window splits for n_samples=%d. "
            "Consider reducing CV_TRAIN_WINDOW (%d) or CV_TEST_WINDOW (%d).",
            n_samples, CV_TRAIN_WINDOW, CV_TEST_WINDOW,
        )

    return splits


def walk_forward_splits(
    n_samples: int,
    train_window: int = WF_TRAIN_WINDOW,
    test_window: int = WF_TEST_WINDOW,
    step: int = WF_STEP,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Walk-forward (sliding-window) cross-validation.

    Unlike the expanding-window scheme, the training window is *fixed* at
    ``train_window`` and slides forward by ``step`` observations each fold.

    Parameters
    ----------
    n_samples : int
        Total number of observations.
    train_window : int
        Fixed number of training observations per fold.
    test_window : int
        Number of test observations per fold.
    step : int
        Number of observations by which the window advances.

    Returns
    -------
    list of (train_indices, test_indices) tuples.
    """
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0

    while True:
        train_start = start
        train_end = train_start + train_window
        test_start = train_end + CV_GAP  # respect gap even in walk-forward
        test_end = test_start + test_window

        if test_end > n_samples:
            break

        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

        start += step

    if len(splits) == 0:
        logger.warning(
            "No valid walk-forward splits for n_samples=%d with "
            "train_window=%d, test_window=%d, step=%d.",
            n_samples, train_window, test_window, step,
        )

    return splits


def purged_cv_splits(
    n_samples: int,
    n_splits: int = CV_N_SPLITS,
    embargo: int = CV_GAP,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Purged group time-series cross-validation with an embargo period.

    This implementation follows de Prado's *purged k-fold* methodology:
    - Data is divided into *n_splits* contiguous groups.
    - For each fold the test group is one of these blocks.
    - Training data consists of all groups that are *before* the test group
      (respecting temporal order).
    - An *embargo* of length ``embargo`` observations is removed from the end
      of the training data to prevent information leakage from overlapping
      labels.

    Parameters
    ----------
    n_samples : int
        Total observations.
    n_splits : int
        Number of folds.
    embargo : int
        Number of observations to drop between training and test.

    Returns
    -------
    list of (train_indices, test_indices) tuples.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    indices = np.arange(n_samples)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    fold_boundaries = np.cumsum(fold_sizes)
    fold_boundaries = np.insert(fold_boundaries, 0, 0)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for i in range(1, n_splits):
        test_start = fold_boundaries[i]
        test_end = fold_boundaries[i + 1] if i + 1 <= n_splits else n_samples
        test_idx = indices[test_start:test_end]

        # Training: everything strictly before the test group minus embargo
        train_end = max(0, test_start - embargo)
        if train_end <= 0:
            continue
        train_idx = indices[:train_end]

        splits.append((train_idx, test_idx))

    return splits


class BlockingTimeSeriesSplit(BaseCrossValidator):
    """Sklearn-compatible time-series splitter with blocking (non-overlapping)
    train-test windows.

    This splitter divides the data into *n_splits + 1* contiguous blocks.
    Each fold uses one block as test and all *preceding* blocks as training
    data, with an optional gap (``embargo``).

    Compatible with ``sklearn.model_selection.cross_val_score`` and
    ``RandomizedSearchCV``.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    train_window : int or None
        If given, use a fixed training window.  Otherwise expanding window.
    gap : int
        Gap between the end of training and start of testing.
    """

    def __init__(
        self,
        n_splits: int = CV_N_SPLITS,
        train_window: Optional[int] = None,
        gap: int = CV_GAP,
    ):
        self.n_splits = n_splits
        self.train_window = train_window
        self.gap = gap

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
    ):
        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        indices = np.arange(n_samples)

        # Determine block boundaries
        n_blocks = self.n_splits + 1
        block_size = n_samples // n_blocks
        boundaries = [i * block_size for i in range(n_blocks)]
        boundaries.append(n_samples)

        for i in range(1, n_blocks):
            test_start = boundaries[i]
            test_end = boundaries[i + 1] if i + 1 < len(boundaries) else n_samples

            if self.train_window is not None:
                train_start = max(0, test_start - self.gap - self.train_window)
            else:
                train_start = 0

            train_end = max(0, test_start - self.gap)

            if train_end <= train_start or test_end <= test_start:
                continue

            yield indices[train_start:train_end], indices[test_start:test_end]


def _resolve_cv_splits(
    n_samples: int,
    cv_method: str = "expanding",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Internal helper to resolve a CV method name to actual splits."""
    if cv_method == "expanding":
        return time_series_cv_splits(n_samples)
    elif cv_method == "walk_forward":
        return walk_forward_splits(n_samples)
    elif cv_method == "purged":
        return purged_cv_splits(n_samples)
    elif cv_method == "blocking":
        splitter = BlockingTimeSeriesSplit()
        dummy = np.zeros(n_samples)
        return list(splitter.split(dummy))
    else:
        raise ValueError(
            f"Unknown cv_method '{cv_method}'. "
            f"Use 'expanding', 'walk_forward', 'purged', or 'blocking'."
        )


# ============================================================================
# SECTION 3 -- EVALUATION METRICS
# ============================================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute a comprehensive set of regression evaluation metrics.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted values.

    Returns
    -------
    dict with keys:
        mse, rmse, mae, r2, mape, directional_accuracy, ic
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE -- guard against zeros in y_true
    nonzero = y_true != 0.0
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])))
    else:
        mape = np.nan

    da = evaluate_directional_accuracy(y_true, y_pred)
    ic = compute_information_coefficient(y_true, y_pred)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
        "directional_accuracy": float(da),
        "ic": float(ic),
    }


def evaluate_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Fraction of observations where predicted and actual signs agree.

    Parameters
    ----------
    y_true, y_pred : array-like

    Returns
    -------
    float in [0, 1].  Returns ``nan`` if arrays are empty.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if len(y_true) == 0:
        return np.nan

    correct = np.sign(y_true) == np.sign(y_pred)
    return float(correct.mean())


def compute_information_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Rank information coefficient (Spearman correlation).

    This is the standard measure used in quantitative finance to assess the
    quality of cross-sectional return forecasts.

    Parameters
    ----------
    y_true, y_pred : array-like

    Returns
    -------
    float : Spearman rank correlation.  ``nan`` if computation fails.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if len(y_true) < 3:
        return np.nan

    try:
        corr, _ = scipy_stats.spearmanr(y_true, y_pred)
        return float(corr)
    except Exception:
        return np.nan


def dm_test(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    h: int = 1,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Diebold-Mariano test for comparing two forecast models.

    Tests the null hypothesis that the two forecasts have equal predictive
    accuracy (measured by squared-error loss).

    Parameters
    ----------
    y_true : array-like
        Actual values.
    pred1, pred2 : array-like
        Predictions from model 1 and model 2.
    h : int
        Forecast horizon (used for Newey-West correction).
    alternative : str
        ``'two-sided'``, ``'less'`` (pred1 better), or ``'greater'``
        (pred2 better).

    Returns
    -------
    dict with ``dm_stat`` and ``p_value``.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    pred1 = np.asarray(pred1, dtype=float).ravel()
    pred2 = np.asarray(pred2, dtype=float).ravel()

    e1 = (y_true - pred1) ** 2
    e2 = (y_true - pred2) ** 2
    d = e1 - e2
    n = len(d)

    if n < 2:
        return {"dm_stat": np.nan, "p_value": np.nan}

    d_mean = d.mean()
    d_var = d.var(ddof=1)

    # Newey-West style correction for serial correlation
    for lag in range(1, h):
        gamma = np.cov(d[lag:], d[:-lag])[0, 1] if n > lag else 0.0
        d_var += 2.0 * gamma

    d_var = max(d_var, 1e-15)  # numerical guard

    dm_stat = d_mean / np.sqrt(d_var / n)

    if alternative == "two-sided":
        p_value = 2.0 * scipy_stats.t.sf(np.abs(dm_stat), df=n - 1)
    elif alternative == "less":
        p_value = scipy_stats.t.cdf(dm_stat, df=n - 1)
    elif alternative == "greater":
        p_value = scipy_stats.t.sf(dm_stat, df=n - 1)
    else:
        raise ValueError(f"Unknown alternative '{alternative}'")

    return {"dm_stat": float(dm_stat), "p_value": float(p_value)}


# ============================================================================
# SECTION 4 -- FEATURE IMPORTANCE
# ============================================================================

def get_feature_importance(
    model: BaseEstimator,
    feature_names: List[str],
    model_name: str,
) -> pd.DataFrame:
    """Extract feature importance from a fitted model.

    Handles tree-based (``feature_importances_``), linear (``coef_``), and
    unsupported model types gracefully.

    Parameters
    ----------
    model : fitted sklearn estimator
    feature_names : list of str
    model_name : str  (used only for logging)

    Returns
    -------
    pd.DataFrame with columns ``feature``, ``importance``, ``importance_pct``,
    sorted descending.
    """
    importance: Optional[np.ndarray] = None

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        if coef.shape[0] == len(feature_names):
            importance = np.abs(coef)
        else:
            logger.warning(
                "Coefficient shape mismatch for %s: coef=%s, features=%d",
                model_name, coef.shape, len(feature_names),
            )
    else:
        logger.info(
            "Model '%s' does not expose feature importances.", model_name
        )

    if importance is None or len(importance) != len(feature_names):
        return pd.DataFrame({
            "feature": feature_names,
            "importance": [np.nan] * len(feature_names),
            "importance_pct": [np.nan] * len(feature_names),
        })

    fi = pd.DataFrame({"feature": feature_names, "importance": importance})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    total = fi["importance"].sum()
    fi["importance_pct"] = (
        fi["importance"] / total * 100.0 if total > 0 else np.nan
    )
    return fi


def permutation_importance_custom(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    n_repeats: int = 10,
    scoring_fn=None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compute permutation importance by shuffling each feature and measuring
    the change in prediction error.

    Parameters
    ----------
    model : fitted estimator
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    n_repeats : int
        Number of times to permute each feature.
    scoring_fn : callable or None
        If ``None``, uses negative MSE.  Must accept (y_true, y_pred) and
        return a scalar where *higher is better*.
    random_state : int

    Returns
    -------
    pd.DataFrame with columns ``feature``, ``importance_mean``,
    ``importance_std``.
    """
    rng = np.random.RandomState(random_state)
    X_arr = np.asarray(X)
    y_arr = np.asarray(y).ravel()

    if scoring_fn is None:
        def scoring_fn(yt, yp):
            return -mean_squared_error(yt, yp)

    baseline_score = scoring_fn(y_arr, model.predict(X_arr))

    feature_names = (
        list(X.columns) if isinstance(X, pd.DataFrame)
        else [f"feature_{i}" for i in range(X_arr.shape[1])]
    )

    results = []
    for j in range(X_arr.shape[1]):
        scores = np.empty(n_repeats)
        saved_col = X_arr[:, j].copy()

        for r in range(n_repeats):
            X_arr[:, j] = rng.permutation(saved_col)
            perm_score = scoring_fn(y_arr, model.predict(X_arr))
            scores[r] = baseline_score - perm_score

        X_arr[:, j] = saved_col  # restore

        results.append({
            "feature": feature_names[j],
            "importance_mean": float(scores.mean()),
            "importance_std": float(scores.std()),
        })

    df = pd.DataFrame(results).sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)
    return df


def shap_importance(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    model_name: str,
    max_samples: int = 500,
) -> Optional[pd.DataFrame]:
    """Compute SHAP-based feature importance.

    Falls back gracefully if the ``shap`` package is not installed.

    Parameters
    ----------
    model : fitted estimator
    X : array-like of shape (n_samples, n_features)
    model_name : str
        Used to choose the right SHAP explainer (tree vs kernel).
    max_samples : int
        Maximum background samples for KernelExplainer.

    Returns
    -------
    pd.DataFrame or None if shap is unavailable.
    """
    try:
        import shap  # type: ignore
    except ImportError:
        logger.warning(
            "shap package is not installed. Skipping SHAP importance. "
            "Install via: pip install shap"
        )
        return None

    feature_names = (
        list(X.columns) if isinstance(X, pd.DataFrame)
        else [f"feature_{i}" for i in range(X.shape[1])]
    )
    X_arr = np.asarray(X)

    tree_models = {"random_forest", "gradient_boosting", "xgboost", "lightgbm",
                   "adaboost", "bagging"}

    try:
        if model_name in tree_models:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_arr)
        else:
            # KernelExplainer for non-tree models (slower)
            bg = X_arr if X_arr.shape[0] <= max_samples else shap.sample(X_arr, max_samples)
            explainer = shap.KernelExplainer(model.predict, bg)
            shap_values = explainer.shap_values(X_arr, nsamples=100)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature": feature_names,
            "shap_importance": mean_abs_shap,
        })
        df = df.sort_values("shap_importance", ascending=False).reset_index(drop=True)
        total = df["shap_importance"].sum()
        df["shap_importance_pct"] = (
            df["shap_importance"] / total * 100.0 if total > 0 else np.nan
        )
        return df

    except Exception as exc:
        logger.warning("SHAP computation failed for %s: %s", model_name, exc)
        return None


# ============================================================================
# SECTION 5 -- TRAINING FUNCTIONS
# ============================================================================

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "xgboost",
    cv_method: str = "expanding",
    scale: bool = True,
    return_oof_predictions: bool = True,
) -> Dict[str, Any]:
    """Train a single model with time-series cross-validation and return
    comprehensive results.

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
    y : pd.Series of shape (n_samples,)
    model_name : str
        Key in ``MODELS``.
    cv_method : str
        One of ``'expanding'``, ``'walk_forward'``, ``'purged'``, ``'blocking'``.
    scale : bool
        Whether to standardise features with ``StandardScaler``.
    return_oof_predictions : bool
        If True, collect out-of-fold predictions.

    Returns
    -------
    dict with keys:
        model, scaler, cv_metrics (DataFrame), model_name, feature_importance,
        oof_predictions (optional), fold_details.
    """
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {sorted(MODELS.keys())}"
        )

    splits = _resolve_cv_splits(len(X), cv_method)

    if len(splits) == 0:
        raise RuntimeError(
            f"No valid CV splits for cv_method='{cv_method}' with "
            f"n_samples={len(X)}. Check config parameters."
        )

    fold_metrics: List[Dict[str, Any]] = []
    fold_details: List[Dict[str, Any]] = []
    oof_indices: List[np.ndarray] = []
    oof_preds: List[np.ndarray] = []

    scaler = StandardScaler() if scale else None

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if scale:
            X_train_proc = scaler.fit_transform(X_train)
            X_test_proc = scaler.transform(X_test)
        else:
            X_train_proc = X_train.values
            X_test_proc = X_test.values

        model = create_model(model_name)
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)

        metrics = evaluate_predictions(y_test.values, y_pred)
        metrics["fold"] = fold_idx
        metrics["train_size"] = len(train_idx)
        metrics["test_size"] = len(test_idx)
        fold_metrics.append(metrics)

        fold_details.append({
            "fold": fold_idx,
            "train_start": int(train_idx[0]),
            "train_end": int(train_idx[-1]),
            "test_start": int(test_idx[0]),
            "test_end": int(test_idx[-1]),
        })

        if return_oof_predictions:
            oof_indices.append(test_idx)
            oof_preds.append(y_pred)

    # Final model trained on all data
    if scale:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    final_model = create_model(model_name)
    final_model.fit(X_scaled, y)

    # Feature importance from final model
    fi = get_feature_importance(final_model, list(X.columns), model_name)

    result: Dict[str, Any] = {
        "model": final_model,
        "scaler": scaler,
        "cv_metrics": pd.DataFrame(fold_metrics),
        "model_name": model_name,
        "feature_importance": fi,
        "fold_details": fold_details,
        "cv_method": cv_method,
        "n_features": X.shape[1],
        "n_samples": len(X),
        "timestamp": datetime.now().isoformat(),
    }

    if return_oof_predictions and oof_indices:
        all_idx = np.concatenate(oof_indices)
        all_preds = np.concatenate(oof_preds)
        oof_df = pd.DataFrame(
            {"index": all_idx, "prediction": all_preds}
        ).sort_values("index").reset_index(drop=True)
        result["oof_predictions"] = oof_df

    return result


def train_with_hyperparameter_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    param_grid: Optional[Dict[str, Any]] = None,
    n_iter: int = 50,
    cv_method: str = "expanding",
    scoring: str = "neg_mean_squared_error",
    scale: bool = True,
    verbose: int = 0,
) -> Dict[str, Any]:
    """Train with RandomizedSearchCV using time-series splits.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    model_name : str
    param_grid : dict or None
        If None, uses ``PARAM_GRIDS`` from ``src.config``.
    n_iter : int
        Number of parameter settings sampled.
    cv_method : str
        Cross-validation method to use.
    scoring : str
        Sklearn scoring string.
    scale : bool
        Whether to standardise features.
    verbose : int
        Verbosity level for RandomizedSearchCV.

    Returns
    -------
    dict with ``best_model``, ``best_params``, ``cv_results``, etc.
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    if param_grid is None:
        param_grid = PARAM_GRIDS.get(model_name, {})
        if not param_grid:
            logger.warning(
                "No param_grid found for '%s' in PARAM_GRIDS. "
                "Training with defaults.",
                model_name,
            )
            return train_and_evaluate(X, y, model_name, cv_method, scale)

    # Build CV splitter
    splits = _resolve_cv_splits(len(X), cv_method)
    if len(splits) == 0:
        raise RuntimeError(f"No valid splits for {cv_method}")

    # Prepare data
    scaler = StandardScaler() if scale else None
    if scale:
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns, index=X.index,
        )
    else:
        X_scaled = X

    base_model = create_model(model_name)

    # Clip n_iter to the total number of possible combinations
    from sklearn.model_selection import ParameterSampler
    total_candidates = 1
    for values in param_grid.values():
        if isinstance(values, list):
            total_candidates *= len(values)
    n_iter_actual = min(n_iter, total_candidates)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter_actual,
        scoring=scoring,
        cv=splits,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=verbose,
        refit=True,
        return_train_score=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_scaled, y)

    best_model = search.best_estimator_
    fi = get_feature_importance(best_model, list(X.columns), model_name)

    cv_results_df = pd.DataFrame(search.cv_results_).sort_values(
        "rank_test_score"
    )

    return {
        "model": best_model,
        "scaler": scaler,
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "cv_results": cv_results_df,
        "model_name": model_name,
        "feature_importance": fi,
        "n_iter": n_iter_actual,
        "scoring": scoring,
        "cv_method": cv_method,
        "timestamp": datetime.now().isoformat(),
    }


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    model_names: Optional[List[str]] = None,
    cv_method: str = "expanding",
    scale: bool = True,
) -> Dict[str, Any]:
    """Train all (or a subset of) registered models and return a comparative
    summary.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    model_names : list of str or None
        If None, trains everything except ``stacking`` (which requires
        explicit configuration via ``train_stacking_ensemble``).
    cv_method : str
    scale : bool

    Returns
    -------
    dict with:
        models -- dict mapping name -> train_and_evaluate result
        comparison -- pd.DataFrame ranking models by RMSE
        feature_importance -- dict mapping name -> importance DataFrame
    """
    if model_names is None:
        # Exclude stacking by default as it needs special setup
        model_names = [n for n in MODELS if n != "stacking"]

    all_results: Dict[str, Any] = {}
    comparison_rows: List[Dict[str, Any]] = []
    feature_importances: Dict[str, pd.DataFrame] = {}

    for name in model_names:
        logger.info("Training %s ...", name)
        print(f"Training {name}...")
        try:
            result = train_and_evaluate(
                X, y, model_name=name, cv_method=cv_method, scale=scale,
            )
            all_results[name] = result

            cv = result["cv_metrics"]
            row = {
                "model": name,
                "rmse_mean": cv["rmse"].mean(),
                "rmse_std": cv["rmse"].std(),
                "mae_mean": cv["mae"].mean(),
                "mae_std": cv["mae"].std(),
                "r2_mean": cv["r2"].mean(),
                "r2_std": cv["r2"].std(),
                "mape_mean": cv["mape"].mean(),
                "da_mean": cv["directional_accuracy"].mean(),
                "ic_mean": cv["ic"].mean(),
            }
            comparison_rows.append(row)

            fi = result.get("feature_importance")
            if fi is not None:
                feature_importances[name] = fi

        except Exception as exc:
            logger.error("Error training %s: %s", name, exc, exc_info=True)
            print(f"  Error training {name}: {exc}")

    comparison_df = (
        pd.DataFrame(comparison_rows)
        .sort_values("rmse_mean")
        .reset_index(drop=True)
    )

    return {
        "models": all_results,
        "comparison": comparison_df,
        "feature_importance": feature_importances,
    }


def train_stacking_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    base_models: Optional[List[str]] = None,
    meta_model: str = "ridge",
    cv_method: str = "expanding",
    scale: bool = True,
) -> Dict[str, Any]:
    """Train a stacking ensemble with configurable base learners and
    meta-learner.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    base_models : list of str or None
        Names from ``MODELS``.  Defaults to a balanced mix of model types.
    meta_model : str
        Name of the meta-learner.
    cv_method : str
    scale : bool

    Returns
    -------
    dict compatible with ``train_and_evaluate`` output.
    """
    if base_models is None:
        base_models = ["ridge", "random_forest", "xgboost", "lightgbm"]

    # Build estimator list for StackingRegressor
    estimators = []
    for name in base_models:
        est = create_model(name)
        estimators.append((name, est))

    final_est = create_model(meta_model)

    # Use internal CV for stacking's cross_val_predict
    stacking_cv = BlockingTimeSeriesSplit(n_splits=3)

    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=final_est,
        cv=stacking_cv,
        n_jobs=-1,
        passthrough=False,
    )

    # Evaluate the stacking model using our external CV
    splits = _resolve_cv_splits(len(X), cv_method)

    if len(splits) == 0:
        raise RuntimeError("No valid CV splits.")

    scaler = StandardScaler() if scale else None
    fold_metrics: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if scale:
            X_train_proc = scaler.fit_transform(X_train)
            X_test_proc = scaler.transform(X_test)
        else:
            X_train_proc = X_train.values
            X_test_proc = X_test.values

        stacking_clone = clone(stacking_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stacking_clone.fit(X_train_proc, y_train)
        y_pred = stacking_clone.predict(X_test_proc)

        metrics = evaluate_predictions(y_test.values, y_pred)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

    # Final fit on all data
    if scale:
        X_all = scaler.fit_transform(X)
    else:
        X_all = X.values

    final_stacking = clone(stacking_model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_stacking.fit(X_all, y)

    return {
        "model": final_stacking,
        "scaler": scaler,
        "cv_metrics": pd.DataFrame(fold_metrics),
        "model_name": "stacking",
        "base_models": base_models,
        "meta_model": meta_model,
        "cv_method": cv_method,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# SECTION 6 -- MODEL PERSISTENCE
# ============================================================================

def save_model(
    result: Dict[str, Any],
    target_name: str,
) -> Path:
    """Persist a trained model result to disk with metadata.

    The saved artefact includes the model, scaler, CV metrics, parameters,
    and a timestamp for auditability.

    Parameters
    ----------
    result : dict
        Output of ``train_and_evaluate`` or ``train_with_hyperparameter_tuning``.
    target_name : str
        Name of the prediction target (e.g. ``'SPY_ret_21d'``).

    Returns
    -------
    pathlib.Path pointing to the saved file.
    """
    model_name = result.get("model_name", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build metadata
    metadata = {
        "model_name": model_name,
        "target_name": target_name,
        "timestamp": timestamp,
        "n_features": result.get("n_features"),
        "n_samples": result.get("n_samples"),
        "cv_method": result.get("cv_method"),
    }

    # Extract params from fitted model
    model_obj = result.get("model")
    if model_obj is not None and hasattr(model_obj, "get_params"):
        try:
            metadata["params"] = model_obj.get_params()
        except Exception:
            metadata["params"] = {}

    # Aggregate metrics
    cv_metrics = result.get("cv_metrics")
    if cv_metrics is not None and isinstance(cv_metrics, pd.DataFrame):
        summary = {}
        for col in cv_metrics.columns:
            if col == "fold":
                continue
            try:
                summary[f"{col}_mean"] = float(cv_metrics[col].mean())
                summary[f"{col}_std"] = float(cv_metrics[col].std())
            except Exception:
                pass
        metadata["metrics_summary"] = summary

    payload = {
        "model": result.get("model"),
        "scaler": result.get("scaler"),
        "cv_metrics": cv_metrics,
        "feature_importance": result.get("feature_importance"),
        "metadata": metadata,
    }

    # Save with best_params if available
    if "best_params" in result:
        payload["best_params"] = result["best_params"]

    # Ensure directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    path = MODELS_DIR / f"model_{model_name}_{target_name}_{timestamp}.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Also save a JSON sidecar for quick inspection without unpickling
    json_path = path.with_suffix(".json")
    json_metadata = {k: v for k, v in metadata.items() if _is_json_serialisable(k, v)}
    try:
        with open(json_path, "w") as jf:
            json.dump(json_metadata, jf, indent=2, default=str)
    except Exception:
        pass  # non-critical

    logger.info("Saved model to %s", path)
    print(f"Saved model to {path}")
    return path


def _is_json_serialisable(key: str, value: Any) -> bool:
    """Quick check so we don't blow up the JSON sidecar."""
    try:
        json.dumps({key: value}, default=str)
        return True
    except (TypeError, ValueError):
        return False


def load_model_result(
    model_name: str,
    target_name: str,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a persisted model result from disk.

    If ``timestamp`` is None, loads the most recent save for the given
    model / target combination.

    Parameters
    ----------
    model_name : str
    target_name : str
    timestamp : str or None

    Returns
    -------
    dict  (same structure as saved by ``save_model``).

    Raises
    ------
    FileNotFoundError if no matching file is found.
    """
    pattern = f"model_{model_name}_{target_name}_*.pkl"
    candidates = sorted(MODELS_DIR.glob(pattern))

    if timestamp is not None:
        candidates = [c for c in candidates if timestamp in c.stem]

    if not candidates:
        raise FileNotFoundError(
            f"No saved model found for model_name='{model_name}', "
            f"target_name='{target_name}' in {MODELS_DIR}"
        )

    # Pick the latest (last in sorted order thanks to timestamp format)
    path = candidates[-1]
    logger.info("Loading model from %s", path)

    with open(path, "rb") as f:
        payload = pickle.load(f)

    # Validate structure
    expected_keys = {"model", "metadata"}
    if not expected_keys.issubset(payload.keys()):
        logger.warning(
            "Loaded payload from %s is missing expected keys: %s",
            path,
            expected_keys - payload.keys(),
        )

    return payload


def list_saved_models() -> pd.DataFrame:
    """List all persisted model artefacts.

    Returns
    -------
    pd.DataFrame with columns:
        file, model_name, target_name, timestamp, size_mb.
    """
    rows: List[Dict[str, Any]] = []
    for pkl_path in sorted(MODELS_DIR.glob("model_*.pkl")):
        parts = pkl_path.stem.split("_", 2)  # model_<name>_<rest>
        name_rest = pkl_path.stem[len("model_"):]  # drop 'model_'
        size_mb = pkl_path.stat().st_size / (1024 * 1024)

        # Try to parse the JSON sidecar for richer info
        json_path = pkl_path.with_suffix(".json")
        meta: Dict[str, Any] = {}
        if json_path.exists():
            try:
                with open(json_path) as jf:
                    meta = json.load(jf)
            except Exception:
                pass

        rows.append({
            "file": pkl_path.name,
            "model_name": meta.get("model_name", name_rest),
            "target_name": meta.get("target_name", ""),
            "timestamp": meta.get("timestamp", ""),
            "size_mb": round(size_mb, 3),
        })

    return pd.DataFrame(rows)


class ModelRegistry:
    """Manage all trained models, compare versions, and retrieve the best.

    This is a thin convenience layer that wraps the module-level persistence
    functions and keeps an in-memory cache of loaded results.

    Usage
    -----
    >>> registry = ModelRegistry()
    >>> registry.register(result, target_name="SPY_ret_21d")
    >>> best = registry.get_best("SPY_ret_21d", metric="rmse_mean")
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    def register(
        self,
        result: Dict[str, Any],
        target_name: str,
    ) -> Path:
        """Save a result and add it to the in-memory cache."""
        path = save_model(result, target_name)
        key = path.stem
        self._cache[key] = result
        return path

    # ------------------------------------------------------------------
    def load(
        self,
        model_name: str,
        target_name: str,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load a model (with caching)."""
        cache_key = f"{model_name}_{target_name}_{timestamp or 'latest'}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        payload = load_model_result(model_name, target_name, timestamp)
        self._cache[cache_key] = payload
        return payload

    # ------------------------------------------------------------------
    def list_models(self) -> pd.DataFrame:
        """Delegate to ``list_saved_models``."""
        return list_saved_models()

    # ------------------------------------------------------------------
    def get_best(
        self,
        target_name: str,
        metric: str = "rmse_mean",
        lower_is_better: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Scan all saved models for *target_name* and return the one with
        the best value of *metric* (reading from JSON sidecars).

        Returns None if no models found.
        """
        best_path: Optional[Path] = None
        best_value: Optional[float] = None

        pattern = f"model_*_{target_name}_*.json"
        for json_path in MODELS_DIR.glob(pattern):
            try:
                with open(json_path) as jf:
                    meta = json.load(jf)
            except Exception:
                continue

            val = (meta.get("metrics_summary") or {}).get(metric)
            if val is None:
                continue
            val = float(val)

            if best_value is None:
                best_value = val
                best_path = json_path
            elif lower_is_better and val < best_value:
                best_value = val
                best_path = json_path
            elif not lower_is_better and val > best_value:
                best_value = val
                best_path = json_path

        if best_path is None:
            return None

        pkl_path = best_path.with_suffix(".pkl")
        if not pkl_path.exists():
            return None

        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    def compare_models(
        self,
        target_name: str,
    ) -> pd.DataFrame:
        """Build a comparison DataFrame from all saved models for a target."""
        rows: List[Dict[str, Any]] = []
        pattern = f"model_*_{target_name}_*.json"
        for json_path in sorted(MODELS_DIR.glob(pattern)):
            try:
                with open(json_path) as jf:
                    meta = json.load(jf)
            except Exception:
                continue

            row = {
                "model_name": meta.get("model_name"),
                "timestamp": meta.get("timestamp"),
                "cv_method": meta.get("cv_method"),
            }
            row.update(meta.get("metrics_summary", {}))
            rows.append(row)

        df = pd.DataFrame(rows)
        if "rmse_mean" in df.columns:
            df = df.sort_values("rmse_mean").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    def clear_cache(self):
        """Release all cached payloads from memory."""
        self._cache.clear()

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"ModelRegistry(cached={len(self._cache)}, "
            f"saved_on_disk={len(list_saved_models())})"
        )


# ============================================================================
# SECTION 7 -- MODEL DIAGNOSTICS
# ============================================================================

def residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_lags: int = 20,
) -> Dict[str, Any]:
    """Compute comprehensive residual diagnostics.

    Parameters
    ----------
    y_true, y_pred : array-like
    max_lags : int
        Number of lags for auto-correlation analysis.

    Returns
    -------
    dict with:
        residuals, mean, std, skewness, kurtosis,
        jarque_bera_stat, jarque_bera_pvalue,
        shapiro_stat, shapiro_pvalue,
        acf (list of autocorrelations), ljung_box_stat, ljung_box_pvalue
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    residuals = y_true - y_pred

    n = len(residuals)
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals, ddof=1)) if n > 1 else 0.0

    # Moments
    skew = float(scipy_stats.skew(residuals)) if n > 2 else np.nan
    kurt = float(scipy_stats.kurtosis(residuals)) if n > 3 else np.nan

    # Normality -- Jarque-Bera
    if n > 7:
        jb_stat, jb_p = scipy_stats.jarque_bera(residuals)
    else:
        jb_stat, jb_p = np.nan, np.nan

    # Normality -- Shapiro-Wilk (limited to 5000 samples)
    if 3 <= n <= 5000:
        sw_stat, sw_p = scipy_stats.shapiro(residuals)
    else:
        sw_stat, sw_p = np.nan, np.nan

    # Autocorrelation function
    acf_values = _compute_acf(residuals, max_lags)

    # Ljung-Box test (portmanteau test for residual autocorrelation)
    lb_stat, lb_p = _ljung_box(residuals, max_lags)

    # Durbin-Watson statistic
    if n > 1:
        dw = float(np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2))
    else:
        dw = np.nan

    return {
        "residuals": residuals,
        "mean": res_mean,
        "std": res_std,
        "skewness": skew,
        "kurtosis": kurt,
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_p),
        "shapiro_stat": float(sw_stat),
        "shapiro_pvalue": float(sw_p),
        "acf": acf_values,
        "ljung_box_stat": float(lb_stat),
        "ljung_box_pvalue": float(lb_p),
        "durbin_watson": dw,
    }


def _compute_acf(x: np.ndarray, max_lags: int) -> List[float]:
    """Compute the autocorrelation function up to *max_lags*."""
    n = len(x)
    x_centered = x - x.mean()
    var = np.sum(x_centered ** 2) / n
    if var == 0:
        return [np.nan] * (max_lags + 1)

    acf = []
    for lag in range(max_lags + 1):
        if lag >= n:
            acf.append(np.nan)
        else:
            cov = np.sum(x_centered[:n - lag] * x_centered[lag:]) / n
            acf.append(float(cov / var))
    return acf


def _ljung_box(x: np.ndarray, max_lags: int) -> Tuple[float, float]:
    """Compute the Ljung-Box Q statistic for autocorrelation."""
    n = len(x)
    if n <= max_lags + 1:
        return np.nan, np.nan

    acf = _compute_acf(x, max_lags)
    q_stat = 0.0
    for k in range(1, max_lags + 1):
        rk = acf[k] if k < len(acf) and not np.isnan(acf[k]) else 0.0
        q_stat += (rk ** 2) / (n - k)

    q_stat *= n * (n + 2)
    p_value = 1.0 - scipy_stats.chi2.cdf(q_stat, df=max_lags)
    return float(q_stat), float(p_value)


def learning_curve_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "xgboost",
    train_sizes: Optional[Sequence[float]] = None,
    cv_method: str = "expanding",
    scale: bool = True,
    n_subsets: int = 10,
) -> pd.DataFrame:
    """Generate learning curve data by training on increasingly large subsets.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    model_name : str
    train_sizes : sequence of float, optional
        Fractions of the full training data to use (e.g. ``[0.1, 0.2, ..., 1.0]``).
        If None, ``n_subsets`` equally-spaced fractions are used.
    cv_method : str
    scale : bool
    n_subsets : int
        Number of training-size fractions when ``train_sizes`` is None.

    Returns
    -------
    pd.DataFrame with columns:
        train_fraction, train_size, train_rmse, test_rmse, train_r2, test_r2.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, n_subsets)

    splits = _resolve_cv_splits(len(X), cv_method)
    if len(splits) == 0:
        raise RuntimeError("No valid CV splits for learning curve.")

    scaler = StandardScaler() if scale else None
    results: List[Dict[str, Any]] = []

    for frac in train_sizes:
        frac = float(frac)
        fold_train_rmse: List[float] = []
        fold_test_rmse: List[float] = []
        fold_train_r2: List[float] = []
        fold_test_r2: List[float] = []

        for train_idx, test_idx in splits:
            # Subsample training data
            n_train = max(2, int(len(train_idx) * frac))
            sub_train_idx = train_idx[:n_train]

            X_train = X.iloc[sub_train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[sub_train_idx]
            y_test = y.iloc[test_idx]

            if scale:
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
            else:
                X_train_s = X_train.values
                X_test_s = X_test.values

            model = create_model(model_name)
            model.fit(X_train_s, y_train)

            y_pred_train = model.predict(X_train_s)
            y_pred_test = model.predict(X_test_s)

            fold_train_rmse.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
            fold_test_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            fold_train_r2.append(r2_score(y_train, y_pred_train))
            fold_test_r2.append(r2_score(y_test, y_pred_test))

        results.append({
            "train_fraction": frac,
            "train_size": int(len(splits[0][0]) * frac),
            "train_rmse_mean": float(np.mean(fold_train_rmse)),
            "train_rmse_std": float(np.std(fold_train_rmse)),
            "test_rmse_mean": float(np.mean(fold_test_rmse)),
            "test_rmse_std": float(np.std(fold_test_rmse)),
            "train_r2_mean": float(np.mean(fold_train_r2)),
            "train_r2_std": float(np.std(fold_train_r2)),
            "test_r2_mean": float(np.mean(fold_test_r2)),
            "test_r2_std": float(np.std(fold_test_r2)),
        })

    return pd.DataFrame(results)


def calibration_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Analyse prediction calibration by binning predicted values and
    comparing mean predicted vs mean actual within each bin.

    Good calibration means the scatter of (mean_predicted, mean_actual) lies
    close to the 45-degree line.

    Parameters
    ----------
    y_true, y_pred : array-like
    n_bins : int
        Number of quantile-based bins.

    Returns
    -------
    pd.DataFrame with columns:
        bin, pred_mean, actual_mean, pred_std, actual_std, count, bias.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    # Use quantile binning so each bin has roughly equal counts
    try:
        bin_labels = pd.qcut(y_pred, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        # Fallback to equal-width bins if too few unique predictions
        bin_labels = pd.cut(y_pred, bins=n_bins, labels=False)

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "bin": bin_labels,
    })

    rows: List[Dict[str, Any]] = []
    for b, grp in df.groupby("bin"):
        rows.append({
            "bin": int(b),
            "pred_mean": float(grp["y_pred"].mean()),
            "actual_mean": float(grp["y_true"].mean()),
            "pred_std": float(grp["y_pred"].std()),
            "actual_std": float(grp["y_true"].std()),
            "count": len(grp),
            "bias": float(grp["y_pred"].mean() - grp["y_true"].mean()),
        })

    return pd.DataFrame(rows)


# ============================================================================
# CONVENIENCE / SUMMARY UTILITIES
# ============================================================================

def summarise_cv_results(result: Dict[str, Any]) -> pd.Series:
    """Return a one-row summary of a model's CV performance.

    Parameters
    ----------
    result : dict
        Output of ``train_and_evaluate``.

    Returns
    -------
    pd.Series with mean and std of each metric across folds.
    """
    cv = result.get("cv_metrics")
    if cv is None or not isinstance(cv, pd.DataFrame):
        return pd.Series(dtype=float)

    summary = {}
    for col in cv.columns:
        if col in ("fold", "train_size", "test_size"):
            continue
        summary[f"{col}_mean"] = cv[col].mean()
        summary[f"{col}_std"] = cv[col].std()

    summary["model_name"] = result.get("model_name", "unknown")
    return pd.Series(summary)


def compare_forecasts(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    baseline: str = "ridge",
) -> pd.DataFrame:
    """Compare multiple forecasts against a baseline using the Diebold-Mariano
    test and standard metrics.

    Parameters
    ----------
    y_true : array-like
    predictions : dict mapping model name -> predicted values
    baseline : str
        Name of the baseline model in ``predictions``.

    Returns
    -------
    pd.DataFrame with metrics for each model plus DM test vs baseline.
    """
    if baseline not in predictions:
        raise ValueError(f"Baseline '{baseline}' not found in predictions.")

    base_pred = predictions[baseline]
    rows: List[Dict[str, Any]] = []

    for name, y_pred in predictions.items():
        metrics = evaluate_predictions(y_true, y_pred)
        metrics["model"] = name

        if name != baseline:
            dm = dm_test(y_true, base_pred, y_pred)
            metrics["dm_stat_vs_baseline"] = dm["dm_stat"]
            metrics["dm_pvalue_vs_baseline"] = dm["p_value"]
        else:
            metrics["dm_stat_vs_baseline"] = np.nan
            metrics["dm_pvalue_vs_baseline"] = np.nan

        rows.append(metrics)

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def print_model_summary(result: Dict[str, Any]) -> None:
    """Print a human-readable summary of a training result to stdout."""
    name = result.get("model_name", "unknown")
    cv = result.get("cv_metrics")
    print(f"\n{'=' * 60}")
    print(f"  Model: {name}")
    print(f"{'=' * 60}")

    if cv is not None and isinstance(cv, pd.DataFrame):
        print(f"  CV Method : {result.get('cv_method', 'N/A')}")
        print(f"  Folds     : {len(cv)}")
        print(f"  Features  : {result.get('n_features', 'N/A')}")
        print(f"  Samples   : {result.get('n_samples', 'N/A')}")
        print()
        print("  Metric               Mean        Std")
        print("  " + "-" * 44)
        for col in ["rmse", "mae", "r2", "mape", "directional_accuracy", "ic"]:
            if col in cv.columns:
                mean_val = cv[col].mean()
                std_val = cv[col].std()
                print(f"  {col:<22s} {mean_val:>10.6f}  {std_val:>10.6f}")

    fi = result.get("feature_importance")
    if fi is not None and isinstance(fi, pd.DataFrame) and not fi["importance"].isna().all():
        print(f"\n  Top 10 Features:")
        for _, row in fi.head(10).iterrows():
            pct = row.get("importance_pct", np.nan)
            if not np.isnan(pct):
                print(f"    {row['feature']:<30s}  {pct:>6.2f}%")

    print(f"{'=' * 60}\n")
