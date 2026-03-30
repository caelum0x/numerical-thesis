"""
Model loading, prediction, ensemble methods, expected-return construction,
prediction diagnostics, and signal generation utilities.

This module sits between the trained ML models (produced by ``trainer.py``)
and the portfolio optimiser (``optimizer.py``).  It converts raw model
outputs into the expected-return vectors, covariance matrices, and
Black--Litterman views that the optimiser consumes, while providing
comprehensive diagnostics so that every step is auditable.

References
----------
* Markowitz, H. (1952).  Portfolio Selection.
* Black, F. & Litterman, R. (1992).  Global Portfolio Optimization.
* Ledoit, O. & Wolf, M. (2004).  A well-conditioned estimator for
  large-dimensional covariance matrices.
* Hoeting, J. et al. (1999).  Bayesian Model Averaging: A Tutorial.

Author : Arhan Subasi
Course : Industrial Engineering MSc Thesis
"""

from __future__ import annotations

import logging
import pickle
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.config import (
    PREDICTION_HORIZON,
    PREDICTION_HORIZONS,
    RANDOM_STATE,
    RESULTS_DIR,
    TICKER_LIST,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ANNUALISATION_FACTOR = 252  # trading days in a year


# ============================================================================
# Section 1 -- Model Loading & Management
# ============================================================================


def load_model(model_name: str, target_name: str) -> dict:
    """Load a trained model artefact from disk.

    The saved artefact is a dictionary produced by ``trainer.save_model``
    containing at minimum the keys ``"model"`` and ``"scaler"``, plus
    optional keys like ``"cv_metrics"`` and ``"feature_importance"``.

    Parameters
    ----------
    model_name : str
        Algorithm identifier, e.g. ``"xgboost"``, ``"ridge"``.
    target_name : str
        Target column name, e.g. ``"SPY_ret_21d"``.

    Returns
    -------
    dict
        Loaded model artefact.

    Raises
    ------
    FileNotFoundError
        If no saved model file is found at the expected path.
    """
    path = RESULTS_DIR / f"model_{model_name}_{target_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")
    with open(path, "rb") as f:
        artefact = pickle.load(f)
    logger.debug("Loaded model from %s", path)
    return artefact


def list_saved_models() -> List[str]:
    """List all saved model files under ``RESULTS_DIR``.

    Returns
    -------
    list[str]
        Stems of every ``model_*.pkl`` file, sorted alphabetically.
    """
    paths = sorted(RESULTS_DIR.glob("model_*.pkl"))
    return [p.stem for p in paths]


def get_best_model_for_asset(
    ticker: str,
    horizon: int = PREDICTION_HORIZON,
    metric: str = "rmse",
) -> Optional[str]:
    """Identify the best model for a given asset by cross-validation metric.

    Scans all saved model artefacts whose target matches
    ``{ticker}_ret_{horizon}d`` and returns the model name that achieved the
    lowest (for error metrics) or highest (for R-squared) CV score.

    Parameters
    ----------
    ticker : str
        Asset ticker, e.g. ``"SPY"``.
    horizon : int
        Prediction horizon in trading days.
    metric : str
        Metric name stored in the ``cv_metrics`` DataFrame.  Common choices:
        ``"rmse"``, ``"mae"``, ``"r2"``.

    Returns
    -------
    str or None
        Model name of the best model, or ``None`` if no models exist.
    """
    target_name = f"{ticker}_ret_{horizon}d"
    best_model_name: Optional[str] = None
    best_score: Optional[float] = None
    higher_is_better = metric in ("r2",)

    for stem in list_saved_models():
        if not stem.endswith(target_name):
            continue
        try:
            artefact = load_model(
                stem.replace("model_", "").replace(f"_{target_name}", ""),
                target_name,
            )
        except (FileNotFoundError, Exception):
            continue

        cv = artefact.get("cv_metrics")
        if cv is None or metric not in cv.columns:
            continue

        score = cv[metric].mean()
        if best_score is None:
            best_score = score
            best_model_name = artefact.get("model_name", stem)
        elif higher_is_better and score > best_score:
            best_score = score
            best_model_name = artefact.get("model_name", stem)
        elif not higher_is_better and score < best_score:
            best_score = score
            best_model_name = artefact.get("model_name", stem)

    if best_model_name is not None:
        logger.info(
            "Best model for %s (horizon=%dd, metric=%s): %s (%.6f)",
            ticker,
            horizon,
            metric,
            best_model_name,
            best_score,
        )
    return best_model_name


class ModelCache:
    """LRU cache for loaded model artefacts.

    Avoids repeated disk reads when the same model is requested multiple
    times during a single pipeline run.  The cache is keyed by
    ``(model_name, target_name)`` and evicts the least-recently-used entry
    when ``maxsize`` is exceeded.

    Parameters
    ----------
    maxsize : int
        Maximum number of artefacts held in memory.  Defaults to 64.

    Examples
    --------
    >>> cache = ModelCache(maxsize=32)
    >>> artefact = cache.get("xgboost", "SPY_ret_21d")
    """

    def __init__(self, maxsize: int = 64) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[Tuple[str, str], dict] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    # -- public interface ---------------------------------------------------

    def get(self, model_name: str, target_name: str) -> dict:
        """Retrieve a model artefact, loading from disk on cache miss.

        Parameters
        ----------
        model_name : str
            Algorithm identifier.
        target_name : str
            Target column name.

        Returns
        -------
        dict
            Model artefact dictionary.
        """
        key = (model_name, target_name)
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        artefact = load_model(model_name, target_name)
        self._misses += 1
        self._cache[key] = artefact
        self._cache.move_to_end(key)

        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

        return artefact

    def invalidate(self, model_name: str, target_name: str) -> None:
        """Remove a specific entry from the cache."""
        key = (model_name, target_name)
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Flush the entire cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> Dict[str, int]:
        """Return hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._cache),
            "maxsize": self._maxsize,
        }

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"ModelCache(size={s['size']}/{s['maxsize']}, "
            f"hit_rate={s['hit_rate']:.1%})"
        )


# Module-level default cache instance
_default_cache = ModelCache(maxsize=64)


# ============================================================================
# Section 2 -- Single-Model Predictions
# ============================================================================


def predict_returns(
    features: pd.DataFrame,
    model_name: str = "xgboost",
    tickers: List[str] = TICKER_LIST,
    horizon: int = PREDICTION_HORIZON,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Generate return predictions for every asset in *tickers*.

    For each ticker the function loads the corresponding trained model
    (``model_{model_name}_{ticker}_ret_{horizon}d.pkl``), scales the
    features using the stored ``StandardScaler``, and produces predictions
    over the full date range of *features*.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix with a ``DatetimeIndex``.  Must contain the same
        columns used during training.
    model_name : str
        Algorithm name, e.g. ``"xgboost"``, ``"ridge"``.
    tickers : list[str]
        List of asset tickers to predict.
    horizon : int
        Forward-return horizon in trading days.
    use_cache : bool
        If ``True`` (default), use the module-level ``ModelCache``.

    Returns
    -------
    pd.DataFrame
        Columns are *tickers*; index matches ``features.index``.
    """
    predictions: Dict[str, np.ndarray] = {}

    for ticker in tickers:
        target_name = f"{ticker}_ret_{horizon}d"
        try:
            if use_cache:
                artefact = _default_cache.get(model_name, target_name)
            else:
                artefact = load_model(model_name, target_name)

            model = artefact["model"]
            scaler = artefact["scaler"]

            X_scaled = scaler.transform(features)
            predictions[ticker] = model.predict(X_scaled)
        except FileNotFoundError:
            logger.warning("No model found for %s -- skipping", target_name)
        except Exception as exc:
            logger.error(
                "Error predicting %s with %s: %s", target_name, model_name, exc
            )

    if not predictions:
        logger.warning(
            "predict_returns produced no predictions for model=%s horizon=%d",
            model_name,
            horizon,
        )
    return pd.DataFrame(predictions, index=features.index)


def predict_at_date(
    features: pd.DataFrame,
    date: str,
    model_name: str = "xgboost",
    tickers: List[str] = TICKER_LIST,
    horizon: int = PREDICTION_HORIZON,
) -> pd.Series:
    """Return predictions for a single calendar date.

    If *date* is not in the index, the nearest available date is used
    (with a warning).

    Parameters
    ----------
    features : pd.DataFrame
        Full feature matrix.
    date : str
        Target date string, e.g. ``"2023-06-30"``.
    model_name : str
        Algorithm name.
    tickers : list[str]
        Asset tickers.
    horizon : int
        Prediction horizon in trading days.

    Returns
    -------
    pd.Series
        Predicted returns keyed by ticker for the requested date.
    """
    date_ts = pd.Timestamp(date)
    if date_ts not in features.index:
        idx = features.index.get_indexer([date_ts], method="nearest")[0]
        date_ts = features.index[idx]
        logger.info("Date %s not in index; snapped to %s", date, date_ts)

    row = features.loc[[date_ts]]
    preds = predict_returns(
        row, model_name=model_name, tickers=tickers, horizon=horizon
    )
    return preds.iloc[0]


def predict_with_confidence(
    features: pd.DataFrame,
    model_name: str,
    ticker: str,
    horizon: int = PREDICTION_HORIZON,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Produce bootstrap confidence intervals around point predictions.

    The bootstrap resamples the training residuals stored in the artefact
    (if available) or perturbs input features with Gaussian noise scaled
    to each feature's standard deviation.  This gives a non-parametric
    estimate of prediction uncertainty.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.
    model_name : str
        Algorithm name.
    ticker : str
        Single asset ticker.
    horizon : int
        Prediction horizon.
    n_bootstrap : int
        Number of bootstrap replications.
    confidence_level : float
        Width of the confidence interval, e.g. 0.95 for 95 %.

    Returns
    -------
    pd.DataFrame
        Columns ``["prediction", "lower", "upper", "std"]``.
    """
    target_name = f"{ticker}_ret_{horizon}d"
    artefact = _default_cache.get(model_name, target_name)
    model = artefact["model"]
    scaler = artefact["scaler"]

    X_scaled = scaler.transform(features)
    point_pred = model.predict(X_scaled)

    rng = np.random.RandomState(RANDOM_STATE)
    alpha = 1.0 - confidence_level

    # Collect bootstrap predictions
    bootstrap_preds = np.empty((n_bootstrap, len(features)))

    # If the artefact stores training residuals, resample those
    residuals = artefact.get("residuals")
    if residuals is not None and len(residuals) > 0:
        for b in range(n_bootstrap):
            noise = rng.choice(residuals, size=len(features), replace=True)
            bootstrap_preds[b] = point_pred + noise
    else:
        # Perturbation-based bootstrap: add small Gaussian noise to features
        feature_std = np.std(X_scaled, axis=0, keepdims=True)
        feature_std = np.where(feature_std == 0, 1e-8, feature_std)
        for b in range(n_bootstrap):
            noise = rng.normal(0, 0.05, size=X_scaled.shape) * feature_std
            X_perturbed = X_scaled + noise
            bootstrap_preds[b] = model.predict(X_perturbed)

    lower = np.percentile(bootstrap_preds, 100 * alpha / 2, axis=0)
    upper = np.percentile(bootstrap_preds, 100 * (1 - alpha / 2), axis=0)
    std = np.std(bootstrap_preds, axis=0)

    result = pd.DataFrame(
        {
            "prediction": point_pred,
            "lower": lower,
            "upper": upper,
            "std": std,
        },
        index=features.index,
    )
    return result


# ============================================================================
# Section 3 -- Ensemble Methods
# ============================================================================


def predict_returns_ensemble(
    features: pd.DataFrame,
    model_names: Optional[List[str]] = None,
    tickers: List[str] = TICKER_LIST,
    horizon: int = PREDICTION_HORIZON,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Weighted-average ensemble of multiple models.

    If *weights* is ``None`` a simple equal-weight average is used.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.
    model_names : list[str] or None
        Model identifiers.  Defaults to ``["ridge", "xgboost", "random_forest"]``.
    tickers : list[str]
        Asset tickers.
    horizon : int
        Prediction horizon.
    weights : dict[str, float] or None
        Mapping from model name to weight.  Weights are normalised
        internally so they need not sum to one.

    Returns
    -------
    pd.DataFrame
        Ensemble predictions.
    """
    if model_names is None:
        model_names = ["ridge", "xgboost", "random_forest"]

    all_preds: List[pd.DataFrame] = []
    valid_models: List[str] = []

    for mname in model_names:
        pred = predict_returns(
            features, model_name=mname, tickers=tickers, horizon=horizon
        )
        if len(pred.columns) > 0:
            all_preds.append(pred)
            valid_models.append(mname)

    if not all_preds:
        raise ValueError("No models produced predictions for the ensemble")

    if weights is not None:
        w = np.array([weights.get(m, 1.0) for m in valid_models])
        w = w / w.sum()
        ensemble = sum(p * wi for p, wi in zip(all_preds, w))
    else:
        ensemble = sum(all_preds) / len(all_preds)

    return ensemble


def predict_returns_stacking(
    features: pd.DataFrame,
    base_models: List[str],
    meta_model: Any,
    tickers: List[str] = TICKER_LIST,
    horizon: int = PREDICTION_HORIZON,
) -> pd.DataFrame:
    """Two-level stacking ensemble.

    Level-0 predictions from each base model are concatenated into a
    meta-feature matrix.  A pre-trained *meta_model* (e.g. ``RidgeCV``)
    then produces the final prediction.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.
    base_models : list[str]
        Names of level-0 models.
    meta_model
        A fitted scikit-learn estimator with a ``.predict`` method.  If
        ``None``, a ``RidgeCV`` is fitted on the base predictions
        themselves (i.e. an in-sample stacking -- suitable only for
        exploratory work).
    tickers : list[str]
        Asset tickers.
    horizon : int
        Prediction horizon.

    Returns
    -------
    pd.DataFrame
        Stacked predictions (columns = tickers).
    """
    stacked: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        base_preds_list: List[np.ndarray] = []
        for mname in base_models:
            target_name = f"{ticker}_ret_{horizon}d"
            try:
                artefact = _default_cache.get(mname, target_name)
                model = artefact["model"]
                scaler = artefact["scaler"]
                X_scaled = scaler.transform(features)
                base_preds_list.append(model.predict(X_scaled))
            except (FileNotFoundError, Exception) as exc:
                logger.warning(
                    "Stacking: skipping %s for %s (%s)", mname, ticker, exc
                )

        if not base_preds_list:
            continue

        meta_X = np.column_stack(base_preds_list)

        if meta_model is None:
            # Fallback: fit a simple RidgeCV on the meta-features
            _meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
            # Use the first base model's predictions as target proxy (rough)
            _meta.fit(meta_X, base_preds_list[0])
            stacked[ticker] = _meta.predict(meta_X)
        else:
            stacked[ticker] = meta_model.predict(meta_X)

    return pd.DataFrame(stacked, index=features.index)


def compute_ensemble_weights_from_cv(
    model_results: Dict[str, dict],
    method: str = "inverse_rmse",
) -> Dict[str, float]:
    """Compute optimal ensemble weights from cross-validation performance.

    Parameters
    ----------
    model_results : dict[str, dict]
        Mapping ``model_name -> artefact`` where each artefact must
        contain a ``"cv_metrics"`` ``DataFrame`` with an ``"rmse"`` column.
    method : str
        * ``"inverse_rmse"`` -- weight inversely proportional to mean RMSE
        * ``"inverse_mse"`` -- weight inversely proportional to mean MSE
        * ``"softmax"`` -- softmax of negative mean RMSE (temperature=1)
        * ``"rank"`` -- weight inversely proportional to RMSE rank

    Returns
    -------
    dict[str, float]
        Normalised weights that sum to one.
    """
    scores: Dict[str, float] = {}
    for mname, artefact in model_results.items():
        cv = artefact.get("cv_metrics")
        if cv is None or "rmse" not in cv.columns:
            logger.warning(
                "Skipping %s in ensemble weight computation (no cv_metrics)", mname
            )
            continue
        scores[mname] = cv["rmse"].mean()

    if not scores:
        raise ValueError("No valid CV metrics found in model_results")

    names = list(scores.keys())
    rmse_arr = np.array([scores[n] for n in names])

    if method == "inverse_rmse":
        raw = 1.0 / np.maximum(rmse_arr, 1e-12)
    elif method == "inverse_mse":
        raw = 1.0 / np.maximum(rmse_arr ** 2, 1e-24)
    elif method == "softmax":
        # Temperature-scaled softmax of negative RMSE
        neg_rmse = -rmse_arr
        exp_vals = np.exp(neg_rmse - neg_rmse.max())
        raw = exp_vals
    elif method == "rank":
        ranks = stats.rankdata(rmse_arr, method="ordinal")
        raw = 1.0 / ranks.astype(float)
    else:
        raise ValueError(f"Unknown method: {method}")

    total = raw.sum()
    weights = {n: float(w / total) for n, w in zip(names, raw)}
    logger.info("Ensemble weights (%s): %s", method, weights)
    return weights


def predict_returns_bayesian_average(
    features: pd.DataFrame,
    model_names: List[str],
    tickers: List[str] = TICKER_LIST,
    horizon: int = PREDICTION_HORIZON,
    criterion: str = "bic",
) -> pd.DataFrame:
    """Bayesian Model Averaging using AIC or BIC approximation.

    Each model's posterior probability is approximated by:

    .. math::

        w_k \\propto \\exp\\bigl(-\\tfrac{1}{2}\\,\\text{IC}_k\\bigr)

    where IC is the information criterion.  This is the standard BMA
    weight derivation when flat model priors are assumed (Hoeting et al.,
    1999).

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.
    model_names : list[str]
        Model identifiers.
    tickers : list[str]
        Asset tickers.
    horizon : int
        Prediction horizon.
    criterion : str
        ``"aic"`` or ``"bic"``.

    Returns
    -------
    pd.DataFrame
        BMA predictions.
    """
    all_preds: List[pd.DataFrame] = []
    ic_values: List[float] = []
    valid_models: List[str] = []

    for mname in model_names:
        pred = predict_returns(
            features, model_name=mname, tickers=tickers, horizon=horizon
        )
        if len(pred.columns) == 0:
            continue

        # Approximate information criterion from stored CV metrics
        artefact_ics: List[float] = []
        for ticker in pred.columns:
            target_name = f"{ticker}_ret_{horizon}d"
            try:
                artefact = _default_cache.get(mname, target_name)
                cv = artefact.get("cv_metrics")
                if cv is not None and "mse" in cv.columns:
                    mse = cv["mse"].mean()
                    n_obs = cv.get("n_test", pd.Series([100])).mean()
                    # Approximate number of parameters
                    model_obj = artefact["model"]
                    n_params = _estimate_n_params(model_obj)

                    if criterion == "aic":
                        ic = n_obs * np.log(max(mse, 1e-20)) + 2 * n_params
                    else:  # bic
                        ic = n_obs * np.log(max(mse, 1e-20)) + n_params * np.log(
                            max(n_obs, 1)
                        )
                    artefact_ics.append(ic)
            except Exception:
                pass

        if artefact_ics:
            ic_values.append(float(np.mean(artefact_ics)))
        else:
            # Fallback: use mean squared prediction as rough proxy
            ic_values.append(float(pred.var().mean()) * 1000)

        all_preds.append(pred)
        valid_models.append(mname)

    if not all_preds:
        raise ValueError("No models produced predictions for BMA")

    # Compute BMA weights
    ic_arr = np.array(ic_values)
    log_weights = -0.5 * (ic_arr - ic_arr.min())  # shift for numerical stability
    weights = np.exp(log_weights)
    weights /= weights.sum()

    logger.info(
        "BMA weights (%s): %s",
        criterion,
        dict(zip(valid_models, np.round(weights, 4))),
    )

    ensemble = sum(p * w for p, w in zip(all_preds, weights))
    return ensemble


def _estimate_n_params(model: Any) -> int:
    """Heuristic for the effective number of model parameters."""
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        return int(np.count_nonzero(coef))
    if hasattr(model, "n_estimators") and hasattr(model, "max_depth"):
        depth = model.max_depth if model.max_depth is not None else 6
        return int(model.n_estimators * (2 ** depth))
    if hasattr(model, "get_num_trees"):
        return int(model.get_num_trees())
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    return 10  # conservative fallback


# ============================================================================
# Section 4 -- Expected Return Construction
# ============================================================================


def build_expected_returns(
    predictions: pd.DataFrame,
    horizon: int = PREDICTION_HORIZON,
) -> np.ndarray:
    """Annualise the mean predicted return for each asset.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predicted returns (columns = tickers).
    horizon : int
        Prediction horizon in trading days.

    Returns
    -------
    np.ndarray
        Annualised expected-return vector (length = number of tickers).
    """
    mean_pred = predictions.mean().values
    annualised = mean_pred * (_ANNUALISATION_FACTOR / horizon)
    return annualised


def build_expected_returns_shrinkage(
    predictions: pd.DataFrame,
    historical_returns: pd.DataFrame,
    alpha: float = 0.5,
    horizon: int = PREDICTION_HORIZON,
) -> np.ndarray:
    """Shrink model predictions toward the historical mean.

    .. math::

        \\hat{\\mu} = \\alpha\\,\\mu_{\\text{model}}
                    + (1 - \\alpha)\\,\\mu_{\\text{hist}}

    This James--Stein-style shrinkage reduces the impact of extreme
    model forecasts and pulls the expected-return vector toward a more
    stable historical baseline.

    Parameters
    ----------
    predictions : pd.DataFrame
        Model predictions (columns = tickers).
    historical_returns : pd.DataFrame
        Historical daily returns (columns = tickers).
    alpha : float
        Shrinkage intensity.  ``alpha=1`` means pure model;
        ``alpha=0`` means pure historical.
    horizon : int
        Prediction horizon (for annualisation).

    Returns
    -------
    np.ndarray
        Shrunk, annualised expected-return vector.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    # Align tickers
    common = predictions.columns.intersection(historical_returns.columns)
    if len(common) == 0:
        raise ValueError("No overlapping tickers between predictions and history")

    mu_model = predictions[common].mean().values
    mu_hist = historical_returns[common].mean().values

    blended = alpha * mu_model + (1.0 - alpha) * mu_hist
    annualised = blended * (_ANNUALISATION_FACTOR / horizon)
    return annualised


def build_covariance_from_predictions(
    predictions: pd.DataFrame,
    returns_history: pd.DataFrame,
    method: str = "shrinkage",
    ewma_halflife: int = 63,
    blend_alpha: float = 0.3,
) -> np.ndarray:
    """Build a forward-looking covariance matrix.

    Three strategies are available:

    * ``"sample"`` -- sample covariance of historical returns (annualised).
    * ``"shrinkage"`` -- Ledoit-Wolf shrinkage estimator.
    * ``"ewma"`` -- exponentially weighted moving average covariance.
    * ``"blend"`` -- blend of prediction-implied covariance and historical
      shrinkage covariance (controlled by *blend_alpha*).

    Parameters
    ----------
    predictions : pd.DataFrame
        Model predictions.  Used only in ``"blend"`` mode.
    returns_history : pd.DataFrame
        Historical daily returns.
    method : str
        Estimation method.
    ewma_halflife : int
        Half-life in days for the EWMA estimator.
    blend_alpha : float
        Weight on prediction-implied covariance in ``"blend"`` mode.

    Returns
    -------
    np.ndarray
        Annualised covariance matrix (N x N).
    """
    common = predictions.columns.intersection(returns_history.columns)
    hist = returns_history[common].dropna()

    if method == "sample":
        cov = hist.cov().values * _ANNUALISATION_FACTOR

    elif method == "shrinkage":
        lw = LedoitWolf().fit(hist.values)
        cov = lw.covariance_ * _ANNUALISATION_FACTOR

    elif method == "ewma":
        ewma_cov = (
            hist.ewm(halflife=ewma_halflife, min_periods=max(ewma_halflife, 30))
            .cov()
            .iloc[-len(common) :]
            .values.reshape(len(common), len(common))
        )
        cov = ewma_cov * _ANNUALISATION_FACTOR

    elif method == "blend":
        # Prediction-implied covariance (from cross-sectional dispersion)
        pred_cov = predictions[common].cov().values * _ANNUALISATION_FACTOR
        lw = LedoitWolf().fit(hist.values)
        hist_cov = lw.covariance_ * _ANNUALISATION_FACTOR
        cov = blend_alpha * pred_cov + (1.0 - blend_alpha) * hist_cov

    else:
        raise ValueError(
            f"Unknown covariance method: {method}. "
            "Choose from 'sample', 'shrinkage', 'ewma', 'blend'."
        )

    # Ensure positive semi-definiteness via eigenvalue clipping
    cov = _ensure_psd(cov)
    return cov


def _ensure_psd(matrix: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix onto the positive semi-definite cone."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def build_views_for_black_litterman(
    predictions: pd.DataFrame,
    confidence: Union[pd.Series, np.ndarray, float],
    tickers: List[str] = TICKER_LIST,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct Black--Litterman view matrices from model predictions.

    Each asset with a prediction becomes an absolute view:
    ``E[r_i] = q_i`` with confidence ``c_i``.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predicted returns (columns = tickers).  Uses the last row
        (most recent prediction) as the view vector.
    confidence : float, pd.Series, or np.ndarray
        Per-asset confidence in [0, 1].  A scalar applies uniformly.
    tickers : list[str]
        Ordered list of tickers in the universe.

    Returns
    -------
    P : np.ndarray
        Pick matrix (K x N).
    Q : np.ndarray
        View return vector (K,).
    omega_diag : np.ndarray
        Diagonal of the view uncertainty matrix (K,).  Lower values
        indicate higher confidence.
    """
    # Use the most recent prediction row
    view_returns = predictions.iloc[-1]
    view_tickers = [t for t in tickers if t in view_returns.index]
    K = len(view_tickers)
    N = len(tickers)

    if K == 0:
        raise ValueError("No overlapping tickers between predictions and universe")

    P = np.zeros((K, N))
    Q = np.zeros(K)

    for k, vticker in enumerate(view_tickers):
        asset_idx = tickers.index(vticker)
        P[k, asset_idx] = 1.0
        Q[k] = view_returns[vticker]

    # Build uncertainty diagonal from confidence
    if isinstance(confidence, (int, float)):
        conf_arr = np.full(K, float(confidence))
    elif isinstance(confidence, pd.Series):
        conf_arr = np.array([confidence.get(t, 0.5) for t in view_tickers])
    else:
        conf_arr = np.asarray(confidence)[:K]

    # Map confidence [0, 1] to uncertainty: higher confidence -> lower omega
    conf_arr = np.clip(conf_arr, 0.01, 0.99)
    omega_diag = (1.0 - conf_arr) / conf_arr  # simple odds-ratio transform

    return P, Q, omega_diag


# ============================================================================
# Section 5 -- Prediction Diagnostics
# ============================================================================


def prediction_summary(
    predictions: pd.DataFrame,
    actual_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compare predictions to realised returns across all assets.

    Computes RMSE, MAE, R-squared, information coefficient (rank
    correlation), and directional accuracy for each ticker.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predicted returns.
    actual_returns : pd.DataFrame
        Realised returns (same shape and index as *predictions*).

    Returns
    -------
    pd.DataFrame
        Summary statistics indexed by ticker.
    """
    common = predictions.columns.intersection(actual_returns.columns)
    aligned_pred = predictions[common].dropna()
    aligned_actual = actual_returns[common].reindex(aligned_pred.index).dropna()

    # Intersect indices
    idx = aligned_pred.index.intersection(aligned_actual.index)
    aligned_pred = aligned_pred.loc[idx]
    aligned_actual = aligned_actual.loc[idx]

    if len(idx) == 0:
        logger.warning("No overlapping dates for prediction_summary")
        return pd.DataFrame()

    records = []
    for ticker in common:
        y_pred = aligned_pred[ticker].values
        y_true = aligned_actual[ticker].values

        mask = np.isfinite(y_pred) & np.isfinite(y_true)
        if mask.sum() < 5:
            continue

        y_pred_clean = y_pred[mask]
        y_true_clean = y_true[mask]

        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)

        # Information Coefficient (Spearman rank correlation)
        ic, ic_pval = stats.spearmanr(y_pred_clean, y_true_clean)

        # Directional accuracy
        correct_dir = np.sign(y_pred_clean) == np.sign(y_true_clean)
        dir_accuracy = correct_dir.mean()

        # Hit rate (predicted positive and was positive)
        pos_mask = y_pred_clean > 0
        if pos_mask.sum() > 0:
            hit_rate = (y_true_clean[pos_mask] > 0).mean()
        else:
            hit_rate = np.nan

        records.append(
            {
                "ticker": ticker,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "ic": ic,
                "ic_pvalue": ic_pval,
                "directional_accuracy": dir_accuracy,
                "hit_rate": hit_rate,
                "n_observations": int(mask.sum()),
                "mean_prediction": float(y_pred_clean.mean()),
                "mean_actual": float(y_true_clean.mean()),
                "std_prediction": float(y_pred_clean.std()),
                "std_actual": float(y_true_clean.std()),
            }
        )

    summary = pd.DataFrame(records).set_index("ticker")
    logger.info(
        "Prediction summary: mean IC=%.4f, mean dir_acc=%.2f%%",
        summary["ic"].mean(),
        summary["directional_accuracy"].mean() * 100,
    )
    return summary


def rolling_prediction_accuracy(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    model_name: str = "xgboost",
    horizon: int = PREDICTION_HORIZON,
    window: int = 63,
    tickers: List[str] = TICKER_LIST,
) -> pd.DataFrame:
    """Rolling information coefficient and directional accuracy.

    At each date *t*, predictions made at dates
    ``[t - window + 1, ..., t]`` are compared against the realised
    forward returns.  This produces a time series of local IC and
    directional accuracy that reveals when the model works and when it
    does not.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.
    prices : pd.DataFrame
        Asset price DataFrame (columns = tickers) used to compute
        realised forward returns.
    model_name : str
        Algorithm name.
    horizon : int
        Prediction horizon.
    window : int
        Rolling window length in trading days.
    tickers : list[str]
        Asset tickers.

    Returns
    -------
    pd.DataFrame
        Columns ``["rolling_ic", "rolling_dir_acc"]``.
    """
    preds = predict_returns(
        features, model_name=model_name, tickers=tickers, horizon=horizon
    )

    # Realised forward returns
    common = preds.columns.intersection(prices.columns)
    fwd_ret = prices[common].pct_change(horizon).shift(-horizon)

    idx = preds.index.intersection(fwd_ret.dropna(how="all").index)
    preds = preds.loc[idx, common]
    fwd_ret = fwd_ret.loc[idx, common]

    rolling_ic = []
    rolling_dir_acc = []
    dates = []

    for end in range(window, len(idx)):
        start = end - window
        p_slice = preds.iloc[start:end].values.ravel()
        a_slice = fwd_ret.iloc[start:end].values.ravel()

        mask = np.isfinite(p_slice) & np.isfinite(a_slice)
        if mask.sum() < 10:
            rolling_ic.append(np.nan)
            rolling_dir_acc.append(np.nan)
        else:
            ic, _ = stats.spearmanr(p_slice[mask], a_slice[mask])
            da = (np.sign(p_slice[mask]) == np.sign(a_slice[mask])).mean()
            rolling_ic.append(ic)
            rolling_dir_acc.append(da)
        dates.append(idx[end])

    return pd.DataFrame(
        {"rolling_ic": rolling_ic, "rolling_dir_acc": rolling_dir_acc},
        index=pd.DatetimeIndex(dates),
    )


def prediction_decay_analysis(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    model_name: str = "xgboost",
    horizons: Optional[List[int]] = None,
    tickers: List[str] = TICKER_LIST,
) -> pd.DataFrame:
    """Analyse how prediction accuracy decays with horizon length.

    For each horizon in *horizons*, the function trains/loads the
    corresponding model (if available), generates predictions, and
    computes aggregate IC and directional accuracy against realised
    returns.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.
    prices : pd.DataFrame
        Asset prices.
    model_name : str
        Algorithm name.
    horizons : list[int] or None
        Horizons to evaluate.  Defaults to ``PREDICTION_HORIZONS``.
    tickers : list[str]
        Asset tickers.

    Returns
    -------
    pd.DataFrame
        One row per horizon with columns
        ``["horizon", "mean_ic", "mean_dir_acc", "mean_rmse"]``.
    """
    if horizons is None:
        horizons = list(PREDICTION_HORIZONS)

    records = []
    for h in horizons:
        try:
            preds = predict_returns(
                features, model_name=model_name, tickers=tickers, horizon=h
            )
        except Exception:
            continue

        if preds.empty:
            continue

        common = preds.columns.intersection(prices.columns)
        fwd_ret = prices[common].pct_change(h).shift(-h)
        idx = preds.index.intersection(fwd_ret.dropna(how="all").index)

        if len(idx) < 20:
            continue

        preds_aligned = preds.loc[idx, common]
        fwd_aligned = fwd_ret.loc[idx, common]

        ics = []
        dir_accs = []
        rmses = []
        for ticker in common:
            p = preds_aligned[ticker].values
            a = fwd_aligned[ticker].values
            mask = np.isfinite(p) & np.isfinite(a)
            if mask.sum() < 10:
                continue
            ic, _ = stats.spearmanr(p[mask], a[mask])
            da = (np.sign(p[mask]) == np.sign(a[mask])).mean()
            rmse = np.sqrt(mean_squared_error(a[mask], p[mask]))
            ics.append(ic)
            dir_accs.append(da)
            rmses.append(rmse)

        if ics:
            records.append(
                {
                    "horizon": h,
                    "mean_ic": float(np.mean(ics)),
                    "std_ic": float(np.std(ics)),
                    "mean_dir_acc": float(np.mean(dir_accs)),
                    "mean_rmse": float(np.mean(rmses)),
                    "n_assets": len(ics),
                }
            )

    return pd.DataFrame(records)


def compute_prediction_turnover(
    predictions: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """Measure how much predictions change from day to day.

    High turnover may indicate an unstable model or noisy features.
    Low turnover suggests smooth, persistent forecasts -- desirable for
    portfolio construction because it reduces trading costs.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predicted returns (columns = tickers).
    window : int
        Lag (in trading days) over which to measure change.

    Returns
    -------
    pd.DataFrame
        Columns ``["mean_abs_change", "rank_correlation"]`` indexed by
        date.
    """
    shifted = predictions.shift(window)
    idx = predictions.index[window:]

    mean_abs_change = []
    rank_corr = []

    for date in idx:
        current = predictions.loc[date].values
        previous = shifted.loc[date].values
        mask = np.isfinite(current) & np.isfinite(previous)
        if mask.sum() < 3:
            mean_abs_change.append(np.nan)
            rank_corr.append(np.nan)
            continue
        mean_abs_change.append(float(np.mean(np.abs(current[mask] - previous[mask]))))
        rc, _ = stats.spearmanr(current[mask], previous[mask])
        rank_corr.append(rc)

    return pd.DataFrame(
        {"mean_abs_change": mean_abs_change, "rank_correlation": rank_corr},
        index=idx,
    )


# ============================================================================
# Section 6 -- Signal Generation
# ============================================================================


def generate_alpha_signals(
    predictions: pd.DataFrame,
    method: str = "zscore",
    clip: float = 3.0,
    expanding_min: int = 63,
) -> pd.DataFrame:
    """Convert raw return predictions to standardised alpha signals.

    Parameters
    ----------
    predictions : pd.DataFrame
        Raw predicted returns (columns = tickers).
    method : str
        Standardisation method:

        * ``"zscore"`` -- rolling z-score within each cross-section.
        * ``"rank"`` -- cross-sectional percentile rank in [0, 1].
        * ``"minmax"`` -- min-max scaling to [-1, 1].
        * ``"raw"`` -- no transformation (pass-through).
    clip : float
        Clip z-scores at +/- this value (only for ``"zscore"``).
    expanding_min : int
        Minimum number of observations for expanding-window statistics.

    Returns
    -------
    pd.DataFrame
        Standardised signals, same shape as *predictions*.
    """
    if method == "zscore":
        # Cross-sectional z-score at each date
        cs_mean = predictions.mean(axis=1)
        cs_std = predictions.std(axis=1)
        cs_std = cs_std.replace(0, np.nan)
        signals = predictions.subtract(cs_mean, axis=0).divide(cs_std, axis=0)
        signals = signals.clip(lower=-clip, upper=clip)

    elif method == "rank":
        signals = predictions.rank(axis=1, pct=True)

    elif method == "minmax":
        row_min = predictions.min(axis=1)
        row_max = predictions.max(axis=1)
        row_range = row_max - row_min
        row_range = row_range.replace(0, np.nan)
        # Scale to [-1, 1]
        signals = 2 * predictions.subtract(row_min, axis=0).divide(
            row_range, axis=0
        ) - 1

    elif method == "raw":
        signals = predictions.copy()

    else:
        raise ValueError(
            f"Unknown signal method: {method}. "
            "Choose from 'zscore', 'rank', 'minmax', 'raw'."
        )

    return signals


def combine_signals(
    signals_dict: Dict[str, pd.DataFrame],
    combination_method: str = "equal",
    weights: Optional[Dict[str, float]] = None,
    rank_before_combine: bool = False,
) -> pd.DataFrame:
    """Combine multiple alpha signal DataFrames into one composite signal.

    Parameters
    ----------
    signals_dict : dict[str, pd.DataFrame]
        Mapping ``signal_name -> DataFrame`` of standardised signals.
    combination_method : str
        * ``"equal"`` -- equal-weight average.
        * ``"weighted"`` -- user-supplied weights via *weights*.
        * ``"rank_average"`` -- rank each signal cross-sectionally, then
          average ranks.
    weights : dict[str, float] or None
        Required when ``combination_method="weighted"``.
    rank_before_combine : bool
        If ``True``, convert each signal to ranks before combining.

    Returns
    -------
    pd.DataFrame
        Composite signal.
    """
    if not signals_dict:
        raise ValueError("signals_dict is empty")

    names = list(signals_dict.keys())
    dfs = list(signals_dict.values())

    # Optionally rank each signal
    if rank_before_combine:
        dfs = [df.rank(axis=1, pct=True) for df in dfs]

    if combination_method == "equal":
        combined = sum(dfs) / len(dfs)

    elif combination_method == "weighted":
        if weights is None:
            raise ValueError("weights required for method='weighted'")
        w = np.array([weights.get(n, 1.0) for n in names])
        w = w / w.sum()
        combined = sum(df * wi for df, wi in zip(dfs, w))

    elif combination_method == "rank_average":
        ranked = [df.rank(axis=1, pct=True) for df in dfs]
        combined = sum(ranked) / len(ranked)

    else:
        raise ValueError(
            f"Unknown combination_method: {combination_method}. "
            "Choose from 'equal', 'weighted', 'rank_average'."
        )

    return combined


def apply_signal_decay(
    signals: pd.DataFrame,
    halflife: int = 5,
) -> pd.DataFrame:
    """Apply exponential decay to stale signals.

    When a model is not re-run daily, yesterday's signal is decayed
    toward zero.  This prevents old forecasts from retaining full
    influence on portfolio construction.

    The decay kernel is:

    .. math::

        s_t^{\\text{decayed}} = s_t \\cdot \\exp\\!\\bigl(
            -\\ln(2)\\,\\Delta t / \\text{halflife}
        \\bigr)

    where :math:`\\Delta t` is the number of periods since the signal
    was last refreshed.

    Parameters
    ----------
    signals : pd.DataFrame
        Raw signals (columns = tickers).
    halflife : int
        Decay half-life in trading days.  After *halflife* days a stale
        signal retains 50 % of its original magnitude.

    Returns
    -------
    pd.DataFrame
        Decayed signals, same shape as input.
    """
    if halflife <= 0:
        raise ValueError(f"halflife must be positive, got {halflife}")

    decay_factor = np.log(2) / halflife
    decayed = signals.copy()
    n_rows = len(decayed)

    if n_rows <= 1:
        return decayed

    for i in range(1, n_rows):
        prev = decayed.iloc[i - 1]
        curr = signals.iloc[i]

        # Where the current signal is NaN (stale), decay the previous value
        stale_mask = curr.isna()
        fresh = curr.copy()
        fresh[stale_mask] = prev[stale_mask] * np.exp(-decay_factor)
        decayed.iloc[i] = fresh

    # Forward-fill any remaining NaNs using exponential decay from last
    # known value
    for col in decayed.columns:
        vals = decayed[col].values.astype(float)
        last_valid_idx = -1
        for i in range(len(vals)):
            if np.isfinite(vals[i]):
                last_valid_idx = i
            elif last_valid_idx >= 0:
                gap = i - last_valid_idx
                vals[i] = vals[last_valid_idx] * np.exp(-decay_factor * gap)
        decayed[col] = vals

    return decayed


# ============================================================================
# Convenience / Pipeline Helpers
# ============================================================================


def run_full_prediction_pipeline(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    model_names: Optional[List[str]] = None,
    tickers: List[str] = TICKER_LIST,
    horizon: int = PREDICTION_HORIZON,
    ensemble_method: str = "weighted",
    shrinkage_alpha: float = 0.5,
    signal_method: str = "zscore",
) -> Dict[str, Any]:
    """End-to-end prediction pipeline returning all intermediate artefacts.

    This helper orchestrates the common workflow:

    1. Generate predictions from each model.
    2. Compute ensemble weights from CV performance.
    3. Produce an ensemble prediction.
    4. Shrink toward historical returns.
    5. Build annualised expected returns and covariance matrix.
    6. Generate alpha signals.
    7. Construct Black--Litterman views.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.
    prices : pd.DataFrame
        Asset prices (for realised returns and covariance).
    model_names : list[str] or None
        Model identifiers.
    tickers : list[str]
        Asset tickers.
    horizon : int
        Prediction horizon.
    ensemble_method : str
        How to combine models: ``"weighted"`` or ``"equal"``.
    shrinkage_alpha : float
        Shrinkage parameter for expected returns.
    signal_method : str
        Signal generation method.

    Returns
    -------
    dict
        Keys: ``"individual_predictions"``, ``"ensemble_predictions"``,
        ``"expected_returns"``, ``"covariance"``, ``"signals"``,
        ``"bl_views"``, ``"diagnostics"``.
    """
    if model_names is None:
        model_names = ["ridge", "xgboost", "random_forest"]

    # Step 1: individual predictions
    individual = {}
    model_results = {}
    for mname in model_names:
        pred = predict_returns(
            features, model_name=mname, tickers=tickers, horizon=horizon
        )
        if not pred.empty:
            individual[mname] = pred
            # Attempt to load CV metrics for weight computation
            for ticker in pred.columns:
                target_name = f"{ticker}_ret_{horizon}d"
                try:
                    artefact = _default_cache.get(mname, target_name)
                    if mname not in model_results:
                        model_results[mname] = artefact
                except Exception:
                    pass

    # Step 2: ensemble weights
    if ensemble_method == "weighted" and model_results:
        try:
            weights = compute_ensemble_weights_from_cv(model_results)
        except ValueError:
            weights = None
    else:
        weights = None

    # Step 3: ensemble prediction
    ensemble = predict_returns_ensemble(
        features,
        model_names=list(individual.keys()) or model_names,
        tickers=tickers,
        horizon=horizon,
        weights=weights,
    )

    # Step 4: historical returns
    common = ensemble.columns.intersection(prices.columns)
    hist_returns = prices[common].pct_change().dropna()

    # Step 5: expected returns and covariance
    mu = build_expected_returns_shrinkage(
        ensemble, hist_returns, alpha=shrinkage_alpha, horizon=horizon
    )
    cov = build_covariance_from_predictions(
        ensemble, hist_returns, method="shrinkage"
    )

    # Step 6: signals
    signals = generate_alpha_signals(ensemble, method=signal_method)

    # Step 7: Black--Litterman views
    try:
        P, Q, omega = build_views_for_black_litterman(
            ensemble, confidence=0.6, tickers=list(common)
        )
        bl_views = {"P": P, "Q": Q, "omega_diag": omega}
    except Exception as exc:
        logger.warning("Could not build BL views: %s", exc)
        bl_views = None

    return {
        "individual_predictions": individual,
        "ensemble_predictions": ensemble,
        "ensemble_weights": weights,
        "expected_returns": mu,
        "covariance": cov,
        "signals": signals,
        "bl_views": bl_views,
        "tickers": list(common),
    }
