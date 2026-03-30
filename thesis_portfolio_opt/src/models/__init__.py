"""Predictive modeling: training, evaluation, and prediction."""

from src.models.trainer import (
    MODELS,
    time_series_cv_splits,
    evaluate_predictions,
    get_feature_importance,
    train_and_evaluate,
    train_all_models,
    save_model,
)
from src.models.predict import (
    load_model,
    list_saved_models,
    predict_returns,
    predict_returns_ensemble,
    build_expected_returns,
    predict_at_date,
)

__all__ = [
    "MODELS",
    "time_series_cv_splits",
    "evaluate_predictions",
    "get_feature_importance",
    "train_and_evaluate",
    "train_all_models",
    "save_model",
    "load_model",
    "list_saved_models",
    "predict_returns",
    "predict_returns_ensemble",
    "build_expected_returns",
    "predict_at_date",
]
