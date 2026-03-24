import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import logging


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    if logger:
        logger.info("[EVAL] Regression metrics: %s", metrics)

    return metrics


def baseline_metrics(
    y_true: np.ndarray,
    y_train: np.ndarray,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    zero_pred = np.zeros_like(y_true, dtype=float)
    mean_pred = np.full_like(y_true, fill_value=float(np.mean(y_train)), dtype=float)

    metrics = {
        "baseline_zero_mae": float(mean_absolute_error(y_true, zero_pred)),
        "baseline_zero_r2": float(r2_score(y_true, zero_pred)),
        "baseline_mean_mae": float(mean_absolute_error(y_true, mean_pred)),
        "baseline_mean_r2": float(r2_score(y_true, mean_pred)),
    }

    if logger:
        logger.info("[EVAL] Baseline metrics: %s", metrics)

    return metrics


def directional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    true_up = y_true > 0
    pred_up = y_pred > 0

    direction_acc = float((true_up == pred_up).mean())

    predicted_positive_mask = pred_up
    predicted_negative_mask = ~pred_up

    positive_hit_rate = (
        float((y_true[predicted_positive_mask] > 0).mean())
        if predicted_positive_mask.sum() > 0
        else np.nan
    )
    negative_hit_rate = (
        float((y_true[predicted_negative_mask] <= 0).mean())
        if predicted_negative_mask.sum() > 0
        else np.nan
    )

    metrics = {
        "directional_accuracy": direction_acc,
        "predicted_positive_rate": float(pred_up.mean()),
        "predicted_negative_rate": float((~pred_up).mean()),
        "positive_hit_rate": float(positive_hit_rate) if not np.isnan(positive_hit_rate) else np.nan,
        "negative_hit_rate": float(negative_hit_rate) if not np.isnan(negative_hit_rate) else np.nan,
    }

    if logger:
        logger.info("[EVAL] Directional metrics: %s", metrics)

    return metrics


def correlation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    y_true_s = pd.Series(y_true)
    y_pred_s = pd.Series(y_pred)

    pearson = y_true_s.corr(y_pred_s, method="pearson")
    spearman = y_true_s.corr(y_pred_s, method="spearman")

    metrics = {
        "pearson_corr": float(pearson) if pd.notna(pearson) else np.nan,
        "spearman_corr": float(spearman) if pd.notna(spearman) else np.nan,
    }

    if logger:
        logger.info("[EVAL] Correlation metrics: %s", metrics)

    return metrics