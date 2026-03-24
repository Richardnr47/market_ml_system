import numpy as np
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
        logger.info("[EVAL] Metrics: %s", metrics)

    return metrics