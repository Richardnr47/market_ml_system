import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    print("[EVAL] Calculating regression metrics...")
    print(f"[EVAL] y_true shape: {y_true.shape}")
    print(f"[EVAL] y_pred shape: {y_pred.shape}")

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    print(f"[EVAL] Metrics: {metrics}")
    return metrics