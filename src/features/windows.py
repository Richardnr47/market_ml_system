from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WindowBatch:
    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    feature_names: list[str]


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    timestamp_col: str,
    window_size: int,
    stride: int = 1,
) -> WindowBatch:
    X_values = df[feature_cols].to_numpy(dtype=float)
    y_values = df[target_col].to_numpy(dtype=float)
    ts_values = df[timestamp_col].to_numpy()

    X_list, y_list, ts_list = [], [], []

    for end_idx in range(window_size, len(df), stride):
        start_idx = end_idx - window_size
        x_window = X_values[start_idx:end_idx]
        y_val = y_values[end_idx]
        ts_val = ts_values[end_idx]

        if np.isnan(x_window).any() or np.isnan(y_val):
            continue

        X_list.append(x_window)
        y_list.append(y_val)
        ts_list.append(ts_val)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=float)
    ts = np.array(ts_list)

    return WindowBatch(X=X, y=y, timestamps=ts, feature_names=feature_cols)


def flatten_windows(X: np.ndarray) -> np.ndarray:
    n, w, f = X.shape
    return X.reshape(n, w * f)