from dataclasses import dataclass

import numpy as np
import pandas as pd
import logging


@dataclass
class WindowBatch:
    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    feature_names: list[str]
    tickers: np.ndarray


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    timestamp_col: str,
    ticker_col: str,
    window_size: int,
    stride: int = 1,
    logger: logging.Logger | None = None,
) -> WindowBatch:
    if logger:
        logger.info("[WINDOWS] Starting rolling window creation per ticker...")
        logger.info("[WINDOWS] Input rows: %s", len(df))
        logger.info("[WINDOWS] Number of feature columns: %s", len(feature_cols))
        logger.info("[WINDOWS] window_size=%s stride=%s", window_size, stride)

    X_list, y_list, ts_list, ticker_list = [], [], [], []

    tickers = df[ticker_col].dropna().unique()

    for ticker in tickers:
        ticker_df = df[df[ticker_col] == ticker].sort_values(timestamp_col).reset_index(drop=True)

        X_values = ticker_df[feature_cols].to_numpy(dtype=float)
        y_values = ticker_df[target_col].to_numpy(dtype=float)
        ts_values = ticker_df[timestamp_col].to_numpy()

        created_for_ticker = 0

        for end_idx in range(window_size, len(ticker_df), stride):
            start_idx = end_idx - window_size
            x_window = X_values[start_idx:end_idx]
            y_val = y_values[end_idx]
            ts_val = ts_values[end_idx]

            if np.isnan(x_window).any() or np.isnan(y_val):
                continue

            X_list.append(x_window)
            y_list.append(y_val)
            ts_list.append(ts_val)
            ticker_list.append(ticker)
            created_for_ticker += 1

        if logger:
            logger.info(
                "[WINDOWS] Ticker=%s rows=%s valid_windows=%s",
                ticker,
                len(ticker_df),
                created_for_ticker,
            )

    if not X_list:
        raise ValueError("No valid windows were created")

    X = np.stack(X_list)
    y = np.array(y_list, dtype=float)
    ts = np.array(ts_list)
    ticker_arr = np.array(ticker_list)

    if logger:
        logger.info("[WINDOWS] X shape: %s", X.shape)
        logger.info("[WINDOWS] y shape: %s", y.shape)
        logger.info("[WINDOWS] timestamps shape: %s", ts.shape)
        logger.info("[WINDOWS] tickers shape: %s", ticker_arr.shape)

    return WindowBatch(
        X=X,
        y=y,
        timestamps=ts,
        feature_names=feature_cols,
        tickers=ticker_arr,
    )


def flatten_windows(X: np.ndarray, logger: logging.Logger | None = None) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {X.shape}")

    n, w, f = X.shape
    out = X.reshape(n, w * f)

    if logger:
        logger.info("[WINDOWS] Flattened shape: %s", out.shape)

    return out