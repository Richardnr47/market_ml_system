import pandas as pd
import logging
import numpy as np


def build_forward_return_target(
    df: pd.DataFrame,
    price_col: str,
    horizon: int,
    ticker_col: str = "ticker",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    if logger:
        logger.info("[TARGET] Building forward return target per ticker with horizon=%s", horizon)

    out = df.copy()

    future_price = out.groupby(ticker_col)[price_col].shift(-horizon)
    out["target_forward_return"] = future_price / out[price_col] - 1.0
    out["target_direction"] = (out["target_forward_return"] > 0).astype(int)

    if logger:
        logger.info("[TARGET] Target columns created")
        valid_target = out["target_forward_return"].dropna()
        if len(valid_target) > 0:
            logger.info("[TARGET] Target stats:\n%s", valid_target.describe().to_string())
        else:
            logger.warning("[TARGET] No valid target values found")

    return out


def _forward_window_max(values: np.ndarray, horizon: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if horizon <= 0:
        return out
    for i in range(n):
        end = i + 1 + horizon
        if end > n:
            continue
        out[i] = float(np.nanmax(values[i + 1 : end]))
    return out


def build_event_label_target(
    df: pd.DataFrame,
    price_col: str,
    horizon: int,
    up_pct: float,
    ticker_col: str = "ticker",
    high_col: str = "high",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    if logger:
        logger.info(
            "[TARGET] Building event label target: up_%.4f_within_%s_bars",
            up_pct,
            horizon,
        )

    out = df.copy()
    future_max = out.groupby(ticker_col)[high_col].transform(
        lambda s: pd.Series(_forward_window_max(s.to_numpy(dtype=float), horizon), index=s.index)
    )
    out["target_future_max_return"] = future_max / out[price_col] - 1.0
    out["target_event_up"] = (out["target_future_max_return"] >= up_pct).astype(float)
    out.loc[out["target_future_max_return"].isna(), "target_event_up"] = np.nan

    if logger:
        valid = out["target_event_up"].dropna()
        if len(valid) > 0:
            logger.info(
                "[TARGET] Event label positives=%s negatives=%s positive_rate=%.4f",
                int((valid > 0.5).sum()),
                int((valid <= 0.5).sum()),
                float(valid.mean()),
            )
        else:
            logger.warning("[TARGET] No valid event labels found")

    return out