import numpy as np
import pandas as pd
import logging


def build_market_features(
    df: pd.DataFrame,
    price_col: str,
    return_lags: list[int],
    vol_windows: list[int],
    momentum_windows: list[int],
    volume_windows: list[int],
    ticker_col: str = "ticker",
    timestamp_col: str = "timestamp",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    if logger:
        logger.info("[FEATURES] Starting feature engineering (grouped by ticker)...")

    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    out = out.sort_values([ticker_col, timestamp_col]).reset_index(drop=True)

    grouped = out.groupby(ticker_col, group_keys=False)

    out["ret_1"] = grouped[price_col].pct_change()

    for lag in return_lags:
        out[f"ret_{lag}"] = grouped[price_col].pct_change(lag)

    for w in vol_windows:
        out[f"rv_{w}"] = grouped["ret_1"].transform(lambda s: s.rolling(w).std())

    for w in momentum_windows:
        out[f"mom_{w}"] = grouped[price_col].transform(lambda s: s / s.shift(w) - 1.0)

    for w in volume_windows:
        rolling_mean = grouped["volume"].transform(lambda s: s.rolling(w).mean())
        rolling_std = grouped["volume"].transform(lambda s: s.rolling(w).std())
        out[f"vol_z_{w}"] = (out["volume"] - rolling_mean) / rolling_std.replace(0, np.nan)

    out["hl_range"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["oc_change"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)
    out["body_to_range"] = (out["close"] - out["open"]).abs() / (
        (out["high"] - out["low"]).replace(0, np.nan)
    )

    if logger:
        logger.info("[FEATURES] Feature engineering complete")
        logger.info("[FEATURES] Output shape after features: %s", out.shape)

    return out