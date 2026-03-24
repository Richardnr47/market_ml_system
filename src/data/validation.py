import pandas as pd
import logging


REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def validate_market_df(df: pd.DataFrame, logger: logging.Logger | None = None) -> None:
    if logger:
        logger.info("[VALIDATION] Validating input dataframe...")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Input dataframe is empty")

    if (df["close"] <= 0).any():
        raise ValueError("Found non-positive close prices")

    if (df["volume"] < 0).any():
        raise ValueError("Found negative volume")

    if logger:
        logger.info("[VALIDATION] Validation passed")