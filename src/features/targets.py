import pandas as pd
import logging


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