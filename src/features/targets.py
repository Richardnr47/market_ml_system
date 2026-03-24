import pandas as pd
import logging


def build_forward_return_target(
    df: pd.DataFrame,
    price_col: str,
    horizon: int,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    if logger:
        logger.info("[TARGET] Building forward return target with horizon=%s", horizon)

    out = df.copy()
    out["target_forward_return"] = out[price_col].shift(-horizon) / out[price_col] - 1.0
    out["target_direction"] = (out["target_forward_return"] > 0).astype(int)

    if logger:
        logger.info("[TARGET] Target columns created")

    return out