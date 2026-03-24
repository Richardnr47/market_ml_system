import pandas as pd


def build_forward_return_target(
    df: pd.DataFrame,
    price_col: str,
    horizon: int,
) -> pd.DataFrame:
    out = df.copy()
    out["target_forward_return"] = out[price_col].shift(-horizon) / out[price_col] - 1.0
    out["target_direction"] = (out["target_forward_return"] > 0).astype(int)
    return out