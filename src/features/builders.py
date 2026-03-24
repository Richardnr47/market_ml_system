import numpy as np
import pandas as pd


def build_market_features(
    df: pd.DataFrame,
    price_col: str,
    return_lags: list[int],
    vol_windows: list[int],
    momentum_windows: list[int],
    volume_windows: list[int],
) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    out["ret_1"] = out[price_col].pct_change()

    for lag in return_lags:
        out[f"ret_{lag}"] = out[price_col].pct_change(lag)

    for w in vol_windows:
        out[f"rv_{w}"] = out["ret_1"].rolling(w).std()

    for w in momentum_windows:
        out[f"mom_{w}"] = out[price_col] / out[price_col].shift(w) - 1.0

    for w in volume_windows:
        mean_ = out["volume"].rolling(w).mean()
        std_ = out["volume"].rolling(w).std()
        out[f"vol_z_{w}"] = (out["volume"] - mean_) / std_.replace(0, np.nan)

    out["hl_range"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["oc_change"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)
    out["body_to_range"] = (out["close"] - out["open"]).abs() / (
        (out["high"] - out["low"]).replace(0, np.nan)
    )

    return out