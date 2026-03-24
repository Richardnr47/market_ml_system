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
    print("[FEATURES] Starting feature engineering...")
    out = df.copy()

    print("[FEATURES] Converting timestamp to datetime")
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    print("[FEATURES] Creating base return ret_1")
    out["ret_1"] = out[price_col].pct_change()

    print(f"[FEATURES] Creating lagged returns: {return_lags}")
    for lag in return_lags:
        out[f"ret_{lag}"] = out[price_col].pct_change(lag)

    print(f"[FEATURES] Creating rolling volatility windows: {vol_windows}")
    for w in vol_windows:
        out[f"rv_{w}"] = out["ret_1"].rolling(w).std()

    print(f"[FEATURES] Creating momentum windows: {momentum_windows}")
    for w in momentum_windows:
        out[f"mom_{w}"] = out[price_col] / out[price_col].shift(w) - 1.0

    print(f"[FEATURES] Creating volume z-score windows: {volume_windows}")
    for w in volume_windows:
        mean_ = out["volume"].rolling(w).mean()
        std_ = out["volume"].rolling(w).std()
        out[f"vol_z_{w}"] = (out["volume"] - mean_) / std_.replace(0, np.nan)

    print("[FEATURES] Creating candle structure features")
    out["hl_range"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["oc_change"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)
    out["body_to_range"] = (out["close"] - out["open"]).abs() / (
        (out["high"] - out["low"]).replace(0, np.nan)
    )

    print("[FEATURES] Feature engineering complete")
    return out