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
    feature_flags: dict,
    ticker_col: str = "ticker",
    timestamp_col: str = "timestamp",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    if logger:
        logger.info("[FEATURES] Starting feature engineering (grouped by ticker)...")
        logger.info("[FEATURES] Feature flags: %s", feature_flags)

    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    out = out.sort_values([ticker_col, timestamp_col]).reset_index(drop=True)

    grouped = out.groupby(ticker_col, group_keys=False)

    # --------------------------------
    # Price basic
    # --------------------------------
    if feature_flags.get("price_basic", True):
        out["ret_1"] = grouped[price_col].pct_change()

        for lag in return_lags:
            out[f"ret_{lag}"] = grouped[price_col].pct_change(lag)

        for w in vol_windows:
            out[f"rv_{w}"] = grouped["ret_1"].transform(lambda s: s.rolling(w).std())

        for w in momentum_windows:
            out[f"mom_{w}"] = grouped[price_col].transform(lambda s: s / s.shift(w) - 1.0)

    # --------------------------------
    # Candle structure
    # --------------------------------
    if feature_flags.get("candle_structure", True):
        raw_range = (out["high"] - out["low"]).replace(0, np.nan)

        out["hl_range"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
        out["oc_change"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)
        out["body_to_range"] = (out["close"] - out["open"]).abs() / raw_range
        out["close_location"] = (out["close"] - out["low"]) / raw_range
        out["buy_pressure_proxy"] = (out["close"] - out["low"]) / raw_range
        out["sell_pressure_proxy"] = (out["high"] - out["close"]) / raw_range

    # --------------------------------
    # Volume regime
    # --------------------------------
    if feature_flags.get("volume_regime", True):
        for w in volume_windows:
            rolling_mean = grouped["volume"].transform(lambda s: s.rolling(w).mean())
            rolling_std = grouped["volume"].transform(lambda s: s.rolling(w).std())
            out[f"vol_z_{w}"] = (out["volume"] - rolling_mean) / rolling_std.replace(0, np.nan)

        vol_mean_5 = grouped["volume"].transform(lambda s: s.rolling(5).mean())
        vol_mean_20 = grouped["volume"].transform(lambda s: s.rolling(20).mean())
        out["vol_ratio_5_20"] = vol_mean_5 / vol_mean_20.replace(0, np.nan)
        out["vol_ratio_1_20"] = out["volume"] / vol_mean_20.replace(0, np.nan)

    # --------------------------------
    # Dollar volume
    # --------------------------------
    if feature_flags.get("dollar_volume", True):
        out["dollar_volume"] = out["close"] * out["volume"]
        dv_mean_20 = grouped["dollar_volume"].transform(lambda s: s.rolling(20).mean())
        dv_std_20 = grouped["dollar_volume"].transform(lambda s: s.rolling(20).std())
        out["dollar_vol_z_20"] = (out["dollar_volume"] - dv_mean_20) / dv_std_20.replace(0, np.nan)

    # --------------------------------
    # Flow / CVD proxies
    # --------------------------------
    if feature_flags.get("flow_proxies", True):
        raw_range = (out["high"] - out["low"]).replace(0, np.nan)
        close_location = (out["close"] - out["low"]) / raw_range

        out["delta_proxy"] = out["volume"] * (2.0 * close_location - 1.0)
        out["signed_volume_proxy"] = np.sign(
            ((out["close"] - out["open"]) / out["open"].replace(0, np.nan)).fillna(0.0)
        ) * out["volume"]

        out["cvd_proxy"] = grouped["delta_proxy"].cumsum()
        out["cvd_proxy_change_5"] = grouped["cvd_proxy"].transform(lambda s: s - s.shift(5))
        out["cvd_proxy_slope_5"] = grouped["cvd_proxy"].transform(lambda s: (s - s.shift(5)) / 5.0)

        cvd_mean_20 = grouped["cvd_proxy"].transform(lambda s: s.rolling(20).mean())
        cvd_std_20 = grouped["cvd_proxy"].transform(lambda s: s.rolling(20).std())
        out["cvd_proxy_z_20"] = (out["cvd_proxy"] - cvd_mean_20) / cvd_std_20.replace(0, np.nan)

        out["delta_proxy_mean_5"] = grouped["delta_proxy"].transform(lambda s: s.rolling(5).mean())
        out["delta_proxy_mean_20"] = grouped["delta_proxy"].transform(lambda s: s.rolling(20).mean())
        out["delta_proxy_std_20"] = grouped["delta_proxy"].transform(lambda s: s.rolling(20).std())

    # --------------------------------
    # Effort vs result
    # --------------------------------
    if feature_flags.get("effort_result", True):
        if "ret_1" not in out.columns:
            out["ret_1"] = grouped[price_col].pct_change()

        if "oc_change" not in out.columns:
            out["oc_change"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)

        abs_ret_1 = out["ret_1"].abs()
        abs_oc_change = out["oc_change"].abs()

        out["effort_result_ratio"] = out["volume"] / abs_oc_change.replace(0, np.nan)
        out["volume_per_abs_ret"] = out["volume"] / abs_ret_1.replace(0, np.nan)
        out["abs_ret_per_volume"] = abs_ret_1 / out["volume"].replace(0, np.nan)
        out["range_per_volume"] = (out["high"] - out["low"]) / out["volume"].replace(0, np.nan)

    if logger:
        logger.info("[FEATURES] Feature engineering complete")
        logger.info("[FEATURES] Output shape after features: %s", out.shape)

    return out