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
    # Technical indicators
    # --------------------------------
    if feature_flags.get("technical_indicators", False):
        # Rolling VWAP over short/medium windows.
        for w in [10, 20]:
            pv_roll = grouped.apply(
                lambda g: (g["close"] * g["volume"]).rolling(w).sum()
            ).reset_index(level=0, drop=True)
            vol_roll = grouped["volume"].transform(lambda s: s.rolling(w).sum())
            vwap_col = f"vwap_{w}"
            out[vwap_col] = pv_roll / vol_roll.replace(0, np.nan)
            out[f"vwap_dist_{w}"] = out["close"] / out[vwap_col].replace(0, np.nan) - 1.0

        # ATR and normalized ATR proxies.
        prev_close = grouped["close"].shift(1)
        tr = pd.concat(
            [
                out["high"] - out["low"],
                (out["high"] - prev_close).abs(),
                (out["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["tr"] = tr
        out["atr_14"] = grouped["tr"].transform(lambda s: s.rolling(14).mean())
        out["natr_14"] = out["atr_14"] / out["close"].replace(0, np.nan)

        # EMA trend and slope features.
        ema_12 = grouped["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
        ema_26 = grouped["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
        out["ema_fast_12"] = ema_12
        out["ema_slow_26"] = ema_26
        out["ema_spread_12_26"] = (ema_12 - ema_26) / out["close"].replace(0, np.nan)
        out["ema_fast_slope_3"] = grouped["ema_fast_12"].transform(lambda s: s.diff(3))

    # --------------------------------
    # Signal lab v1 (candidate alpha pack)
    # --------------------------------
    if feature_flags.get("signal_lab_v1", False):
        # Mean-reversion proxy via Bollinger-style z-score.
        close_ma_20 = grouped["close"].transform(lambda s: s.rolling(20).mean())
        close_std_20 = grouped["close"].transform(lambda s: s.rolling(20).std())
        out["bb_z_20"] = (out["close"] - close_ma_20) / close_std_20.replace(0, np.nan)

        # RSI(14) momentum oscillator.
        out["_delta_close"] = grouped["close"].diff()
        out["_up_move"] = out["_delta_close"].clip(lower=0)
        out["_down_move"] = (-out["_delta_close"]).clip(lower=0)
        avg_up_14 = grouped["_up_move"].transform(lambda s: s.rolling(14).mean())
        avg_down_14 = grouped["_down_move"].transform(lambda s: s.rolling(14).mean())
        rs_14 = avg_up_14 / avg_down_14.replace(0, np.nan)
        out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs_14))

        # Stochastic %K(14) for range position.
        low_14 = grouped["low"].transform(lambda s: s.rolling(14).min())
        high_14 = grouped["high"].transform(lambda s: s.rolling(14).max())
        out["stoch_k_14"] = (out["close"] - low_14) / (high_14 - low_14).replace(0, np.nan)

        # Breakout / breakdown distance from rolling extremes.
        roll_high_20 = grouped["high"].transform(lambda s: s.rolling(20).max())
        roll_low_20 = grouped["low"].transform(lambda s: s.rolling(20).min())
        out["breakout_up_20"] = out["close"] / roll_high_20.replace(0, np.nan) - 1.0
        out["breakout_down_20"] = out["close"] / roll_low_20.replace(0, np.nan) - 1.0

        # Volatility compression and trend-to-volatility ratio.
        if "rv_5" not in out.columns:
            out["rv_5"] = grouped["ret_1"].transform(lambda s: s.rolling(5).std())
        if "rv_20" not in out.columns:
            out["rv_20"] = grouped["ret_1"].transform(lambda s: s.rolling(20).std())
        out["rv_ratio_5_20"] = out["rv_5"] / out["rv_20"].replace(0, np.nan)

        if "ema_spread_12_26" in out.columns and "natr_14" in out.columns:
            out["trend_to_vol_ema"] = out["ema_spread_12_26"] / out["natr_14"].replace(0, np.nan)

        # Cross-sectional relative-strength ranks per timestamp.
        by_ts = out.groupby(timestamp_col, group_keys=False)
        if "ret_1" in out.columns:
            out["cs_rank_ret_1"] = by_ts["ret_1"].rank(pct=True)
        if "mom_5" in out.columns:
            out["cs_rank_mom_5"] = by_ts["mom_5"].rank(pct=True)
        if "natr_14" in out.columns:
            out["cs_rank_natr_14"] = by_ts["natr_14"].rank(pct=True)
        if "vwap_dist_10" in out.columns:
            out["cs_rank_vwap_dist_10"] = by_ts["vwap_dist_10"].rank(pct=True)
        out["cs_rank_volume"] = by_ts["volume"].rank(pct=True)

        out = out.drop(columns=["_delta_close", "_up_move", "_down_move"], errors="ignore")

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