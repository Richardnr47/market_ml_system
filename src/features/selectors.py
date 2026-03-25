def select_feature_columns(df_columns, feature_flags: dict) -> list[str]:
    selected = []

    if feature_flags.get("price_basic", True):
        selected.extend(
            [c for c in df_columns if c.startswith("ret_") or c.startswith("rv_") or c.startswith("mom_")]
        )

    if feature_flags.get("candle_structure", True):
        selected.extend(
            [
                c
                for c in [
                    "hl_range",
                    "oc_change",
                    "body_to_range",
                    "close_location",
                    "buy_pressure_proxy",
                    "sell_pressure_proxy",
                ]
                if c in df_columns
            ]
        )

    if feature_flags.get("volume_regime", True):
        selected.extend(
            [c for c in df_columns if c.startswith("vol_z_") or c.startswith("vol_ratio_")]
        )

    if feature_flags.get("dollar_volume", True):
        selected.extend(
            [c for c in df_columns if c.startswith("dollar_")]
        )

    if feature_flags.get("flow_proxies", True):
        selected.extend(
            [
                c
                for c in df_columns
                if c.startswith("cvd_")
                or c.startswith("delta_")
                or c == "signed_volume_proxy"
            ]
        )

    if feature_flags.get("effort_result", True):
        selected.extend(
            [
                c
                for c in [
                    "effort_result_ratio",
                    "volume_per_abs_ret",
                    "abs_ret_per_volume",
                    "range_per_volume",
                ]
                if c in df_columns
            ]
        )

    if feature_flags.get("technical_indicators", False):
        selected.extend(
            [
                c
                for c in [
                    "vwap_10",
                    "vwap_20",
                    "vwap_dist_10",
                    "vwap_dist_20",
                    "tr",
                    "atr_14",
                    "natr_14",
                    "ema_fast_12",
                    "ema_slow_26",
                    "ema_spread_12_26",
                    "ema_fast_slope_3",
                ]
                if c in df_columns
            ]
        )

    if feature_flags.get("compression", False):
        selected.extend([c for c in df_columns if c.startswith("compression_")])

    if feature_flags.get("range_position", False):
        selected.extend([c for c in df_columns if c.startswith("range_pos_") or c.startswith("range_dist_")])

    if feature_flags.get("trend_quality", False):
        selected.extend([c for c in df_columns if c.startswith("trend_efficiency_") or c.startswith("trend_quality_") or c.startswith("trend_slope_")])

    if feature_flags.get("market_context", False):
        selected.extend([c for c in df_columns if c.startswith("mkt_") or c.startswith("rel_")])

    if feature_flags.get("signal_lab_v1", False):
        mean_rev_on = feature_flags.get("signal_lab_mean_reversion", True)
        osc_on = feature_flags.get("signal_lab_oscillator", True)
        breakout_on = feature_flags.get("signal_lab_breakout", True)
        vol_regime_on = feature_flags.get("signal_lab_volatility", True)
        cross_sectional_on = feature_flags.get("signal_lab_cross_sectional", True)
        selected.extend(
            [
                c
                for c in df_columns
                if (
                    mean_rev_on and c.startswith("bb_z_")
                    or osc_on and (c.startswith("rsi_") or c.startswith("stoch_"))
                    or breakout_on and c.startswith("breakout_")
                    or vol_regime_on and (c.startswith("rv_ratio_") or c.startswith("trend_to_vol_"))
                    or cross_sectional_on and c.startswith("cs_rank_")
                )
            ]
        )

    # preserve order and uniqueness
    seen = set()
    ordered_unique = []
    for c in selected:
        if c not in seen:
            ordered_unique.append(c)
            seen.add(c)

    return ordered_unique