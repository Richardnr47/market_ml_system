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

    # preserve order and uniqueness
    seen = set()
    ordered_unique = []
    for c in selected:
        if c not in seen:
            ordered_unique.append(c)
            seen.add(c)

    return ordered_unique