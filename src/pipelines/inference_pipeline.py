import numpy as np
import pandas as pd
import logging

from src.features.builders import build_market_features


def run_inference_pipeline(
    input_df: pd.DataFrame,
    feature_cols: list[str],
    state_model,
    predictor,
    price_col: str,
    window_size: int,
    logger: logging.Logger | None = None,
) -> dict:
    if logger:
        logger.info("[INFERENCE] Starting inference pipeline...")
        logger.info("[INFERENCE] Input rows: %s", len(input_df))

    df = input_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = build_market_features(
        df=df,
        price_col=price_col,
        return_lags=[1, 2, 5],
        vol_windows=[5, 10, 20],
        momentum_windows=[5, 10, 20],
        volume_windows=[5, 20],
        logger=logger,
    )

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    feat_df = df[feature_cols].copy().dropna().reset_index(drop=True)

    if logger:
        logger.info("[INFERENCE] Feature-ready rows after dropna: %s", len(feat_df))

    if len(feat_df) < window_size:
        raise ValueError(
            f"Not enough rows after feature engineering. Need at least {window_size}, got {len(feat_df)}"
        )

    X_latest_window = feat_df.iloc[-window_size:].to_numpy(dtype=float)
    X_latest = X_latest_window.reshape(1, -1)

    state = int(state_model.predict(X_latest)[0])
    probs = state_model.predict_proba(X_latest)
    X_meta = np.column_stack([X_latest, np.array([state]), probs])

    pred = float(predictor.predict(X_meta)[0])

    result = {
        "predicted_forward_return": pred,
        "state": state,
        "timestamp": str(df["timestamp"].iloc[-1]),
    }

    if logger:
        logger.info("[INFERENCE] Result: %s", result)

    return result