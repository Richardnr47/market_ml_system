import numpy as np
import pandas as pd

from src.features.builders import build_market_features


def run_inference_pipeline(
    input_df: pd.DataFrame,
    feature_cols: list[str],
    state_model,
    predictor,
    price_col: str,
    window_size: int,
) -> dict:
    print("[INFERENCE] Starting inference pipeline...")
    print(f"[INFERENCE] Input rows: {len(input_df)}")

    df = input_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("[INFERENCE] Building market features...")
    df = build_market_features(
        df=df,
        price_col=price_col,
        return_lags=[1, 2, 5],
        vol_windows=[5, 10, 20],
        momentum_windows=[5, 10, 20],
        volume_windows=[5, 20],
    )

    print("[INFERENCE] Selecting required feature columns...")
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"[INFERENCE] Missing required features: {missing_features}")

    feat_df = df[feature_cols].copy()
    feat_df = feat_df.dropna().reset_index(drop=True)

    print(f"[INFERENCE] Feature-ready rows after dropna: {len(feat_df)}")
    if len(feat_df) < window_size:
        raise ValueError(
            f"[INFERENCE] Not enough rows after feature engineering. "
            f"Need at least {window_size}, got {len(feat_df)}"
        )

    X_latest_window = feat_df.iloc[-window_size:].to_numpy(dtype=float)
    X_latest = X_latest_window.reshape(1, -1)

    print(f"[INFERENCE] Latest flattened window shape: {X_latest.shape}")

    print("[INFERENCE] Predicting state and state probabilities...")
    state = int(state_model.predict(X_latest)[0])
    probs = state_model.predict_proba(X_latest)

    X_meta = np.column_stack([X_latest, np.array([state]), probs])
    print(f"[INFERENCE] Meta feature shape: {X_meta.shape}")

    print("[INFERENCE] Running predictor...")
    pred = float(predictor.predict(X_meta)[0])

    result = {
        "predicted_forward_return": pred,
        "state": state,
        "timestamp": str(df["timestamp"].iloc[-1]),
    }

    print(f"[INFERENCE] Result: {result}")
    return result