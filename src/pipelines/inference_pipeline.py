import joblib
import numpy as np
import pandas as pd

from src.features.builders import build_market_features
from src.features.windows import make_windows, flatten_windows


def run_inference_pipeline(
    input_df: pd.DataFrame,
    feature_cols: list[str],
    state_model,
    predictor,
    price_col: str,
    window_size: int,
) -> dict:
    df = build_market_features(
        df=input_df,
        price_col=price_col,
        return_lags=[1, 2, 5],
        vol_windows=[5, 10, 20],
        momentum_windows=[5, 10, 20],
        volume_windows=[5, 20],
    )

    wb = make_windows(
        df=df,
        feature_cols=feature_cols,
        target_col=feature_cols[0],  # dummy, not used at final step
        timestamp_col="timestamp",
        window_size=window_size,
        stride=1,
    )

    X_latest = flatten_windows(wb.X)[-1:].copy()

    X_meta = np.column_stack([
        X_latest,
        state_model.predict(X_latest),
        state_model.predict_proba(X_latest),
    ])

    pred = float(predictor.predict(X_meta)[0])
    state = int(state_model.predict(X_latest)[0])

    return {
        "predicted_forward_return": pred,
        "state": state,
        "timestamp": str(wb.timestamps[-1]),
    }