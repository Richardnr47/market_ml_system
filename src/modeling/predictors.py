import lightgbm as lgb
import numpy as np


class LGBMReturnPredictor:
    def __init__(
        self,
        learning_rate: float,
        n_estimators: int,
        num_leaves: int,
        max_depth: int,
        subsample: float,
        colsample_bytree: float,
        random_state: int,
    ) -> None:
        print("[PREDICTOR] Initializing LightGBM predictor...")
        self.model = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGBMReturnPredictor":
        print("[PREDICTOR] Fitting LightGBM predictor...")
        print(f"[PREDICTOR] X shape: {X.shape}")
        print(f"[PREDICTOR] y shape: {y.shape}")
        self.model.fit(X, y)
        print("[PREDICTOR] Predictor fitted successfully")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        print(f"[PREDICTOR] Predicting for X shape: {X.shape}")
        preds = self.model.predict(X)
        print(f"[PREDICTOR] Prediction vector shape: {preds.shape}")
        return preds