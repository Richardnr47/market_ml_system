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
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)