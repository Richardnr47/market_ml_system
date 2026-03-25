import lightgbm as lgb
import numpy as np
import logging


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
        objective: str = "regression",
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger
        self.objective = objective
        common = {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
        }
        if objective == "classification":
            self.model = lgb.LGBMClassifier(objective="binary", **common)
        else:
            self.model = lgb.LGBMRegressor(objective="regression", **common)

        if self.logger:
            self.logger.info("[PREDICTOR] Initialized LightGBM predictor objective=%s", objective)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGBMReturnPredictor":
        if self.logger:
            self.logger.info("[PREDICTOR] Fitting predictor on X=%s y=%s", X.shape, y.shape)

        fit_y = y.astype(int) if self.objective == "classification" else y
        self.model.fit(X, fit_y)

        if self.logger:
            self.logger.info("[PREDICTOR] Fit complete")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.logger:
            self.logger.info("[PREDICTOR] Predicting on X=%s", X.shape)
        return self.model.predict(X)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if self.objective == "classification":
            if self.logger:
                self.logger.info("[PREDICTOR] Predicting probabilities on X=%s", X.shape)
            proba = self.model.predict_proba(X)
            return proba[:, 1]
        return self.predict(X)