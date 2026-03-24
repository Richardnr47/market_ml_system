import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class GMMStateModel:
    def __init__(self, n_components: int, random_state: int = 42) -> None:
        self.scaler = StandardScaler()
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
        )
        self._fitted = False
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> "GMMStateModel":
        print("[STATE] Fitting GMM state model...")
        print(f"[STATE] Input X shape: {X.shape}")
        print(f"[STATE] n_components={self.n_components}")

        Xs = self.scaler.fit_transform(X)
        print(f"[STATE] Scaled X shape: {Xs.shape}")

        self.model.fit(Xs)
        self._fitted = True

        print("[STATE] GMM state model fitted successfully")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check()
        print(f"[STATE] Predicting states for X shape: {X.shape}")
        Xs = self.scaler.transform(X)
        preds = self.model.predict(Xs)
        print(f"[STATE] Predicted state vector shape: {preds.shape}")
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check()
        print(f"[STATE] Predicting state probabilities for X shape: {X.shape}")
        Xs = self.scaler.transform(X)
        probs = self.model.predict_proba(Xs)
        print(f"[STATE] Predicted state probability matrix shape: {probs.shape}")
        return probs

    def _check(self) -> None:
        if not self._fitted:
            raise ValueError("State model is not fitted")