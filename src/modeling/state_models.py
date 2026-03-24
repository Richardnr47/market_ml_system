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

    def fit(self, X: np.ndarray) -> "GMMStateModel":
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check()
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check()
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)

    def _check(self) -> None:
        if not self._fitted:
            raise ValueError("State model is not fitted")