import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import logging
from hmmlearn.hmm import GaussianHMM


class GMMStateModel:
    def __init__(
        self,
        n_components: int,
        random_state: int = 42,
        logger: logging.Logger | None = None,
    ) -> None:
        self.scaler = StandardScaler()
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
        )
        self._fitted = False
        self.n_components = n_components
        self.logger = logger

    def fit(self, X: np.ndarray) -> "GMMStateModel":
        if self.logger:
            self.logger.info("[STATE] Fitting GMM state model on X=%s", X.shape)

        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self._fitted = True

        if self.logger:
            self.logger.info("[STATE] GMM fitted successfully")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check()
        Xs = self.scaler.transform(X)
        preds = self.model.predict(Xs)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check()
        Xs = self.scaler.transform(X)
        probs = self.model.predict_proba(Xs)
        return probs

    def _check(self) -> None:
        if not self._fitted:
            raise ValueError("State model is not fitted")


class HMMStateModel:
    def __init__(
        self,
        n_components: int,
        random_state: int = 42,
        logger: logging.Logger | None = None,
    ) -> None:
        self.scaler = StandardScaler()
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=200,
            random_state=random_state,
        )
        self._fitted = False
        self.n_components = n_components
        self.logger = logger

    def fit(self, X: np.ndarray) -> "HMMStateModel":
        if self.logger:
            self.logger.info("[STATE] Fitting HMM state model on X=%s", X.shape)

        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self._fitted = True

        if self.logger:
            self.logger.info("[STATE] HMM fitted successfully")

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