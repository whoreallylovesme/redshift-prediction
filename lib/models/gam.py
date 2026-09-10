from typing import Self

import numpy as np
from numpy.typing import NDArray
from pygam import LinearGAM
from sklearn.base import BaseEstimator, RegressorMixin


class GAMRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_splines: int = 20,
        lam: float = 0.6,
        max_iter: int = 1000,
    ) -> None:
        self.n_splines = n_splines
        self.lam = lam
        self.max_iter = max_iter

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> Self:
        self.gam_ = LinearGAM(
            n_splines=self.n_splines,
            lam=self.lam,
            max_iter=self.max_iter,
        ).fit(X, y)

        self.n_features_in_ = X.shape[1]
        return self

    def predict(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        pred_any = self.gam_.predict(X)
        pred: NDArray[np.float64] = np.asarray(pred_any, dtype=np.float64).ravel()
        return pred