from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted


class LocalLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, k: int = 50) -> None:
        self.k = k

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> Self:
        X = check_array(X)
        y = np.asarray(y).ravel()

        self.X_train_ = X
        self.y_train_ = y

        self._nn = NearestNeighbors(n_neighbors=self.k)
        self._nn.fit(self.X_train_)

        return self

    def predict(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        check_is_fitted(self, ["X_train_", "y_train_"])
        X = check_array(X)

        idx = self._nn.kneighbors(X, return_distance=False)
        preds = np.empty(X.shape[0], dtype=float)

        for i in range(X.shape[0]):
            Xn = self.X_train_[idx[i]]
            yn = self.y_train_[idx[i]]
            lr = LinearRegression()
            lr.fit(Xn, yn)
            preds[i] = lr.predict(X[i].reshape(1, -1))[0]

        return preds
