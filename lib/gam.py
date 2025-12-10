from sklearn.base import BaseEstimator, RegressorMixin
from pygam import LinearGAM


class GAMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_splines=20, lam=0.6, max_iter=1000):
        self.n_splines = n_splines
        self.lam = lam
        self.max_iter = max_iter

    def fit(self, X, y):
        self._gam = LinearGAM(
            n_splines=self.n_splines,
            lam=self.lam,
            max_iter=self.max_iter,
        )
        self._gam.fit(X, y)
        return self

    def predict(self, X):
        return self._gam.predict(X)
