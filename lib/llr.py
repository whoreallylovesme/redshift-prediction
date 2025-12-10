# >> DEPRECATED <<

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression

class LocalLinearRegression:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).ravel()
        return self

    def predict(self, X):
        X = np.asarray(X)
        n_test = X.shape[0]
        preds = np.zeros(n_test, dtype=float)

        dists = pairwise_distances(X, self.X_train)

        for i in range(n_test):
            idx = np.argsort(dists[i])[:self.k]
            X_neighbors = self.X_train[idx]
            y_neighbors = self.y_train[idx]

            lr = LinearRegression()
            lr.fit(X_neighbors, y_neighbors)
            preds[i] = lr.predict(X[i].reshape(1, -1))[0]

        return preds