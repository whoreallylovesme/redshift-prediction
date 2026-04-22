from sklearn.base import BaseEstimator


def load_estimator(estimator_code: str) -> BaseEstimator:
    scope: dict[str, object] = {}
    exec(estimator_code, scope)

    est = scope["estimator"]

    if not isinstance(est, BaseEstimator):
        raise TypeError("Loaded object is not a BaseEstimator")

    return est
