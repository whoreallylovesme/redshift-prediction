from sklearn.base import BaseEstimator


def load_estimator(estimator_code: str, expected_type: type = BaseEstimator) -> object:
    scope: dict[str, object] = {}
    exec(estimator_code, scope)

    est = scope["estimator"]

    if not isinstance(est, expected_type):
        raise TypeError(f"Loaded object is not a {expected_type.__name__}")

    return est
