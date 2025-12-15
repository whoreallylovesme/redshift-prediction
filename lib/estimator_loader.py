from sklearn.base import clone

def load_estimator(cfg):
    scope = {}
    exec(cfg["estimator"], scope)
    est = clone(scope["estimator"])
    est.set_params(**cfg["estimator_params"])

    return est
