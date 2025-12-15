from sklearn.model_selection import train_test_split, GridSearchCV
from .metrics import evaluate_regression


def feature_selection(df, features):
    cols = set(df.columns)
    selected = []
    missing = []
    for f in features:
        if f in cols:
            selected.append(f)
        else:
            missing.append(f)

    return selected, missing


def split_data(df, features, target, split_cfg):
    X = df[features]
    y = df[target]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_seed"],
        shuffle=split_cfg["shuffle"],
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=split_cfg["val_size"],
        random_state=split_cfg["random_seed"],
        shuffle=split_cfg["shuffle"],
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_full, y_train_full


def hyperparameter_optimization(estimator, X_train, y_train, cfg):
    if cfg["hyperopt"]["enabled"] == False:
        return {}

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=cfg["param_grid"],
        cv=cfg["hyperopt"]["cv"],
        scoring=cfg["hyperopt"]["scoring"],
        n_jobs=cfg["hyperopt"]["n_jobs"],
        verbose=cfg["hyperopt"]["verbose"],
    )
    gs.fit(X_train, y_train)

    return gs.best_params_


def train_model(estimator, X_train_full, y_train_full, best_params):
    if best_params:
        estimator.set_params(**best_params)
    estimator.fit(X_train_full, y_train_full)

    return estimator


def evaluate(model, X, y):
    pred = model.predict(X)

    return evaluate_regression(y, pred)
