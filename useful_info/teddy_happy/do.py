import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pathlib import Path
from plots import redshift_scatter

from load import load
from llr import LocalLinearRegression
from metrics import evaluate_regression

def do(dataset):
    if dataset == "happy":
        X_train, y_train = load("Happy/happy_A")
        X_test_B, y_test_B = load("Happy/happy_B")
        X_test_C, y_test_C = load("Happy/happy_C")
        X_test_D, y_test_D = load("Happy/happy_D")
        print("===HAPPY===")
    else:
        X_train, y_train = load("Teddy/teddy_A")
        X_test_B, y_test_B = load("Teddy/teddy_B")
        X_test_C, y_test_C = load("Teddy/teddy_C")
        X_test_D, y_test_D = load("Teddy/teddy_D")
        print("===TEDDY===")

    y_test_B = pd.Series(y_test_B)
    y_test_C = pd.Series(y_test_C)
    y_test_D = pd.Series(y_test_D)

    model_catboost = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    
    model_catboost.fit(X_train, y_train)
    
    y_pred_B = model_catboost.predict(X_test_B)
    y_pred_C = model_catboost.predict(X_test_C)
    y_pred_D = model_catboost.predict(X_test_D)
    
    eval_B = evaluate_regression(y_test_B, y_pred_B)
    eval_C = evaluate_regression(y_test_C, y_pred_C)
    eval_D = evaluate_regression(y_test_D, y_pred_D)

    print("===CATBOOST===")
    print("EVAL B:", eval_B)
    print("EVAL C:", eval_C)
    print("EVAL D:", eval_D)



    model_mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
        )),
    ])
    
    model_mlp.fit(X_train, y_train)
    
    y_pred_B_mlp = model_mlp.predict(X_test_B)
    y_pred_C_mlp = model_mlp.predict(X_test_C)
    y_pred_D_mlp = model_mlp.predict(X_test_D)
    
    eval_B_mlp = evaluate_regression(y_test_B, y_pred_B_mlp)
    eval_C_mlp = evaluate_regression(y_test_C, y_pred_C_mlp)
    eval_D_mlp = evaluate_regression(y_test_D, y_pred_D_mlp)

    print("===MLP===")
    print("EVAL B:", eval_B_mlp)
    print("EVAL C:", eval_C_mlp)
    print("EVAL D:", eval_D_mlp)

    model_rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    
    model_rf.fit(X_train, y_train)
    
    y_pred_B_rf = model_rf.predict(X_test_B)
    y_pred_C_rf = model_rf.predict(X_test_C)
    y_pred_D_rf = model_rf.predict(X_test_D)
    
    eval_B_rf = evaluate_regression(y_test_B, y_pred_B_rf)
    eval_C_rf = evaluate_regression(y_test_C, y_pred_C_rf)
    eval_D_rf = evaluate_regression(y_test_D, y_pred_D_rf)

    print("===RF===")
    print("EVAL B:", eval_B_rf)
    print("EVAL C:", eval_C_rf)
    print("EVAL D:", eval_D_rf)

    model_llr = LocalLinearRegression(k=1000)
    model_llr.fit(X_train, y_train)
    
    y_pred_B_llr = model_llr.predict(X_test_B)
    y_pred_C_llr = model_llr.predict(X_test_C)
    y_pred_D_llr = model_llr.predict(X_test_D)
    
    eval_B_llr = evaluate_regression(y_test_B, y_pred_B_llr)
    eval_C_llr = evaluate_regression(y_test_C, y_pred_C_llr)
    eval_D_llr = evaluate_regression(y_test_D, y_pred_D_llr)

    print("===LLR===")
    print("EVAL B:", eval_B_llr)
    print("EVAL C:", eval_C_llr)
    print("EVAL D:", eval_D_llr)
    
    outdir = Path("plots")
    outdir.mkdir(exist_ok=True)
    
    models = {
        "catboost": (y_pred_B, y_pred_C, y_pred_D),
        "rf": (y_pred_B_rf, y_pred_C_rf, y_pred_D_rf),
        "mlp": (y_pred_B_mlp, y_pred_C_mlp, y_pred_D_mlp),
        "llr": (y_pred_B_llr, y_pred_C_llr, y_pred_D_llr),
    }
    
    for name, (pred_B, pred_C, pred_D) in models.items():
        if name == "llr":
            redshift_scatter(y_test_B, pred_B, outdir / dataset / f"{name}_B.png", 0.0, 1.0)
            redshift_scatter(y_test_C, pred_C, outdir / dataset / f"{name}_C.png", 0.0, 1.0)
            redshift_scatter(y_test_D, pred_D, outdir / dataset / f"{name}_D.png", 0.0, 1.0)
        else:
            redshift_scatter(y_test_B, pred_B, outdir / dataset / f"{name}_B.png")
            redshift_scatter(y_test_C, pred_C, outdir / dataset / f"{name}_C.png")
            redshift_scatter(y_test_D, pred_D, outdir / dataset / f"{name}_D.png")
