"""
regression.py
Train a Ridge regression with polynomial features and evaluate MAE, RMSE, R2.
Uses GridSearchCV to tune alpha (regularization strength).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def train_and_evaluate_regression(X_train, y_train, X_test, y_test, degrees=[1,2], alphas=[0.1,1.0,10.0], cv=5):
    """
    Trains regression models with combinations of polynomial degree and Ridge alpha using GridSearchCV.
    Returns best_estimator_ and a dict of test metrics.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(include_bias=False)),
        ("model", Ridge())
    ])

    param_grid = {
        "poly__degree": degrees,
        "model__alpha": alphas
    }

    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    print("Starting GridSearchCV for regression...")
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"Best regression params: {grid.best_params_}")
    # Predict
    y_pred = best.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return best, metrics, grid
