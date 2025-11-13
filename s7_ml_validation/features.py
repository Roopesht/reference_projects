"""
features.py
Feature engineering helpers. Demonstrates PolynomialFeatures for regression.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def add_polynomial_features(X: pd.DataFrame, degree=2, include_bias=False):
    """
    Returns a pandas DataFrame with polynomial features.
    WARNING: can increase number of columns quickly.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    return pd.DataFrame(X_poly, columns=feature_names, index=X.index)

def create_regression_pipeline(degree=2, alpha=1.0, model=None):
    """
    Returns a Pipeline (scaler -> poly -> model). Model should accept alpha param if using Ridge/Lasso.
    If model is None, default to Ridge(alpha=alpha).
    """
    from sklearn.linear_model import Ridge
    if model is None:
        model = Ridge(alpha=alpha)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("model", model)
    ])
    return pipeline
