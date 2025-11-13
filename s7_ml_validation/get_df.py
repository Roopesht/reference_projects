"""
get_df.py
Functions to load the California Housing dataset and return pandas DataFrame + target.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_california_housing(as_frame=True):
    """
    Loads the California Housing dataset and returns (X_df, y_array, feature_names).
    as_frame=True returns a pandas DataFrame for X.
    """
    data = fetch_california_housing(as_frame=as_frame)
    X = data.frame if as_frame else pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target.values
    return X, y, data.feature_names

def dataset_summary(X, y, n_rows=5):
    """Print basic dataset information and return a dict summary."""
    print("=== Dataset Summary ===")
    print(f"Rows: {X.shape[0]}, Columns: {X.shape[1]}")
    print("\nColumn names:")
    print(list(X.columns))
    print("\nFirst few rows:")
    print(X.head(n_rows))
    print("\nTarget sample (first 10):")
    print(y[:10])
    print("=======================\n")
    return {
        "n_rows": X.shape[0],
        "n_cols": X.shape[1],
        "columns": list(X.columns)
    }
