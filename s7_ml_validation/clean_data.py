"""
clean_data.py
Lightweight cleaning utilities and train/test splitting.
Designed for this dataset which is already clean.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def simple_clean(X: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder cleaning step. For California Housing the dataset is clean,
    but this function demonstrates what you'd normally do.
    - Check missing values
    - Fill or drop if needed
    """
    X_clean = X.copy()
    # Print missingness summary
    missing = X_clean.isnull().sum()
    if missing.sum() > 0:
        print("Missing values found. Filling with median for each column.")
        for c in X_clean.columns:
            X_clean[c] = X_clean[c].fillna(X_clean[c].median())
    else:
        print("No missing values detected.")
    return X_clean

def split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    """Wrapper around sklearn train_test_split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    print(f"Train rows: {X_train.shape[0]}, Test rows: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test
