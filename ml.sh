#!/usr/bin/env bash
set -e

ROOT="ml_validation"
mkdir -p "$ROOT"
cd "$ROOT"

# requirements
cat > requirements.txt <<'REQ'
numpy
pandas
scikit-learn
matplotlib
REQ

# README
cat > README.md <<'MD'
# ml_validation

Beginner-friendly project demonstrating:
- Loading the California Housing dataset
- Creating regression and classification tasks (binary target derived from median house value)
- Feature engineering (polynomial features)
- Training a Ridge regression (regularized) and a Decision Tree classifier with hyperparameter tuning
- Evaluations and printed summaries of dataset and model performance

Run:
1. Install requirements:
   pip install -r requirements.txt
2. Run the pipeline:
   python main.py
MD

# 1. get_df.py - load dataset and provide basic info
cat > get_df.py <<'PY'
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
PY

# 2. clean_data.py - trivial cleaning (placeholder) and train/test split
cat > clean_data.py <<'PY'
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
PY

# 3. features.py - feature engineering utilities
cat > features.py <<'PY'
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
PY

# 4. regression.py - train and evaluate regression model
cat > regression.py <<'PY'
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
PY

# 5. classification.py - create binary target, train classifier with tuning, evaluate
cat > classification.py <<'PY'
"""
classification.py
Create binary target from regression target (threshold = median).
Train a classifier (DecisionTree by default) and tune 2+ hyperparameters with GridSearchCV.
Evaluate Precision, Recall, F1.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

def create_binary_target(y, threshold=None):
    """
    Creates binary target. If threshold is None, uses median.
    Returns numpy array of 0/1.
    """
    if threshold is None:
        threshold = np.median(y)
    y_bin = (y > threshold).astype(int)
    return y_bin, threshold

def train_and_evaluate_classifier(X_train, y_train, X_test, y_test, cv=5):
    """
    Trains DecisionTreeClassifier and tunes max_depth and min_samples_split.
    Returns best_estimator and metrics dict.
    """
    clf = DecisionTreeClassifier(random_state=42)
    param_grid = {
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10]
    }
    grid = GridSearchCV(clf, param_grid, cv=cv, scoring="f1", n_jobs=-1)
    print("Starting GridSearchCV for classifier...")
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"Best classifier params: {grid.best_params_}")
    y_pred = best.predict(X_test)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    metrics = {"Precision": precision, "Recall": recall, "F1": f1, "Accuracy": accuracy, "Report": report}
    return best, metrics, grid
PY

# 6. evaluate.py - helper printing results
cat > evaluate.py <<'PY'
"""
evaluate.py
Helpers to pretty-print evaluation metrics and insights.
"""
import json

def print_regression_metrics(metrics: dict):
    print("\n=== Regression Results ===")
    print(f"MAE : {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R2  : {metrics['R2']:.4f}")

def print_classification_metrics(metrics: dict):
    print("\n=== Classification Results ===")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1       : {metrics['F1']:.4f}")
    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print("\nClassification report:")
    print(metrics.get("Report", ""))

def summarize_insights(regression_grid, classifier_grid):
    """
    Create a short list of insights based on grid search results.
    This is a small automated summary; users should add domain interpretation.
    """
    insights = []
    # Regression insight
    try:
        best_deg = regression_grid.best_params_.get("poly__degree")
        best_alpha = regression_grid.best_params_.get("model__alpha")
        insights.append(f"Regression: Best polynomial degree = {best_deg}, Ridge alpha = {best_alpha}.")
    except Exception:
        insights.append("Regression: Grid search summary not available.")
    # Classifier insight
    try:
        clf_params = classifier_grid.best_params_
        insights.append(f"Classifier: Best params = {clf_params}.")
    except Exception:
        insights.append("Classifier: Grid search summary not available.")
    return insights
PY

# 7. main.py - orchestrates everything and prints dataset info and results
cat > main.py <<'PY'
"""
main.py
Run end-to-end: load data, clean, split, train regression and classifier,
evaluate and print results and insights.

Designed for beginners: prints shapes, columns, heads, and evaluation metrics.
"""
import numpy as np
import pandas as pd

from get_df import load_california_housing, dataset_summary
from clean_data import simple_clean, split_data
from regression import train_and_evaluate_regression
from classification import create_binary_target, train_and_evaluate_classifier
from evaluate import print_regression_metrics, print_classification_metrics, summarize_insights

def main():
    # 1. Load data
    X, y, feature_names = load_california_housing(as_frame=True)
    print("Loaded California Housing dataset.")
    summary = dataset_summary(X, y, n_rows=5)

    # 2. Clean data (dataset is mostly clean; this prints status)
    X_clean = simple_clean(X)

    # 3. Create classification target
    y_class, threshold = create_binary_target(y)
    print(f"Binary classification threshold (median of target) = {threshold:.4f}")
    # Ensure consistent split between regression and classification by using same indices
    # 4. Split data (80/20)
    X_train, X_test, y_train_reg, y_test_reg = split_data(X_clean, y, test_size=0.2, random_state=42)
    # For classification we need binary target split using same indices - recompute split
    # We'll use the same split indices by splitting X and using the same random_state and test_size
    # Simpler: create class labels for all, then split with same random_state
    from sklearn.model_selection import train_test_split
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_clean, y_class, test_size=0.2, random_state=42, stratify=None
    )

    # 5. Train regression (feature engineering + Ridge)
    degrees = [1, 2]   # try linear and quadratic
    alphas = [0.1, 1.0, 10.0]
    best_reg, reg_metrics, reg_grid = train_and_evaluate_regression(
        X_train, y_train_reg, X_test, y_test_reg, degrees=degrees, alphas=alphas, cv=5
    )
    print_regression_metrics(reg_metrics)

    # 6. Train classification (Decision Tree + tuning)
    best_clf, clf_metrics, clf_grid = train_and_evaluate_classifier(
        X_train_cls, y_train_cls, X_test_cls, y_test_cls, cv=5
    )
    print_classification_metrics(clf_metrics)

    # 7. Insights summary
    print("\n=== Insights ===")
    insights = summarize_insights(reg_grid, clf_grid)
    for i, ins in enumerate(insights, 1):
        print(f"{i}. {ins}")

    # 8. Simple recommendation based on metrics (beginner-friendly)
    print("\n=== Recommendation ===")
    if reg_metrics["R2"] > 0.6:
        print("- Regression model explains a fair amount of variance (R2 > 0.6). Polynomial features helped.")
    else:
        print("- Regression R2 is modest; consider more features, transformations, or regularization tuning.")

    if clf_metrics["F1"] > 0.7:
        print("- Classifier performs reasonably well (F1 > 0.7).")
    else:
        print("- Classifier needs improvement: tune more hyperparameters, try different models (LogisticRegression, KNN), or balance classes.")

if __name__ == "__main__":
    main()
PY

# Make script executable (optional)
chmod +x main.py

echo "Created ml_validation project."
echo "To run:"
echo "  cd $ROOT"
echo "  pip install -r requirements.txt"
echo "  python main.py"