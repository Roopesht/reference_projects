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
    print (y_class)
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
