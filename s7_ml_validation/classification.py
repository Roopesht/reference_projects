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
