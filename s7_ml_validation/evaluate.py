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
