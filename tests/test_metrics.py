"""
tests/test_metrics.py
Unit tests verifying accuracy of performance and fairness calculations.
"""

import numpy as np

from src.fairness_analysis import compute_bootstrap_ci
from src.train_models import compute_metrics


def test_standard_metrics_calculation():
    """Verify standard classification metrics against hand-calculated ground truths."""
    # Ground truth: 2 positive, 2 negative
    # Pred: 1 TP, 1 FN, 1 FP, 1 TN
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.2, 0.8, 0.1])

    m = compute_metrics(y_true, y_pred, y_prob)

    assert m["Accuracy"] == 0.50
    assert m["Precision"] == 0.50
    assert m["Recall"] == 0.50
    assert m["Specificity"] == 0.50
    assert m["F1"] == 0.50
    assert m["FPR"] == 0.50
    assert m["FNR"] == 0.50
    assert m["TP"] == 1
    assert m["TN"] == 1
    assert m["FP"] == 1
    assert m["FN"] == 1
    assert m["ROC_AUC"] == 0.75


def test_extreme_zero_division():
    """Verify metrics handle edge cases without crashing or invalid divisions."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.1, 0.2])

    m = compute_metrics(y_true, y_pred, y_prob)
    assert m["Accuracy"] == 1.0
    assert m["Recall"] == 0.0
    assert m["Precision"] == 0.0
    assert m["Specificity"] == 1.0


def test_bootstrap_confidence_intervals():
    """Verify that bootstrap confidence intervals produce ordered, bounded limits."""
    y_true = np.random.binomial(1, 0.2, 500)
    y_pred = y_true.copy()  # perfect predictions
    ci = compute_bootstrap_ci(y_true, y_pred, n_bootstraps=50)

    assert ci["Recall_CI_Lower"] <= ci["Recall_CI_Upper"]
    assert 0.0 <= ci["Recall_CI_Lower"] <= 1.0
    assert 0.0 <= ci["Recall_CI_Upper"] <= 1.0
    assert ci["FNR_CI_Lower"] <= ci["FNR_CI_Upper"]
