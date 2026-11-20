"""
src/train_models.py
Trains and cross-validates Logistic Regression, Random Forest, SVM, and XGBoost models
using stratified 5-fold cross-validation strictly on the training partition.
"""

import json
import logging
import os
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

# Ensure local imports work cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import prepare_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict[str, float]:
    """Compute standard classification metrics including healthcare-critical specificity and FNR."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_prob)
    except (ValueError, TypeError):
        auc = 0.5

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Specificity": spec,
        "F1": f1,
        "ROC_AUC": auc,
        "FPR": fpr,
        "FNR": fnr,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def get_model_definitions(scale_pos_weight: float) -> dict[str, Any]:
    """Return model architectures configured with class balancing and reproducibility."""
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "SVM": CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced",
                dual=False,
                C=1.0,
                max_iter=2000,
                random_state=42,
            ),
            method="sigmoid",
            cv=3,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        ),
    }
    return models


def train_and_cross_validate(
    X_train: np.ndarray, y_train: np.ndarray, n_splits: int = 5, random_state: int = 42
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Run Stratified 5-Fold Cross Validation on the training set,
    record fold metrics, and fit final models on the entire training set.
    """
    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    scale_pos_weight = float(neg_count / pos_count)
    logger.info(
        f"Class counts - Negative: {neg_count}, Positive: {pos_count}. scale_pos_weight: {scale_pos_weight:.2f}"
    )

    models = get_model_definitions(scale_pos_weight)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_results = []

    for name, model in models.items():
        logger.info(f"Starting {n_splits}-Fold Cross-Validation for {name}...")
        fold_metrics = []
        start_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            # Clone / re-instantiate fresh model for fold
            fold_model = (
                joblib.load(joblib.dump(model, "temp_model.joblib")[0])
                if os.path.exists("temp_model.joblib")
                else None
            )
            if fold_model is None:
                fold_model = get_model_definitions(scale_pos_weight)[name]

            fold_model.fit(X_tr, y_tr)
            y_pred = fold_model.predict(X_val)

            if hasattr(fold_model, "predict_proba"):
                y_prob = fold_model.predict_proba(X_val)[:, 1]
            elif hasattr(fold_model, "decision_function"):
                df_scores = fold_model.decision_function(X_val)
                y_prob = 1 / (1 + np.exp(-df_scores))
            else:
                y_prob = y_pred.astype(float)

            metrics = compute_metrics(y_val, y_pred, y_prob)
            metrics["Model"] = name
            metrics["Fold"] = fold
            fold_metrics.append(metrics)

        elapsed = time.time() - start_time
        logger.info(f"Completed CV for {name} in {elapsed:.1f}s")
        cv_results.extend(fold_metrics)

    if os.path.exists("temp_model.joblib"):
        os.remove("temp_model.joblib")

    cv_df = pd.DataFrame(cv_results)

    # Compute summary mean and std
    summary_list = []
    metric_cols = [
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "ROC_AUC",
        "FPR",
        "FNR",
    ]
    for name in models:
        sub = cv_df[cv_df["Model"] == name]
        row = {"Model": name}
        for m in metric_cols:
            row[f"{m}_Mean"] = sub[m].mean()
            row[f"{m}_Std"] = sub[m].std()
        summary_list.append(row)
    cv_summary_df = pd.DataFrame(summary_list)

    # Train final models on entire training population
    trained_models = {}
    logger.info("Fitting final models on 100% of training data...")
    for name, model in models.items():
        t0 = time.time()
        logger.info(f"Fitting final {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        logger.info(f"Final {name} fitted in {time.time() - t0:.1f}s")

    return trained_models, cv_df, cv_summary_df


def save_models_and_configs(
    trained_models: dict[str, Any], output_root: str = "models"
):
    """Persist models into algorithm-specific directories alongside hyperparameter configurations."""
    folder_mapping = {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "SVM": "svm",
        "XGBoost": "xgboost",
    }

    configs = {}
    for name, model in trained_models.items():
        folder = folder_mapping[name]
        save_dir = os.path.join(output_root, folder)
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, "model.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Saved {name} to {model_path}")

        # Extract config details
        if name == "SVM":
            params = model.get_params()
            inner_params = model.estimator.get_params()
            configs[name] = {
                "wrapper": "CalibratedClassifierCV",
                "method": params.get("method"),
                "cv": params.get("cv"),
                "base_estimator": "LinearSVC",
                "C": inner_params.get("C"),
                "class_weight": inner_params.get("class_weight"),
                "dual": inner_params.get("dual"),
                "max_iter": inner_params.get("max_iter"),
            }
        else:
            configs[name] = {
                k: str(v) for k, v in model.get_params().items() if not callable(v)
            }

    config_path = os.path.join("results", "model_configurations.json")
    os.makedirs("results", exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(configs, f, indent=2)
    logger.info(f"Saved model configurations to {config_path}")


def run_training_pipeline() -> tuple[dict[str, Any], dict[str, Any]]:
    """Complete execution pipeline for model training and CV evaluation."""
    data = prepare_data()
    X_train = data["X_train"]
    y_train = data["y_train"]

    trained_models, cv_df, cv_summary_df = train_and_cross_validate(
        X_train, y_train, n_splits=5
    )

    cv_df.to_csv("results/cv_scores.csv", index=False)
    cv_summary_df.to_csv("results/cv_summary.csv", index=False)
    logger.info("Saved cv_scores.csv and cv_summary.csv to results/")

    save_models_and_configs(trained_models)
    return trained_models, data


if __name__ == "__main__":
    trained_models, data = run_training_pipeline()
    print("All four models trained and persisted successfully.")
