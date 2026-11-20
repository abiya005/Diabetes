"""
src/evaluate_models.py
Evaluates trained models on the held-out test partition across all 9 required metrics,
generates comparison tables, and plots research-quality evaluation figures.
"""

import os
import sys
import tempfile

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
import matplotlib

matplotlib.use("Agg")
import logging
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Ensure local imports work cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import prepare_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Colors for models
MODEL_COLORS = {
    "Logistic Regression": "#2b5c8f",
    "Random Forest": "#2ca02c",
    "SVM": "#9467bd",
    "XGBoost": "#d95f02",
}


def load_all_models(models_dir: str = "models") -> dict[str, Any]:
    """Load all four serialized models."""
    model_paths = {
        "Logistic Regression": os.path.join(
            models_dir, "logistic_regression", "model.joblib"
        ),
        "Random Forest": os.path.join(models_dir, "random_forest", "model.joblib"),
        "SVM": os.path.join(models_dir, "svm", "model.joblib"),
        "XGBoost": os.path.join(models_dir, "xgboost", "model.joblib"),
    }

    models = {}
    for name, path in model_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}. Train models first."
            )
        models[name] = joblib.load(path)
        logger.info(f"Loaded {name} from {path}")
    return models


def evaluate_on_test_set(
    models: dict[str, Any], X_test: np.ndarray, y_test: np.ndarray
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Evaluate all models on test set and return metrics dataframe, y_pred dict, y_prob dict.
    """
    rows = []
    y_pred_dict = {}
    y_prob_dict = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-scores))
        else:
            y_prob = y_pred.astype(float)

        y_pred_dict[name] = y_pred
        y_prob_dict[name] = y_prob

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_test, y_pred, zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        auc = roc_auc_score(y_test, y_prob)

        rows.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "Specificity": spec,
                "F1": f1,
                "ROC_AUC": auc,
                "False Positive Rate": fpr,
                "False Negative Rate": fnr,
                "True Negatives": int(tn),
                "False Positives": int(fp),
                "False Negatives": int(fn),
                "True Positives": int(tp),
            }
        )

    perf_df = pd.DataFrame(rows)
    return perf_df, y_pred_dict, y_prob_dict


def plot_overall_performance(
    perf_df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_dict: dict[str, np.ndarray],
    y_prob_dict: dict[str, np.ndarray],
    output_dir: str = "figures/model_performance",
):
    """Generate all required performance charts."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Accuracy Comparison
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x="Model",
        y="Accuracy",
        data=perf_df,
        palette=[MODEL_COLORS[m] for m in perf_df["Model"]],
        hue="Model",
        legend=False,
    )
    plt.title("Test Accuracy Comparison Across Models", fontsize=13, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim(0.6, 0.9)
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(
            f"{h * 100:.2f}%",
            (p.get_x() + p.get_width() / 2.0, h + 0.005),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=300)
    plt.close()

    # 2. Precision & Recall Comparison
    plt.figure(figsize=(9, 5))
    melted = perf_df.melt(
        id_vars=["Model"],
        value_vars=["Precision", "Recall"],
        var_name="Metric",
        value_name="Score",
    )
    ax = sns.barplot(
        x="Model", y="Score", hue="Metric", data=melted, palette=["#3274a1", "#e1812c"]
    )
    plt.title(
        "Precision vs Recall (Sensitivity) on Test Partition",
        fontsize=13,
        fontweight="bold",
    )
    plt.ylabel("Score", fontsize=11)
    plt.ylim(0, 1.0)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h * 100:.1f}%",
                (p.get_x() + p.get_width() / 2.0, h + 0.02),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_comparison.png"), dpi=300)
    plt.close()

    # 3. F1-Score Comparison
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x="Model",
        y="F1",
        data=perf_df,
        palette=[MODEL_COLORS[m] for m in perf_df["Model"]],
        hue="Model",
        legend=False,
    )
    plt.title("F1-Score Comparison Across Models", fontsize=13, fontweight="bold")
    plt.ylabel("F1-Score (Harmonic Mean)", fontsize=11)
    plt.ylim(0.2, 0.6)
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(
            f"{h:.4f}",
            (p.get_x() + p.get_width() / 2.0, h + 0.01),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_comparison.png"), dpi=300)
    plt.close()

    # 4. ROC Curves
    plt.figure(figsize=(8, 6))
    for name in perf_df["Model"]:
        y_prob = y_prob_dict[name]
        fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(
            fpr_arr,
            tpr_arr,
            label=f"{name} (AUC = {auc:.4f})",
            color=MODEL_COLORS[name],
            linewidth=2,
        )
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.5000)")
    plt.title(
        "Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold"
    )
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300)
    plt.close()

    # 5. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    baseline = np.mean(y_test)
    for name in perf_df["Model"]:
        y_prob = y_prob_dict[name]
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        plt.plot(
            rec_arr,
            prec_arr,
            label=f"{name} (AP = {ap:.4f})",
            color=MODEL_COLORS[name],
            linewidth=2,
        )
    plt.axhline(
        baseline,
        color="k",
        linestyle="--",
        alpha=0.5,
        label=f"No-skill Baseline ({baseline * 100:.1f}%)",
    )
    plt.title("Precision-Recall (PR) Curves", fontsize=13, fontweight="bold")
    plt.xlabel("Recall / Sensitivity", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.png"), dpi=300)
    plt.close()

    # 6. Confusion Matrices (2x2 grid)
    _fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    for idx, name in enumerate(perf_df["Model"]):
        y_pred = y_pred_dict[name]
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Annotations with counts and percentages
        annot = np.empty_like(cm, dtype=object)
        for i in range(2):
            for j in range(2):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j] * 100:.1f}%)"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            xticklabels=["Pred: No", "Pred: Yes"],
            yticklabels=["Actual: No", "Actual: Yes"],
        )
        axes[idx].set_title(
            f"{name}\nAcc: {accuracy_score(y_test, y_pred):.3f} | Recall: {recall_score(y_test, y_pred):.3f}",
            fontweight="bold",
            fontsize=11,
        )
    plt.suptitle(
        "Test Confusion Matrices with Class-Normalized Percentages",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=300)
    plt.close()
    logger.info(f"Performance plots saved to {output_dir}")


def run_evaluation_pipeline() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Complete evaluation pipeline."""
    data = prepare_data()
    models = load_all_models()

    perf_df, y_pred_dict, y_prob_dict = evaluate_on_test_set(
        models, data["X_test"], data["y_test"]
    )

    perf_path = os.path.join("results", "overall_performance.csv")
    perf_df.to_csv(perf_path, index=False)
    logger.info(f"Saved test performance results to {perf_path}")

    plot_overall_performance(perf_df, data["y_test"], y_pred_dict, y_prob_dict)
    return perf_df, {
        "models": models,
        "data": data,
        "y_pred": y_pred_dict,
        "y_prob": y_prob_dict,
    }


if __name__ == "__main__":
    perf_df, _ = run_evaluation_pipeline()
    print("\n=== CONSOLIDATED MODEL PERFORMANCE COMPARISON ===")
    display_cols = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "ROC_AUC",
        "False Negative Rate",
    ]
    print(perf_df[display_cols].to_string(index=False))
