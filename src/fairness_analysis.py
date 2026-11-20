"""
src/fairness_analysis.py
Comprehensive demographic fairness evaluation across Age and Sex groups,
computing subgroup performance metrics, Fairlearn disparities, and bootstrap confidence intervals.
"""

import os
import sys
import tempfile

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
import matplotlib

matplotlib.use("Agg")
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Ensure local imports work cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate_models import MODEL_COLORS, load_all_models
from src.preprocessing import prepare_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def compute_group_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict[str, float]:
    """Calculate all 11 required subgroup metrics safely."""
    sample_size = len(y_true)
    if sample_size == 0:
        return {}

    pos_prevalence = float(np.mean(y_true))
    pred_pos_rate = float(np.mean(y_pred))

    # Check if confusion matrix can be computed
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    try:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = np.nan
    except (ValueError, TypeError, ZeroDivisionError):
        auc = np.nan

    return {
        "Sample_Size": int(sample_size),
        "Actual_Prevalence": pos_prevalence,
        "Predicted_Positive_Rate": pred_pos_rate,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Specificity": spec,
        "F1": f1,
        "False_Positive_Rate": fpr,
        "False_Negative_Rate": fnr,
        "ROC_AUC": auc,
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals for recall and false negative rate."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    recalls = []
    fnrs = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt_boot = y_true[idx]
        yp_boot = y_pred[idx]
        pos_count = np.sum(yt_boot == 1)
        if pos_count > 0:
            tp_count = np.sum((yt_boot == 1) & (yp_boot == 1))
            rec = float(tp_count / pos_count)
            recalls.append(rec)
            fnrs.append(1.0 - rec)

    if len(recalls) > 0:
        rec_lower = float(np.percentile(recalls, 100 * (alpha / 2)))
        rec_upper = float(np.percentile(recalls, 100 * (1 - alpha / 2)))
        fnr_lower = float(np.percentile(fnrs, 100 * (alpha / 2)))
        fnr_upper = float(np.percentile(fnrs, 100 * (1 - alpha / 2)))
    else:
        rec_lower, rec_upper, fnr_lower, fnr_upper = np.nan, np.nan, np.nan, np.nan

    return {
        "Recall_CI_Lower": rec_lower,
        "Recall_CI_Upper": rec_upper,
        "FNR_CI_Lower": fnr_lower,
        "FNR_CI_Upper": fnr_upper,
    }


def evaluate_demographic_subgroups(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    demo_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate subgroup performance for Age and Sex across all models,
    and compute standard demographic fairness disparities.
    """
    age_groups = ["18-34", "35-49", "50-64", "65+"]
    sex_groups = ["Female", "Male"]

    age_rows = []
    sex_rows = []
    fairness_rows = []

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-scores))
        else:
            y_prob = y_pred.astype(float)

        # 1. Age-Based Subgroup Evaluation
        for ag in age_groups:
            mask = demo_test["Age_Group"].values == ag
            yt_sub = y_test[mask]
            yp_sub = y_pred[mask]
            ypr_sub = y_prob[mask]

            m = compute_group_metrics(yt_sub, yp_sub, ypr_sub)
            ci = compute_bootstrap_ci(yt_sub, yp_sub)
            m.update(ci)
            m["Model"] = model_name
            m["Age_Group"] = ag
            age_rows.append(m)

        # 2. Sex-Based Subgroup Evaluation
        for sx in sex_groups:
            mask = demo_test["Sex_Label"].values == sx
            yt_sub = y_test[mask]
            yp_sub = y_pred[mask]
            ypr_sub = y_prob[mask]

            m = compute_group_metrics(yt_sub, yp_sub, ypr_sub)
            ci = compute_bootstrap_ci(yt_sub, yp_sub)
            m.update(ci)
            m["Model"] = model_name
            m["Sex"] = sx
            sex_rows.append(m)

        # 3. Fairness Metrics Computation using Fairlearn & exact formulas
        # Sex fairness (binary sensitive attribute)
        sex_sens = demo_test["Sex_Label"].values
        dpd_sex = demographic_parity_difference(
            y_test, y_pred, sensitive_features=sex_sens
        )
        eodd_sex = equalized_odds_difference(
            y_test, y_pred, sensitive_features=sex_sens
        )

        # Sex subgroup recalls & FNRs
        female_mask = sex_sens == "Female"
        male_mask = sex_sens == "Male"
        rec_female = recall_score(
            y_test[female_mask], y_pred[female_mask], zero_division=0
        )
        rec_male = recall_score(y_test[male_mask], y_pred[male_mask], zero_division=0)
        eod_sex = abs(rec_male - rec_female)
        fnrd_sex = abs((1 - rec_male) - (1 - rec_female))
        sex_disparity_dir = (
            "Higher Recall for Male"
            if rec_male > rec_female
            else "Higher Recall for Female"
        )

        # Age fairness (multiclass sensitive attribute)
        age_sens = demo_test["Age_Group"].values
        dpd_age = demographic_parity_difference(
            y_test, y_pred, sensitive_features=age_sens
        )
        eodd_age = equalized_odds_difference(
            y_test, y_pred, sensitive_features=age_sens
        )

        # Age group max disparity in recall and FNR
        age_recalls = {
            ag: recall_score(
                y_test[age_sens == ag], y_pred[age_sens == ag], zero_division=0
            )
            for ag in age_groups
        }
        min_age = min(age_recalls, key=age_recalls.get)
        max_age = max(age_recalls, key=age_recalls.get)
        eod_age = age_recalls[max_age] - age_recalls[min_age]
        fnrd_age = eod_age
        age_disparity_dir = f"Highest: {max_age} ({age_recalls[max_age] * 100:.1f}%), Lowest: {min_age} ({age_recalls[min_age] * 100:.1f}%)"

        fairness_rows.append(
            {
                "Model": model_name,
                "Attribute": "Sex (Male vs Female)",
                "Demographic_Parity_Difference": dpd_sex,
                "Equal_Opportunity_Difference": eod_sex,
                "Equalized_Odds_Difference": eodd_sex,
                "False_Negative_Rate_Difference": fnrd_sex,
                "Disparity_Direction": sex_disparity_dir,
                "Group_A_Recall": rec_female,
                "Group_B_Recall": rec_male,
                "Group_A_Name": "Female",
                "Group_B_Name": "Male",
                "Sample_Sizes": f"Female: {int(np.sum(female_mask)):,}, Male: {int(np.sum(male_mask)):,}",
            }
        )

        fairness_rows.append(
            {
                "Model": model_name,
                "Attribute": "Age Groups (18-34, 35-49, 50-64, 65+)",
                "Demographic_Parity_Difference": dpd_age,
                "Equal_Opportunity_Difference": eod_age,
                "Equalized_Odds_Difference": eodd_age,
                "False_Negative_Rate_Difference": fnrd_age,
                "Disparity_Direction": age_disparity_dir,
                "Group_A_Recall": age_recalls[min_age],
                "Group_B_Recall": age_recalls[max_age],
                "Group_A_Name": min_age,
                "Group_B_Name": max_age,
                "Sample_Sizes": ", ".join(
                    [f"{ag}: {int(np.sum(age_sens == ag)):,}" for ag in age_groups]
                ),
            }
        )

    age_df = pd.DataFrame(age_rows)
    sex_df = pd.DataFrame(sex_rows)
    fairness_df = pd.DataFrame(fairness_rows)

    return age_df, sex_df, fairness_df


def plot_fairness_visualizations(
    age_df: pd.DataFrame,
    sex_df: pd.DataFrame,
    fairness_df: pd.DataFrame,
    output_dir: str = "figures/fairness",
):
    """Generate all 7 required fairness visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Age-Group vs Recall
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        x="Age_Group",
        y="Recall",
        hue="Model",
        data=age_df,
        palette=[MODEL_COLORS[m] for m in age_df["Model"].unique()],
    )
    plt.title(
        "Model Recall (Sensitivity) Across Age Groups", fontsize=13, fontweight="bold"
    )
    plt.xlabel("Age Bracket", fontsize=11)
    plt.ylabel("Recall (True Positive Rate)", fontsize=11)
    plt.ylim(0, 1.0)
    plt.legend(title="Algorithm", loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "age_group_recall.png"), dpi=300)
    plt.close()

    # 2. Sex-Group vs Recall
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x="Sex",
        y="Recall",
        hue="Model",
        data=sex_df,
        palette=[MODEL_COLORS[m] for m in sex_df["Model"].unique()],
    )
    plt.title(
        "Model Recall (Sensitivity) Across Sex Groups", fontsize=13, fontweight="bold"
    )
    plt.xlabel("Sex", fontsize=11)
    plt.ylabel("Recall (True Positive Rate)", fontsize=11)
    plt.ylim(0, 1.0)
    plt.legend(title="Algorithm", loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sex_group_recall.png"), dpi=300)
    plt.close()

    # 3. Age-Group vs False Negative Rate
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        x="Age_Group",
        y="False_Negative_Rate",
        hue="Model",
        data=age_df,
        palette=[MODEL_COLORS[m] for m in age_df["Model"].unique()],
    )
    plt.title(
        "False Negative Rate (Missed Cases) Across Age Groups",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Age Bracket", fontsize=11)
    plt.ylabel("False Negative Rate (1 - Recall)", fontsize=11)
    plt.ylim(0, 1.0)
    plt.legend(title="Algorithm", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "age_group_fnr.png"), dpi=300)
    plt.close()

    # 4. Sex-Group vs False Negative Rate
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x="Sex",
        y="False_Negative_Rate",
        hue="Model",
        data=sex_df,
        palette=[MODEL_COLORS[m] for m in sex_df["Model"].unique()],
    )
    plt.title(
        "False Negative Rate (Missed Cases) Across Sex Groups",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Sex", fontsize=11)
    plt.ylabel("False Negative Rate (1 - Recall)", fontsize=11)
    plt.ylim(0, 1.0)
    plt.legend(title="Algorithm", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sex_group_fnr.png"), dpi=300)
    plt.close()

    # 5. Demographic Parity Comparison
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        x="Model",
        y="Demographic_Parity_Difference",
        hue="Attribute",
        data=fairness_df,
        palette=["#3470a3", "#d95f02"],
    )
    plt.title(
        "Demographic Parity Difference Across Models (Lower is Fairer)",
        fontsize=13,
        fontweight="bold",
    )
    plt.ylabel("Demographic Parity Difference (DPD)", fontsize=11)
    plt.ylim(0, max(fairness_df["Demographic_Parity_Difference"]) * 1.25)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h + 0.01),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "demographic_parity_comparison.png"), dpi=300)
    plt.close()

    # 6. Equal Opportunity Comparison
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        x="Model",
        y="Equal_Opportunity_Difference",
        hue="Attribute",
        data=fairness_df,
        palette=["#3470a3", "#d95f02"],
    )
    plt.title(
        "Equal Opportunity Difference (Recall Disparity Across Groups)",
        fontsize=13,
        fontweight="bold",
    )
    plt.ylabel("Equal Opportunity Difference (EOD)", fontsize=11)
    plt.ylim(0, max(fairness_df["Equal_Opportunity_Difference"]) * 1.25)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h + 0.01),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "equal_opportunity_comparison.png"), dpi=300)
    plt.close()

    # 7. Equalized Odds Comparison
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        x="Model",
        y="Equalized_Odds_Difference",
        hue="Attribute",
        data=fairness_df,
        palette=["#3470a3", "#d95f02"],
    )
    plt.title(
        "Equalized Odds Difference (Max of TPR & FPR Disparity)",
        fontsize=13,
        fontweight="bold",
    )
    plt.ylabel("Equalized Odds Difference (EODD)", fontsize=11)
    plt.ylim(0, max(fairness_df["Equalized_Odds_Difference"]) * 1.25)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h + 0.01),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "equalized_odds_comparison.png"), dpi=300)
    plt.close()
    logger.info(f"Fairness visualizations generated in {output_dir}")


def run_fairness_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete demographic fairness audit."""
    data = prepare_data()
    models = load_all_models()

    age_df, sex_df, fairness_df = evaluate_demographic_subgroups(
        models, data["X_test"], data["y_test"], data["demo_test"]
    )

    age_path = os.path.join("results", "age_fairness.csv")
    sex_path = os.path.join("results", "sex_fairness.csv")
    fairness_path = os.path.join("results", "fairness_metrics.csv")

    age_df.to_csv(age_path, index=False)
    sex_df.to_csv(sex_path, index=False)
    fairness_df.to_csv(fairness_path, index=False)

    logger.info(
        "Saved age_fairness.csv, sex_fairness.csv, and fairness_metrics.csv to results/"
    )
    plot_fairness_visualizations(age_df, sex_df, fairness_df)

    return age_df, sex_df, fairness_df


if __name__ == "__main__":
    age_df, sex_df, fairness_df = run_fairness_pipeline()
    print("\n=== SEX FAIRNESS METRICS ===")
    print(
        fairness_df[fairness_df["Attribute"].str.startswith("Sex")][
            [
                "Model",
                "Demographic_Parity_Difference",
                "Equal_Opportunity_Difference",
                "Equalized_Odds_Difference",
                "False_Negative_Rate_Difference",
                "Disparity_Direction",
            ]
        ].to_string(index=False)
    )

    print("\n=== AGE FAIRNESS METRICS ===")
    print(
        fairness_df[fairness_df["Attribute"].str.startswith("Age")][
            [
                "Model",
                "Demographic_Parity_Difference",
                "Equal_Opportunity_Difference",
                "Equalized_Odds_Difference",
                "False_Negative_Rate_Difference",
                "Disparity_Direction",
            ]
        ].to_string(index=False)
    )
