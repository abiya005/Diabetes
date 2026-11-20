"""
src/robustness_analysis.py
Implements rigorous robustness stress-testing:
  Experiment 1: Controlled Missing Data (MCAR at 5%, 10%, 20%) with train-fitted imputer pipeline.
  Experiment 2: Controlled Feature Corruption (5%, 10%, 20% binary flips, ordinal perturbations, numerical noise).
  Experiment 3: Subgroup-Based Distribution Shift (stratified covariate shift across clinical risk strata).
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
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

# Ensure local imports work cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate_models import MODEL_COLORS, load_all_models
from src.preprocessing import (
    ALL_MODEL_FEATURES,
    BINARY_FEATURES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
    prepare_data,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_metrics_single(
    model: Any, X: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """Helper to evaluate standard robustness metrics on perturbed array."""
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        y_prob = 1 / (1 + np.exp(-scores))
    else:
        y_prob = y_pred.astype(float)

    acc = accuracy_score(y, y_pred)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y, y_prob)
    except (ValueError, TypeError):
        auc = 0.5

    return {"Accuracy": acc, "Recall": rec, "F1": f1, "ROC_AUC": auc}


# ==========================================
# EXPERIMENT 1: MISSING DATA ROBUSTNESS
# ==========================================
def run_missing_data_experiment(
    models: dict[str, Any],
    preprocessor: Any,
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    missing_rates: list[float] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Introduce reproducible MCAR missingness into test features,
    pass through the train-fitted imputer pipeline, and measure performance drop.
    """
    if missing_rates is None:
        missing_rates = [0.05, 0.10, 0.20]
    logger.info("Executing Experiment 1: Missing Data Robustness...")
    rng = np.random.RandomState(random_state)
    results = []

    # Baseline performance (0% missing)
    X_base_trans = preprocessor.transform(X_test_raw)
    for model_name, model in models.items():
        base_m = evaluate_metrics_single(model, X_base_trans, y_test)
        for metric, val in base_m.items():
            results.append(
                {
                    "Experiment": "Missing Data",
                    "Model": model_name,
                    "Perturbation_Level": 0.0,
                    "Metric": metric,
                    "Perturbed_Value": val,
                    "Original_Value": val,
                    "Absolute_Drop": 0.0,
                    "Relative_Drop_Pct": 0.0,
                }
            )

    # Perturbed evaluations
    for rate in missing_rates:
        logger.info(f"Evaluating {int(rate * 100)}% Missing Data...")
        X_perturbed_raw = X_test_raw.copy()

        # Apply missingness mask across all model features
        mask = rng.rand(*X_perturbed_raw.shape) < rate
        X_perturbed_raw[mask] = np.nan

        # Transform using preprocessor fitted ONLY on training data
        X_pert_trans = preprocessor.transform(X_perturbed_raw)

        for model_name, model in models.items():
            base_m = evaluate_metrics_single(model, X_base_trans, y_test)
            pert_m = evaluate_metrics_single(model, X_pert_trans, y_test)

            for metric in ["Accuracy", "Recall", "F1", "ROC_AUC"]:
                orig_v = base_m[metric]
                pert_v = pert_m[metric]
                abs_drop = orig_v - pert_v
                rel_drop = (abs_drop / orig_v * 100.0) if orig_v != 0 else 0.0

                results.append(
                    {
                        "Experiment": "Missing Data",
                        "Model": model_name,
                        "Perturbation_Level": rate,
                        "Metric": metric,
                        "Perturbed_Value": pert_v,
                        "Original_Value": orig_v,
                        "Absolute_Drop": abs_drop,
                        "Relative_Drop_Pct": rel_drop,
                    }
                )

    return pd.DataFrame(results)


# ==========================================
# EXPERIMENT 2: NOISY DATA ROBUSTNESS
# ==========================================
def run_noisy_data_experiment(
    models: dict[str, Any],
    preprocessor: Any,
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    noise_rates: list[float] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Introduce realistic feature-type-specific corruption:
      - Binary features: Bit-flipping (0 -> 1, 1 -> 0)
      - Ordinal features: Perturb to adjacent valid integer categories (+/- 1)
      - Numerical features: Proportional measurement noise (+/- 10%)
      - Demographics (Age, Sex) and Target kept strictly UNCHANGED
    """
    if noise_rates is None:
        noise_rates = [0.05, 0.10, 0.20]
    logger.info("Executing Experiment 2: Noisy Data Robustness...")
    rng = np.random.RandomState(random_state)
    results = []

    X_base_trans = preprocessor.transform(X_test_raw)
    base_metrics = {
        name: evaluate_metrics_single(m, X_base_trans, y_test)
        for name, m in models.items()
    }

    # Record 0% baseline
    for model_name in models:
        for metric, val in base_metrics[model_name].items():
            results.append(
                {
                    "Experiment": "Noisy Data",
                    "Model": model_name,
                    "Perturbation_Level": 0.0,
                    "Metric": metric,
                    "Perturbed_Value": val,
                    "Original_Value": val,
                    "Absolute_Drop": 0.0,
                    "Relative_Drop_Pct": 0.0,
                }
            )

    for rate in noise_rates:
        logger.info(f"Evaluating {int(rate * 100)}% Noisy Data...")
        X_corrupt = X_test_raw.copy()

        # 1. Binary feature corruption: bit flips on selected proportion
        for col in BINARY_FEATURES:
            flip_mask = rng.rand(len(X_corrupt)) < rate
            X_corrupt.loc[flip_mask, col] = 1.0 - X_corrupt.loc[flip_mask, col]

        # 2. Ordinal feature corruption: adjacent category shift
        for col in ORDINAL_FEATURES:
            shift_mask = rng.rand(len(X_corrupt)) < rate
            directions = rng.choice([-1.0, 1.0], size=len(X_corrupt))

            # Bounds
            min_val = X_test_raw[col].min()
            max_val = X_test_raw[col].max()
            new_vals = X_corrupt[col] + directions * shift_mask
            X_corrupt[col] = np.clip(new_vals, min_val, max_val)

        # 3. Numerical feature corruption: proportional Gaussian noise
        for col in NUMERICAL_FEATURES:
            noise_mask = rng.rand(len(X_corrupt)) < rate
            std_col = X_test_raw[col].std()
            jitter = rng.normal(0, 0.25 * std_col, size=len(X_corrupt)) * noise_mask
            X_corrupt[col] = np.maximum(0.0, X_corrupt[col] + jitter)

        # Note: Demographics (Sex, Age) are kept uncorrupted so cohort identities remain stable

        X_corr_trans = preprocessor.transform(X_corrupt)

        for model_name, model in models.items():
            pert_m = evaluate_metrics_single(model, X_corr_trans, y_test)
            orig_m = base_metrics[model_name]

            for metric in ["Accuracy", "Recall", "F1", "ROC_AUC"]:
                orig_v = orig_m[metric]
                pert_v = pert_m[metric]
                abs_drop = orig_v - pert_v
                rel_drop = (abs_drop / orig_v * 100.0) if orig_v != 0 else 0.0

                results.append(
                    {
                        "Experiment": "Noisy Data",
                        "Model": model_name,
                        "Perturbation_Level": rate,
                        "Metric": metric,
                        "Perturbed_Value": pert_v,
                        "Original_Value": orig_v,
                        "Absolute_Drop": abs_drop,
                        "Relative_Drop_Pct": rel_drop,
                    }
                )

    return pd.DataFrame(results)


# ==========================================
# EXPERIMENT 3: DISTRIBUTION SHIFT
# ==========================================
def run_distribution_shift_experiment(
    models: dict[str, Any],
    preprocessor: Any,
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    demo_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Subgroup-based distribution shift analysis across distinct clinical risk profiles:
      - Low-Metabolic Risk Subpopulation (Normal BMI < 25, No HighBP)
      - High-Metabolic Risk Subpopulation (Obese BMI >= 30, HighBP = 1)
    Calculates Kolmogorov-Smirnov distribution divergence and compares predictive degradation.
    """
    logger.info("Executing Experiment 3: Subgroup-based Distribution Shift...")

    # Define clinical subpopulation masks
    low_risk_mask = (X_test_raw["BMI"] < 25.0) & (X_test_raw["HighBP"] == 0.0)
    high_risk_mask = (X_test_raw["BMI"] >= 30.0) & (X_test_raw["HighBP"] == 1.0)

    # Calculate Feature Distribution Divergence using Kolmogorov-Smirnov Test
    ks_results = []
    for col in ALL_MODEL_FEATURES:
        stat, p_val = ks_2samp(
            X_test_raw.loc[low_risk_mask, col], X_test_raw.loc[high_risk_mask, col]
        )
        ks_results.append(
            {
                "Feature": col,
                "KS_Statistic": stat,
                "P_Value": p_val,
                "Distribution_Divergence": "Severe (KS > 0.3)"
                if stat > 0.3
                else ("Moderate (KS > 0.1)" if stat > 0.1 else "Slight"),
            }
        )
    ks_df = pd.DataFrame(ks_results).sort_values(by="KS_Statistic", ascending=False)

    # Evaluate models on shifted cohorts
    subpopulations = {
        "Overall Test Cohort": np.ones(len(y_test), dtype=bool),
        "Low-Metabolic Risk Subgroup": low_risk_mask.values,
        "High-Metabolic Risk Subgroup": high_risk_mask.values,
    }

    shift_results = []
    for sub_name, mask in subpopulations.items():
        X_sub_raw = X_test_raw.loc[mask]
        y_sub = y_test[mask]
        X_sub_trans = preprocessor.transform(X_sub_raw)

        for model_name, model in models.items():
            m = evaluate_metrics_single(model, X_sub_trans, y_sub)
            m["Model"] = model_name
            m["Subpopulation"] = sub_name
            m["Cohort_Size"] = int(np.sum(mask))
            m["Diabetes_Prevalence"] = float(np.mean(y_sub))
            shift_results.append(m)

    shift_df = pd.DataFrame(shift_results)
    return shift_df, ks_df


# ==========================================
# ROBUSTNESS VISUALIZATIONS
# ==========================================
def plot_robustness_visualizations(
    missing_df: pd.DataFrame,
    noisy_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    output_dir: str = "figures/robustness",
):
    """Plot degradation graphs for missingness, noise, and distribution shift."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Missing Data Performance Degradation (ROC-AUC & F1)
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, metric in enumerate(["ROC_AUC", "F1"]):
        sub = missing_df[missing_df["Metric"] == metric]
        sns.lineplot(
            data=sub,
            x="Perturbation_Level",
            y="Perturbed_Value",
            hue="Model",
            marker="o",
            linewidth=2.5,
            palette=MODEL_COLORS,
            ax=axes[idx],
        )
        axes[idx].set_title(
            f"Missing Data Impact on {metric}", fontsize=12, fontweight="bold"
        )
        axes[idx].set_xlabel("Missing Data Level", fontsize=11)
        axes[idx].set_ylabel(metric, fontsize=11)
        axes[idx].set_xticks([0.0, 0.05, 0.10, 0.20])
        axes[idx].set_xticklabels(["0%", "5%", "10%", "20%"])
    plt.suptitle(
        "Model Performance Degradation Under Missing Values (MCAR)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missing_data_degradation.png"), dpi=300)
    plt.close()

    # 2. Noisy Data Performance Degradation (ROC-AUC & Recall)
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, metric in enumerate(["ROC_AUC", "Recall"]):
        sub = noisy_df[noisy_df["Metric"] == metric]
        sns.lineplot(
            data=sub,
            x="Perturbation_Level",
            y="Perturbed_Value",
            hue="Model",
            marker="s",
            linewidth=2.5,
            palette=MODEL_COLORS,
            ax=axes[idx],
        )
        axes[idx].set_title(
            f"Noisy Input Impact on {metric}", fontsize=12, fontweight="bold"
        )
        axes[idx].set_xlabel("Feature Corruption Level", fontsize=11)
        axes[idx].set_ylabel(metric, fontsize=11)
        axes[idx].set_xticks([0.0, 0.05, 0.10, 0.20])
        axes[idx].set_xticklabels(["0%", "5%", "10%", "20%"])
    plt.suptitle(
        "Model Performance Degradation Under Input Noise",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "noise_data_degradation.png"), dpi=300)
    plt.close()

    # 3. Model-Wise Robustness Comparison (Absolute Drop at 20% Perturbation)
    plt.figure(figsize=(9, 5))
    drop_20_missing = missing_df[
        (missing_df["Perturbation_Level"] == 0.20) & (missing_df["Metric"] == "ROC_AUC")
    ].copy()
    drop_20_missing["Perturbation_Type"] = "20% Missing"

    drop_20_noisy = noisy_df[
        (noisy_df["Perturbation_Level"] == 0.20) & (noisy_df["Metric"] == "ROC_AUC")
    ].copy()
    drop_20_noisy["Perturbation_Type"] = "20% Noise"

    drop_combined = pd.concat([drop_20_missing, drop_20_noisy])
    ax = sns.barplot(
        x="Model",
        y="Absolute_Drop",
        hue="Perturbation_Type",
        data=drop_combined,
        palette=["#e7298a", "#7570b3"],
    )
    plt.title(
        "ROC-AUC Absolute Drop at 20% Stress Level (Lower Drop = More Robust)",
        fontsize=13,
        fontweight="bold",
    )
    plt.ylabel("ROC-AUC Drop (Original - Perturbed)", fontsize=11)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.4f}",
                (p.get_x() + p.get_width() / 2.0, h + 0.001),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_robustness_comparison.png"), dpi=300)
    plt.close()

    # 4. Distribution Shift Performance
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        x="Model",
        y="ROC_AUC",
        hue="Subpopulation",
        data=shift_df,
        palette=["#2b5c8f", "#7570b3", "#d95f02"],
    )
    plt.title(
        "Model ROC-AUC Under Subgroup Covariate Shift", fontsize=13, fontweight="bold"
    )
    plt.ylabel("ROC-AUC", fontsize=11)
    plt.ylim(0.5, 0.9)
    plt.legend(title="Target Subpopulation", loc="lower left")
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h + 0.005),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_shift_degradation.png"), dpi=300)
    plt.close()
    logger.info(f"Robustness figures generated in {output_dir}")


def run_robustness_pipeline() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Execute all three robustness experiments and persist results."""
    data = prepare_data()
    models = load_all_models()
    preprocessor = data["preprocessor"]
    X_test_raw = data["X_test_raw"]
    y_test = data["y_test"]
    demo_test = data["demo_test"]

    missing_df = run_missing_data_experiment(models, preprocessor, X_test_raw, y_test)
    noisy_df = run_noisy_data_experiment(models, preprocessor, X_test_raw, y_test)
    shift_df, ks_df = run_distribution_shift_experiment(
        models, preprocessor, X_test_raw, y_test, demo_test
    )

    # Combine into unified robustness table
    robustness_all = pd.concat([missing_df, noisy_df], ignore_index=True)
    robustness_path = os.path.join("results", "robustness_results.csv")
    robustness_all.to_csv(robustness_path, index=False)

    shift_path = os.path.join("results", "distribution_shift_results.csv")
    shift_df.to_csv(shift_path, index=False)

    ks_path = os.path.join("results", "feature_divergence_ks.csv")
    ks_df.to_csv(ks_path, index=False)

    logger.info(
        "Saved robustness_results.csv, distribution_shift_results.csv, and feature_divergence_ks.csv"
    )
    plot_robustness_visualizations(missing_df, noisy_df, shift_df)

    return missing_df, noisy_df, shift_df, ks_df


if __name__ == "__main__":
    missing_df, noisy_df, shift_df, ks_df = run_robustness_pipeline()
    print("\n=== MISSING DATA ROBUSTNESS SUMMARY (ROC-AUC DROP AT 20%) ===")
    print(
        missing_df[
            (missing_df["Perturbation_Level"] == 0.20)
            & (missing_df["Metric"] == "ROC_AUC")
        ][
            [
                "Model",
                "Original_Value",
                "Perturbed_Value",
                "Absolute_Drop",
                "Relative_Drop_Pct",
            ]
        ].to_string(index=False)
    )

    print("\n=== NOISY DATA ROBUSTNESS SUMMARY (ROC-AUC DROP AT 20%) ===")
    print(
        noisy_df[
            (noisy_df["Perturbation_Level"] == 0.20) & (noisy_df["Metric"] == "ROC_AUC")
        ][
            [
                "Model",
                "Original_Value",
                "Perturbed_Value",
                "Absolute_Drop",
                "Relative_Drop_Pct",
            ]
        ].to_string(index=False)
    )
