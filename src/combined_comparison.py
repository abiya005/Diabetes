"""
src/combined_comparison.py
Synthesizes the three research dimensions:
  1. Predictive Performance
  2. Demographic Fairness
  3. Robustness
Produces results/combined_comparison.csv without arbitrary hardcoded scores.
"""

import pandas as pd


def generate_combined_comparison():
    perf_df = pd.read_csv("results/overall_performance.csv").set_index("Model")
    fair_df = pd.read_csv("results/fairness_metrics.csv")
    rob_df = pd.read_csv("results/robustness_results.csv")
    shift_df = pd.read_csv("results/distribution_shift_results.csv")

    # Extract Age Fairness metrics (multigroup disparity)
    age_fair = fair_df[fair_df["Attribute"].str.startswith("Age")].set_index("Model")
    # Extract Sex Fairness metrics (binary disparity)
    sex_fair = fair_df[fair_df["Attribute"].str.startswith("Sex")].set_index("Model")

    # Extract Robustness Drops at 20% stress level for ROC-AUC
    missing_20 = rob_df[
        (rob_df["Experiment"] == "Missing Data")
        & (rob_df["Perturbation_Level"] == 0.20)
        & (rob_df["Metric"] == "ROC_AUC")
    ].set_index("Model")
    noisy_20 = rob_df[
        (rob_df["Experiment"] == "Noisy Data")
        & (rob_df["Perturbation_Level"] == 0.20)
        & (rob_df["Metric"] == "ROC_AUC")
    ].set_index("Model")

    # Subgroup shift drop (Overall test cohort vs High-Risk subgroup)
    shift_pivot = shift_df.pivot(
        index="Model", columns="Subpopulation", values="ROC_AUC"
    )
    shift_drop = (
        shift_pivot["Overall Test Cohort"] - shift_pivot["High-Metabolic Risk Subgroup"]
    )

    models = perf_df.index.tolist()
    rows = []

    for m in models:
        row = {
            "Model": m,
            # Dimension 1: Predictive Performance
            "Accuracy": perf_df.loc[m, "Accuracy"],
            "Recall": perf_df.loc[m, "Recall"],
            "F1_Score": perf_df.loc[m, "F1"],
            "ROC_AUC": perf_df.loc[m, "ROC_AUC"],
            "False_Negative_Rate": perf_df.loc[m, "False Negative Rate"],
            # Dimension 2: Demographic Fairness (Disparities - Lower is Fairer)
            "Sex_Equal_Opportunity_Diff": sex_fair.loc[
                m, "Equal_Opportunity_Difference"
            ],
            "Sex_Demographic_Parity_Diff": sex_fair.loc[
                m, "Demographic_Parity_Difference"
            ],
            "Age_Equal_Opportunity_Diff": age_fair.loc[
                m, "Equal_Opportunity_Difference"
            ],
            "Age_Demographic_Parity_Diff": age_fair.loc[
                m, "Demographic_Parity_Difference"
            ],
            # Dimension 3: Robustness (Drop at 20% - Lower Drop is More Robust)
            "Missing_Data_20pct_AUC_Drop": missing_20.loc[m, "Absolute_Drop"],
            "Noisy_Data_20pct_AUC_Drop": noisy_20.loc[m, "Absolute_Drop"],
            "Covariate_Shift_AUC_Drop": shift_drop.loc[m],
        }
        rows.append(row)

    combined_df = pd.DataFrame(rows)
    combined_df.to_csv("results/combined_comparison.csv", index=False)
    print("Saved results/combined_comparison.csv")
    print(combined_df.to_string(index=False))
    return combined_df


if __name__ == "__main__":
    generate_combined_comparison()
