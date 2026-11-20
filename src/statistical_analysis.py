"""
src/statistical_analysis.py
Performs statistical validation:
  1. Multi-seed repeated cross-validation across models
  2. Paired statistical testing (Paired t-test, Wilcoxon signed-rank test)
  3. McNemar's test on test-set prediction discordance
  4. Documentation of formal statistical assumptions and test limitations
"""

import os
import sys
import tempfile

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# Ensure local imports work cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate_models import load_all_models
from src.preprocessing import prepare_data
from src.train_models import get_model_definitions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_repeated_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seeds: list[int] | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Run repeated stratified cross-validation across multiple random seeds."""
    if seeds is None:
        seeds = [42, 123, 456]
    logger.info(
        f"Running Repeated CV across {len(seeds)} seeds ({len(seeds) * n_splits} total evaluations per model)..."
    )
    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    scale_pos_weight = float(neg_count / pos_count)

    records = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            models = get_model_definitions(scale_pos_weight)
            for name, model in models.items():
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_val)[:, 1]
                elif hasattr(model, "decision_function"):
                    df_scores = model.decision_function(X_val)
                    y_prob = 1 / (1 + np.exp(-df_scores))
                else:
                    y_prob = y_pred.astype(float)

                acc = accuracy_score(y_val, y_pred)
                rec = recall_score(y_val, y_pred, zero_division=0)
                auc = roc_auc_score(y_val, y_prob)

                records.append(
                    {
                        "Seed": seed,
                        "Fold": fold,
                        "Model": name,
                        "Accuracy": acc,
                        "Recall": rec,
                        "ROC_AUC": auc,
                    }
                )

    return pd.DataFrame(records)


def perform_paired_tests(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Perform Paired t-tests and Wilcoxon signed-rank tests across paired fold runs."""
    pairs = [
        ("XGBoost", "Logistic Regression"),
        ("XGBoost", "Random Forest"),
        ("Random Forest", "Logistic Regression"),
        ("SVM", "Logistic Regression"),
        ("SVM", "Random Forest"),
        ("SVM", "XGBoost"),
    ]

    test_results = []

    for m1, m2 in pairs:
        sub1 = cv_df[cv_df["Model"] == m1].sort_values(by=["Seed", "Fold"])
        sub2 = cv_df[cv_df["Model"] == m2].sort_values(by=["Seed", "Fold"])

        for metric in ["ROC_AUC", "Recall", "Accuracy"]:
            scores1 = sub1[metric].values
            scores2 = sub2[metric].values
            diffs = scores1 - scores2

            # Paired Student's t-test
            t_stat, t_pval = stats.ttest_rel(scores1, scores2)

            # Wilcoxon signed-rank test (non-parametric)
            try:
                w_stat, w_pval = stats.wilcoxon(scores1, scores2)
            except (ValueError, TypeError):
                w_stat, w_pval = np.nan, np.nan

            mean_diff = float(np.mean(diffs))
            ci_low, ci_high = stats.t.interval(
                0.95, len(diffs) - 1, loc=mean_diff, scale=stats.sem(diffs)
            )

            test_results.append(
                {
                    "Model_1": m1,
                    "Model_2": m2,
                    "Metric": metric,
                    "Mean_1": float(np.mean(scores1)),
                    "Mean_2": float(np.mean(scores2)),
                    "Mean_Difference": mean_diff,
                    "CI_95_Lower": float(ci_low),
                    "CI_95_Upper": float(ci_high),
                    "Paired_t_Statistic": float(t_stat),
                    "t_test_p_value": float(t_pval),
                    "Wilcoxon_Statistic": float(w_stat),
                    "Wilcoxon_p_value": float(w_pval),
                    "Statistically_Significant (p < 0.05)": (
                        t_pval < 0.05 and w_pval < 0.05
                    ),
                }
            )

    return pd.DataFrame(test_results)


def perform_mcnemar_tests(
    models: dict[str, Any], X_test: np.ndarray, y_test: np.ndarray
) -> pd.DataFrame:
    """Perform McNemar's test with continuity correction on held-out test predictions."""
    logger.info("Computing McNemar test on test set predictions...")
    preds = {name: model.predict(X_test) for name, model in models.items()}

    pairs = [
        ("XGBoost", "Logistic Regression"),
        ("XGBoost", "Random Forest"),
        ("Random Forest", "Logistic Regression"),
        ("SVM", "Logistic Regression"),
        ("SVM", "Random Forest"),
        ("SVM", "XGBoost"),
    ]

    mcnemar_rows = []

    for m1, m2 in pairs:
        y1 = preds[m1]
        y2 = preds[m2]

        correct1 = y1 == y_test
        correct2 = y2 == y_test

        # Contingency table
        # a: both correct
        # b: m1 correct, m2 wrong
        # c: m1 wrong, m2 correct
        # d: both wrong
        b = int(np.sum(correct1 & ~correct2))
        c = int(np.sum(~correct1 & correct2))

        # McNemar statistic with Edwards continuity correction
        if (b + c) > 0:
            chi2_stat = float(((abs(b - c) - 1.0) ** 2) / (b + c))
            p_val = float(stats.chi2.sf(chi2_stat, df=1))
        else:
            chi2_stat, p_val = 0.0, 1.0

        mcnemar_rows.append(
            {
                "Model_1": m1,
                "Model_2": m2,
                "M1_Correct_M2_Incorrect (b)": b,
                "M1_Incorrect_M2_Correct (c)": c,
                "Chi2_Statistic": chi2_stat,
                "p_value": p_val,
                "Significant_Discordance (p < 0.05)": p_val < 0.05,
                "Superior_Model": m1 if b > c else m2,
            }
        )

    return pd.DataFrame(mcnemar_rows)


def run_statistical_pipeline(
    rerun_cv: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete statistical validation pipeline."""
    data = prepare_data()
    models = load_all_models()

    # 1. Repeated CV across seeds (loads existing if available to avoid redundant recomputation)
    repeated_cv_path = "results/repeated_cv_scores.csv"
    if not rerun_cv and os.path.exists(repeated_cv_path):
        logger.info(f"Loading existing repeated CV scores from {repeated_cv_path}")
        cv_df = pd.read_csv(repeated_cv_path)
    else:
        cv_df = run_repeated_cv(
            data["X_train"], data["y_train"], seeds=[42, 123], n_splits=5
        )
        cv_df.to_csv(repeated_cv_path, index=False)

    # 2. Paired t-tests and Wilcoxon tests
    paired_df = perform_paired_tests(cv_df)

    # 3. McNemar's tests on held-out test predictions
    mcnemar_df = perform_mcnemar_tests(models, data["X_test"], data["y_test"])

    # Persist results
    paired_df.to_csv("results/statistical_results.csv", index=False)
    mcnemar_df.to_csv("results/mcnemar_results.csv", index=False)

    # Document Assumptions & Limitations
    notes = (
        "STATISTICAL VALIDATION PROTOCOL & LIMITATIONS\n"
        "============================================\n"
        "1. Paired Student's t-test:\n"
        "   - Assumptions: Differences between paired fold scores are normally distributed.\n"
        "   - Limitation: Overlapping training folds in repeated CV introduce mild dependence,\n"
        "     which can result in slightly liberal p-values. Hence, reported alongside Wilcoxon non-parametric test.\n\n"
        "2. Wilcoxon Signed-Rank Test:\n"
        "   - Non-parametric counterpart relaxing normality assumption.\n"
        "   - Tests median rank difference across matched fold evaluations.\n\n"
        "3. McNemar's Chi-Squared Test:\n"
        "   - Measures whether the discordant prediction errors between two models on the test set differ significantly.\n"
        "   - Edwards continuity correction (|b - c| - 1)^2 / (b + c) was applied to avoid Type I inflation.\n\n"
        "4. Subgroup Uncertainty:\n"
        "   - Smaller demographic brackets (e.g. 18-34 young adults) have higher variance in positive prevalence,\n"
        "     which is accounted for via 500-sample bootstrap percentile confidence intervals.\n"
    )
    with open("results/statistical_assumptions_and_limitations.txt", "w") as f:
        f.write(notes)

    logger.info(
        "Saved statistical_results.csv, repeated_cv_scores.csv, mcnemar_results.csv"
    )
    return cv_df, paired_df, mcnemar_df


if __name__ == "__main__":
    cv_df, paired_df, mcnemar_df = run_statistical_pipeline()
    print("\n=== PAIRED STATISTICAL COMPARISONS (ROC-AUC) ===")
    print(
        paired_df[paired_df["Metric"] == "ROC_AUC"][
            [
                "Model_1",
                "Model_2",
                "Mean_1",
                "Mean_2",
                "Mean_Difference",
                "Paired_t_Statistic",
                "t_test_p_value",
                "Statistically_Significant (p < 0.05)",
            ]
        ].to_string(index=False)
    )

    print("\n=== MCNEMAR TEST (TEST SET DISCORDANCE) ===")
    print(mcnemar_df.to_string(index=False))
