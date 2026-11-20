"""
src/data_loader.py
Dataset ingestion, schema validation, demographic decoding, profiling, and visualization.
"""

import os
import tempfile

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
import matplotlib

matplotlib.use("Agg")
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# CDC BRFSS 2015 Codebook Mappings
AGE_CODE_MAP = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80+",
}

AGE_GROUP_MAP = {
    1: "18-34",
    2: "18-34",
    3: "18-34",
    4: "35-49",
    5: "35-49",
    6: "35-49",
    7: "50-64",
    8: "50-64",
    9: "50-64",
    10: "65+",
    11: "65+",
    12: "65+",
    13: "65+",
}

SEX_MAP = {0: "Female", 1: "Male"}

GEN_HLTH_MAP = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}


def load_dataset(
    filepath: str = "data/raw/diabetes_012_health_indicators_BRFSS2015.csv",
) -> pd.DataFrame:
    """
    Load raw CDC BRFSS dataset, validate columns, and standardise target class.

    Target interpretation:
      0 = No diabetes / negative class
      1 = Positive class combining prediabetes (1.0) and diabetes (2.0)
    """
    if not os.path.exists(filepath):
        alt_path = os.path.join("Diabetes_Data", os.path.basename(filepath))
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {filepath} or {alt_path}")

    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded raw records: {df.shape[0]}, features: {df.shape[1]}")

    # Standardize column naming
    df.columns = [c.strip() for c in df.columns]

    # Target harmonization
    if "Diabetes_012" in df.columns:
        # 0 = No diabetes, 1 = prediabetes, 2 = diabetes
        # Map 0 -> 0, 1 -> 1, 2 -> 1 (combining prediabetes and diabetes)
        df["Diabetes_binary"] = (
            df["Diabetes_012"].apply(lambda x: 1 if x in [1.0, 2.0] else 0).astype(int)
        )
        logger.info(
            "Combined Diabetes_012: class 0 -> 0 (No diabetes), classes 1 & 2 -> 1 (Prediabetes/Diabetes)"
        )
    elif "Diabetes_binary" in df.columns:
        df["Diabetes_binary"] = df["Diabetes_binary"].astype(int)
        logger.info("Using existing Diabetes_binary column")
    else:
        raise ValueError(
            f"No recognised target column found. Available columns: {list(df.columns)}"
        )

    # Add decoded demographic columns for transparent downstream auditing
    df["Sex_Label"] = df["Sex"].map(SEX_MAP)
    df["Age_Label"] = df["Age"].map(AGE_CODE_MAP)
    df["Age_Group"] = df["Age"].map(AGE_GROUP_MAP)

    return df


def generate_dataset_profile(
    df: pd.DataFrame, output_dir: str = "results"
) -> pd.DataFrame:
    """
    Generate rigorous dataset profile including distributions, types, missingness, and duplicates.
    """
    os.makedirs(output_dir, exist_ok=True)

    total_records = len(df)
    features = [
        c
        for c in df.columns
        if c
        not in [
            "Diabetes_012",
            "Diabetes_binary",
            "Sex_Label",
            "Age_Label",
            "Age_Group",
        ]
    ]
    total_features = len(features)

    target_counts = df["Diabetes_binary"].value_counts().to_dict()
    neg_count = target_counts.get(0, 0)
    pos_count = target_counts.get(1, 0)

    missing_counts = df[features + ["Diabetes_binary"]].isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())

    # Check duplicates across the original feature space
    duplicate_records = int(df.duplicated(subset=features + ["Diabetes_binary"]).sum())

    profile_data = {
        "Metric": [
            "Total Records",
            "Total Input Features",
            "Negative Class (0 - No Diabetes)",
            "Positive Class (1 - Prediabetes/Diabetes)",
            "Positive Class Prevalence (%)",
            "Total Missing Values",
            "Duplicate Records Count",
            "Duplicate Records (%)",
            "Female Count",
            "Male Count",
            "Female Prevalence (%)",
            "Male Prevalence (%)",
            "Age Group 18-34 Count",
            "Age Group 35-49 Count",
            "Age Group 50-64 Count",
            "Age Group 65+ Count",
        ],
        "Value": [
            total_records,
            total_features,
            neg_count,
            pos_count,
            round(pos_count / total_records * 100, 2),
            total_missing,
            duplicate_records,
            round(duplicate_records / total_records * 100, 2),
            int((df["Sex"] == 0).sum()),
            int((df["Sex"] == 1).sum()),
            round(df[df["Sex"] == 0]["Diabetes_binary"].mean() * 100, 2),
            round(df[df["Sex"] == 1]["Diabetes_binary"].mean() * 100, 2),
            int((df["Age_Group"] == "18-34").sum()),
            int((df["Age_Group"] == "35-49").sum()),
            int((df["Age_Group"] == "50-64").sum()),
            int((df["Age_Group"] == "65+").sum()),
        ],
    }

    profile_df = pd.DataFrame(profile_data)
    profile_path = os.path.join(output_dir, "dataset_profile.csv")
    profile_df.to_csv(profile_path, index=False)
    logger.info(f"Dataset profile saved to {profile_path}")

    # Also save detailed feature statistics
    num_cols = ["BMI", "MentHlth", "PhysHlth"]
    ord_cols = ["GenHlth", "Age", "Education", "Income"]

    feature_meta = []
    for col in features:
        feature_meta.append(
            {
                "Feature": col,
                "Type": "Numerical"
                if col in num_cols
                else (
                    "Ordinal"
                    if col in ord_cols
                    else ("Demographic" if col in ["Sex", "Age"] else "Binary")
                ),
                "Data_Type": str(df[col].dtype),
                "Unique_Values_Count": df[col].nunique(),
                "Missing_Count": int(df[col].isnull().sum()),
                "Sample_Values": str(sorted(df[col].dropna().unique()[:6].tolist())),
            }
        )
    feature_meta_df = pd.DataFrame(feature_meta)
    feature_meta_df.to_csv(
        os.path.join(output_dir, "feature_metadata.csv"), index=False
    )

    return profile_df


def plot_dataset_visualizations(
    df: pd.DataFrame, output_dir: str = "figures/dataset_analysis"
):
    """
    Generate research-grade exploratory visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Diabetes Class Distribution
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        x="Diabetes_binary",
        data=df,
        palette=["#4a7bb0", "#d95f02"],
        hue="Diabetes_binary",
        legend=False,
    )
    plt.title(
        "Diabetes Risk Class Distribution (0 = Negative, 1 = Positive)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Diabetes Status", fontsize=11)
    plt.ylabel("Number of Observations", fontsize=11)
    plt.xticks([0, 1], ["0: No Diabetes (84.2%)", "1: Prediabetes / Diabetes (15.8%)"])
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height):,} ({height / len(df) * 100:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, height / 2),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=11,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diabetes_class_distribution.png"), dpi=300)
    plt.close()

    # 2. Age Distribution (Original 13 categories)
    plt.figure(figsize=(10, 5))
    age_counts = df["Age"].value_counts().sort_index()
    age_labels = [AGE_CODE_MAP[int(i)] for i in age_counts.index]
    sns.barplot(x=age_labels, y=age_counts.values, color="#3470a3")
    plt.title(
        "Distribution Across Age Categories (BRFSS 2015 13-Level Coding)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Age Bracket", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "age_distribution.png"), dpi=300)
    plt.close()

    # 3. Sex Distribution
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(
        x="Sex_Label",
        data=df,
        palette=["#e7298a", "#2b5c8f"],
        hue="Sex_Label",
        legend=False,
    )
    plt.title("Sex Distribution in BRFSS Cohort", fontsize=13, fontweight="bold")
    plt.xlabel("Sex", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(
            f"{int(h):,} ({h / len(df) * 100:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, h / 2),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sex_distribution.png"), dpi=300)
    plt.close()

    # 4. Diabetes Prevalence Across Age Groups
    plt.figure(figsize=(8, 5))
    age_group_order = ["18-34", "35-49", "50-64", "65+"]
    age_prev = (
        df.groupby("Age_Group")["Diabetes_binary"]
        .agg(["mean", "count"])
        .reindex(age_group_order)
        .reset_index()
    )
    age_prev["prevalence_pct"] = age_prev["mean"] * 100

    ax = sns.barplot(
        x="Age_Group",
        y="prevalence_pct",
        data=age_prev,
        palette="Blues_d",
        hue="Age_Group",
        legend=False,
    )
    plt.title(
        "Empirical Diabetes Prevalence Across Research Age Groups",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Age Group", fontsize=11)
    plt.ylabel("Positive Class Prevalence (%)", fontsize=11)
    plt.ylim(0, max(age_prev["prevalence_pct"]) * 1.25)
    for idx, p in enumerate(ax.patches):
        h = p.get_height()
        count = age_prev.loc[idx, "count"]
        ax.annotate(
            f"{h:.1f}%\n(N={count:,})",
            (p.get_x() + p.get_width() / 2.0, h + 0.8),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prevalence_by_age.png"), dpi=300)
    plt.close()

    # 5. Diabetes Prevalence Across Sex Groups
    plt.figure(figsize=(6, 5))
    sex_prev = (
        df.groupby("Sex_Label")["Diabetes_binary"].agg(["mean", "count"]).reset_index()
    )
    sex_prev["prevalence_pct"] = sex_prev["mean"] * 100

    ax = sns.barplot(
        x="Sex_Label",
        y="prevalence_pct",
        data=sex_prev,
        palette=["#d95f02", "#2b5c8f"],
        hue="Sex_Label",
        legend=False,
    )
    plt.title(
        "Empirical Diabetes Prevalence Across Sex Groups",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Sex", fontsize=11)
    plt.ylabel("Positive Class Prevalence (%)", fontsize=11)
    plt.ylim(0, max(sex_prev["prevalence_pct"]) * 1.3)
    for idx, p in enumerate(ax.patches):
        h = p.get_height()
        count = sex_prev.loc[idx, "count"]
        ax.annotate(
            f"{h:.1f}%\n(N={count:,})",
            (p.get_x() + p.get_width() / 2.0, h + 0.6),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prevalence_by_sex.png"), dpi=300)
    plt.close()

    # 6. Feature Correlation Heatmap
    plt.figure(figsize=(14, 11))
    numeric_df = df.drop(
        columns=["Diabetes_012", "Sex_Label", "Age_Label", "Age_Group"], errors="ignore"
    )
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.4,
        vmax=0.6,
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    plt.title(
        "Feature Correlation Matrix (BRFSS 2015 Indicators)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()
    logger.info(f"Exploratory figures generated in {output_dir}")


if __name__ == "__main__":
    df = load_dataset()
    profile = generate_dataset_profile(df)
    plot_dataset_visualizations(df)
    print("Dataset loading and profiling complete.")
    print(profile)
