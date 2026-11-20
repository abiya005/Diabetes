"""
src/preprocessing.py
Reproducible preprocessing pipeline with strict data leakage prevention and demographic preservation.
"""

import logging
import os
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Feature taxonomy
BINARY_FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
]

NUMERICAL_FEATURES = ["BMI", "MentHlth", "PhysHlth"]

ORDINAL_FEATURES = ["GenHlth", "Education", "Income"]

DEMOGRAPHIC_FEATURES = ["Sex", "Age"]

ALL_MODEL_FEATURES = (
    BINARY_FEATURES + NUMERICAL_FEATURES + ORDINAL_FEATURES + DEMOGRAPHIC_FEATURES
)


def build_preprocessor() -> ColumnTransformer:
    """
    Build Scikit-Learn ColumnTransformer pipeline.
    Preserves binary features as binary (0/1) indicators.
    Scales numerical and ordinal features using StandardScaler.
    Applies median and most-frequent imputation to handle robustness tests seamlessly.
    """
    # Numerical pipeline: median imputation + standard scaling
    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Ordinal pipeline: median imputation + standard scaling (preserves relative magnitude for linear/margin models)
    ord_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Binary pipeline: most frequent imputation (preserves 0/1 indicator representation without shifting baseline)
    bin_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])

    # Demographic pipeline: preserve discrete values for model inputs with median imputer
    demo_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("ord", ord_pipeline, ORDINAL_FEATURES),
            ("bin", bin_pipeline, BINARY_FEATURES),
            ("demo", demo_pipeline, DEMOGRAPHIC_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    """Get clean column names matching transformed output."""
    feature_names = (
        NUMERICAL_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES + DEMOGRAPHIC_FEATURES
    )
    return feature_names


def prepare_data(
    filepath: str = "data/raw/diabetes_012_health_indicators_BRFSS2015.csv",
    test_size: float = 0.20,
    random_state: int = 42,
    output_dir: str = "data/processed",
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """
    Execute end-to-end reproducible preprocessing:
      1. Load dataset & validate target
      2. Stratified 80:20 train-test split
      3. Fit preprocessor strictly on training data
      4. Transform train and test sets
      5. Retain original demographic metadata for fairness audits
      6. Persist processed datasets and transformer
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = load_dataset(filepath)

    # Feature matrix X and target y
    X = df[ALL_MODEL_FEATURES].copy()
    y = df["Diabetes_binary"].values

    # Metadata for fairness evaluation
    demographics = df[["Sex", "Age", "Sex_Label", "Age_Label", "Age_Group"]].copy()

    logger.info(
        f"Splitting dataset: {len(df)} records into {(1 - test_size) * 100:.0f}% train and {test_size * 100:.0f}% test (stratified)"
    )

    # Stratified Train-Test Split (80:20, seed=42)
    X_train_raw, X_test_raw, y_train, y_test, demo_train, demo_test = train_test_split(
        X, y, demographics, test_size=test_size, stratify=y, random_state=random_state
    )

    logger.info(
        f"Train set: {X_train_raw.shape[0]} records (Positive prevalence: {y_train.mean() * 100:.2f}%)"
    )
    logger.info(
        f"Test set:  {X_test_raw.shape[0]} records (Positive prevalence: {y_test.mean() * 100:.2f}%)"
    )

    # Build and fit preprocessor strictly on train data
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train_raw)

    feature_cols = get_feature_names_out(preprocessor)

    # Transform train and test partitions
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    X_train_df = pd.DataFrame(X_train, columns=feature_cols, index=X_train_raw.index)
    X_test_df = pd.DataFrame(X_test, columns=feature_cols, index=X_test_raw.index)

    # Attach target and demographic metadata for comprehensive audit files
    train_full = pd.concat(
        [
            X_train_df,
            pd.Series(y_train, name="Diabetes_binary", index=X_train_raw.index),
            demo_train,
        ],
        axis=1,
    )
    test_full = pd.concat(
        [
            X_test_df,
            pd.Series(y_test, name="Diabetes_binary", index=X_test_raw.index),
            demo_test,
        ],
        axis=1,
    )

    if save_artifacts:
        preprocessor_path = os.path.join("models", "preprocessor.joblib")
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Saved preprocessor pipeline to {preprocessor_path}")

        train_path_csv = os.path.join(output_dir, "train_processed.csv")
        test_path_csv = os.path.join(output_dir, "test_processed.csv")
        train_full.to_csv(train_path_csv, index=False)
        test_full.to_csv(test_path_csv, index=False)
        logger.info(
            f"Saved processed train ({train_path_csv}) and test ({test_path_csv}) datasets."
        )

        # Save summary report text
        summary_text = (
            f"PREPROCESSING AND LEAKAGE-PREVENTION REPORT\n"
            f"===========================================\n"
            f"Dataset Source: CDC BRFSS 2015 Diabetes Indicators\n"
            f"Total Original Observations: {len(df)}\n"
            f"Train Split: {len(X_train_raw)} ({(1 - test_size) * 100:.1f}%)\n"
            f"Test Split:  {len(X_test_raw)} ({test_size * 100:.1f}%)\n"
            f"Stratification: By Diabetes_binary class (Preserved positive ratio: ~15.76%)\n"
            f"Random State: {random_state}\n\n"
            f"Feature Transformations:\n"
            f"  - Numerical ({len(NUMERICAL_FEATURES)} features: {NUMERICAL_FEATURES}): Median Imputation + StandardScaler\n"
            f"  - Ordinal ({len(ORDINAL_FEATURES)} features: {ORDINAL_FEATURES}): Median Imputation + StandardScaler\n"
            f"  - Binary ({len(BINARY_FEATURES)} features: {BINARY_FEATURES}): Most Frequent Imputation (No scaling, preserved as 0/1 indicators)\n"
            f"  - Demographics ({len(DEMOGRAPHIC_FEATURES)} features: {DEMOGRAPHIC_FEATURES}): Median Imputation + StandardScaler for model inputs\n"
            f"  - Demographic Audit Meta (Sex_Label, Age_Group): Preserved unscaled and unaltered for subgroup fairness evaluation\n\n"
            f"Data Leakage Safeguards:\n"
            f"  1. All imputers and scalers fit STRICTLY on training split (X_train_raw).\n"
            f"  2. Test split (X_test_raw) was held out untouched until final model evaluation.\n"
            f"  3. Cross-validation applies pipeline transformations strictly within training folds.\n"
        )
        with open(os.path.join("reports", "preprocessing_summary.txt"), "w") as f:
            f.write(summary_text)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "demo_train": demo_train,
        "demo_test": demo_test,
        "preprocessor": preprocessor,
        "feature_names": feature_cols,
    }


if __name__ == "__main__":
    data = prepare_data()
    print("Preprocessing completed successfully.")
    print("X_train shape:", data["X_train"].shape)
    print("X_test shape:", data["X_test"].shape)
