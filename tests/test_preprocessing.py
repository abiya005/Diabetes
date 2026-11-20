"""
tests/test_preprocessing.py
Unit tests verifying data loading, target harmonization, split stratification, and transformer isolation.
"""

import numpy as np

from src.data_loader import load_dataset
from src.preprocessing import ALL_MODEL_FEATURES, build_preprocessor, prepare_data


def test_dataset_loading_and_target():
    """Verify raw dataset loading and correct target binarization."""
    df = load_dataset("data/raw/diabetes_012_health_indicators_BRFSS2015.csv")
    assert "Diabetes_binary" in df.columns
    assert set(df["Diabetes_binary"].unique()).issubset({0, 1})
    assert len(df) == 253680
    assert "Sex_Label" in df.columns
    assert "Age_Group" in df.columns


def test_stratified_split_integrity():
    """Verify that train and test sets are strictly partitioned without data leakage."""
    data = prepare_data(save_artifacts=False)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    assert len(X_train) == 202944
    assert len(X_test) == 50736
    assert X_train.shape[1] == len(ALL_MODEL_FEATURES)
    assert X_test.shape[1] == len(ALL_MODEL_FEATURES)

    # Verify stratified proportion preservation
    train_prev = np.mean(y_train)
    test_prev = np.mean(y_test)
    assert abs(train_prev - test_prev) < 0.005

    # Check disjoint index sets (No data leakage)
    train_idx = set(data["X_train_raw"].index)
    test_idx = set(data["X_test_raw"].index)
    assert len(train_idx.intersection(test_idx)) == 0


def test_preprocessor_pipeline_transformation():
    """Verify that ColumnTransformer produces valid finite numerical arrays."""
    preprocessor = build_preprocessor()
    data = prepare_data(save_artifacts=False)

    # Fit strictly on train
    preprocessor.fit(data["X_train_raw"])

    trans_train = preprocessor.transform(data["X_train_raw"])
    trans_test = preprocessor.transform(data["X_test_raw"])

    assert not np.isnan(trans_train).any()
    assert not np.isnan(trans_test).any()
    assert trans_train.shape[1] == len(ALL_MODEL_FEATURES)
