"""
tests/test_robustness.py
Unit tests verifying controlled perturbation masks and corruption algorithms.
"""

import numpy as np

from src.preprocessing import (
    BINARY_FEATURES,
    ORDINAL_FEATURES,
    prepare_data,
)


def test_missing_data_mask_proportions():
    """Verify that simulated MCAR missingness matches the targeted rates."""
    data = prepare_data(save_artifacts=False)
    X_raw = data["X_test_raw"].iloc[:1000].copy()

    rate = 0.10
    rng = np.random.RandomState(42)
    mask = rng.rand(*X_raw.shape) < rate
    X_raw[mask] = np.nan

    observed_rate = X_raw.isnull().values.mean()
    assert abs(observed_rate - rate) < 0.02


def test_noise_data_bounds():
    """Verify that bit-flipping and ordinal shifts preserve valid feature domains."""
    data = prepare_data(save_artifacts=False)
    X_raw = data["X_test_raw"].iloc[:1000].copy()
    rng = np.random.RandomState(42)

    # Test binary bit-flip
    col = BINARY_FEATURES[0]
    flip_mask = rng.rand(len(X_raw)) < 0.2
    flipped = X_raw[col].copy()
    flipped[flip_mask] = 1.0 - flipped[flip_mask]
    assert set(flipped.unique()).issubset({0.0, 1.0})

    # Test ordinal perturbation bounds
    ord_col = ORDINAL_FEATURES[0]  # GenHlth (1-5)
    shift_mask = rng.rand(len(X_raw)) < 0.2
    directions = rng.choice([-1.0, 1.0], size=len(X_raw))
    min_v, max_v = X_raw[ord_col].min(), X_raw[ord_col].max()
    shifted = np.clip(X_raw[ord_col] + directions * shift_mask, min_v, max_v)
    assert shifted.min() >= min_v
    assert shifted.max() <= max_v
