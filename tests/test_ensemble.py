"""
Tests for the majority-vote ensemble defense in src/defense/ensemble.py.
Run with: pytest tests/test_ensemble.py -v
"""

import numpy as np
import pytest
from pathlib import Path

SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models")


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_data():
    """Load a small slice of the CICIDS2017 test split for prediction tests."""
    data = np.load(SPLITS_DIR / "cicids2017.npz", allow_pickle=True)
    label_map = np.load(SPLITS_DIR / "cicids2017_label_map.npy", allow_pickle=True).item()
    X_test = data["X_test"][:200]
    y_test = data["y_test"][:200]
    return X_test, y_test, label_map


@pytest.fixture(scope="module")
def baseline_ensemble():
    """Load the baseline ensemble once for the whole test session."""
    from src.defense.ensemble import Ensemble
    return Ensemble.baseline()


# ── Model file existence ───────────────────────────────────────────────────────

def test_baseline_model_files_exist():
    for filename in [
        "rf_cicids2017.pkl",
        "xgb_cicids2017.pkl",
        "mlp_cicids2017.h5",
        "scaler_cicids2017.pkl",
    ]:
        assert (MODELS_DIR / filename).exists(), f"Missing baseline model file: {filename}"

def test_adversarial_model_files_exist():
    """
    Will fail until adversarial retrained models are committed.
    Skipped automatically if the files are not present yet.
    """
    missing = [
        f for f in [
            "adv_rf_cicids2017.pkl",
            "adv_xgb_cicids2017.pkl",
            "adv_mlp_cicids2017.h5",
        ]
        if not (MODELS_DIR / f).exists()
    ]
    if missing:
        pytest.skip(f"Adversarial model files not yet available: {missing}")


# ── Ensemble loading ───────────────────────────────────────────────────────────

def test_baseline_loads_without_error():
    from src.defense.ensemble import Ensemble
    ensemble = Ensemble.baseline()
    assert ensemble is not None


def test_baseline_repr_contains_name():
    from src.defense.ensemble import Ensemble
    ensemble = Ensemble.baseline()
    assert "baseline" in repr(ensemble)


def test_adversarial_loads_without_error():
    """Skipped until adversarial retrained model files exist."""
    missing = [
        f for f in ["adv_rf_cicids2017.pkl", "adv_xgb_cicids2017.pkl", "adv_mlp_cicids2017.h5"]
        if not (MODELS_DIR / f).exists()
    ]
    if missing:
        pytest.skip(f"Adversarial model files not yet available: {missing}")

    from src.defense.ensemble import Ensemble
    ensemble = Ensemble.adversarial()
    assert ensemble is not None


# ── Prediction output shape and type ──────────────────────────────────────────

def test_predict_returns_1d_array(baseline_ensemble, test_data):
    X, _, _ = test_data
    preds = baseline_ensemble.predict(X)
    assert preds.ndim == 1, f"Expected 1D output, got shape {preds.shape}"


def test_predict_length_matches_input(baseline_ensemble, test_data):
    X, _, _ = test_data
    preds = baseline_ensemble.predict(X)
    assert len(preds) == len(X), (
        f"Prediction length {len(preds)} != input length {len(X)}"
    )


def test_predict_dtype_is_int(baseline_ensemble, test_data):
    X, _, _ = test_data
    preds = baseline_ensemble.predict(X)
    assert np.issubdtype(preds.dtype, np.integer), (
        f"Expected integer dtype, got {preds.dtype}"
    )


def test_predict_labels_are_valid(baseline_ensemble, test_data):
    """All predicted label IDs must exist in the label map."""
    X, _, label_map = test_data
    preds = baseline_ensemble.predict(X)
    valid_ids = set(label_map.keys())
    predicted = set(np.unique(preds).tolist())
    unknown = predicted - valid_ids
    assert not unknown, f"Predicted unknown label IDs: {unknown}"


def test_predict_no_nan_in_output(baseline_ensemble, test_data):
    X, _, _ = test_data
    preds = baseline_ensemble.predict(X)
    assert not np.isnan(preds.astype(float)).any(), "NaN found in predictions"


# ── Prediction sanity ──────────────────────────────────────────────────────────

def test_predict_accuracy_above_threshold(baseline_ensemble, test_data):
    """
    Ensemble accuracy on the first 200 test samples should be well above 90%.
    If this fails the ensemble is loading or predicting incorrectly.
    """
    X, y, _ = test_data
    preds = baseline_ensemble.predict(X)
    accuracy = (preds == y).mean()
    assert accuracy >= 0.90, (
        f"Ensemble accuracy {accuracy:.2%} is below 0.90 - check model loading."
    )


def test_predict_not_all_same_class(baseline_ensemble, test_data):
    """
    Predictions should not all collapse to a single class - that would indicate
    a broken model or scaler mismatch.
    """
    X, _, _ = test_data
    preds = baseline_ensemble.predict(X)
    assert len(np.unique(preds)) > 1, (
        "All predictions are the same class - possible scaler or model loading issue."
    )


def test_predict_deterministic(baseline_ensemble, test_data):
    """Running predict twice on the same input should return identical results."""
    X, _, _ = test_data
    preds_1 = baseline_ensemble.predict(X)
    preds_2 = baseline_ensemble.predict(X)
    assert np.array_equal(preds_1, preds_2), "predict() is not deterministic"


# ── Interface consistency ──────────────────────────────────────────────────────

def test_baseline_and_adversarial_same_output_shape(test_data):
    """
    Both ensembles must return the same output shape for the same input.
    Skipped until adversarial models are available.
    """
    missing = [
        f for f in ["adv_rf_cicids2017.pkl", "adv_xgb_cicids2017.pkl", "adv_mlp_cicids2017.h5"]
        if not (MODELS_DIR / f).exists()
    ]
    if missing:
        pytest.skip(f"Adversarial model files not yet available: {missing}")

    from src.defense.ensemble import Ensemble
    X, _, _ = test_data
    baseline_preds = Ensemble.baseline().predict(X)
    adversarial_preds = Ensemble.adversarial().predict(X)
    assert baseline_preds.shape == adversarial_preds.shape, (
        f"Shape mismatch: baseline={baseline_preds.shape}, adversarial={adversarial_preds.shape}"
    )