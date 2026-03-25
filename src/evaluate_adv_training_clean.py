"""
Evaluate adversarially retrained models on clean test splits.

Outputs:
- results/adv_training_clean_metrics.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

ROOT = Path(".")
SPLITS_DIR = ROOT / "data/splits"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["cicids2017", "nslkdd", "unswnb15"]
MAX_ALLOWED_DROP = 0.03  # 3 percentage points


def majority_vote(preds_list: list[np.ndarray]) -> np.ndarray:
    preds = np.stack(preds_list, axis=1)
    out = np.zeros(preds.shape[0], dtype=np.int64)
    for i in range(preds.shape[0]):
        values, counts = np.unique(preds[i], return_counts=True)
        out[i] = values[np.argmax(counts)]
    return out


def load_clean_test(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    path = SPLITS_DIR / f"{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    with np.load(path, allow_pickle=True) as data:
        return data["X_test"], data["y_test"]


def evaluate_dataset(dataset: str) -> dict:
    X_test, y_test = load_clean_test(dataset)

    report: dict[str, dict] = {}
    baseline_vote_preds: list[np.ndarray] = []
    retrained_vote_preds: list[np.ndarray] = []

    # RF
    rf_base_path = MODELS_DIR / f"rf_{dataset}.pkl"
    rf_path = MODELS_DIR / f"adv_rf_{dataset}.pkl"
    rf_base = joblib.load(rf_base_path)
    rf = joblib.load(rf_path)
    rf_base_pred = rf_base.predict(X_test)
    rf_pred = rf.predict(X_test)
    rf_base_acc = float(accuracy_score(y_test, rf_base_pred))
    rf_acc = float(accuracy_score(y_test, rf_pred))
    report["random_forest"] = {
        "baseline_accuracy": rf_base_acc,
        "retrained_clean_test_accuracy": rf_acc,
        "accuracy_drop_pp": (rf_base_acc - rf_acc) * 100.0,
        "within_3pp_target": (rf_base_acc - rf_acc) <= MAX_ALLOWED_DROP,
        "baseline_model_path": str(rf_base_path),
        "model_path": str(rf_path),
    }
    baseline_vote_preds.append(rf_base_pred)
    retrained_vote_preds.append(rf_pred)

    # XGB
    xgb_base_path = MODELS_DIR / f"xgb_{dataset}.pkl"
    xgb_path = MODELS_DIR / f"adv_xgb_{dataset}.pkl"
    xgb_base = joblib.load(xgb_base_path)
    xgb = joblib.load(xgb_path)
    xgb_base_pred = xgb_base.predict(X_test)
    xgb_pred = xgb.predict(X_test)
    xgb_base_acc = float(accuracy_score(y_test, xgb_base_pred))
    xgb_acc = float(accuracy_score(y_test, xgb_pred))
    report["xgboost"] = {
        "baseline_accuracy": xgb_base_acc,
        "retrained_clean_test_accuracy": xgb_acc,
        "accuracy_drop_pp": (xgb_base_acc - xgb_acc) * 100.0,
        "within_3pp_target": (xgb_base_acc - xgb_acc) <= MAX_ALLOWED_DROP,
        "baseline_model_path": str(xgb_base_path),
        "model_path": str(xgb_path),
    }
    baseline_vote_preds.append(xgb_base_pred)
    retrained_vote_preds.append(xgb_pred)

    # MLP
    mlp_base_scaler_path = MODELS_DIR / f"scaler_{dataset}.pkl"
    mlp_base_path = MODELS_DIR / f"mlp_{dataset}.h5"
    mlp_scaler_path = MODELS_DIR / f"adv_scaler_{dataset}.pkl"
    mlp_path = MODELS_DIR / f"adv_mlp_{dataset}.h5"
    mlp_base_scaler = joblib.load(mlp_base_scaler_path)
    X_test_mlp_base = mlp_base_scaler.transform(X_test)
    mlp_base = tf.keras.models.load_model(mlp_base_path, compile=False)
    mlp_base_pred = np.argmax(mlp_base.predict(X_test_mlp_base, verbose=0), axis=1)
    mlp_scaler = joblib.load(mlp_scaler_path)
    X_test_mlp = mlp_scaler.transform(X_test)
    mlp = tf.keras.models.load_model(mlp_path, compile=False)
    mlp_pred = np.argmax(mlp.predict(X_test_mlp, verbose=0), axis=1)
    mlp_base_acc = float(accuracy_score(y_test, mlp_base_pred))
    mlp_acc = float(accuracy_score(y_test, mlp_pred))
    report["mlp"] = {
        "baseline_accuracy": mlp_base_acc,
        "retrained_clean_test_accuracy": mlp_acc,
        "accuracy_drop_pp": (mlp_base_acc - mlp_acc) * 100.0,
        "within_3pp_target": (mlp_base_acc - mlp_acc) <= MAX_ALLOWED_DROP,
        "baseline_model_path": str(mlp_base_path),
        "baseline_scaler_path": str(mlp_base_scaler_path),
        "model_path": str(mlp_path),
        "scaler_path": str(mlp_scaler_path),
    }
    baseline_vote_preds.append(mlp_base_pred)
    retrained_vote_preds.append(mlp_pred)

    # Optional LSTM (if adversarial retrained LSTM exists)
    lstm_base_path = MODELS_DIR / f"lstm_{dataset}.h5"
    lstm_base_scaler_path = MODELS_DIR / f"scaler_lstm_{dataset}.pkl"
    lstm_path = MODELS_DIR / f"adv_lstm_{dataset}.h5"
    lstm_scaler_path = MODELS_DIR / f"adv_scaler_lstm_{dataset}.pkl"
    if (
        lstm_base_path.exists()
        and lstm_base_scaler_path.exists()
        and lstm_path.exists()
        and lstm_scaler_path.exists()
    ):
        lstm_base_scaler = joblib.load(lstm_base_scaler_path)
        X_test_lstm_base = lstm_base_scaler.transform(X_test).astype(np.float32)[..., np.newaxis]
        lstm_base_model = tf.keras.models.load_model(lstm_base_path, compile=False)
        lstm_base_pred = np.argmax(lstm_base_model.predict(X_test_lstm_base, verbose=0), axis=1)
        lstm_base_acc = float(accuracy_score(y_test, lstm_base_pred))

        lstm_scaler = joblib.load(lstm_scaler_path)
        X_test_lstm = lstm_scaler.transform(X_test).astype(np.float32)[..., np.newaxis]
        lstm = tf.keras.models.load_model(lstm_path, compile=False)
        lstm_pred = np.argmax(lstm.predict(X_test_lstm, verbose=0), axis=1)
        lstm_acc = float(accuracy_score(y_test, lstm_pred))
        report["lstm"] = {
            "baseline_accuracy": lstm_base_acc,
            "retrained_clean_test_accuracy": lstm_acc,
            "accuracy_drop_pp": (lstm_base_acc - lstm_acc) * 100.0,
            "within_3pp_target": (lstm_base_acc - lstm_acc) <= MAX_ALLOWED_DROP,
            "baseline_model_path": str(lstm_base_path),
            "baseline_scaler_path": str(lstm_base_scaler_path),
            "model_path": str(lstm_path),
            "scaler_path": str(lstm_scaler_path),
        }
    elif lstm_path.exists() and lstm_scaler_path.exists():
        report["lstm"] = {
            "baseline_accuracy": None,
            "retrained_clean_test_accuracy": None,
            "accuracy_drop_pp": None,
            "within_3pp_target": None,
            "model_path": str(lstm_path),
            "scaler_path": str(lstm_scaler_path),
            "baseline_note": "Baseline LSTM model/scaler not found in models/.",
        }

    # Majority voting compared on the same model family as baseline scripts (RF+XGB+MLP).
    mv_base_pred = majority_vote(baseline_vote_preds)
    mv_pred = majority_vote(retrained_vote_preds)
    mv_base_acc = float(accuracy_score(y_test, mv_base_pred))
    mv_acc = float(accuracy_score(y_test, mv_pred))
    report["majority_voting"] = {
        "baseline_accuracy": mv_base_acc,
        "retrained_clean_test_accuracy": mv_acc,
        "accuracy_drop_pp": (mv_base_acc - mv_acc) * 100.0,
        "within_3pp_target": (mv_base_acc - mv_acc) <= MAX_ALLOWED_DROP,
        "member_models": ["random_forest", "xgboost", "mlp"],
        "note": "Majority vote compares baseline vs retrained using RF+XGB+MLP only.",
    }

    overall_pass = all(
        meta["within_3pp_target"]
        for name, meta in report.items()
        if meta["within_3pp_target"] is not None and name != "lstm"
    )
    return {
        "n_test_samples": int(len(y_test)),
        "target_max_drop_pp": 3.0,
        "overall_pass": overall_pass,
        "models": report,
    }


def main() -> None:
    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_mode": (
            "Baseline and retrained accuracies are both computed directly from model files "
            "on the same clean X_test split."
        ),
        "datasets": {},
    }
    for dataset in DATASETS:
        out["datasets"][dataset] = evaluate_dataset(dataset)

    # Global pass across datasets (ignoring optional LSTM when no baseline exists).
    out["overall_pass"] = all(
        out["datasets"][d]["overall_pass"] for d in DATASETS
    )

    out_path = RESULTS_DIR / "adv_training_clean_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved clean retraining evaluation to {out_path}")
    print(f"Overall pass (<=3pp drop): {out['overall_pass']}")


if __name__ == "__main__":
    main()
