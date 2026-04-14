"""
Train baseline models (RF, XGBoost, MLP + majority-voting ensemble) on NIDS datasets.
"""
import json
import argparse
from pathlib import Path
from collections import Counter
import sys

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Allow both:
# - python -m src.train_baseline
# - python src/train_baseline.py
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.dotenv_utils import get_env_int, get_env_str

RF_N_ESTIMATORS = get_env_int("RF_N_ESTIMATORS")
RF_RANDOM_STATE = get_env_int("RF_RANDOM_STATE")
RF_N_JOBS = get_env_int("RF_N_JOBS")
RF_CLASS_WEIGHT = get_env_str("RF_CLASS_WEIGHT")
XGB_RANDOM_STATE = get_env_int("XGB_RANDOM_STATE")
XGB_N_JOBS = get_env_int("XGB_N_JOBS")
MLP_HIDDEN_1 = get_env_int("MLP_HIDDEN_1")
MLP_HIDDEN_2 = get_env_int("MLP_HIDDEN_2")
MLP_HIDDEN_3 = get_env_int("MLP_HIDDEN_3")
MLP_OPTIMIZER = get_env_str("MLP_OPTIMIZER")
MLP_EARLYSTOP_PATIENCE = get_env_int("MLP_EARLYSTOP_PATIENCE")
MLP_EPOCHS = get_env_int("MLP_EPOCHS")
MLP_BATCH_SIZE = get_env_int("MLP_BATCH_SIZE")

# Paths
SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["cicids2017", "nslkdd", "unswnb15"]


def load_split_stats():
    """Load split_stats.json (row counts, label_map, feature_names)."""
    path = SPLITS_DIR / "split_stats.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_label_map(dataset_name, split_stats):
    """Label map as dict int -> str. Prefer split_stats, else .npy."""
    if split_stats and dataset_name in split_stats:
        raw = split_stats[dataset_name]["label_map"]
        return {int(k): v for k, v in raw.items()}
    path = SPLITS_DIR / f"{dataset_name}_label_map.npy"
    if path.exists():
        return np.load(path, allow_pickle=True).item()
    raise FileNotFoundError(f"No label map for {dataset_name}")


def load_dataset(dataset_name, split_stats=None):
    """Load train/val and label map from data .npz (with optional feature_names)."""
    data = np.load(SPLITS_DIR / f"{dataset_name}.npz", allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    label_map = get_label_map(dataset_name, split_stats)
    feature_names = None
    if "feature_names" in data.files:
        feature_names = data["feature_names"].tolist()
    return X_train, y_train, X_val, y_val, label_map, feature_names


def compute_metrics(y_true, y_pred, label_map):
    """Accuracy, per-class precision/recall/F1, confusion matrix."""
    n_classes = len(label_map)
    labels = list(range(n_classes))
    target_names = [label_map[i] for i in labels]

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {
        name: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
        }
        for name, p, r, f in zip(target_names, prec, rec, f1)
    }
    return {
        "accuracy": acc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": target_names,
    }


def train_rf(X_train, y_train, X_val, y_val, label_map):
    """Random Forest n_estimators=100, class_weight=balanced per feature_notes."""
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RF_RANDOM_STATE,
        n_jobs=RF_N_JOBS,
        class_weight=RF_CLASS_WEIGHT,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return model, compute_metrics(y_val, y_pred, label_map)


def train_xgb(X_train, y_train, X_val, y_val, label_map):
    """XGBoost default params; sample_weight from class_weight for imbalance."""
    cw = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    sample_weight = np.array([cw[c] for c in y_train])
    model = xgb.XGBClassifier(
        random_state=XGB_RANDOM_STATE,
        n_jobs=XGB_N_JOBS,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = model.predict(X_val)
    return model, compute_metrics(y_val, y_pred, label_map)


def train_mlp(X_train, y_train, X_val, y_val, label_map):
    """3-layer MLP 128-64-32, ReLU, Adam; class_weight for imbalance (feature_notes)."""
    n_classes = len(label_map)
    n_features = X_train.shape[1]

    cw = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(cw))

    model = Sequential([
        Dense(MLP_HIDDEN_1, activation="relu", input_shape=(n_features,)),
        Dense(MLP_HIDDEN_2, activation="relu"),
        Dense(MLP_HIDDEN_3, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=MLP_OPTIMIZER,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=MLP_EARLYSTOP_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    model.fit(
        X_train,
        y_train,
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH_SIZE,
        class_weight=class_weights,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1,
    )
    y_pred_proba = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    return model, compute_metrics(y_val, y_pred, label_map)


def majority_vote(preds_list):
    """preds_list: list of (n_samples,) arrays. Returns (n_samples,) int array."""
    preds = np.stack(preds_list, axis=1)
    out = np.zeros(preds.shape[0], dtype=np.int64)
    for i in range(preds.shape[0]):
        out[i] = Counter(preds[i]).most_common(1)[0][0]
    return out


def run_one_dataset(dataset_name, split_stats):
    """Train RF, XGBoost, MLP + majority voting; evaluate on validation; return metrics."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print("Loading from patched .npz and split_stats.json...")
    X_train, y_train, X_val, y_val, label_map, _ = load_dataset(
        dataset_name, split_stats
    )
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Classes: {len(label_map)}")

    all_metrics = {}
    preds_for_vote = []

    ### Random Forest ###
    print(f"\n### Random Forest (n_estimators=100, class_weight=balanced) ###")
    rf_model, rf_metrics = train_rf(X_train, y_train, X_val, y_val, label_map)
    all_metrics["random_forest"] = rf_metrics
    preds_for_vote.append(rf_model.predict(X_val))
    print(f"  Validation accuracy: {rf_metrics['accuracy']:.4f}")
    joblib.dump(rf_model, MODELS_DIR / f"rf_{dataset_name}.pkl")

    ### XGBoost ###
    print(f"\n--- XGBoost (default params, sample_weight=balanced) ---")
    xgb_model, xgb_metrics = train_xgb(X_train, y_train, X_val, y_val, label_map)
    all_metrics["xgboost"] = xgb_metrics
    preds_for_vote.append(xgb_model.predict(X_val))
    print(f"  Validation accuracy: {xgb_metrics['accuracy']:.4f}")
    joblib.dump(xgb_model, MODELS_DIR / f"xgb_{dataset_name}.pkl")

    ### MLP (scaled features) ###
    print("\n--- Scaling for MLP ---")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    joblib.dump(scaler, MODELS_DIR / f"scaler_{dataset_name}.pkl")

    print("\n--- MLP (128-64-32, ReLU, Adam, class_weight=balanced) ---")
    mlp_model, mlp_metrics = train_mlp(
        X_train_s, y_train, X_val_s, y_val, label_map
    )
    all_metrics["mlp"] = mlp_metrics
    preds_for_vote.append(np.argmax(mlp_model.predict(X_val_s, verbose=0), axis=1))
    print(f"  Validation accuracy: {mlp_metrics['accuracy']:.4f}")
    mlp_model.save(MODELS_DIR / f"mlp_{dataset_name}.h5", save_format="h5")

    ### Majority voting ensemble ###
    print("\n--- Majority voting ensemble ---")
    ensemble_pred = majority_vote(preds_for_vote)
    all_metrics["majority_voting"] = compute_metrics(y_val, ensemble_pred, label_map)
    print(f"  Validation accuracy: {all_metrics['majority_voting']['accuracy']:.4f}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Train baseline NIDS models.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS,
        help="Train and evaluate on this dataset only.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on all three datasets (CICIDS2017, NSL-KDD, UNSW-NB15).",
    )
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Provide either --dataset <name> or --all")

    split_stats = load_split_stats()
    if split_stats:
        print("Loaded data/splits/split_stats.json (row counts, label_map, feature_names).")
    else:
        print("split_stats.json not found; using .npy label maps only.")

    if args.all:
        datasets_to_run = DATASETS
        combined = {}
    else:
        datasets_to_run = [args.dataset]
        combined = None

    for dataset_name in datasets_to_run:
        metrics = run_one_dataset(dataset_name, split_stats)
        out_path = RESULTS_DIR / f"baseline_metrics_{dataset_name}.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {out_path}")

        if args.all:
            combined[dataset_name] = metrics

    if combined is not None:
        summary_path = RESULTS_DIR / "baseline_metrics.json"
        with open(summary_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nCombined metrics (all datasets) saved to {summary_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Validation accuracy (target ≥90%)")
    print("=" * 60)
    for d in datasets_to_run:
        path = RESULTS_DIR / f"baseline_metrics_{d}.json"
        if path.exists():
            with open(path) as f:
                m = json.load(f)
            print(f"\n{d}:")
            for model_name, meta in m.items():
                acc = meta["accuracy"]
                status = "OK" if acc >= 0.90 else "BELOW TARGET"
                print(f"  {model_name}: {acc:.2%} {status}")


if __name__ == "__main__":
    main()
