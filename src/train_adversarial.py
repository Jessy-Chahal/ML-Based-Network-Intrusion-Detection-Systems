"""
Retrain baseline models (RF, XGBoost, MLP, LSTM + majority-voting ensemble)
on the adversarial training set from data/adversarial/ (output of
gen_adversarial_dataset.py: adv_train_*.npz). Validation still comes from
data/splits/<dataset>.npz.

LSTM matches train_lstm.py: stacked LSTM (64→32), dense head, scaled tabular
features as a sequence (n_features, 1), warmup epochs then class-weighted fit.

Metrics JSON: results/retrained_adversarial_metrics_<dataset>.json and
results/retrained_adversarial_metrics.json (--all).
"""
import json
import argparse
from pathlib import Path
from scipy.stats import mode

import numpy as np
import joblib
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Defaults aligned with train_lstm.py
LSTM_LR = 8e-4
LSTM_EPOCHS = 4
LSTM_WARMUP_EPOCHS = 2
LSTM_BATCH_SIZE = 8192
LSTM_PATIENCE = 2
LSTM_SEED = 42
LSTM_CLASS_WEIGHT_ALPHA = 0.05
LSTM_MAX_CLASS_WEIGHT = 20.0

# Directory Setup
SPLITS_DIR = Path("data/splits")
ADV_DIR = Path("data/adversarial")
ADV_SUMMARY_PATH = ADV_DIR / "adv_generation_summary.json"
# File stems after adv_train_ — must match src/gen_adversarial_dataset.py DATASET_NAME_MAP
ADV_TRAIN_STEM = {"cicids2017": "cicids", "nslkdd": "nslkdd", "unswnb15": "unswnb15"}

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["cicids2017", "nslkdd", "unswnb15"]


def load_split_stats():
    """Fetch metadata like row counts, label maps, and feature names."""
    path = SPLITS_DIR / "split_stats.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_label_map(dataset_name, split_stats):
    """Retrieve the mapping of integer labels to string class names."""
    if split_stats and dataset_name in split_stats:
        return {int(k): v for k, v in split_stats[dataset_name]["label_map"].items()}
    
    fallback_path = SPLITS_DIR / f"{dataset_name}_label_map.npy"
    if fallback_path.exists():
        return np.load(fallback_path, allow_pickle=True).item()
        
    raise FileNotFoundError(f"Could not find a label map for {dataset_name}. Did you run the preprocessor?")


def _resolve_adv_train_npz(dataset_name: str) -> Path:
    """
    Path to adv_train_*.npz from gen_adversarial_dataset.py.
    Uses adv_generation_summary.json output_path first, then default names under data/adversarial/.
    """
    candidates: list[Path] = []
    if ADV_SUMMARY_PATH.exists():
        with open(ADV_SUMMARY_PATH, encoding="utf-8") as f:
            summary = json.load(f)
        op = (summary.get(dataset_name) or {}).get("output_path")
        if op:
            candidates.append(Path(str(op).replace("\\", "/")))
    stem = ADV_TRAIN_STEM.get(dataset_name)
    if stem:
        candidates.append(ADV_DIR / f"adv_train_{stem}.npz")
    if dataset_name == "unswnb15":
        candidates.append(ADV_DIR / "adv_train_unsw_nb15.npz")

    for p in candidates:
        if p.is_file():
            return p

    tried = ", ".join(str(p) for p in candidates) if candidates else "(none)"
    raise FileNotFoundError(
        f"No adversarial training .npz found for {dataset_name}. "
        f"Tried: {tried}. Run: python src/gen_adversarial_dataset.py"
    )


def load_adversarial_mixed_dataset(dataset_name, split_stats=None):
    """
    Training: data/adversarial/adv_train_*.npz (70% stratified clean + generated adversarial).
    Validation: data/splits/<dataset>.npz (unchanged split).
    """
    adv_path = _resolve_adv_train_npz(dataset_name)
    base_path = SPLITS_DIR / f"{dataset_name}.npz"

    if not base_path.exists():
        raise FileNotFoundError(
            f"Missing validation split for {dataset_name}: {base_path}"
        )

    adv_data = np.load(adv_path, allow_pickle=True)
    base_data = np.load(base_path, allow_pickle=True)

    if "X_train" not in adv_data.files or "y_train" not in adv_data.files:
        raise KeyError(f"{adv_path} must contain X_train and y_train")

    label_map = get_label_map(dataset_name, split_stats)

    return (
        adv_data["X_train"],
        adv_data["y_train"],
        base_data["X_val"],
        base_data["y_val"],
        label_map,
        adv_path,
    )


def compute_metrics(y_true, y_pred, label_map):
    """Calculates accuracy, per-class stats (Precision/Recall/F1), and generates a confusion matrix."""
    labels = list(range(len(label_map)))
    target_names = [label_map[i] for i in labels]

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {
        name: {"precision": float(p), "recall": float(r), "f1": float(f)}
        for name, p, r, f in zip(target_names, prec, rec, f1)
    }
    
    return {
        "accuracy": acc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": target_names,
    }


def train_rf(X_train, y_train, X_val, y_val, label_map, dataset_name):
    """Trains and saves a balanced Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)
    
    joblib.dump(model, MODELS_DIR / f"adv_rf_{dataset_name}.pkl")
    return model, compute_metrics(y_val, model.predict(X_val), label_map)


def train_xgb(X_train, y_train, X_val, y_val, label_map, dataset_name):
    """Trains and saves an XGBoost classifier, applying sample weights to handle class imbalance."""
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    sample_weights = np.array([cw[c] for c in y_train])
    
    model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    joblib.dump(model, MODELS_DIR / f"adv_xgb_{dataset_name}.pkl")
    return model, compute_metrics(y_val, model.predict(X_val), label_map)


def train_mlp(X_train, y_train, X_val, y_val, label_map, dataset_name):
    """
    Trains and saves a 3-layer MLP. 
    Includes data scaling and uses EarlyStopping to mitigate training instability on rare classes.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, MODELS_DIR / f"adv_scaler_{dataset_name}.pkl")

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(cw))

    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(len(label_map), activation="softmax"),
    ])
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    early_stop = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1)
    
    model.fit(
        X_train_scaled, y_train,
        epochs=20, batch_size=256,
        class_weight=class_weights,
        validation_data=(X_val_scaled, y_val),
        callbacks=[early_stop],
        verbose=1,
    )
    
    model.save(MODELS_DIR / f"adv_mlp_{dataset_name}.h5", save_format="h5")
    
    y_pred = np.argmax(model.predict(X_val_scaled, verbose=0), axis=1)
    return model, compute_metrics(y_val, y_pred, label_map)


def as_sequence(X: np.ndarray) -> np.ndarray:
    """(n_samples, n_features) -> (n_samples, n_features, 1) — same as train_lstm.py."""
    return X[..., np.newaxis]


def lstm_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Smoothed + capped balanced weights (train_lstm.py DEFAULT_CLASS_WEIGHT_ALPHA / MAX)."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weights = 1.0 + LSTM_CLASS_WEIGHT_ALPHA * (weights - 1.0)
    weights = np.minimum(weights, LSTM_MAX_CLASS_WEIGHT)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def train_lstm(X_train, y_train, X_val, y_val, label_map, dataset_name):
    """Train LSTM on mixed data; save adv_lstm_*.h5 and adv_scaler_lstm_*.pkl."""
    tf.keras.utils.set_random_seed(LSTM_SEED)
    n_features = X_train.shape[1]
    n_classes = len(label_map)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    joblib.dump(scaler, MODELS_DIR / f"adv_scaler_lstm_{dataset_name}.pkl")

    X_train_seq = as_sequence(X_train_scaled)
    X_val_seq = as_sequence(X_val_scaled)
    cw = lstm_class_weights(y_train)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features, 1)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LSTM_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=LSTM_PATIENCE,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, LSTM_PATIENCE // 2),
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    if LSTM_WARMUP_EPOCHS > 0:
        model.fit(
            X_train_seq,
            y_train,
            validation_data=(X_val_seq, y_val),
            epochs=LSTM_WARMUP_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            verbose=2,
        )

    model.fit(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        class_weight=cw,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(MODELS_DIR / f"adv_lstm_{dataset_name}.h5")
    y_pred = np.argmax(model.predict(X_val_seq, verbose=0), axis=1)
    return model, compute_metrics(y_val, y_pred, label_map)


def majority_vote(preds_list):
    """
    Vectorized majority voting across all model predictions using SciPy.
    Substantially faster than looping with a Counter.
    """
    preds = np.stack(preds_list, axis=1)
    return mode(preds, axis=1, keepdims=False).mode


def run_one_dataset(dataset_name, split_stats):
    """Executes the full adversarial training pipeline for a single dataset."""
    print(f"\n{'='*60}")
    print(f"Starting Adversarial Training for: {dataset_name.upper()}")
    
    X_train, y_train, X_val, y_val, label_map, adv_train_path = load_adversarial_mixed_dataset(
        dataset_name, split_stats
    )
    print(f"  Train file: {adv_train_path}")
    print(
        f"Data loaded -> Train: {X_train.shape[0]:,} samples | Val: {X_val.shape[0]:,} samples | Classes: {len(label_map)}"
    )

    all_metrics = {}
    preds_for_vote = []

    # 1. Random Forest
    print("\nTraining Random Forest...")
    rf_model, rf_metrics = train_rf(X_train, y_train, X_val, y_val, label_map, dataset_name)
    all_metrics["random_forest"] = rf_metrics
    preds_for_vote.append(rf_model.predict(X_val))
    print(f"   RF Validation Accuracy: {rf_metrics['accuracy']:.4f}")

    # 2. XGBoost
    print("\nTraining XGBoost...")
    xgb_model, xgb_metrics = train_xgb(X_train, y_train, X_val, y_val, label_map, dataset_name)
    all_metrics["xgboost"] = xgb_metrics
    preds_for_vote.append(xgb_model.predict(X_val))
    print(f"   XGB Validation Accuracy: {xgb_metrics['accuracy']:.4f}")

    # 3. Multi-Layer Perceptron
    print("\nTraining 3-Layer MLP...")
    mlp_model, mlp_metrics = train_mlp(X_train, y_train, X_val, y_val, label_map, dataset_name)
    all_metrics["mlp"] = mlp_metrics
    
    # Reload scaler to make the prediction for the vote array
    scaler = joblib.load(MODELS_DIR / f"adv_scaler_{dataset_name}.pkl")
    preds_for_vote.append(np.argmax(mlp_model.predict(scaler.transform(X_val), verbose=0), axis=1))
    print(f"   MLP Validation Accuracy: {mlp_metrics['accuracy']:.4f}")

    # 4. LSTM (same architecture & training recipe as train_lstm.py)
    print("\nTraining LSTM...")
    lstm_model, lstm_metrics = train_lstm(
        X_train, y_train, X_val, y_val, label_map, dataset_name
    )
    all_metrics["lstm"] = lstm_metrics
    lstm_scaler = joblib.load(MODELS_DIR / f"adv_scaler_lstm_{dataset_name}.pkl")
    X_val_lstm = as_sequence(lstm_scaler.transform(X_val).astype(np.float32))
    preds_for_vote.append(
        np.argmax(lstm_model.predict(X_val_lstm, verbose=0), axis=1)
    )
    print(f"   LSTM Validation Accuracy: {lstm_metrics['accuracy']:.4f}")

    # 5. Ensemble voting (RF, XGB, MLP, LSTM)
    print("\nCalculating Majority Vote Ensemble...")
    ensemble_pred = majority_vote(preds_for_vote)
    all_metrics["majority_voting"] = compute_metrics(y_val, ensemble_pred, label_map)
    print(f"   Ensemble Validation Accuracy: {all_metrics['majority_voting']['accuracy']:.4f}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Retrain NIDS models on adversarially mixed data to improve robustness.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=str, choices=DATASETS, help="Train and evaluate on a specific dataset.")
    group.add_argument("--all", action="store_true", help="Run the pipeline on all three datasets.")
    
    args = parser.parse_args()
    split_stats = load_split_stats()

    if split_stats:
        print("Loaded split_stats.json successfully.")
    else:
        print("split_stats.json not found. Falling back to .npy label maps.")

    datasets_to_run = DATASETS if args.all else [args.dataset]
    combined_metrics = {}

    for dataset_name in datasets_to_run:
        metrics = run_one_dataset(dataset_name, split_stats)
        
        # Save individual dataset metrics
        out_path = RESULTS_DIR / f"retrained_adversarial_metrics_{dataset_name}.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved individual metrics to {out_path}")
        
        combined_metrics[dataset_name] = metrics

    # Save master metrics file if running all datasets
    if args.all:
        summary_path = RESULTS_DIR / "retrained_adversarial_metrics.json"
        with open(summary_path, "w") as f:
            json.dump(combined_metrics, f, indent=2)
        print(f"\nCombined master metrics saved to {summary_path}")


if __name__ == "__main__":
    main()