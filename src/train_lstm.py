"""
Train an LSTM classifier on CICIDS2017 tabular flow features.

Design choice:
- Treat each feature as a timestep in a univariate sequence:
  input shape = (n_features, 1)

Outputs:
- models/lstm_<dataset>.h5
- results/lstm_baseline_metrics_<dataset>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Reproducible default settings (edit here to change baseline run behavior).
DATASET_NAMES = ["cicids2017", "nslkdd", "unswnb15"]
DEFAULT_DATASET = "cicids2017"
MODEL_FILENAME_TEMPLATE = "lstm_{dataset}.h5"
SCALER_FILENAME_TEMPLATE = "scaler_lstm_{dataset}.pkl"
METRICS_FILENAME_TEMPLATE = "lstm_baseline_metrics_{dataset}.json"

DEFAULT_LEARNING_RATE = 8e-4
DEFAULT_EPOCHS = 4
DEFAULT_WARMUP_EPOCHS = 2
DEFAULT_BATCH_SIZE = 8192
DEFAULT_PATIENCE = 2
DEFAULT_SEED = 42
DEFAULT_CLASS_WEIGHT_ALPHA = 0.05
DEFAULT_MAX_CLASS_WEIGHT = 20.0


def load_dataset(dataset_name: str):
    npz_path = SPLITS_DIR / f"{dataset_name}.npz"
    label_map_path = SPLITS_DIR / f"{dataset_name}_label_map.npy"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing split file: {npz_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing label map: {label_map_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        X_train = data["X_train"].astype(np.float32)
        y_train = data["y_train"].astype(np.int64)
        X_val = data["X_val"].astype(np.float32)
        y_val = data["y_val"].astype(np.int64)
        X_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.int64)

    label_map = np.load(label_map_path, allow_pickle=True).item()
    return X_train, y_train, X_val, y_val, X_test, y_test, label_map


def as_sequence(X: np.ndarray) -> np.ndarray:
    # (n_samples, n_features) -> (n_samples, n_features, 1)
    return X[..., np.newaxis]


def build_model(n_features: int, n_classes: int) -> tf.keras.Model:
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def class_weights(
    y: np.ndarray,
    alpha: float = 1.0,
    max_weight: float | None = None,
) -> dict[int, float]:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    # Smooth toward uniform weights to avoid instability from extremely rare classes.
    weights = 1.0 + alpha * (weights - 1.0)
    if max_weight is not None:
        weights = np.minimum(weights, max_weight)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_map: dict[int, str]) -> dict:
    labels = sorted(label_map.keys())
    names = [label_map[i] for i in labels]
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "per_class": {
            name: {"precision": float(pi), "recall": float(ri), "f1": float(fi)}
            for name, pi, ri, fi in zip(names, p, r, f1)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": names,
    }


def main():
    parser = argparse.ArgumentParser(description="Train an LSTM classifier on NIDS datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASET_NAMES,
        default=DEFAULT_DATASET,
        help="Dataset split to train/evaluate on.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=DEFAULT_WARMUP_EPOCHS,
        help="Initial epochs without class weights to stabilize optimization.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-class-weight",
        type=float,
        default=DEFAULT_MAX_CLASS_WEIGHT,
        help="Cap for balanced class weights to avoid instability from ultra-rare labels.",
    )
    parser.add_argument(
        "--class-weight-alpha",
        type=float,
        default=DEFAULT_CLASS_WEIGHT_ALPHA,
        help="Interpolation toward balanced weights: 0=uniform, 1=fully balanced.",
    )
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    X_train, y_train, X_val, y_val, X_test, y_test, label_map = load_dataset(args.dataset)
    n_features = X_train.shape[1]
    n_classes = len(label_map)

    # Match the MLP baseline preprocessing: fit scaler on train split only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    X_train_seq = as_sequence(X_train_scaled)
    X_val_seq = as_sequence(X_val_scaled)
    X_test_seq = as_sequence(X_test_scaled)

    cw = class_weights(
        y_train,
        alpha=float(args.class_weight_alpha),
        max_weight=None if args.max_class_weight is None else float(args.max_class_weight),
    )
    model = build_model(n_features=n_features, n_classes=n_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=args.patience,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, args.patience // 2),
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    warmup_history = None
    if args.warmup_epochs > 0:
        warmup_history = model.fit(
            X_train_seq,
            y_train,
            validation_data=(X_val_seq, y_val),
            epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            verbose=2,
        )

    history = model.fit(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=cw,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test_seq, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test_seq, verbose=0), axis=1)
    details = per_class_metrics(y_test, y_pred, label_map)
    warmup_best_val = (
        max(warmup_history.history["val_accuracy"]) if warmup_history is not None else 0.0
    )
    weighted_best_val = max(history.history["val_accuracy"])
    class_weight_mode = (
        "balanced"
        if float(args.class_weight_alpha) == 1.0 and args.max_class_weight is None
        else "balanced_smoothed"
    )

    model_path = MODELS_DIR / MODEL_FILENAME_TEMPLATE.format(dataset=args.dataset)
    scaler_path = MODELS_DIR / SCALER_FILENAME_TEMPLATE.format(dataset=args.dataset)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    metrics = {
        "dataset": args.dataset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "warmup_epochs_ran": int(0 if warmup_history is None else len(warmup_history.history["loss"])),
        "weighted_epochs_ran": int(len(history.history["loss"])),
        "best_val_accuracy": float(max(warmup_best_val, weighted_best_val)),
        "best_val_accuracy_warmup": float(warmup_best_val),
        "best_val_accuracy_weighted": float(weighted_best_val),
        "class_weight_mode": class_weight_mode,
        "class_weight_alpha": float(args.class_weight_alpha),
        "max_class_weight": float(args.max_class_weight) if args.max_class_weight is not None else None,
        "batch_size": int(args.batch_size),
        "epochs_requested": int(args.epochs),
        "warmup_epochs_requested": int(args.warmup_epochs),
        "patience": int(args.patience),
        **details,
    }

    metrics_path = RESULTS_DIR / METRICS_FILENAME_TEMPLATE.format(dataset=args.dataset)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved scaler: {scaler_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Test accuracy: {test_acc:.4f}")

    if test_acc < 0.90:
        print(
            "WARNING: Test accuracy is below 0.90. "
            "Consider tuning class weights and early stopping patience."
        )


if __name__ == "__main__":
    main()
