"""
Train baseline models (RF, XGBoost, MLP) on specified dataset.
Evaluate on validation split during development; log metrics; save models.
"""
import json
import argparse
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import xgboost as xgb

# Keras/TF for MLP
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Paths
SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(dataset_name):
    """Load specified dataset train/val and label map."""
    data = np.load(SPLITS_DIR / f"{dataset_name}.npz", allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    
    # Using Validation set for development tuning per data dictionary
    X_val = data["X_val"]
    y_val = data["y_val"]
    
    label_map = np.load(
        SPLITS_DIR / f"{dataset_name}_label_map.npy", allow_pickle=True
    ).item()
    
    return X_train, y_train, X_val, y_val, label_map


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
    """Random Forest n_estimators=100."""
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val, y_pred, label_map)
    return model, metrics


def train_xgb(X_train, y_train, X_val, y_val, label_map):
    """XGBoost with default params."""
    model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val, y_pred, label_map)
    return model, metrics


def train_mlp(X_train, y_train, X_val, y_val, label_map):
    """3-layer MLP: 128-64-32, ReLU, Adam. Optimized for stability."""
    n_classes = len(label_map)
    n_features = X_train.shape[1]

    model = Sequential([
        Dense(128, activation="relu", input_shape=(n_features,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(
        monitor='val_accuracy', 
        patience=3, 
        restore_best_weights=True,
        verbose=1
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=256,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1,
    )

    y_pred_proba = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    metrics = compute_metrics(y_val, y_pred, label_map)
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train baseline NIDS models.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=["cicids2017", "nslkdd", "unswnb15"],
        help="The name of the dataset to train on."
    )
    args = parser.parse_args()
    dataset = args.dataset

    print(f"Loading {dataset.upper()} splits...")
    X_train, y_train, X_val, y_val, label_map = load_dataset(dataset)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Classes: {len(label_map)}")

    all_metrics = {}

    # --- Random Forest ---
    print(f"\n--- Training Random Forest on {dataset.upper()} ---")
    rf_model, rf_metrics = train_rf(X_train, y_train, X_val, y_val, label_map)
    all_metrics["random_forest"] = rf_metrics
    print(f"  Validation accuracy: {rf_metrics['accuracy']:.4f}")
    joblib.dump(rf_model, MODELS_DIR / f"rf_{dataset}.pkl")
    print(f"  Saved {MODELS_DIR / f'rf_{dataset}.pkl'}")

    # --- XGBoost ---
    print(f"\n--- Training XGBoost on {dataset.upper()} ---")
    xgb_model, xgb_metrics = train_xgb(X_train, y_train, X_val, y_val, label_map)
    all_metrics["xgboost"] = xgb_metrics
    print(f"  Validation accuracy: {xgb_metrics['accuracy']:.4f}")
    joblib.dump(xgb_model, MODELS_DIR / f"xgb_{dataset}.pkl")
    print(f"  Saved {MODELS_DIR / f'xgb_{dataset}.pkl'}")

    # --- MLP (scale features) ---
    print("\n--- Scaling features for MLP ---")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    joblib.dump(scaler, MODELS_DIR / f"scaler_{dataset}.pkl")

    print(f"\n--- Training MLP on {dataset.upper()} ---")
    mlp_model, mlp_metrics = train_mlp(
        X_train_s, y_train, X_val_s, y_val, label_map
    )
    all_metrics["mlp"] = mlp_metrics
    print(f"  Validation accuracy: {mlp_metrics['accuracy']:.4f}")
    mlp_model.save(MODELS_DIR / f"mlp_{dataset}.h5", save_format="h5")
    print(f"  Saved {MODELS_DIR / f'mlp_{dataset}.h5'}")

    # --- Save metrics ---
    out_path = RESULTS_DIR / f"baseline_metrics_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

    # Summary
    print("\n--- Summary (target >=90% accuracy) ---")
    for name, m in all_metrics.items():
        status = "OK" if m["accuracy"] >= 0.90 else "BELOW TARGET"
        print(f"  {name}: {m['accuracy']:.2%} {status}")


if __name__ == "__main__":
    main()