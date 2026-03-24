"""
Retrain baseline models (RF, XGBoost, MLP + majority-voting ensemble)
on adversarially mixed training data to build evasion resistance.
"""
import json
import argparse
from pathlib import Path
from scipy.stats import mode

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

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Directory Setup
SPLITS_DIR = Path("data/splits")
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


def load_adversarial_mixed_dataset(dataset_name, split_stats=None):
    """
    Loads the 70% clean / 30% perturbed training mix alongside the clean validation set.
    """
    mixed_path = SPLITS_DIR / f"adversarial_mixed_{dataset_name}.npz"
    base_path = SPLITS_DIR / f"{dataset_name}.npz"

    if not mixed_path.exists() or not base_path.exists():
        raise FileNotFoundError(f"Missing required .npz splits for {dataset_name}. Check your data/splits folder.")

    # Load data
    mixed_data = np.load(mixed_path, allow_pickle=True)
    base_data = np.load(base_path, allow_pickle=True)

    label_map = get_label_map(dataset_name, split_stats)
    
    return mixed_data["X_train"], mixed_data["y_train"], base_data["X_val"], base_data["y_val"], label_map


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
    
    X_train, y_train, X_val, y_val, label_map = load_adversarial_mixed_dataset(dataset_name, split_stats)
    print(f"Data loaded -> Train: {X_train.shape[0]:,} samples | Val: {X_val.shape[0]:,} samples | Classes: {len(label_map)}")

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

    # 4. Ensemble Voting
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
        out_path = RESULTS_DIR / f"adversarial_metrics_{dataset_name}.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved individual metrics to {out_path}")
        
        combined_metrics[dataset_name] = metrics

    # Save master metrics file if running all datasets
    if args.all:
        summary_path = RESULTS_DIR / "adversarial_metrics.json"
        with open(summary_path, "w") as f:
            json.dump(combined_metrics, f, indent=2)
        print(f"\nCombined master metrics saved to {summary_path}")


if __name__ == "__main__":
    main()