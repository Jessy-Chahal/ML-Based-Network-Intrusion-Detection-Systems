"""
RF + XGBoost + MLP: each predicts a class; the final label is the majority vote.

Ways to load models (all use predict(X) the same way):
  baseline_for(dataset)     — normal training
  adversarial_for(dataset) — trained with all adversarial families
  partial_for(dataset, "a"|"b"|"c") — trained on one family only

Files live under models/ with prefixes rf_, xgb_, mlp_, scaler_ (baseline),
adv_* (full adversarial), or adv_{a|b|c}_only_* (partial).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from scipy import stats

MODELS_DIR = Path("models")


class Ensemble:
    def __init__(self, rf, xgb, mlp, mlp_scaler, name="ensemble"):
        """rf / xgb / mlp are the three voters; mlp_scaler is for the MLP input only."""
        self.rf = rf
        self.xgb = xgb
        self.mlp = mlp
        self.mlp_scaler = mlp_scaler
        self.name = name

    @classmethod
    def baseline(cls, models_dir: Path = MODELS_DIR) -> "Ensemble":
        """Shortcut: baseline CICIDS2017 only."""
        return cls.baseline_for("cicids2017", models_dir)

    @classmethod
    def adversarial(cls, models_dir: Path = MODELS_DIR) -> "Ensemble":
        """Shortcut: adversarial CICIDS2017 only."""
        return cls.adversarial_for("cicids2017", models_dir)

    @classmethod
    def baseline_for(cls, dataset: str, models_dir: Path = MODELS_DIR) -> "Ensemble":
        """Load rf / xgb / mlp / scaler for one dataset name."""
        rf = joblib.load(models_dir / f"rf_{dataset}.pkl")
        xgb = joblib.load(models_dir / f"xgb_{dataset}.pkl")
        mlp_scaler = joblib.load(models_dir / f"scaler_{dataset}.pkl")
        mlp = tf.keras.models.load_model(
            models_dir / f"mlp_{dataset}.h5", compile=False
        )

        return cls(
            rf=rf, xgb=xgb, mlp=mlp, mlp_scaler=mlp_scaler,
            name=f"baseline_{dataset}",
        )

    @classmethod
    def adversarial_for(cls, dataset: str, models_dir: Path = MODELS_DIR) -> "Ensemble":
        """Same as baseline_for but adv_* filenames (all attack families in training)."""
        rf = joblib.load(models_dir / f"adv_rf_{dataset}.pkl")
        xgb = joblib.load(models_dir / f"adv_xgb_{dataset}.pkl")
        mlp_scaler = joblib.load(models_dir / f"adv_scaler_{dataset}.pkl")
        mlp = tf.keras.models.load_model(
            models_dir / f"adv_mlp_{dataset}.h5", compile=False
        )

        return cls(
            rf=rf, xgb=xgb, mlp=mlp, mlp_scaler=mlp_scaler,
            name=f"adversarial_{dataset}",
        )

    @classmethod
    def partial_for(
        cls, dataset: str, attack: str, models_dir: Path = MODELS_DIR
    ) -> "Ensemble":
        """dataset = cicids2017 | nslkdd | unswnb15; attack = a | b | c.

        Expects adv_{attack}_only_* model files from your partial train pipeline.
        """
        if attack not in ("a", "b", "c"):
            raise ValueError(f"attack must be 'a', 'b', or 'c', however got input {attack!r}")

        prefix = f"adv_{attack}_only"
        rf = joblib.load(models_dir / f"{prefix}_rf_{dataset}.pkl")
        xgb = joblib.load(models_dir / f"{prefix}_xgb_{dataset}.pkl")
        mlp_scaler = joblib.load(models_dir / f"{prefix}_scaler_{dataset}.pkl")
        mlp = tf.keras.models.load_model(
            models_dir / f"{prefix}_mlp_{dataset}.h5", compile=False
        )

        return cls(
            rf=rf, xgb=xgb, mlp=mlp, mlp_scaler=mlp_scaler,
            name=f"partial_attack_{attack}_{dataset}",
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X shape (n_samples, n_features); returns one label id per row."""
        X = np.array(X, dtype=np.float64)

        votes = self._get_votes(X)

        # Shape (3, n_samples); pick the most common label per column
        vote_matrix = np.stack(votes, axis=0)
        majority, _ = stats.mode(vote_matrix, axis=0, keepdims=False)
        return majority.astype(np.int64).flatten()

    def _get_votes(self, X: np.ndarray) -> list[np.ndarray]:
        """One integer prediction vector per model."""
        rf_preds = self.rf.predict(X).astype(np.int64)
        xgb_preds = self.xgb.predict(X).astype(np.int64)

        X_scaled_mlp = self.mlp_scaler.transform(X).astype(np.float32)
        mlp_preds = np.argmax(self.mlp.predict(X_scaled_mlp, verbose=0), axis=1).astype(np.int64)

        voters = [rf_preds, xgb_preds, mlp_preds]

        return voters

    def __repr__(self) -> str:
        voters = "RF + XGBoost + MLP"
        return f"Ensemble(name={self.name!r}, voters={voters})"