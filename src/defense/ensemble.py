"""
Majority-vote ensemble defense.

Loads a set of classifiers and combines their predictions via majority vote.
Each voter gets one vote per sample - the class that appears most often wins.

Two named constructors are provided:
    Ensemble.baseline()    - loads the  baseline models
    Ensemble.adversarial() - loads adversarially retrained models

Both expose the same predict(X) interface so evaluation scripts need no
changes when switching between them.

Model files expected in models/:
    Baseline:
        rf_cicids2017.pkl
        xgb_cicids2017.pkl
        mlp_cicids2017.h5
        scaler_cicids2017.pkl

    Adversarial (needed from adversarial retraining):
        adv_rf_cicids2017.pkl
        adv_xgb_cicids2017.pkl
        adv_mlp_cicids2017.h5
        scaler_cicids2017.pkl       (same scaler - retraining doesn't change it)
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
        """
        Args:
            rf:           Trained RandomForestClassifier.
            xgb:          Trained XGBoost classifier.
            mlp:          Trained Keras MLP model.
            mlp_scaler:   Scaler fitted for the MLP (also used by RF/XGB since
                          those don't need scaling, but kept here for consistency).
            name:         Label for this ensemble, used in __repr__.
        """
        self.rf = rf
        self.xgb = xgb
        self.mlp = mlp
        self.mlp_scaler = mlp_scaler
        self.name = name

    ### Named constructors ###
    @classmethod
    def baseline(cls, models_dir: Path = MODELS_DIR) -> "Ensemble":
        """Load the standard Sprint 2 CICIDS2017 baseline models."""
        return cls.baseline_for("cicids2017", models_dir)

    @classmethod
    def adversarial(cls, models_dir: Path = MODELS_DIR) -> "Ensemble":
        """Load adversarially retrained CICIDS2017 models."""
        return cls.adversarial_for("cicids2017", models_dir)

    @classmethod
    def baseline_for(cls, dataset: str, models_dir: Path = MODELS_DIR) -> "Ensemble":
        """
        Load baseline models for the specified dataset.

        """
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
        """
        Load adversarially retrained models for the specified dataset.
        """
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

    ### Prediction ###
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return majority-vote predictions for each sample in X.

        Args:
            X: 2D array of shape (n_samples, n_features).

        Returns:
            1D array of predicted label IDs, shape (n_samples,).
        """
        X = np.array(X, dtype=np.float64)

        # Get one prediction array per voter, shape (n_samples,) each
        votes = self._get_votes(X)

        # Stack into (n_voters, n_samples), then take the mode across voters
        vote_matrix = np.stack(votes, axis=0)
        majority, _ = stats.mode(vote_matrix, axis=0, keepdims=False)
        return majority.astype(np.int64).flatten()

    def _get_votes(self, X: np.ndarray) -> list[np.ndarray]:
        """Collect one prediction array from each active voter."""
        # RF and XGBoost predict directly from raw features
        rf_preds = self.rf.predict(X).astype(np.int64)
        xgb_preds = self.xgb.predict(X).astype(np.int64)

        # MLP needs scaled input
        X_scaled_mlp = self.mlp_scaler.transform(X).astype(np.float32)
        mlp_preds = np.argmax(self.mlp.predict(X_scaled_mlp, verbose=0), axis=1).astype(np.int64)

        voters = [rf_preds, xgb_preds, mlp_preds]

        return voters

    def __repr__(self) -> str:
        voters = "RF + XGBoost + MLP"
        return f"Ensemble(name={self.name!r}, voters={voters})"