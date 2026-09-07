"""
tree_classifier.py - Tree Ensemble Classifiers (Random Forest & XGBoost).

Implements non-linear tree ensembles to capture high-order feature interactions
between n-grams and handcrafted metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.features.feature_pipeline import UnifiedFeaturePipeline, build_feature_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tree_classifier")


from config.config import RF_N_ESTIMATORS, TFIDF_MAX_FEATURES, RANDOM_SEED

class EnsembleClassifier:
    """
    Wrapper for tree-based ensemble models (Random Forest and XGBoost)
    with consistent scikit-learn API endpoints.

    Parameters:
        model_type: 'random_forest' or 'xgboost'
        n_estimators: Number of trees in forest / boosting stages
        max_depth: Maximum tree depth
        random_state: Reproducibility seed
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = RF_N_ESTIMATORS,
        max_depth: Optional[int] = 12,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.model_type = model_type.lower()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth if self.max_depth is not None else 6,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unsupported model_type: '{model_type}'. Choose 'random_forest' or 'xgboost'.")

        self.is_fitted_ = False

    def fit(self, X: Any, y: Any) -> "EnsembleClassifier":
        """Fit the ensemble classifier on training data."""
        self.model.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict binary class labels (1 for spam, 0 for ham)."""
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities.
        Returns shape (n_samples, 2).
        """
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> np.ndarray:
        """Return Gini / split importance scores for features."""
        return self.model.feature_importances_

    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        """Evaluate performance metrics."""
        preds = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1_score": float(f1_score(y, preds, zero_division=0)),
        }


def build_ensemble_pipeline(
    model_type: str = "random_forest",
    n_estimators: int = RF_N_ESTIMATORS,
    max_depth: Optional[int] = 12,
    max_features: int = TFIDF_MAX_FEATURES,
) -> Pipeline:
    """
    Construct an end-to-end Pipeline combining UnifiedFeaturePipeline with
    a Random Forest or XGBoost estimator.
    """
    if model_type.lower() == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type.lower() == "xgboost":
        estimator = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth is not None else 6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    return Pipeline([
        ("features", build_feature_pipeline(max_features=max_features)),
        ("ensemble", estimator),
    ])


def main() -> None:
    """Self-check and training demo on processed data splits."""
    import os

    train_file = "data/processed/train.csv"
    val_file = "data/processed/val.csv"

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"Error: Datasets not found at {train_file}")
        return

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    for model_name in ["random_forest", "xgboost"]:
        print(f"\n--- TRAINING {model_name.upper()} ---")
        pipeline = build_ensemble_pipeline(model_type=model_name, n_estimators=50, max_features=1000)
        pipeline.fit(train_df["text"], train_df["label"])

        val_preds = pipeline.predict(val_df["text"])
        val_probs = pipeline.predict_proba(val_df["text"])[:, 1]

        acc = accuracy_score(val_df["label"], val_preds)
        prec = precision_score(val_df["label"], val_preds, zero_division=0)
        rec = recall_score(val_df["label"], val_preds, zero_division=0)
        f1 = f1_score(val_df["label"], val_preds, zero_division=0)

        print(f"Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        # Top feature importances
        feature_names = pipeline.named_steps["features"].get_feature_names_out()
        importances = pipeline.named_steps["ensemble"].feature_importances_
        top_indices = np.argsort(importances)[-5:][::-1]
        print("Top 5 Important Features:")
        for idx in top_indices:
            print(f"  * {feature_names[idx]}: {importances[idx]:.4f}")

    print("\nTree Ensemble verification completed successfully!")


if __name__ == "__main__":
    main()
