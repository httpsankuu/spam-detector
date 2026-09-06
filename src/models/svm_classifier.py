"""
svm_classifier.py - Linear Support Vector Machine (LinearSVM) Module.

Maximum-margin separator optimized for high-dimensional TF-IDF and handcrafted text spaces.
Calibrated via Platt scaling (CalibratedClassifierCV) to provide true probability scores.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.features.feature_pipeline import UnifiedFeaturePipeline, build_feature_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("svm_classifier")


class SVMClassifier:
    """
    Linear Support Vector Machine classifier wrapper with probability calibration.

    Parameters:
        C: Regularization parameter
        calibrate_probabilities: If True, wraps LinearSVC in CalibratedClassifierCV
        class_weight: Weighting strategy ('balanced' or None)
        max_iter: Maximum optimization iterations
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        C: float = 1.0,
        calibrate_probabilities: bool = True,
        class_weight: Optional[Union[str, Dict[int, float]]] = "balanced",
        max_iter: int = 2000,
        random_state: int = 42,
    ) -> None:
        self.C = C
        self.calibrate_probabilities = calibrate_probabilities
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

        self.base_svm_ = LinearSVC(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

        if self.calibrate_probabilities:
            # CalibratedClassifierCV with cv="warn" or cv=2 for small datasets
            self.model = CalibratedClassifierCV(estimator=self.base_svm_, cv=2)
        else:
            self.model = self.base_svm_

        self.is_fitted_ = False

    def fit(self, X: Any, y: Any) -> "SVMClassifier":
        """Fit Linear SVM model on feature matrix X."""
        self.model.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict binary class labels (1 for spam, 0 for ham)."""
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict calibrated class probabilities.
        Returns shape (n_samples, 2).
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            # Fallback if uncalibrated: sigmoid over decision function
            df = self.model.decision_function(X)
            prob_spam = 1 / (1 + np.exp(-df))
            prob_ham = 1 - prob_spam
            return np.column_stack([prob_ham, prob_spam])

    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        """Evaluate model metrics on a test/validation split."""
        preds = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1_score": float(f1_score(y, preds, zero_division=0)),
        }


def build_svm_pipeline(
    C: float = 1.0,
    calibrate_probabilities: bool = True,
    max_features: int = 5000,
) -> Pipeline:
    """
    Construct a complete end-to-end scikit-learn Pipeline bundling
    UnifiedFeaturePipeline and Calibrated LinearSVC.
    """
    base_svc = LinearSVC(C=C, class_weight="balanced", max_iter=2000, random_state=42)
    estimator = CalibratedClassifierCV(estimator=base_svc, cv=2) if calibrate_probabilities else base_svc

    return Pipeline([
        ("features", build_feature_pipeline(max_features=max_features)),
        ("svm", estimator),
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

    print("--- TRAINING LINEAR SVM (WITH CALIBRATION) ---")
    pipeline = build_svm_pipeline(C=1.0, max_features=1000)
    pipeline.fit(train_df["text"], train_df["label"])

    val_preds = pipeline.predict(val_df["text"])
    val_probs = pipeline.predict_proba(val_df["text"])[:, 1]

    acc = accuracy_score(val_df["label"], val_preds)
    prec = precision_score(val_df["label"], val_preds, zero_division=0)
    rec = recall_score(val_df["label"], val_preds, zero_division=0)
    f1 = f1_score(val_df["label"], val_preds, zero_division=0)

    print("\n--- VALIDATION METRICS ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Test single-sentence live inference
    sample_text = "Final Notice: Your credit card account has been compromised. Verify immediately!"
    prob = pipeline.predict_proba([sample_text])[0, 1]
    label = "SPAM" if prob >= 0.5 else "HAM"
    print(f"\nLive Test:")
    print(f"Input: '{sample_text}'")
    print(f"Verdict: {label} (Spam Probability: {prob:.4f})")
    print("\nLinear SVM verification completed successfully!")


if __name__ == "__main__":
    main()
