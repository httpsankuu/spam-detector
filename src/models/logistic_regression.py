"""
logistic_regression.py - Logistic Regression Classifier Module.

Regularized linear classifier with calibrated probabilities and transparent
log-odds coefficients for spam vs. ham explainability.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.features.feature_pipeline import UnifiedFeaturePipeline, build_feature_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("logistic_regression")


from config.config import LR_C, LR_MAX_ITER, LR_CLASS_WEIGHT, TFIDF_MAX_FEATURES, RANDOM_SEED

class LogisticRegressionClassifier:
    """
    Scikit-learn Logistic Regression wrapper configured for text classification.

    Parameters:
        C: Inverse regularization strength (smaller values specify stronger regularization)
        max_iter: Maximum iterations for solvers to converge
        class_weight: Strategy for handling class imbalance ('balanced' or None)
        solver: Optimization algorithm ('lbfgs', 'liblinear')
        random_state: Seed for reproducibility
    """

    def __init__(
        self,
        C: float = LR_C,
        max_iter: int = LR_MAX_ITER,
        class_weight: Optional[Union[str, Dict[int, float]]] = LR_CLASS_WEIGHT,
        solver: str = "lbfgs",
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.solver = solver
        self.random_state = random_state

        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            solver=self.solver,
            random_state=self.random_state,
        )
        self.is_fitted_ = False

    def fit(self, X: Any, y: Any) -> "LogisticRegressionClassifier":
        """
        Fit Logistic Regression on feature matrix X against binary labels y.
        """
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
        return self.model.predict_proba(X)

    def decision_function(self, X: Any) -> np.ndarray:
        """Predict raw log-odds decision margin."""
        return self.model.decision_function(X)

    def get_coefficients(self) -> np.ndarray:
        """
        Return the 1D array of learned coefficients (weights) corresponding to each feature.
        Positive values indicate spam; negative values indicate legitimate ham.
        """
        return self.model.coef_.flatten()

    def get_top_features(self, feature_names: Sequence[str], top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract the top most positive (spam-indicative) and negative (ham-indicative) features.
        """
        coefs = self.get_coefficients()
        sorted_indices = np.argsort(coefs)

        top_ham_indices = sorted_indices[:top_n]
        top_spam_indices = sorted_indices[-top_n:][::-1]

        return {
            "spam_indicative": [(feature_names[i], float(coefs[i])) for i in top_spam_indices],
            "ham_indicative": [(feature_names[i], float(coefs[i])) for i in top_ham_indices],
        }

    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        """Evaluate model performance on a test/validation split."""
        preds = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1_score": float(f1_score(y, preds, zero_division=0)),
        }


def build_lr_pipeline(
    C: float = LR_C,
    class_weight: Optional[str] = LR_CLASS_WEIGHT,
    max_features: int = TFIDF_MAX_FEATURES,
) -> Pipeline:
    """
    Construct a complete scikit-learn Pipeline bundling UnifiedFeaturePipeline and LogisticRegression.
    """
    return Pipeline([
        ("features", build_feature_pipeline(max_features=max_features)),
        ("lr", LogisticRegression(C=C, class_weight=class_weight, max_iter=LR_MAX_ITER, random_state=RANDOM_SEED)),
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

    print(f"--- TRAINING LOGISTIC REGRESSION ---")
    pipeline = build_lr_pipeline(C=1.0, max_features=1000)
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

    # Inspect learned feature coefficients
    feature_names = pipeline.named_steps["features"].get_feature_names_out()
    coefs = pipeline.named_steps["lr"].coef_.flatten()

    top_spam_idx = np.argsort(coefs)[-5:][::-1]
    top_ham_idx = np.argsort(coefs)[:5]

    print("\n--- TOP LEARNED COEFFICIENTS ---")
    print("Spam-indicative features:")
    for idx in top_spam_idx:
        print(f"  + {feature_names[idx]}: {coefs[idx]:.4f}")

    print("Ham-indicative features:")
    for idx in top_ham_idx:
        print(f"  - {feature_names[idx]}: {coefs[idx]:.4f}")

    # Test single sentence live inference
    sample_text = "Hey team, the project sync notes are attached. Please check them before 4pm."
    prob = pipeline.predict_proba([sample_text])[0, 1]
    label = "SPAM" if prob >= 0.5 else "HAM"
    print(f"\nLive Test:")
    print(f"Input: '{sample_text}'")
    print(f"Verdict: {label} (Spam Probability: {prob:.4f})")
    print("\nLogistic Regression verification completed successfully!")


if __name__ == "__main__":
    main()
