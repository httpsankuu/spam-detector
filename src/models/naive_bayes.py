"""
naive_bayes.py - Multinomial Naive Bayes Classifier Module.

Baseline probabilistic spam classifier implementing Laplace smoothing (alpha tuning)
and outputting class posterior probabilities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.features.feature_pipeline import UnifiedFeaturePipeline, build_feature_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("naive_bayes")


from config.config import NB_ALPHA, TFIDF_MAX_FEATURES

class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes classifier wrapper with Laplace smoothing and evaluation utilities.

    Parameters:
        alpha: Additive (Laplace/Lidstone) smoothing parameter (1.0 = standard Laplace)
        fit_prior: Whether to learn class prior probabilities from data
    """

    def __init__(self, alpha: float = NB_ALPHA, fit_prior: bool = True) -> None:
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.model = MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior)
        self.is_fitted_ = False

    def fit(self, X: Any, y: Any) -> "NaiveBayesClassifier":
        """
        Fit the Multinomial Naive Bayes model.

        Args:
            X: Feature matrix (e.g. CSR sparse matrix from UnifiedFeaturePipeline)
            y: Binary target vector (1=spam, 0=ham)
        """
        self.model.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict binary class labels (1 for spam, 0 for ham)."""
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probability distributions.
        Returns shape (n_samples, 2) where column 1 is spam probability.
        """
        return self.model.predict_proba(X)

    def get_feature_log_prob(self) -> np.ndarray:
        """Return empirical log probability of features given a class, P(x_i|y)."""
        return self.model.feature_log_prob_

    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        """
        Evaluate classifier performance on a validation/test dataset.
        """
        preds = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1_score": float(f1_score(y, preds, zero_division=0)),
        }


def build_nb_pipeline(alpha: float = NB_ALPHA, max_features: int = TFIDF_MAX_FEATURES) -> Pipeline:
    """
    Construct a complete end-to-end scikit-learn Pipeline bundling
    UnifiedFeaturePipeline and MultinomialNB.
    """
    return Pipeline([
        ("features", build_feature_pipeline(max_features=max_features)),
        ("nb", MultinomialNB(alpha=alpha)),
    ])


def main() -> None:
    """Self-check and training demo on processed data splits."""
    import os

    train_file = "data/processed/train.csv"
    val_file = "data/processed/val.csv"

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"Error: Required datasets not found at {train_file} or {val_file}")
        return

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    print(f"--- TRAINING MULTINOMIAL NAIVE BAYES ---")
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    pipeline = build_nb_pipeline(alpha=1.0, max_features=1000)
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
    sample_text = "CONGRATULATIONS! You have won a $1,000 gift card! Claim now at http://gift.com"
    prob = pipeline.predict_proba([sample_text])[0, 1]
    label = "SPAM" if prob >= 0.5 else "HAM"
    print(f"\nLive Test:")
    print(f"Input: '{sample_text}'")
    print(f"Verdict: {label} (Spam Probability: {prob:.4f})")
    print("\nMultinomial Naive Bayes verification completed successfully!")


if __name__ == "__main__":
    main()
