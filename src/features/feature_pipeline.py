"""
feature_pipeline.py - Unified Composite Feature Pipeline.

Merges text NLP preprocessing, TF-IDF n-gram extraction, and scaled handcrafted
statistical features into a unified sparse feature matrix for model training & inference.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MaxAbsScaler

from src.features.handcrafted_features import HandcraftedFeatureExtractor
from src.features.tfidf_features import TFIDFExtractor
from src.preprocessing.transformer import TextPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("feature_pipeline")


class UnifiedFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Composite feature extraction transformer that harmonizes:
      1. Text preprocessing (cleaning, tokenization, lemmatization) -> TF-IDF word n-grams
      2. Handcrafted statistical spam indicator extraction -> Feature scaling (MaxAbsScaler)
    and stacks them into a single unified scipy.sparse.csr_matrix.

    Exposes full feature name mapping via `get_feature_names_out()` for explainability.
    """

    def __init__(
        self,
        max_features: Optional[int] = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: Union[int, float] = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        scale_handcrafted: bool = True,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.scale_handcrafted = scale_handcrafted

        # Initialize sub-transformers
        self.preprocessor = TextPreprocessor()
        self.tfidf = TFIDFExtractor(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
        )
        self.handcrafted = HandcraftedFeatureExtractor()
        # MaxAbsScaler preserves zero values and scales non-negative metrics into [0, 1] without centering
        self.scaler = MaxAbsScaler() if self.scale_handcrafted else None

    def _normalize_input(self, X: Union[pd.Series, Sequence[str], np.ndarray, str]) -> List[str]:
        """Convert input variations into a clean list of strings."""
        if isinstance(X, str):
            return [X]
        elif isinstance(X, pd.Series):
            return X.fillna("").astype(str).tolist()
        elif isinstance(X, np.ndarray):
            return [str(x) for x in X.flatten()]
        elif isinstance(X, (list, tuple)):
            return [str(x) if x is not None else "" for x in X]
        else:
            return [str(X)]

    def fit(self, X: Any, y: Any = None) -> "UnifiedFeaturePipeline":
        """
        Fit both TF-IDF vocabulary and handcrafted feature scalers on training texts.

        Args:
            X: Raw texts (Series, array, or list of strings)
            y: Ignored

        Returns:
            self
        """
        raw_texts = self._normalize_input(X)
        logger.info(f"Fitting UnifiedFeaturePipeline on {len(raw_texts)} texts...")

        # 1. Clean & lemmatize for TF-IDF
        cleaned_texts = self.preprocessor.transform(raw_texts)
        self.tfidf.fit(cleaned_texts)

        # 2. Extract and fit scaler for handcrafted features (on raw texts to capture caps/raw punctuation)
        handcrafted_matrix = self.handcrafted.transform(raw_texts)
        if self.scaler is not None:
            self.scaler.fit(handcrafted_matrix)

        logger.info(
            f"Fitted successfully! TF-IDF vocabulary size: {self.tfidf.vocabulary_size}, "
            f"Handcrafted features: {len(self.handcrafted.get_feature_names())}"
        )
        return self

    def transform(self, X: Any) -> sp.csr_matrix:
        """
        Transform texts into unified composite sparse matrix.

        Args:
            X: Raw texts

        Returns:
            sp.csr_matrix of shape (n_samples, n_tfidf + n_handcrafted)
        """
        raw_texts = self._normalize_input(X)

        # 1. Text branch: Clean -> TF-IDF sparse matrix
        cleaned_texts = self.preprocessor.transform(raw_texts)
        tfidf_sparse = self.tfidf.transform(cleaned_texts)

        # 2. Handcrafted branch: Extract -> Scale -> Convert to sparse
        handcrafted_dense = self.handcrafted.transform(raw_texts)
        if self.scaler is not None:
            handcrafted_dense = self.scaler.transform(handcrafted_dense)
        handcrafted_sparse = sp.csr_matrix(handcrafted_dense)

        # 3. Stack horizontally
        composite_matrix = sp.hstack([tfidf_sparse, handcrafted_sparse], format="csr")
        return composite_matrix

    def fit_transform(self, X: Any, y: Any = None) -> sp.csr_matrix:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """
        Return the unified list of feature names corresponding to composite matrix columns.
        """
        tfidf_names = self.tfidf.get_feature_names()
        handcrafted_names = [f"hc_{name}" for name in self.handcrafted.get_feature_names()]
        return np.array(tfidf_names + handcrafted_names, dtype=object)


def build_feature_pipeline(
    max_features: Optional[int] = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: Union[int, float] = 2,
    scale_handcrafted: bool = True,
) -> UnifiedFeaturePipeline:
    """
    Convenience factory to build a configured UnifiedFeaturePipeline.
    """
    return UnifiedFeaturePipeline(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        scale_handcrafted=scale_handcrafted,
    )


def main() -> None:
    """Self-check and verification using the processed training split."""
    import os

    train_path = "data/processed/train.csv"
    if os.path.exists(train_path):
        print(f"--- TESTING ON SAVED DATASET: {train_path} ---")
        df = pd.read_csv(train_path)
        texts = df["text"].tolist()
    else:
        print("--- TESTING ON INLINE SAMPLES ---")
        texts = [
            "URGENT! You have won $1,000,000! Claim your reward at http://claim-prize.com immediately!",
            "Hi Ankit, please check the meeting notes from today's engineering sync.",
            "Exclusive bonus voucher code: FREE50. Redeem today only!",
        ]

    pipeline = build_feature_pipeline(max_features=500, ngram_range=(1, 2), min_df=1)
    matrix = pipeline.fit_transform(texts)

    feature_names = pipeline.get_feature_names_out()
    print(f"\nTransformed matrix shape: {matrix.shape}")
    print(f"Total feature names count: {len(feature_names)}")
    assert matrix.shape[1] == len(feature_names), "Mismatch between matrix columns and feature names!"

    # Show handcrafted features at the tail
    print(f"Sample tail feature names (handcrafted): {list(feature_names[-10:])}")
    print("\nUnifiedFeaturePipeline verification passed successfully!")


if __name__ == "__main__":
    main()
