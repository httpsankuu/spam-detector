"""
tfidf_features.py - TF-IDF Word N-Gram Feature Extractor.

Extracts unigrams and bigrams using TF-IDF weighting with configurable
vocabulary caps, frequency cutoffs, and sublinear scaling.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tfidf_features")


class TFIDFExtractor(BaseEstimator, TransformerMixin):
    """
    TF-IDF Vectorizer wrapper implementing scikit-learn's Transformer API.

    Parameters:
        max_features: Maximum vocabulary size (None for unlimited)
        ngram_range: (min_n, max_n) tuple, defaults to unigram + bigram (1, 2)
        min_df: Minimum document frequency (integer or float ratio)
        max_df: Maximum document frequency ratio
        sublinear_tf: If True, apply sublinear scaling: 1 + log(tf)
        use_idf: If True, enable inverse-document-frequency weighting
        norm: Normalization method ('l1', 'l2', or None)
    """

    def __init__(
        self,
        max_features: Optional[int] = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: Union[int, float] = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        use_idf: bool = True,
        norm: str = "l2",
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.use_idf = use_idf
        self.norm = norm

        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            use_idf=self.use_idf,
            norm=self.norm,
        )

    def fit(self, X: Iterable[str], y: Any = None) -> "TFIDFExtractor":
        """
        Fit the TF-IDF vectorizer on training text corpus.

        Args:
            X: Iterable of preprocessed string documents
            y: Ignored (present for scikit-learn API consistency)

        Returns:
            self
        """
        docs = list(X)
        n_docs = len(docs)
        # Prevent ValueError when dataset is tiny (e.g. unit tests or tiny slices)
        effective_min_df = self.min_df
        if isinstance(effective_min_df, int) and n_docs <= effective_min_df:
            effective_min_df = 1

        effective_max_df = self.max_df
        if n_docs <= 2 and isinstance(effective_max_df, float):
            effective_max_df = 1.0

        if effective_min_df != self.min_df or effective_max_df != self.max_df:
            self.vectorizer_ = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=effective_min_df,
                max_df=effective_max_df,
                sublinear_tf=self.sublinear_tf,
                use_idf=self.use_idf,
                norm=self.norm,
            )

        self.vectorizer_.fit(docs)
        return self

    def transform(self, X: Iterable[str]) -> sp.csr_matrix:
        """
        Transform documents into a sparse TF-IDF matrix.

        Args:
            X: Iterable of preprocessed string documents

        Returns:
            Scipy CSR sparse matrix of TF-IDF values
        """
        return self.vectorizer_.transform(X)

    def fit_transform(self, X: Iterable[str], y: Any = None) -> sp.csr_matrix:
        """
        Fit and transform documents in a single optimized pass.
        """
        docs = list(X)
        return self.fit(docs, y).transform(docs)

    def get_feature_names(self) -> List[str]:
        """
        Return the list of vocabulary feature names matching output matrix columns.
        """
        return list(self.vectorizer_.get_feature_names_out())

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """
        Standard scikit-learn feature names method.
        """
        return self.vectorizer_.get_feature_names_out(input_features)

    @property
    def vocabulary_size(self) -> int:
        """Return the number of learned features in the vocabulary."""
        return len(self.vectorizer_.vocabulary_) if hasattr(self.vectorizer_, "vocabulary_") else 0


def build_tfidf_vectorizer(
    max_features: Optional[int] = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: Union[int, float] = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> TFIDFExtractor:
    """
    Factory function to construct a configured TFIDFExtractor instance.
    """
    return TFIDFExtractor(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )


def main() -> None:
    """Self-check and demonstration of TFIDFExtractor."""
    sample_texts = [
        "claim your free prize now visit httpaddr dollar number",
        "urgent meeting scheduled for tomorrow project update",
        "win dollar cash reward today claim exclusive prize",
        "reminder push code review to github repository",
        "free gift card number dollar win free claim",
    ]

    print("--- FITTING TFIDF EXTRACTOR (UNIGRAM + BIGRAM) ---")
    extractor = TFIDFExtractor(max_features=30, ngram_range=(1, 2), min_df=1)
    matrix = extractor.fit_transform(sample_texts)

    print(f"Output sparse matrix shape: {matrix.shape}")
    print(f"Non-zero elements: {matrix.nnz}")

    feature_names = extractor.get_feature_names()
    print(f"\nVocabulary ({len(feature_names)} features):")
    print(feature_names[:15], "..." if len(feature_names) > 15 else "")

    # Inspect bigrams
    bigrams = [f for f in feature_names if " " in f]
    print(f"\nSample extracted bigrams ({len(bigrams)} found): {bigrams[:5]}")

    print("\nTFIDFExtractor self-test passed successfully!")


if __name__ == "__main__":
    main()
