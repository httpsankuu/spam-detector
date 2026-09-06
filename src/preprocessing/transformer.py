"""
transformer.py - Scikit-Learn Pipeline-Compatible Preprocessing Transformer.

Wraps raw text cleaning, tokenization, stopword removal, and lemmatization into a
standard scikit-learn transformer (BaseEstimator, TransformerMixin).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing.cleaner import clean_text
from src.preprocessing.tokenizer import lemmatized_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("transformer")


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that applies full NLP text cleaning and lemmatization.

    Parameters:
        remove_html: If True, remove HTML tags
        remove_headers: If True, strip email headers (Subject:, etc.)
        replace_urls: If True, map URLs to token 'httpaddr'
        replace_emails: If True, map emails to token 'emailaddr'
        replace_currency: If True, map currency signs to 'dollar'
        replace_numbers: If True, map standalone numbers to 'number'
        lowercase: If True, convert characters to lowercase
        remove_stopwords: If True, remove English stopwords
        min_word_length: Minimum token length to retain
    """

    def __init__(
        self,
        remove_html: bool = True,
        remove_headers: bool = True,
        replace_urls: bool = True,
        replace_emails: bool = True,
        replace_currency: bool = True,
        replace_numbers: bool = True,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        min_word_length: int = 2,
    ) -> None:
        self.remove_html = remove_html
        self.remove_headers = remove_headers
        self.replace_urls = replace_urls
        self.replace_emails = replace_emails
        self.replace_currency = replace_currency
        self.replace_numbers = replace_numbers
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length

    def fit(self, X: Any, y: Any = None) -> "TextPreprocessor":
        """Fit method (stateless; returns self)."""
        return self

    def preprocess_single(self, text: str) -> str:
        """Clean and lemmatize a single string."""
        cleaned = clean_text(
            text,
            remove_html=self.remove_html,
            remove_headers=self.remove_headers,
            replace_urls=self.replace_urls,
            replace_emails=self.replace_emails,
            replace_currency=self.replace_currency,
            replace_numbers=self.replace_numbers,
            lowercase=self.lowercase,
        )
        lemmatized = lemmatized_text(
            cleaned,
            remove_stopwords=self.remove_stopwords,
            min_word_length=self.min_word_length,
        )
        return lemmatized

    def transform(self, X: Union[pd.Series, Sequence[str], np.ndarray]) -> List[str]:
        """
        Transform an iterable of raw texts into clean, lemmatized string documents.

        Args:
            X: Input Series, list, or array of strings

        Returns:
            List of processed strings ready for TF-IDF vectorization
        """
        if isinstance(X, pd.Series):
            texts = X.tolist()
        elif isinstance(X, np.ndarray):
            texts = X.flatten().tolist()
        elif isinstance(X, (list, tuple)):
            texts = list(X)
        else:
            texts = [str(X)]

        return [self.preprocess_single(t) for t in texts]


def main() -> None:
    """Self-check and demonstration of TextPreprocessor inside a scikit-learn Pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    raw_samples = [
        "Subject: WINNER! You won $50,000! Visit http://winner.org today to claim your prize.",
        "Hey team, don't forget to push your code to GitHub before the 5pm sprint deadline.",
        "URGENT: Your account balance is low. Click <b>http://mybank.com</b> immediately.",
    ]

    print("--- RAW SAMPLES ---")
    for i, s in enumerate(raw_samples):
        print(f"[{i}] {s}")

    preprocessor = TextPreprocessor()
    transformed = preprocessor.transform(raw_samples)

    print("\n--- TRANSFORMED SAMPLES ---")
    for i, t in enumerate(transformed):
        print(f"[{i}] {t}")

    print("\n--- TESTING SCIKIT-LEARN PIPELINE INTEGRATION ---")
    pipe = Pipeline([
        ("prep", TextPreprocessor()),
        ("tfidf", TfidfVectorizer(max_features=20)),
    ])
    tfidf_matrix = pipe.fit_transform(raw_samples)
    print(f"Pipeline executed successfully! Output matrix shape: {tfidf_matrix.shape}")
    vocab = pipe.named_steps["tfidf"].get_feature_names_out()
    print(f"Sample learned vocabulary ({len(vocab)} words): {list(vocab)}")
    print("\nTextPreprocessor verification passed!")


if __name__ == "__main__":
    main()
