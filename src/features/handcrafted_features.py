"""
handcrafted_features.py - Handcrafted Statistical & Spam Trigger Feature Extractor.

Extracts domain-specific numerical features and statistical metrics from text
including capitalization ratio, link counts, currency symbols, punctuation density,
and spam trigger keyword frequencies.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("handcrafted_features")

# Curated list of high-signal spam trigger words & phrases
SPAM_TRIGGER_WORDS = [
    "free",
    "win",
    "winner",
    "won",
    "prize",
    "claim",
    "urgent",
    "cash",
    "bonus",
    "credit",
    "loan",
    "guarantee",
    "guaranteed",
    "selected",
    "congratulations",
    "congrats",
    "exclusive",
    "offer",
    "limited",
    "expire",
    "expires",
    "call now",
    "act now",
    "click here",
    "instant",
    "refund",
    "million",
    "thousand",
    "invest",
    "investment",
    "risk-free",
    "apply now",
    "verify your account",
    "password",
    "security alert",
]


class HandcraftedFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts statistical and domain-specific handcrafted features from text documents.

    Output features (10 total):
        1. char_count: Total character count
        2. word_count: Total whitespace-separated word count
        3. caps_ratio: Proportion of uppercase alphabetic characters (measures urgency/shouting)
        4. exclamation_count: Count of exclamation marks (!)
        5. question_count: Count of question marks (?)
        6. dollar_count: Count of currency indicators ('$', 'dollar', 'usd', 'eur', 'gbp')
        7. url_count: Count of URLs or 'httpaddr' placeholder tokens
        8. email_count: Count of email addresses or 'emailaddr' placeholder tokens
        9. digit_ratio: Ratio of digits to total characters
        10. trigger_words_count: Total occurrences of curated spam trigger keywords
    """

    FEATURE_NAMES = [
        "char_count",
        "word_count",
        "caps_ratio",
        "exclamation_count",
        "question_count",
        "dollar_count",
        "url_count",
        "email_count",
        "digit_ratio",
        "trigger_words_count",
    ]

    def __init__(self, trigger_words: Union[Sequence[str], None] = None) -> None:
        self.trigger_words = list(trigger_words) if trigger_words is not None else SPAM_TRIGGER_WORDS
        # Compile case-insensitive word boundary regex for trigger words
        escaped_triggers = [re.escape(w) for w in self.trigger_words]
        self._trigger_pattern = re.compile(r"\b(" + "|".join(escaped_triggers) + r")\b", re.IGNORECASE)

    def fit(self, X: Any, y: Any = None) -> "HandcraftedFeatureExtractor":
        """Stateless fit method returning self."""
        return self

    def extract_single(self, text: str) -> List[float]:
        """
        Extract numerical feature vector for a single text document.

        Args:
            text: Raw or partially preprocessed text string

        Returns:
            List of 10 float values
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        char_count = len(text)
        words = text.split()
        word_count = len(words)

        # Capitalization ratio: uppercase letters / total alphabetic letters
        alpha_chars = sum(1 for c in text if c.isalpha())
        upper_chars = sum(1 for c in text if c.isupper())
        caps_ratio = (upper_chars / alpha_chars) if alpha_chars > 0 else 0.0

        # Punctuation counts
        exclamation_count = float(text.count("!"))
        question_count = float(text.count("?"))

        # Currency signals: '$', '€', '£', '¥', and 'dollar' token
        currency_matches = re.findall(r"[\$€£¥]|dollar\b|usd\b", text, flags=re.IGNORECASE)
        dollar_count = float(len(currency_matches))

        # URL signals: explicit URLs or 'httpaddr' token
        url_matches = re.findall(r"https?://\S+|www\.\S+|httpaddr\b", text, flags=re.IGNORECASE)
        url_count = float(len(url_matches))

        # Email signals: email pattern or 'emailaddr' token
        email_matches = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+|emailaddr\b", text, flags=re.IGNORECASE)
        email_count = float(len(email_matches))

        # Digit ratio: digits / total characters
        digits = sum(1 for c in text if c.isdigit())
        # Also check for 'number' placeholder token
        number_tokens = len(re.findall(r"\bnumber\b", text, flags=re.IGNORECASE))
        digit_count = digits + (number_tokens * 3)  # surrogate weight for number tokens
        digit_ratio = (digit_count / char_count) if char_count > 0 else 0.0

        # Spam trigger keyword occurrences
        trigger_matches = self._trigger_pattern.findall(text)
        trigger_words_count = float(len(trigger_matches))

        return [
            float(char_count),
            float(word_count),
            float(caps_ratio),
            exclamation_count,
            question_count,
            dollar_count,
            url_count,
            email_count,
            float(digit_ratio),
            trigger_words_count,
        ]

    def transform(self, X: Union[pd.Series, Sequence[str], np.ndarray]) -> np.ndarray:
        """
        Transform a collection of text documents into a 2D numpy array of handcrafted features.

        Args:
            X: Iterable, Series, or array of strings

        Returns:
            2D numpy array of shape (n_samples, 10)
        """
        if isinstance(X, pd.Series):
            texts = X.tolist()
        elif isinstance(X, np.ndarray):
            texts = X.flatten().tolist()
        elif isinstance(X, (list, tuple)):
            texts = list(X)
        else:
            texts = [str(X)]

        matrix = [self.extract_single(t) for t in texts]
        return np.array(matrix, dtype=np.float64)

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.transform(X)

    def get_feature_names(self) -> List[str]:
        """Return handcrafted feature column names."""
        return list(self.FEATURE_NAMES)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return numpy array of feature names for scikit-learn compatibility."""
        return np.array(self.FEATURE_NAMES, dtype=object)


def main() -> None:
    """Self-check and demonstration of HandcraftedFeatureExtractor."""
    sample_texts = [
        "URGENT: CONGRATULATIONS! You won $10,000 CASH prize! Claim now at http://win.com!",
        "Hey Ankit, can you review the pull request when you get a chance? Thanks.",
        "Free entry into our weekly £1000 competition! Text CLAIM to 87121 now!",
    ]

    extractor = HandcraftedFeatureExtractor()
    features = extractor.transform(sample_texts)

    print("--- EXTRACTED HANDCRAFTED FEATURES ---")
    df = pd.DataFrame(features, columns=extractor.get_feature_names())
    print(df.to_string(index=True))

    print(f"\nMatrix shape: {features.shape}")
    assert features.shape == (3, 10), f"Expected shape (3, 10), got {features.shape}"
    print("HandcraftedFeatureExtractor self-test passed successfully!")


if __name__ == "__main__":
    main()
