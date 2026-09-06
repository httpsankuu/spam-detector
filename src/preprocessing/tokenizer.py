"""
tokenizer.py - Tokenization, Stopword Removal, and Lemmatization.

Extracts tokens from cleaned text, filters common English stopwords (preserving key negations),
and applies NLTK WordNet lemmatization to unify word variations.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tokenizer")

# Global instances (lazy initialized)
_LEMMATIZER = None
_STOPWORDS: Optional[Set[str]] = None

# Words to preserve even if present in default stopwords (negations matter for spam vs ham intent)
PRESERVED_NEGATIONS: Set[str] = {"not", "no", "nor", "neither", "never"}


def _get_lemmatizer():
    """Lazily initialize NLTK WordNetLemmatizer with fallback."""
    global _LEMMATIZER
    if _LEMMATIZER is None:
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            _LEMMATIZER = WordNetLemmatizer()
            # Test lemmatizer to catch missing corpora early
            _LEMMATIZER.lemmatize("testing")
        except Exception:
            logger.debug("NLTK WordNetLemmatizer unavailable; using fallback lemmatizer.")
            class FallbackLemmatizer:
                def lemmatize(self, word: str, pos: str = "n") -> str:
                    # Basic rule-based stemming fallback if WordNet resource not installed
                    if word.endswith("ing") and len(word) > 5:
                        return word[:-3]
                    if word.endswith("ies") and len(word) > 4:
                        return word[:-3] + "y"
                    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
                        return word[:-1]
                    return word
            _LEMMATIZER = FallbackLemmatizer()
    return _LEMMATIZER


def _get_stopwords() -> Set[str]:
    """Lazily load NLTK stopwords with default fallback."""
    global _STOPWORDS
    if _STOPWORDS is None:
        try:
            import nltk
            from nltk.corpus import stopwords
            sw = set(stopwords.words("english"))
            _STOPWORDS = sw - PRESERVED_NEGATIONS
        except Exception:
            logger.debug("NLTK stopwords unavailable; using fallback standard stopword set.")
            # Fallback stopword set
            _STOPWORDS = {
                "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
                "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
                "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
                "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
                "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
                "at", "by", "for", "with", "about", "against", "between", "into", "through",
                "during", "before", "after", "above", "below", "to", "from", "up", "down",
                "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
                "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
                "few", "more", "most", "other", "some", "such", "only", "own", "same", "so",
                "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
            }
    return _STOPWORDS


def tokenize_and_lemmatize(
    text: str,
    remove_stopwords: bool = True,
    min_word_length: int = 2,
) -> List[str]:
    """
    Tokenize text, filter stopwords, and lemmatize tokens.

    Args:
        text: Input string (already cleaned and lowercased)
        remove_stopwords: If True, exclude common English stopwords
        min_word_length: Minimum character length for tokens (default: 2)

    Returns:
        List of processed token strings
    """
    if not text or not isinstance(text, str):
        return []

    # Fast word splitting on whitespace
    raw_tokens = text.split()

    lemmatizer = _get_lemmatizer()
    stop_words = _get_stopwords() if remove_stopwords else set()

    processed: List[str] = []
    for token in raw_tokens:
        token = token.strip()
        if len(token) < min_word_length:
            continue
        if remove_stopwords and token in stop_words:
            continue

        # Lemmatize (verbs first, then nouns)
        lemma = lemmatizer.lemmatize(token, pos="v")
        if lemma == token:
            lemma = lemmatizer.lemmatize(token, pos="n")

        processed.append(lemma)

    return processed


def lemmatized_text(
    text: str,
    remove_stopwords: bool = True,
    min_word_length: int = 2,
) -> str:
    """
    Convenience function that converts text to a single space-separated string of lemmatized tokens.
    """
    tokens = tokenize_and_lemmatize(
        text, remove_stopwords=remove_stopwords, min_word_length=min_word_length
    )
    return " ".join(tokens)


def main() -> None:
    """Self-check demo for tokenizer and lemmatizer."""
    sample = "urgent you won dollar number in the lottery visit httpaddr to get cash now"
    tokens = tokenize_and_lemmatize(sample)
    joined = lemmatized_text(sample)
    print("--- INPUT SAMPLE ---")
    print(sample)
    print("\n--- LEMMATIZED TOKENS ---")
    print(tokens)
    print("\n--- JOINED TEXT ---")
    print(joined)
    assert len(tokens) > 0, "No tokens returned"
    assert "in" not in tokens, "Stopword 'in' was not removed"
    assert "the" not in tokens, "Stopword 'the' was not removed"
    print("\nTokenizer verification passed!")


if __name__ == "__main__":
    main()
