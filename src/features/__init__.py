"""
Features package containing TF-IDF, handcrafted statistical extractors, and unified pipeline.
"""

from src.features.tfidf_features import TFIDFExtractor, build_tfidf_vectorizer
from src.features.handcrafted_features import HandcraftedFeatureExtractor, SPAM_TRIGGER_WORDS
from src.features.feature_pipeline import UnifiedFeaturePipeline, build_feature_pipeline

__all__ = [
    "TFIDFExtractor",
    "build_tfidf_vectorizer",
    "HandcraftedFeatureExtractor",
    "SPAM_TRIGGER_WORDS",
    "UnifiedFeaturePipeline",
    "build_feature_pipeline",
]
