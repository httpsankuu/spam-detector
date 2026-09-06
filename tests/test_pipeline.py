"""
test_pipeline.py - Comprehensive Unit & Integration Tests.

Validates:
1. Data loading and schema integrity.
2. Stratified data splitting with zero leakage.
3. Preprocessing transformation (HTML, lemmatization, tokens).
4. Feature extraction (TF-IDF + handcrafted indicator matrix shapes).
5. Model inference consistency and probability outputs across all 5 models.
6. Explainability engine token-attribution structure.
7. Edge case handling (empty input, very short text, non-English, punctuation).
"""

import os
import pytest
import pandas as pd
import numpy as np
import joblib

from src.data.email_loader import EmailDataLoader
from src.data.sms_loader import SMSDataLoader
from src.data.split_data import split_dataset
from src.preprocessing.cleaner import clean_text
from src.preprocessing.tokenizer import tokenize_and_lemmatize
from src.preprocessing.transformer import TextPreprocessor
from src.features.tfidf_features import TFIDFExtractor
from src.features.handcrafted_features import HandcraftedFeatureExtractor
from src.features.feature_pipeline import build_feature_pipeline
from src.evaluation.explainability import SpamExplainer
from src.evaluation.metrics import ModelEvaluator


class TestDataModule:
    """Tests for dataset ingestion and splitting."""

    def test_email_loader_schema(self):
        loader = EmailDataLoader()
        df = loader.get_sample_corpus()
        assert list(df.columns) == ["text", "label"]
        assert df["label"].isin([0, 1]).all()
        assert len(df) >= 30

    def test_sms_loader_schema(self):
        df = SMSDataLoader.get_sample_sms()
        assert list(df.columns) == ["text", "label"]
        assert df["label"].isin([0, 1]).all()
        assert len(df) >= 16

    def test_stratified_split_no_leakage(self):
        loader = EmailDataLoader()
        df = loader.get_sample_corpus()
        train_df, val_df, test_df = split_dataset(df, test_size=0.2, val_size=0.1, random_state=42)

        # Check disjoint sets
        train_texts = set(train_df["text"])
        val_texts = set(val_df["text"])
        test_texts = set(test_df["text"])

        assert len(train_texts.intersection(val_texts)) == 0
        assert len(train_texts.intersection(test_texts)) == 0
        assert len(val_texts.intersection(test_texts)) == 0
        assert len(train_df) + len(val_df) + len(test_df) == len(df)


class TestNLPPreprocessing:
    """Tests for text cleaning, tokenization, and scikit-learn transformer."""

    def test_clean_text_normalizations(self):
        raw = "Subject: WIN $50,000! Visit <b>http://win.com</b> or email win@promo.org! Call 123456 now."
        cleaned = clean_text(raw)
        assert "httpaddr" in cleaned
        assert "dollar" in cleaned
        assert "emailaddr" in cleaned
        assert "<b>" not in cleaned
        assert "Subject:" not in cleaned

    def test_lemmatizer_negation_preservation(self):
        text = "we do not want this spam message"
        lemmas = tokenize_and_lemmatize(text)
        assert "not" in lemmas  # Negation must be preserved

    def test_text_preprocessor_transformer(self):
        prep = TextPreprocessor()
        out = prep.transform(["Subject: Claim $1000 cash at http://win.com"])
        assert isinstance(out, list)
        assert len(out) == 1
        assert "dollar" in out[0] or "httpaddr" in out[0]


class TestFeatureEngineering:
    """Tests for TF-IDF and handcrafted extractors."""

    def test_tfidf_extractor_shape_and_names(self):
        texts = ["free prize claim now", "urgent bank update", "lunch meeting today"]
        extractor = TFIDFExtractor(max_features=20, ngram_range=(1, 2), min_df=1)
        matrix = extractor.fit_transform(texts)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] <= 20
        assert len(extractor.get_feature_names()) == matrix.shape[1]

    def test_handcrafted_extractor_shape(self):
        texts = ["URGENT! Free $10,000 at http://win.com!", "Normal note here."]
        extractor = HandcraftedFeatureExtractor()
        mat = extractor.transform(texts)
        assert mat.shape == (2, 10)
        assert mat[0, 2] > mat[1, 2]  # caps ratio of first text should exceed second
        assert mat[0, 6] == 1.0  # url count in first text

    def test_unified_feature_pipeline(self):
        pipe = build_feature_pipeline(max_features=100)
        texts = ["Claim your prize now!", "Team sync meeting notes."]
        mat = pipe.fit_transform(texts)
        assert mat.shape[0] == 2
        assert mat.shape[1] == len(pipe.get_feature_names_out())


class TestModelInferenceAndPersistence:
    """Tests that all 5 persisted pipelines load and infer correctly."""

    @pytest.mark.parametrize("model_name", [
        "logistic_regression",
        "linear_svm",
        "naive_bayes",
        "random_forest",
        "xgboost",
    ])
    def test_persisted_model_inference(self, model_name):
        path = f"models/{model_name}_pipeline.joblib"
        assert os.path.exists(path), f"Missing model file {path}"
        model = joblib.load(path)

        # Predict on spam sample
        spam_sample = "URGENT! You won $10,000! Click http://claim.com immediately!"
        pred_spam = model.predict([spam_sample])[0]
        assert pred_spam in [0, 1]

        # Check probability output
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([spam_sample])
            assert probs.shape == (1, 2)
            assert np.isclose(probs.sum(), 1.0)


class TestExplainabilityAndEdgeCases:
    """Tests for SpamExplainer and edge cases."""

    def test_explainer_payload_structure(self):
        explainer = SpamExplainer()
        exp = explainer.explain_text("CONGRATULATIONS! You won $5,000 cash! Claim now at http://win.com!")
        assert "prediction" in exp
        assert "probability" in exp
        assert "risk_level" in exp
        assert "detected_triggers" in exp
        assert "handcrafted_metrics" in exp
        assert exp["prediction"] == "SPAM"

    def test_edge_cases_no_crash(self):
        explainer = SpamExplainer()
        edge_cases = [
            "",  # Empty
            "   \t\n  ",  # Whitespace
            "k",  # Single letter
            "!!! $$$ ???",  # Punctuation only
            "19283019283019",  # Numbers only
            "Bonjour comment allez-vous aujourd'hui?",  # Non-English
            "http://example.com/test",  # Link only
            "Normal text " * 500,  # Long text
        ]
        for ec in edge_cases:
            res = explainer.explain_text(ec)
            assert res["prediction"] in ["SPAM", "HAM"]
            assert 0.0 <= res["probability"] <= 1.0
