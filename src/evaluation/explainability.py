"""
explainability.py - Local & Global Explainability Engine.

Provides model transparency and token/feature attribution:
- Global: Most predictive positive (spam) and negative (ham) features with weights
- Local: Individual email decomposition (x_i * w_i) showing why a specific message
  was flagged, highlighting detected spam triggers and handcrafted metrics.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

from src.features.handcrafted_features import SPAM_TRIGGER_WORDS

from config.config import LOGISTIC_REGRESSION_MODEL_PATH
DEFAULT_MODEL_PATH = str(LOGISTIC_REGRESSION_MODEL_PATH)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("explainability")


class SpamExplainer:
    """
    Explainability engine built upon transparent linear model coefficients.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model pipeline not found at {self.model_path}")

        self.pipeline = joblib.load(self.model_path)
        self.feature_pipeline = self.pipeline.named_steps["features"]
        self.classifier = self.pipeline.steps[-1][1]

        # Cache feature names
        self.feature_names = self.feature_pipeline.get_feature_names_out()
        
        if hasattr(self.classifier, "coef_"):
            self.coefficients = self.classifier.coef_.flatten()
            self.intercept = float(self.classifier.intercept_[0]) if hasattr(self.classifier, "intercept_") else 0.0
            self.is_linear = True
        elif hasattr(self.classifier, "feature_log_prob_"):
            self.coefficients = (self.classifier.feature_log_prob_[1] - self.classifier.feature_log_prob_[0]).flatten()
            self.intercept = 0.0
            self.is_linear = True
        elif hasattr(self.classifier, "calibrated_classifiers_"):
            base_estimator = self.classifier.calibrated_classifiers_[0].estimator
            if hasattr(base_estimator, "coef_"):
                self.coefficients = base_estimator.coef_.flatten()
                self.intercept = float(base_estimator.intercept_[0]) if hasattr(base_estimator, "intercept_") else 0.0
                self.is_linear = True
            else:
                self.coefficients = np.zeros(len(self.feature_names))
                self.intercept = 0.0
                self.is_linear = False
        else:
            self.coefficients = np.zeros(len(self.feature_names))
            self.intercept = 0.0
            self.is_linear = False

    def get_global_feature_importance(self, top_n: int = 15) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract the top most positive (spam-indicative) and negative (ham-indicative) features.
        """
        sorted_indices = np.argsort(self.coefficients)

        top_ham_indices = sorted_indices[:top_n]
        top_spam_indices = sorted_indices[-top_n:][::-1]

        spam_features = [
            {"feature": str(self.feature_names[i]), "weight": round(float(self.coefficients[i]), 4)}
            for i in top_spam_indices
        ]
        ham_features = [
            {"feature": str(self.feature_names[i]), "weight": round(float(self.coefficients[i]), 4)}
            for i in top_ham_indices
        ]

        return {
            "spam_indicative": spam_features,
            "ham_indicative": ham_features,
        }

    def explain_text(self, raw_text: str, top_signals: int = 6) -> Dict[str, Any]:
        """
        Decompose an individual email or SMS into predictive feature contributions.

        Args:
            raw_text: Raw string input
            top_signals: Number of top positive/negative signals to return

        Returns:
            Dictionary payload formatted for explainability display and Streamlit UI.
        """
        if not isinstance(raw_text, str):
            raw_text = str(raw_text) if raw_text is not None else ""

        # 1. Prediction & Probabilities
        prob_spam = float(self.pipeline.predict_proba([raw_text])[0, 1])
        prediction = "SPAM" if prob_spam >= 0.5 else "HAM"

        # Risk classification
        if prob_spam >= 0.80:
            risk_level = "HIGH RISK"
        elif prob_spam >= 0.50:
            risk_level = "MODERATE RISK"
        elif prob_spam >= 0.20:
            risk_level = "LOW RISK"
        else:
            risk_level = "SAFE"

        # 2. Extract feature representation for this text
        feat_vector = self.feature_pipeline.transform([raw_text])
        # Non-zero indices
        non_zero_cols = feat_vector.indices
        feat_values = feat_vector.data

        # 3. Compute contributions: x_i * w_i
        contributions: List[Tuple[str, float, float, float]] = []
        for col_idx, val in zip(non_zero_cols, feat_values):
            name = str(self.feature_names[col_idx])
            weight = float(self.coefficients[col_idx])
            contrib = float(val * weight)
            contributions.append((name, val, weight, contrib))

        # Sort contributions
        contributions.sort(key=lambda x: x[3], reverse=True)

        spam_signals = [
            {"feature": name, "value": round(val, 4), "weight": round(weight, 4), "contribution": round(contrib, 4)}
            for name, val, weight, contrib in contributions
            if contrib > 0
        ][:top_signals]

        ham_signals = [
            {"feature": name, "value": round(val, 4), "weight": round(weight, 4), "contribution": round(contrib, 4)}
            for name, val, weight, contrib in reversed(contributions)
            if contrib < 0
        ][:top_signals]

        # 4. Detect matched trigger keywords directly in text for visual highlighting
        found_triggers = []
        for word in SPAM_TRIGGER_WORDS:
            pattern = r"\b" + re.escape(word) + r"\b"
            if re.search(pattern, raw_text, flags=re.IGNORECASE):
                found_triggers.append(word.lower())

        # 5. Handcrafted structural metrics
        handcrafted_extractor = self.feature_pipeline.handcrafted
        hc_values = handcrafted_extractor.extract_single(raw_text)
        hc_names = handcrafted_extractor.get_feature_names()
        metrics_dict = {name: round(val, 3) for name, val in zip(hc_names, hc_values)}

        return {
            "prediction": prediction,
            "probability": round(prob_spam, 4),
            "risk_level": risk_level,
            "detected_triggers": sorted(list(set(found_triggers))),
            "top_spam_signals": spam_signals,
            "top_ham_signals": ham_signals,
            "handcrafted_metrics": metrics_dict,
            "is_linear": self.is_linear,
        }


def main() -> None:
    """Self-check of the explainability engine."""
    explainer = SpamExplainer()

    print("--- GLOBAL TOP PREDICTIVE FEATURES ---")
    global_importance = explainer.get_global_feature_importance(top_n=5)
    print("Spam-indicative:", global_importance["spam_indicative"])
    print("Ham-indicative:", global_importance["ham_indicative"])

    print("\n--- LOCAL EMAIL EXPLANATION ---")
    phishing_sample = (
        "URGENT: Congratulations! You won a $5,000 cash prize! Claim your prize now at "
        "http://lottery-winner.com before it expires!"
    )
    explanation = explainer.explain_text(phishing_sample)
    print(json.dumps(explanation, indent=2))
    print("\nSpamExplainer self-test passed successfully!")


if __name__ == "__main__":
    main()
