"""
metrics.py - Multi-Metric Model Evaluation Harness.

Loads serialized pipeline models and evaluates them against held-out test datasets
across Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.config import MODELS_DIR, TEST_BENCHMARK_JSON, TEST_DATA_PATH
DEFAULT_MODELS_DIR = str(MODELS_DIR)
DEFAULT_TEST_PATH = str(TEST_DATA_PATH)
DEFAULT_OUTPUT_JSON = str(TEST_BENCHMARK_JSON)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("metrics")


class ModelEvaluator:
    """
    Evaluator harness that tests persisted scikit-learn pipelines on held-out data.
    """

    def __init__(self, models_dir: str = DEFAULT_MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.models_: Dict[str, Any] = {}
        self.load_models()

    def load_models(self) -> Dict[str, Any]:
        """Load all joblib pipeline artifacts found in the models directory."""
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory '{self.models_dir}' does not exist.")
            return {}

        for filename in os.listdir(self.models_dir):
            if filename.endswith("_pipeline.joblib"):
                model_name = filename.replace("_pipeline.joblib", "")
                filepath = os.path.join(self.models_dir, filename)
                try:
                    self.models_[model_name] = joblib.load(filepath)
                    logger.info(f"Loaded model '{model_name}' from {filepath}")
                except Exception as e:
                    logger.error(f"Failed to load {filepath}: {e}")

        return self.models_

    def evaluate_model(
        self,
        model_name: str,
        X_test: pd.Series,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """
        Evaluate a single loaded model against ground truth labels.
        """
        if model_name not in self.models_:
            raise ValueError(f"Model '{model_name}' not loaded.")

        pipeline = self.models_[model_name]
        preds = pipeline.predict(X_test)

        # Probabilities for ROC AUC
        probs: Optional[np.ndarray] = None
        roc_auc = 0.0
        try:
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_test)[:, 1]
                if len(set(y_test)) > 1:
                    roc_auc = float(roc_auc_score(y_test, probs))
        except Exception as e:
            logger.debug(f"Could not compute probabilities for {model_name}: {e}")

        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Per-class metrics
        spam_prec = float(precision_score(y_test, preds, pos_label=1, zero_division=0))
        spam_rec = float(recall_score(y_test, preds, pos_label=1, zero_division=0))
        spam_f1 = float(f1_score(y_test, preds, pos_label=1, zero_division=0))

        ham_prec = float(precision_score(y_test, preds, pos_label=0, zero_division=0))
        ham_rec = float(recall_score(y_test, preds, pos_label=0, zero_division=0))
        ham_f1 = float(f1_score(y_test, preds, pos_label=0, zero_division=0))

        macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))

        return {
            "model_name": model_name,
            "accuracy": float(accuracy_score(y_test, preds)),
            "spam_precision": spam_prec,
            "spam_recall": spam_rec,
            "spam_f1": spam_f1,
            "ham_precision": ham_prec,
            "ham_recall": ham_rec,
            "ham_f1": ham_f1,
            "macro_f1": macro_f1,
            "precision": spam_prec,  # backwards compatibility
            "recall": spam_rec,      # backwards compatibility
            "f1_score": spam_f1,     # backwards compatibility
            "roc_auc": roc_auc,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

    def evaluate_all(
        self,
        test_path: str = DEFAULT_TEST_PATH,
        output_json: str = DEFAULT_OUTPUT_JSON,
    ) -> pd.DataFrame:
        """
        Evaluate all loaded models on test data, print comparison table, and save JSON report.
        """
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test dataset not found at {test_path}")

        test_df = pd.read_csv(test_path)
        X_test = test_df["text"].fillna("")
        y_test = test_df["label"].astype(int)

        logger.info(f"Evaluating {len(self.models_)} models on {len(test_df)} test samples...")
        results: List[Dict[str, Any]] = []

        for model_name in sorted(self.models_.keys()):
            res = self.evaluate_model(model_name, X_test, y_test)
            results.append(res)

        # Create summary DataFrame with both per-class and overall metrics
        summary_rows = []
        for r in results:
            summary_rows.append({
                "Model": r["model_name"],
                "Accuracy": r["accuracy"],
                "Spam Precision": r["spam_precision"],
                "Spam Recall": r["spam_recall"],
                "Spam F1": r["spam_f1"],
                "Ham Precision": r["ham_precision"],
                "Ham Recall": r["ham_recall"],
                "Ham F1": r["ham_f1"],
                "Macro F1": r["macro_f1"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1-Score": r["f1_score"],
                "ROC-AUC": r["roc_auc"],
                "TN": r["confusion_matrix"]["tn"],
                "FP": r["confusion_matrix"]["fp"],
                "FN": r["confusion_matrix"]["fn"],
                "TP": r["confusion_matrix"]["tp"],
            })

        summary_df = pd.DataFrame(summary_rows).sort_values(by="Spam F1", ascending=False)

        print("\n" + "=" * 80)
        print("                HELD-OUT TEST SPLIT BENCHMARK LEADERBOARD                ")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)

        # Save to JSON
        out_dir = os.path.dirname(output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({
                "test_samples": len(test_df),
                "summary": summary_rows,
                "detailed_results": results,
            }, f, indent=2)

        logger.info(f"Test benchmark successfully written to {output_json}")
        return summary_df


def main() -> None:
    evaluator = ModelEvaluator()
    evaluator.evaluate_all()


if __name__ == "__main__":
    main()
