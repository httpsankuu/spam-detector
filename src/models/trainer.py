"""
trainer.py - Multi-Model Training Orchestrator & Persistence Pipeline.

Trains all 5 classifiers (Multinomial NB, Logistic Regression, Linear SVM,
Random Forest, and XGBoost) using the unified feature pipeline, benchmarks their
performance on validation data, and serializes each pipeline to disk along with
leaderboard metadata.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.logistic_regression import build_lr_pipeline
from src.models.naive_bayes import build_nb_pipeline
from src.models.svm_classifier import build_svm_pipeline
from src.models.tree_classifier import build_ensemble_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trainer")


def train_and_persist_all(
    train_path: str = "data/processed/train.csv",
    val_path: str = "data/processed/val.csv",
    output_dir: str = "models",
    max_features: int = 5000,
) -> Dict[str, Any]:
    """
    Train all 5 classifiers, evaluate on validation set, and persist pipelines to disk.

    Args:
        train_path: Path to train.csv
        val_path: Path to val.csv
        output_dir: Destination directory for serialized joblib pipelines and metadata
        max_features: Vocabulary capacity for the feature pipeline

    Returns:
        Dictionary containing leaderboard metrics and metadata
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading datasets: train='{train_path}', val='{val_path}'")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df["text"].fillna("")
    y_train = train_df["label"].astype(int)

    X_val = val_df["text"].fillna("")
    y_val = val_df["label"].astype(int)

    # Define model configurations
    model_factories = {
        "naive_bayes": lambda: build_nb_pipeline(alpha=1.0, max_features=max_features),
        "logistic_regression": lambda: build_lr_pipeline(C=1.0, max_features=max_features),
        "linear_svm": lambda: build_svm_pipeline(C=1.0, calibrate_probabilities=True, max_features=max_features),
        "random_forest": lambda: build_ensemble_pipeline(model_type="random_forest", n_estimators=100, max_features=max_features),
        "xgboost": lambda: build_ensemble_pipeline(model_type="xgboost", n_estimators=100, max_features=max_features),
    }

    leaderboard: List[Dict[str, Any]] = []
    models_saved: Dict[str, str] = {}

    print("\n" + "=" * 70)
    print("           SPAM DETECTION MULTI-MODEL TRAINING HARNESS           ")
    print("=" * 70)

    for model_name, factory in model_factories.items():
        logger.info(f"==> Training pipeline for '{model_name}'...")
        pipeline = factory()
        pipeline.fit(X_train, y_train)

        # Predictions on validation split
        val_preds = pipeline.predict(X_val)
        try:
            val_probs = pipeline.predict_proba(X_val)[:, 1]
            # ROC AUC requires at least one of each class
            if len(set(y_val)) > 1:
                roc_auc = float(roc_auc_score(y_val, val_probs))
            else:
                roc_auc = 0.0
        except Exception:
            val_probs = None
            roc_auc = 0.0

        metrics = {
            "model_name": model_name,
            "accuracy": float(accuracy_score(y_val, val_preds)),
            "precision": float(precision_score(y_val, val_preds, zero_division=0)),
            "recall": float(recall_score(y_val, val_preds, zero_division=0)),
            "f1_score": float(f1_score(y_val, val_preds, zero_division=0)),
            "roc_auc": roc_auc,
        }
        leaderboard.append(metrics)

        # Persist pipeline to disk
        model_filename = f"{model_name}_pipeline.joblib"
        model_filepath = os.path.join(output_dir, model_filename)
        joblib.dump(pipeline, model_filepath)
        models_saved[model_name] = model_filepath
        logger.info(f"Saved {model_name} pipeline -> {model_filepath}")

    # Build summary leaderboard DataFrame
    leaderboard_df = pd.DataFrame(leaderboard).sort_values(by="f1_score", ascending=False)
    print("\n--- VALIDATION LEADERBOARD ---")
    print(leaderboard_df.to_string(index=False))

    # Persist metadata JSON
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "leaderboard": leaderboard,
        "models_saved": models_saved,
    }
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model training metadata saved -> {metadata_path}")
    return metadata


def main() -> None:
    train_and_persist_all()


if __name__ == "__main__":
    main()
