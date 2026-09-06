"""
config.py - Centralized Configuration Module for AI Spam Detection System.

Defines all paths, hyperparameters, random seeds, and feature constants to ensure
complete reproducibility across training, evaluation, and application serving.
"""

from pathlib import Path
from typing import List, Tuple

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------------------
# Reproducibility & Random Seeds
# ------------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ------------------------------------------------------------------------------
# Data Paths & Proportions
# ------------------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

TRAIN_DATA_PATH: Path = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH: Path = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH: Path = PROCESSED_DATA_DIR / "test.csv"

# Dataset splitting proportions
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.10
TEST_RATIO: float = 0.20

# ------------------------------------------------------------------------------
# NLP Preprocessing Settings
# ------------------------------------------------------------------------------
MIN_WORD_LENGTH: int = 2
REMOVE_STOPWORDS: bool = True
REPLACE_URLS: bool = True
REPLACE_EMAILS: bool = True
REPLACE_CURRENCY: bool = True
REPLACE_NUMBERS: bool = True

# ------------------------------------------------------------------------------
# Feature Engineering Settings
# ------------------------------------------------------------------------------
TFIDF_MAX_FEATURES: int = 5000
TFIDF_NGRAM_RANGE: Tuple[int, int] = (1, 2)
TFIDF_MIN_DF: int = 2
TFIDF_MAX_DF: float = 0.95
TFIDF_SUBLINEAR_TF: bool = True
SCALE_HANDCRAFTED: bool = True

# ------------------------------------------------------------------------------
# Model Artifact Paths & Settings
# ------------------------------------------------------------------------------
MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

LOGISTIC_REGRESSION_MODEL_PATH: Path = MODELS_DIR / "logistic_regression_pipeline.joblib"
LINEAR_SVM_MODEL_PATH: Path = MODELS_DIR / "linear_svm_pipeline.joblib"
NAIVE_BAYES_MODEL_PATH: Path = MODELS_DIR / "naive_bayes_pipeline.joblib"
RANDOM_FOREST_MODEL_PATH: Path = MODELS_DIR / "random_forest_pipeline.joblib"
XGBOOST_MODEL_PATH: Path = MODELS_DIR / "xgboost_pipeline.joblib"

# Model hyperparameters
LR_C: float = 1.0
LR_MAX_ITER: int = 1000
LR_CLASS_WEIGHT: str = "balanced"

SVM_C: float = 1.0
SVM_MAX_ITER: int = 2000

NB_ALPHA: float = 1.0

RF_N_ESTIMATORS: int = 100
RF_MAX_DEPTH: int = 15

XGB_N_ESTIMATORS: int = 100
XGB_MAX_DEPTH: int = 6
XGB_LEARNING_RATE: float = 0.1

# ------------------------------------------------------------------------------
# Evaluation & Report Paths
# ------------------------------------------------------------------------------
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TEST_BENCHMARK_JSON: Path = REPORTS_DIR / "test_benchmark.json"

CONFUSION_MATRICES_PNG: Path = FIGURES_DIR / "confusion_matrices.png"
ROC_CURVES_PNG: Path = FIGURES_DIR / "roc_curves.png"
METRICS_COMPARISON_PNG: Path = FIGURES_DIR / "metrics_comparison.png"
