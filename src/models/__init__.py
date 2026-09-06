"""
Models package for spam detection classification.
"""

from src.models.naive_bayes import NaiveBayesClassifier, build_nb_pipeline
from src.models.logistic_regression import LogisticRegressionClassifier, build_lr_pipeline
from src.models.svm_classifier import SVMClassifier, build_svm_pipeline
from src.models.tree_classifier import EnsembleClassifier, build_ensemble_pipeline
from src.models.trainer import train_and_persist_all

__all__ = [
    "NaiveBayesClassifier",
    "build_nb_pipeline",
    "LogisticRegressionClassifier",
    "build_lr_pipeline",
    "SVMClassifier",
    "build_svm_pipeline",
    "EnsembleClassifier",
    "build_ensemble_pipeline",
    "train_and_persist_all",
]
