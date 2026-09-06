"""
Evaluation package containing metrics benchmarks, plotting, and explainability.
"""

from src.evaluation.metrics import ModelEvaluator
from src.evaluation.plots import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_metrics_comparison_bar,
    generate_all_plots,
)
from src.evaluation.explainability import SpamExplainer

__all__ = [
    "ModelEvaluator",
    "SpamExplainer",
    "plot_confusion_matrices",
    "plot_roc_curves",
    "plot_metrics_comparison_bar",
    "generate_all_plots",
]
