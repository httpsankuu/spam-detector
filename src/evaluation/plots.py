"""
plots.py - Diagnostic & Evaluation Visualization Generator.

Generates publication-quality charts for spam classification models:
- Labeled Confusion Matrix Heatmap subplots
- Combined ROC-AUC Curves with threshold lines
- Multi-Metric Bar Chart Comparisons
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

from src.evaluation.metrics import ModelEvaluator

from config.config import (
    CONFUSION_MATRICES_PNG,
    METRICS_COMPARISON_PNG,
    ROC_CURVES_PNG,
    TEST_BENCHMARK_JSON,
    TEST_DATA_PATH,
)

DEFAULT_BENCHMARK_JSON = str(TEST_BENCHMARK_JSON)
DEFAULT_CM_PNG = str(CONFUSION_MATRICES_PNG)
DEFAULT_ROC_PNG = str(ROC_CURVES_PNG)
DEFAULT_METRICS_PNG = str(METRICS_COMPARISON_PNG)
DEFAULT_TEST_PATH = str(TEST_DATA_PATH)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("plots")

# Modern clean aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_confusion_matrices(
    benchmark_json_path: str = DEFAULT_BENCHMARK_JSON,
    output_path: str = DEFAULT_CM_PNG,
) -> str:
    """
    Render a 2x3 grid of annotated confusion matrix heatmaps.
    """
    if not os.path.exists(benchmark_json_path):
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_json_path}")

    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models_data = data["detailed_results"]
    n_models = len(models_data)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, model_res in enumerate(models_data):
        ax = axes[idx]
        cm_dict = model_res["confusion_matrix"]
        cm = np.array([
            [cm_dict["tn"], cm_dict["fp"]],
            [cm_dict["fn"], cm_dict["tp"]],
        ])

        labels = [["TN", "FP"], ["FN", "TP"]]
        annot = np.empty_like(cm, dtype=object)
        for i in range(2):
            for j in range(2):
                annot[i, j] = f"{labels[i][j]}\n{cm[i, j]}"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Ham (0)", "Spam (1)"],
            yticklabels=["Ham (0)", "Spam (1)"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(f"{model_res['model_name'].replace('_', ' ').title()}\n(F1: {model_res['f1_score']:.2f})", fontsize=12, weight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    # Hide extra unused subplot if odd count
    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle("Spam Classifier Confusion Matrices (Held-out Test Split)", fontsize=16, weight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrices -> {output_path}")
    return output_path


def plot_roc_curves(
    evaluator: Optional[ModelEvaluator] = None,
    test_path: str = DEFAULT_TEST_PATH,
    output_path: str = DEFAULT_ROC_PNG,
) -> str:
    """
    Render combined ROC Curves for all classifiers with AUC legend scores.
    """
    if evaluator is None:
        evaluator = ModelEvaluator()

    test_df = pd.read_csv(test_path)
    X_test = test_df["text"].fillna("")
    y_test = test_df["label"].astype(int)

    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance (AUC = 0.50)")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (model_name, pipeline) in enumerate(sorted(evaluator.models_.items())):
        try:
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                color = colors[idx % len(colors)]
                plt.plot(fpr, tpr, lw=2.2, color=color, label=f"{model_name.replace('_', ' ').title()} (AUC = {roc_auc:.3f})")
        except Exception as e:
            logger.warning(f"Could not compute ROC curve for {model_name}: {e}")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=15, weight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved ROC curves -> {output_path}")
    return output_path


def plot_metrics_comparison_bar(
    benchmark_json_path: str = DEFAULT_BENCHMARK_JSON,
    output_path: str = DEFAULT_METRICS_PNG,
) -> str:
    """
    Render grouped bar chart comparing Accuracy, Precision, Recall, and F1 across models.
    """
    if not os.path.exists(benchmark_json_path):
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_json_path}")

    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data["summary"])
    df["Model"] = df["Model"].apply(lambda s: s.replace("_", " ").title())

    melted = pd.melt(
        df,
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=melted,
        x="Model",
        y="Score",
        hue="Metric",
        palette="viridis",
    )
    plt.ylim([0, 1.15])
    plt.title("Model Comparison Across Key Evaluation Metrics", fontsize=15, weight="bold", pad=12)
    plt.xlabel("Classifier Model", fontsize=12, weight="bold")
    plt.ylabel("Score", fontsize=12, weight="bold")
    plt.legend(title="Metric", loc="upper right", frameon=True)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved metrics bar chart -> {output_path}")
    return output_path


def generate_all_plots() -> Dict[str, str]:
    """Generate all three diagnostic evaluation plots."""
    evaluator = ModelEvaluator()
    evaluator.evaluate_all()

    cm_path = plot_confusion_matrices()
    roc_path = plot_roc_curves(evaluator)
    bar_path = plot_metrics_comparison_bar()

    return {
        "confusion_matrices": cm_path,
        "roc_curves": roc_path,
        "metrics_comparison": bar_path,
    }


def main() -> None:
    generate_all_plots()


if __name__ == "__main__":
    main()
