"""
streamlit_app.py - Interactive AI-Based Spam Email Detection Web Application.

Provides an interactive user interface to:
1. Classify any user-submitted email or SMS message in real time.
2. Select among 5 trained ML models (Logistic Regression, Linear SVM, Naive Bayes, Random Forest, XGBoost).
3. Display confidence probability gauges, risk level badges, and explainability breakdown.
4. Inspect model performance comparison tables, confusion matrices, and ROC curves.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root directory is in Python's search path when run via `streamlit run app/streamlit_app.py`
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.evaluation.explainability import SpamExplainer

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="AI Spam Email Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern card styling and visual badges
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-spam {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #F87171;
    }
    .badge-ham {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #4ADE80;
    }
    .trigger-tag {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        display: inline-block;
        border: 1px solid #FCD34D;
    }
    .highlight-spam {
        background-color: #FED7AA;
        font-weight: bold;
        padding: 1px 4px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Preset sample emails for quick one-click testing
SAMPLE_PRESETS = {
    "-- Select a sample preset --": "",
    "🎰 Phishing: Lottery Prize Winner": (
        "Subject: CONGRATULATIONS! You won $5,000,000 in our international lottery! "
        "Visit http://claim-cash-reward.org immediately to verify your identity and receive your payment. "
        "Hurry, this offer expires in 24 hours! Call 1-800-CLAIM-NOW!"
    ),
    "🔒 Urgent: Bank Security Alert": (
        "URGENT: Your bank account security has been compromised! Click http://secure-verify-bank.com "
        "to reset your password immediately and prevent account suspension. Failure to act will result "
        "in total account lock. Security Team."
    ),
    "💼 Normal: Project Sprint Sync": (
        "Hi team, please find attached the meeting notes and action items from today's engineering sprint sync. "
        "Don't forget to push your code commits to GitHub before the 5pm code freeze. "
        "Best regards, Ankit."
    ),
    "☕ Normal: Casual Lunch Invitation": (
        "Hey, are you free for lunch tomorrow around 12:30? Let's check out that new cafe near the library. "
        "Let me know if that time works for you!"
    ),
}

MODEL_DISPLAY_NAMES = {
    "Logistic Regression (Recommended)": "logistic_regression",
    "Linear SVM": "linear_svm",
    "Multinomial Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}


@st.cache_resource
def load_all_models() -> Dict[str, Any]:
    """Cache loaded scikit-learn models from disk."""
    models: Dict[str, Any] = {}
    models_dir = "models"
    for filename in os.listdir(models_dir):
        if filename.endswith("_pipeline.joblib"):
            key = filename.replace("_pipeline.joblib", "")
            filepath = os.path.join(models_dir, filename)
            try:
                models[key] = joblib.load(filepath)
            except Exception as e:
                st.error(f"Error loading {filepath}: {e}")
    return models


@st.cache_resource
def get_explainer() -> SpamExplainer:
    """Initialize cached explainability engine."""
    return SpamExplainer(model_path="models/logistic_regression_pipeline.joblib")


def render_highlighted_text(text: str, triggers: List[str]) -> str:
    """Render HTML with highlighted spam trigger words."""
    highlighted = text
    for t in sorted(triggers, key=len, reverse=True):
        pattern = re.compile(r"\b(" + re.escape(t) + r")\b", re.IGNORECASE)
        highlighted = pattern.sub(r'<span class="highlight-spam">\1</span>', highlighted)
    # Convert newlines to HTML breaks
    highlighted = highlighted.replace("\n", "<br>")
    return highlighted


# ==============================================================================
# Main Application
# ==============================================================================
def main() -> None:
    models = load_all_models()
    explainer = get_explainer()

    # Sidebar Controls
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/shield.png", width=70)
        st.markdown("### Model Configuration")

        selected_model_label = st.selectbox(
            "Select Classifier Model:",
            options=list(MODEL_DISPLAY_NAMES.keys()),
            index=0,
            help="Choose the algorithm used for making real-time predictions.",
        )
        selected_model_key = MODEL_DISPLAY_NAMES[selected_model_label]
        active_pipeline = models.get(selected_model_key)

        st.divider()
        st.markdown("### Quick Presets")
        selected_preset = st.selectbox("Load Example Message:", options=list(SAMPLE_PRESETS.keys()))

        st.divider()
        st.markdown("### Pipeline Architecture")
        st.markdown(
            """
            - **Cleaning**: Regex token normalization
            - **Lemmatization**: NLTK WordNet
            - **Features**: TF-IDF (1,2-grams) + 10 Handcrafted Indicators
            - **Serialization**: Scikit-Learn Joblib
            """
        )
        st.caption("College ML Project • Antigravity Build")

    # App Header
    st.markdown('<div class="main-header">🛡️ AI-Based Spam Email Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">End-to-end Machine Learning system for classifying emails & SMS messages with model comparisons and transparent explainability.</div>',
        unsafe_allow_html=True,
    )

    tab_live, tab_benchmarks, tab_about = st.tabs([
        "🔍 Live Message Classifier",
        "📊 Model Comparison & Benchmarks",
        "ℹ️ System Architecture",
    ])

    # --------------------------------------------------------------------------
    # Tab 1: Live Message Classifier
    # --------------------------------------------------------------------------
    with tab_live:
        st.subheader("Analyze Email or SMS Text")

        default_text = SAMPLE_PRESETS.get(selected_preset, "")
        user_input = st.text_area(
            "Paste your email / message text below:",
            value=default_text,
            height=160,
            placeholder="e.g. Subject: You won $10,000 cash! Claim your prize now...",
        )

        col_btn1, col_btn2 = st.columns([1, 5])
        with col_btn1:
            analyze_clicked = st.button("🚀 Analyze Message", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("Clear Text", use_container_width=False):
                user_input = ""
                st.rerun()

        if (analyze_clicked or default_text) and user_input.strip():
            # Run inference using the active pipeline
            prob_spam = 0.0
            prediction = "HAM"
            if active_pipeline is not None:
                try:
                    prob_spam = float(active_pipeline.predict_proba([user_input])[0, 1])
                    prediction = "SPAM" if prob_spam >= 0.5 else "HAM"
                except Exception:
                    pred_val = active_pipeline.predict([user_input])[0]
                    prediction = "SPAM" if pred_val == 1 else "HAM"
                    prob_spam = 1.0 if prediction == "SPAM" else 0.0

            # Get explainability payload
            exp_data = explainer.explain_text(user_input)

            # --- Results Banner ---
            st.markdown("---")
            res_col1, res_col2, res_col3, res_col4 = st.columns([2, 2, 2, 2])

            with res_col1:
                st.markdown("**Prediction Verdict**")
                if prediction == "SPAM":
                    st.markdown('<div class="badge-spam">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-ham">✅ LEGITIMATE (HAM)</div>', unsafe_allow_html=True)

            with res_col2:
                st.markdown("**Spam Probability**")
                st.progress(prob_spam)
                st.write(f"**{prob_spam * 100:.2f}%** confidence")

            with res_col3:
                st.markdown("**Risk Categorization**")
                risk = exp_data["risk_level"]
                risk_colors = {
                    "HIGH RISK": "🔴",
                    "MODERATE RISK": "🟠",
                    "LOW RISK": "🟡",
                    "SAFE": "🟢",
                }
                st.markdown(f"### {risk_colors.get(risk, '⚪')} {risk}")

            with res_col4:
                st.markdown("**Evaluated By**")
                st.write(f"**{selected_model_label.split(' (')[0]}**")

            # --- Explainability Cards ---
            st.markdown("---")
            col_exp1, col_exp2 = st.columns([1, 1])

            with col_exp1:
                st.markdown("### 🏷️ Detected Spam Trigger Keywords")
                triggers = exp_data.get("detected_triggers", [])
                if triggers:
                    tags_html = "".join([f'<span class="trigger-tag">{t}</span>' for t in triggers])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.info("No overt spam trigger keywords detected.")

                st.markdown("#### Annotated Message Preview")
                annotated_html = render_highlighted_text(user_input, triggers)
                st.markdown(
                    f'<div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:12px; border-radius:8px; font-size:0.95rem; max-height:200px; overflow-y:auto;">{annotated_html}</div>',
                    unsafe_allow_html=True,
                )

            with col_exp2:
                st.markdown("### 📊 Structural Handcrafted Metrics")
                hc = exp_data.get("handcrafted_metrics", {})
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Uppercase Ratio", f"{hc.get('caps_ratio', 0.0) * 100:.1f}%")
                    st.metric("Exclamation Marks", int(hc.get('exclamation_count', 0)))
                with metric_col2:
                    st.metric("Link / URL Count", int(hc.get('url_count', 0)))
                    st.metric("Currency Symbols", int(hc.get('dollar_count', 0)))
                with metric_col3:
                    st.metric("Spam Trigger Count", int(hc.get('trigger_words_count', 0)))
                    st.metric("Digit Ratio", f"{hc.get('digit_ratio', 0.0) * 100:.1f}%")

            # --- Feature Contribution Breakdown ---
            st.markdown("---")
            with st.expander("🔬 Detailed Model Explainability: Feature Attribution ($x_i \\cdot w_i$)", expanded=True):
                st.caption(
                    "Transparent feature contributions based on Logistic Regression weights. "
                    "Positive values drive the spam verdict; negative values push towards ham."
                )

                col_sig1, col_sig2 = st.columns(2)
                with col_sig1:
                    st.markdown("#### 🚩 Top Spam-Indicative Signals (+)")
                    spam_signals = exp_data.get("top_spam_signals", [])
                    if spam_signals:
                        df_spam_sig = pd.DataFrame(spam_signals)[["feature", "contribution"]]
                        st.dataframe(df_spam_sig, use_container_width=True, hide_index=True)
                    else:
                        st.write("None active.")

                with col_sig2:
                    st.markdown("#### 🛡️ Top Ham-Indicative Signals (-)")
                    ham_signals = exp_data.get("top_ham_signals", [])
                    if ham_signals:
                        df_ham_sig = pd.DataFrame(ham_signals)[["feature", "contribution"]]
                        st.dataframe(df_ham_sig, use_container_width=True, hide_index=True)
                    else:
                        st.write("None active.")

    # --------------------------------------------------------------------------
    # Tab 2: Model Comparison & Benchmarks
    # --------------------------------------------------------------------------
    with tab_benchmarks:
        st.subheader("Model Performance Benchmark & Evaluation Suite")
        st.markdown(
            "Comparison of 5 machine learning models trained on the unified NLP and handcrafted feature pipeline, "
            "evaluated on a held-out test split."
        )

        benchmark_path = "reports/test_benchmark.json"
        if os.path.exists(benchmark_path):
            with open(benchmark_path, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
            df_bench = pd.DataFrame(bench_data["summary"])
            df_bench["Model"] = df_bench["Model"].apply(lambda x: x.replace("_", " ").title())

            st.markdown("#### 🏆 Test Set Performance Leaderboard")
            st.dataframe(
                df_bench.style.highlight_max(
                    subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                    color="#DCFCE7",
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")
        st.markdown("#### 📈 Diagnostic Evaluation Plots")

        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            if os.path.exists("reports/figures/roc_curves.png"):
                st.image("reports/figures/roc_curves.png", caption="Combined Multi-Model ROC Curves", use_container_width=True)

        with fig_col2:
            if os.path.exists("reports/figures/metrics_comparison.png"):
                st.image("reports/figures/metrics_comparison.png", caption="Comparative Metrics Bar Chart", use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🎯 Confusion Matrices Across All Models")
        if os.path.exists("reports/figures/confusion_matrices.png"):
            st.image("reports/figures/confusion_matrices.png", caption="Held-out Test Confusion Matrices", use_container_width=True)

    # --------------------------------------------------------------------------
    # Tab 3: System Architecture
    # --------------------------------------------------------------------------
    with tab_about:
        st.subheader("System Design & Architecture")
        st.markdown(
            """
            ### Project Overview
            This system was engineered as an end-to-end college machine learning project showcasing modern NLP and machine learning engineering practices:

            1. **Preprocessing Layer**:
               - Normalization of email headers, HTML tags, URLs (`httpaddr`), currency values (`dollar`), and numerical digits (`number`).
               - Stopword pruning with negation retention (`not`, `no`, `never`).
               - NLTK WordNet lemmatization.

            2. **Feature Engineering Layer**:
               - **TF-IDF**: Sublinear word unigrams & bigrams (`ngram_range=(1, 2)`).
               - **Handcrafted Statistics**: Capitalization ratio, URL count, email count, currency symbols, exclamation marks, question marks, and spam trigger frequency.
               - **Unified Pipeline**: Combined via `scipy.sparse.hstack` with non-negative `MaxAbsScaler`.

            3. **Multi-Model Suite**:
               - **Multinomial Naive Bayes**: Fast probabilistic baseline.
               - **Logistic Regression**: High-accuracy regularized model with transparent log-odds coefficients.
               - **Linear SVM**: Maximum-margin linear hyperplane with Platt scaling calibration.
               - **Random Forest & XGBoost**: Non-linear tree ensembles capturing non-linear interactions.

            4. **Explainability Engine**:
               - Decomposes predictions into active word and structural metric contributions ($x_i \\cdot w_i$).
            """
        )


if __name__ == "__main__":
    main()
