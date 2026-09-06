# AI-Based Spam Email Detection using Machine Learning

## What This Is

An end-to-end college machine learning system that accurately classifies emails (and SMS messages) as spam or legitimate ("ham"). The system features a robust natural language processing pipeline, compares multiple ML algorithms (Naive Bayes, Logistic Regression, Linear SVM, and Random Forest / XGBoost) with explainability, and presents predictions and metrics via an interactive Streamlit web demo.

## Core Value

Accurate, well-evaluated, and explainable spam email classification with an intuitive interactive demo and clear step-by-step model comparisons.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Project directory structure and environment configuration (Python, scikit-learn, XGBoost, NLTK/spaCy, Streamlit)
- [ ] Dataset acquisition and loader for email (e.g., Enron/SpamAssassin/Kaggle email corpus) and SMS (UCI SMS Spam Collection)
- [ ] Text cleaning and NLP preprocessing pipeline (HTML stripping, lowercasing, tokenization, stopword removal, lemmatization)
- [ ] Feature engineering pipeline (TF-IDF unigrams & bigrams, plus hand-crafted features: link count, spam trigger words, capitalization ratio)
- [ ] Model training and benchmarking harness:
  - Multinomial Naive Bayes (baseline)
  - Logistic Regression
  - Linear SVM
  - Random Forest / XGBoost
- [ ] Comprehensive model evaluation suite:
  - Precision, Recall, F1-Score, Accuracy
  - Confusion Matrix visualization
  - ROC-AUC curves and comparative performance tables
- [ ] Prediction explainability component (identifying top keywords and features influencing spam vs. ham decisions)
- [ ] Interactive Streamlit web application:
  - Text input for custom email/SMS testing
  - Instant classification with confidence score
  - Feature & keyword breakdown for prediction
  - Model metrics and comparative charts dashboard
- [ ] Secondary / Stretch: SMS classification support and basic URL/phishing indicator detection

### Out of Scope

- Heavy deep learning or LLM fine-tuning (e.g., fine-tuning BERT/RoBERTa) — classical ML and gradient boosting are prioritized for explainability, fast training, and college project evaluation constraints.
- Real-time mailbox background daemon or production email server integration (IMAP/SMTP hook) — focus is on ML classification and web demonstration.
- Full-fledged enterprise multi-user authentication — simple local web demo suffices.

## Context

- Target audience: Academic evaluation, college project viva, and portfolio demonstration.
- Text data is naturally imbalanced (legitimate ham significantly outnumbers spam), making Precision, Recall, and F1-Score crucial evaluation criteria rather than raw accuracy.
- Explainability is vital to demonstrate ML reasoning during presentations (showing why an email was flagged as spam).

## Constraints

- **Tech Stack**: Python 3.10+, pandas, numpy, scikit-learn, XGBoost, NLTK or spaCy, matplotlib, seaborn, Streamlit.
- **Workflow / Pedagogy**: Step-by-step modular development — pipeline first, individual model walk-throughs next, then web application.
- **Environment**: Windows OS environment, runnable locally without specialized GPU acceleration.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Classical ML + XGBoost over Large LLMs | Fits college project requirements, fast training, lightweight deployment, transparent feature importance | — Pending |
| TF-IDF + Handcrafted Features | Combines statistical n-gram patterns with domain-specific spam signals (URL counts, caps ratio, trigger words) | — Pending |
| Streamlit for Demonstration | Rapid prototyping, native Python integration, easy visualization for confusion matrices and explainability | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-09-06 after initialization*
