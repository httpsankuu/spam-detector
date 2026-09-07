# AI-Based Spam Email Detection using Machine Learning

## What This Is

An end-to-end college machine learning system that accurately classifies emails (and SMS messages) as spam or legitimate ("ham"). The system features a robust natural language processing pipeline, compares multiple ML algorithms (Naive Bayes, Logistic Regression, Linear SVM, and Random Forest / XGBoost) with explainability, and presents predictions and metrics via an interactive Streamlit web demo.

## Core Value

Accurate, well-evaluated, and explainable spam email classification with an intuitive interactive demo and clear step-by-step model comparisons.

## Current Milestone

*(Planning Next Milestone)*

## Requirements

### Validated

**v1.0 MVP**


- [x] Project directory structure and environment configuration (Python, scikit-learn, XGBoost, NLTK, Streamlit) — Phase 1
- [x] Dataset acquisition and loader for email and SMS (UCI SMS Spam Collection) — Phase 2
- [x] Text cleaning and NLP preprocessing pipeline (HTML stripping, lowercasing, tokenization, stopword removal, lemmatization) — Phase 2
- [x] Feature engineering pipeline (TF-IDF unigrams & bigrams, hand-crafted features) — Phase 3
- [x] Model training and benchmarking harness (Naive Bayes, LR, LinearSVC, RF, XGBoost) — Phase 4
- [x] Comprehensive model evaluation suite (Precision, Recall, F1, ROC-AUC, confusion matrices, plots) — Phase 5
- [x] Prediction explainability component (keyword + feature attribution) — Phase 5
- [x] Interactive Streamlit web application with live inference, confidence score, explainability, and benchmark dashboard — Phase 6

**v2.0 Hardened and Production-Ready**

- [x] Bootstrap script (`scripts/bootstrap.py`) that downloads real data → splits → trains all 5 models end-to-end so fresh clones work immediately - v2.0 (FIX-01)
- [x] Real UCI SMS dataset integration (~5.5k messages) replacing the 30-row synthetic training set; re-generate benchmark JSON and figures with honest numbers - v2.0 (FIX-02)
- [x] Fix `SMSDataLoader.download_uci_dataset()` fallback to return the actual bundled sample file instead of a non-existent path - v2.0 (FIX-03)
- [x] Wire `config/config.py` constants (hyperparams, NLP flags, paths) through all model factories and `TextPreprocessor` so the "centralized configuration" is real - v2.0 (FIX-04)
- [x] Remove all `try/except ImportError` config fallback blocks in trainer, metrics, plots, explainability, split_data — replace with direct imports - v2.0 (FIX-05)
- [x] Fix `TFIDFExtractor.fit_transform` to apply the same `min_df`/`max_df` small-dataset guards as `fit` + `transform` - v2.0 (FIX-06)
- [x] Fix Streamlit "Clear Text" button using `key=` on `st.text_area` and `st.session_state` manipulation - v2.0 (FIX-07)
- [x] Add graceful app startup error handling — when models dir is missing, show "run bootstrap first" message instead of crashing - v2.0 (FIX-08)
- [x] Route explainability to the selected model (LR, SVM, NB) rather than always using Logistic Regression regardless of user's choice - v2.0 (FIX-09)
- [x] Pin all versions in `requirements.txt` based on the current working install - v2.0 (FIX-10)
- [x] Scope SSL verification workaround in `setup_env.py` to only the NLTK download call - v2.0 (FIX-11)
- [x] Strip `y_pred`/`y_prob` arrays from `test_benchmark.json` — keep summary metrics only - v2.0 (FIX-12)
- [x] Fix model metadata to use cross-platform forward-slash paths - v2.0 (FIX-13)
- [x] Remove unused typing imports across model modules; fix `digit_ratio` magic constant; swap sidebar remote icon for local emoji; fix `EMAIL_HEADER_RE` anchor; fix `os.makedirs` edge case - v2.0 (FIX-14)

### Active

(None currently)

### Out of Scope

- Heavy deep learning or LLM fine-tuning — classical ML and gradient boosting are prioritized for explainability, fast training, and college project evaluation constraints.
- Real-time mailbox background daemon or production email server integration (IMAP/SMTP hook).
- Full-fledged enterprise multi-user authentication.
- CI/CD pipeline (GitHub Actions) — out of scope for a local college project.

## Context

- Target audience: Academic evaluation, college project viva, and portfolio demonstration.
- Text data is naturally imbalanced; Precision, Recall, and F1-Score are the primary evaluation criteria.
- Explainability is vital during presentations.
- v2.0 focus: correctness and credibility — honest benchmarks on a real dataset, zero crashes on fresh clone.

## Constraints

- **Tech Stack**: Python 3.10+, pandas, numpy, scikit-learn, XGBoost, NLTK, matplotlib, seaborn, Streamlit.
- **Workflow / Pedagogy**: Step-by-step modular development — no new architectural layers, only bug fixes and wiring improvements.
- **Environment**: Windows OS environment, runnable locally without specialized GPU acceleration.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------| 
| Classical ML + XGBoost over Large LLMs | Fits college project requirements, fast training, lightweight deployment, transparent feature importance | Validated v1.0 |
| TF-IDF + Handcrafted Features | Combines statistical n-gram patterns with domain-specific spam signals | Validated v1.0 |
| Streamlit for Demonstration | Rapid prototyping, native Python integration | Validated v1.0 |
| UCI SMS Spam Collection as real training data | ~5.5k messages with balanced labels; publicly available; canonical benchmark dataset | v2.0 |
| Bootstrap script over tracked artifacts | Regenerate-on-demand is better practice than committing binary model files | v2.0 |
| Pin requirements.txt versions | Reproducible installs; avoid future breakage from upstream updates | v2.0 |

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
*Last updated: 2026-09-07 after v2.0 milestone completion*
