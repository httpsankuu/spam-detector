# Project Presentation Summary — AI Spam Detector

## What It Does
Classifies email and SMS messages as Spam or Ham using 5 ML algorithms with explainable predictions and an interactive Streamlit interface.

## Architecture (Why This Design)
- **Preprocessing (`src/preprocessing/`)**: Regex normalization + NLTK lemmatization. Chosen over spaCy because it requires no GPU and is lightweight.
- **Features (`src/features/`)**: TF-IDF (1,2-grams) + 10 handcrafted indicators (caps ratio, URL count, currency symbols, trigger keywords). Hybrid approach improves over pure bag-of-words.
- **Models (`src/models/`)**: Naive Bayes (baseline), Logistic Regression (recommended — transparent weights), Linear SVM (max-margin), Random Forest & XGBoost (non-linear ensembles). Multi-model benchmark demonstrates rigorous evaluation.
- **Evaluation (`src/evaluation/`)**: Stratified split (`70/10/20`), confusion matrices, ROC curves, and feature-attribution explainability (`SpamExplainer`).
- **App (`app/streamlit_app.py`)**: Real-time inference with confidence gauges, risk badges, and annotated text highlights.

## Tools & Justifications
| Tool | Why Used |
|---|---|
| Python 3.10+ | Stable, widely taught |
| scikit-learn | Standard ML contracts (`fit`/`transform`/`predict`) |
| NLTK | Lightweight NLP (no GPU needed) |
| XGBoost | Gradient boosting for non-linear comparison |
| Streamlit | Rapid interactive UI without HTML/CSS |
| joblib | Fast model serialization |
| pytest | Automated regression tests |

## Results
- Best model: **Logistic Regression** (F1 ≈ 0.957, ROC-AUC ≈ 0.998)
- All 5 models evaluated and plotted in `reports/figures/`
- Interactive demo available locally with `streamlit run app/streamlit_app.py`

## Deployment (Manual Steps)
1. Push repository (with `models/` included) to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Select `app/streamlit_app.py`
4. Ensure `requirements.txt` is present (it is)
