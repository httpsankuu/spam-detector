# AI-Based Spam Email Detection using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8+-yellow.svg)](https://www.nltk.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end machine learning system that accurately classifies emails (and SMS messages) as **Spam** or legitimate **Ham**. Developed as a comprehensive academic ML capstone project, this system features a modular NLP preprocessing pipeline, benchmarks multiple classical and ensemble algorithms (Naive Bayes, Logistic Regression, Linear SVM, Random Forest, and XGBoost), provides prediction explainability, and hosts an interactive Streamlit web application.

---

## 📌 Key Objectives

- **Robust NLP Pipeline**: Custom text cleaning, tokenization, stopword removal, and lemmatization tailored for unstructured email and SMS text.
- **Hybrid Feature Engineering**: TF-IDF unigram and bigram representation combined with domain-specific handcrafted spam signals (link count, capitalization ratio, spam keyword frequency).
- **Multi-Model Benchmark**: Rigorous comparative evaluation of classical probabilistic models, linear classifiers, and gradient-boosted decision trees.
- **Model Explainability**: Transparent insights into top predictive tokens influencing spam vs. ham classifications.
- **Interactive Web Demo**: Clean Streamlit web interface for single-message testing, confidence assessment, and comparative performance dashboard.

---

## 📂 Project Architecture

```
spam-detector/
├── app/                      # Streamlit web application & interactive dashboards
│   └── streamlit_app.py      # Main web application entry point
├── config/                   # Centralized configuration & hyperparameters
│   ├── __init__.py
│   └── config.py             # Paths, seeds, ratios, and model hyperparameters
├── data/
│   ├── raw/                  # Raw starter datasets (sample_emails.csv, sample_sms.csv)
│   └── processed/            # Cleaned and stratified splits (train.csv, val.csv, test.csv)
├── models/                   # Serialized ML pipelines (.joblib) and model_metadata.json
├── notebooks/                # Exploratory Data Analysis & step-by-step walkthroughs
├── reports/                  # Benchmark metrics JSON & publication figures (300 DPI)
│   ├── test_benchmark.json   # Full held-out test evaluation scores
│   └── figures/              # Confusion matrices, ROC curves, metrics bar charts
├── src/                      # Core modular Python packages
│   ├── __init__.py
│   ├── data/                 # Dataset downloaders, loaders, and splitters
│   ├── preprocessing/        # Text cleaners, tokenizers, and lemmatizer transformers
│   ├── features/             # TF-IDF n-grams, handcrafted features & composite pipeline
│   ├── models/               # Model wrappers (NB, LR, SVM, RF, XGBoost) and training harness
│   ├── evaluation/           # Benchmarking metrics, plotting, and explainability engine
│   └── utils/                # Environment verification and helper scripts
│       └── setup_env.py      # Automated health check & NLTK corpus bootstrap
├── tests/                    # Automated pytest test suite
│   └── test_pipeline.py      # Unit and integration tests for all modules
├── .gitignore                # Git exclusions (virtualenvs, cache, binary models)
├── AGENTS.md                 # Project architecture & development conventions
├── README.md                 # Project documentation and setup guide
└── requirements.txt          # Pinned Python package dependencies
```

---

## 🛠 Tech Stack

| Domain | Libraries / Tools |
|---|---|
| **Language & Runtime** | Python 3.10+ (standard `venv` + `pip`) |
| **Data Manipulation** | `pandas`, `numpy` |
| **Natural Language Processing** | `nltk` (WordNetLemmatizer, punkt tokenizers, stopwords) |
| **Machine Learning** | `scikit-learn` (MultinomialNB, LogisticRegression, LinearSVC, RandomForest) |
| **Ensemble Gradient Boosting** | `xgboost` |
| **Visualizations** | `matplotlib`, `seaborn` |
| **Web Interface** | `streamlit` |
| **Model Serialization** | `joblib` |
| **Testing** | `pytest` |

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/httpsankuu/spam-detector.git
cd spam-detector
```

### 2. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Automated Setup & Verification
Run the built-in health-check utility to verify that all dependencies are installed and automatically bootstrap required NLTK corpora (`punkt`, `stopwords`, `wordnet`, `omw-1.4`):
```bash
python -m src.utils.setup_env
```

### 5. Launch the Interactive Web App
Launch the interactive Streamlit interface:
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### 6. Retrain Models & Regenerate Figures (Optional)
To retrain all 5 models and re-evaluate benchmarks:
```bash
# 1. Train and serialize all 5 models
python -m src.models.trainer

# 2. Evaluate held-out test split and regenerate diagnostic plots
python -m src.evaluation.plots

# 3. Run automated tests
pytest -v
```

---

## 🗺 Development Roadmap

- [x] **Phase 1: Environment & Project Foundation** — Project scaffolding, `requirements.txt`, and automated dependency/NLTK verification utility.
- [x] **Phase 2: Data Ingestion & NLP Preprocessing Pipeline** — Dataset loaders (Email & SMS), stratified splits, and text cleaning/lemmatization transformer.
- [x] **Phase 3: Feature Engineering & Feature Union** — Word n-gram TF-IDF vectorizer combined with handcrafted indicator extractors (links, caps ratio, trigger keywords).
- [x] **Phase 4: Model Training Harness & Persistence** — Step-by-step training and serialization of Multinomial Naive Bayes, Logistic Regression, Linear SVM, and Random Forest / XGBoost.
- [x] **Phase 5: Evaluation Suite & Explainability Engine** — Multi-metric benchmark (Precision, Recall, F1, ROC-AUC), confusion matrices/ROC curves, and top keyword explainability.
- [x] **Phase 6: Interactive Streamlit Web Application** — Interactive demo with live message testing, confidence scores, keyword highlights, and comparative charts.

---

## 📄 License
This project is licensed under the MIT License.
