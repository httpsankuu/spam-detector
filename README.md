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
├── app/                      # Streamlit web application & dashboards
├── data/
│   ├── raw/                  # Raw downloaded datasets (Enron, SMS Spam, etc.)
│   └── processed/            # Cleaned and stratified train/val/test splits
├── models/                   # Serialized ML pipelines (.joblib) and metadata
├── notebooks/                # Exploratory Data Analysis & step-by-step walkthroughs
├── src/                      # Core modular Python packages
│   ├── __init__.py
│   ├── data/                 # Dataset downloaders, loaders, and splitters
│   ├── preprocessing/        # Text cleaners, tokenizers, and lemmatizer transformers
│   ├── features/             # TF-IDF extractors and handcrafted feature union
│   ├── models/               # Model training harnesses and predictors
│   └── utils/                # Environment verification and helper scripts
│       └── setup_env.py      # Automated health check & NLTK corpus bootstrap
├── .gitignore                # Git exclusions (virtualenvs, cache, binary models)
├── AGENTS.md                 # Project agent instructions & development conventions
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

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
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

---

## 🗺 Development Roadmap

- [x] **Phase 1: Environment & Project Foundation** — Project scaffolding, `requirements.txt`, and automated dependency/NLTK verification utility.
- [ ] **Phase 2: Data Ingestion & NLP Preprocessing Pipeline** — Dataset loaders (Email & SMS), stratified splits, and text cleaning/lemmatization transformer.
- [ ] **Phase 3: Feature Engineering & Feature Union** — Word n-gram TF-IDF vectorizer combined with handcrafted indicator extractors (links, caps ratio, trigger keywords).
- [ ] **Phase 4: Model Training Harness & Persistence** — Step-by-step training and serialization of Multinomial Naive Bayes, Logistic Regression, Linear SVM, and Random Forest / XGBoost.
- [ ] **Phase 5: Evaluation Suite & Explainability Engine** — Multi-metric benchmark (Precision, Recall, F1, ROC-AUC), confusion matrices/ROC curves, and top keyword explainability.
- [ ] **Phase 6: Interactive Streamlit Web Application** — Interactive demo with live message testing, confidence scores, keyword highlights, and comparative charts.

---

## 📄 License
This project is licensed under the MIT License.
