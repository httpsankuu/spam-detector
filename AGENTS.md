<!-- GSD:project-start source:PROJECT.md -->

## Project

**AI-Based Spam Email Detection using Machine Learning**

An end-to-end college machine learning system that accurately classifies emails (and SMS messages) as spam or legitimate ("ham"). The system features a robust natural language processing pipeline, compares multiple ML algorithms (Naive Bayes, Logistic Regression, Linear SVM, and Random Forest / XGBoost) with explainability, and presents predictions and metrics via an interactive Streamlit web demo.

**Core Value:** Accurate, well-evaluated, and explainable spam email classification with an intuitive interactive demo and clear step-by-step model comparisons.

### Constraints

- **Tech Stack**: Python 3.10+, pandas, numpy, scikit-learn, XGBoost, NLTK or spaCy, matplotlib, seaborn, Streamlit.
- **Workflow / Pedagogy**: Step-by-step modular development — pipeline first, individual model walk-throughs next, then web application.
- **Environment**: Windows OS environment, runnable locally without specialized GPU acceleration.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:STACK.md -->

## Technology Stack

- **Runtime & Environment**: Python 3.10+ on Windows, virtualenv + pip.
- **Data & Numeric**: `pandas`, `numpy`, `scipy`.
- **Natural Language Processing**: `nltk` (WordNetLemmatizer, stopwords, punkt).
- **Machine Learning**: `scikit-learn` (MultinomialNB, LogisticRegression, LinearSVC, RandomForestClassifier, Pipeline, metrics), `xgboost` (XGBClassifier).
- **Visualization**: `matplotlib` (Agg headless backend), `seaborn`.
- **Web Application**: `streamlit`.
- **Artifact Serialization**: `joblib`.
- **Testing**: `pytest`.

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

- **Centralized Configuration**: All paths, seeds, thresholds, and hyperparameters are declared in `config/config.py`.
- **Standardized Schema**: Every dataset loader normalizes tabular outputs to `['text', 'label']` where 1=spam, 0=ham.
- **Feature Preservation**: Raw URLs, emails, currencies, and numbers are replaced with semantic tokens (`httpaddr`, `emailaddr`, `dollar`, `number`) during text cleaning.
- **Pipeline Architecture**: All components implement scikit-learn's `BaseEstimator` and `TransformerMixin` contracts so feature transformations and estimators bundle into single `.joblib` files.
- **Model Explainability**: Predictions are decomposed into word contributions and structural metric contributions ($x_i \cdot w_i$).
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

```
spam-detector/
├── app/streamlit_app.py      # Streamlit web UI (tabs: live classifier, benchmarks, architecture)
├── config/config.py          # Central configuration & parameters
├── data/
│   ├── raw/                  # Starter sample emails & SMS
│   └── processed/            # Stratified train, val, test splits
├── models/                   # Serialized ML pipelines (*.joblib) and model_metadata.json
├── reports/
│   ├── test_benchmark.json   # Multi-metric test evaluation scores
│   └── figures/              # confusion_matrices.png, roc_curves.png, metrics_comparison.png
├── src/
│   ├── data/                 # EmailDataLoader, SMSDataLoader, split_dataset
│   ├── preprocessing/        # clean_text, tokenize_and_lemmatize, TextPreprocessor
│   ├── features/             # TFIDFExtractor, HandcraftedFeatureExtractor, UnifiedFeaturePipeline
│   ├── models/               # NaiveBayes, LogisticRegression, SVM, Ensembles, trainer
│   ├── evaluation/           # ModelEvaluator, plot generators, SpamExplainer
│   └── utils/                # setup_env.py
└── tests/
    └── test_pipeline.py      # Automated pytest unit and integration test suite
```
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.agents/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
