# Roadmap: AI-Based Spam Email Detection using Machine Learning

## Overview

This roadmap delivers a complete college machine learning project following a structured, horizontal layers approach. We start by building a rock-solid environment and automated dataset ingestion and NLP preprocessing pipeline. Next, we engineer TF-IDF and handcrafted statistical features, train and persist multiple ML models (Naive Bayes baseline, Logistic Regression, Linear SVM, and Random Forest / XGBoost) with modular explanations, build a rigorous evaluation and explainability suite, and culminate in an interactive Streamlit web application for real-time testing and presentation.

## Phases

- [x] **Phase 1: Environment & Project Foundation** - Project directory structure, virtualenv requirements, and dependency verification. (completed 2026-09-06)
- [x] **Phase 2: Data Ingestion & NLP Preprocessing Pipeline** - Dataset loaders (Email & SMS), stratified splits, and text cleaning/lemmatization transformer. (completed 2026-09-06)
- [x] **Phase 3: Feature Engineering & Feature Union** - Word n-gram TF-IDF vectorizer and handcrafted indicator feature union. (completed 2026-09-06)
- [x] **Phase 4: Model Training Harness & Persistence** - Step-by-step training of MNB, Logistic Regression, Linear SVM, and XGBoost/RF with serialization. (completed 2026-09-06)
- [ ] **Phase 5: Evaluation Suite & Explainability Engine** - Comprehensive multi-metric benchmark, ROC/confusion plots, and keyword explainability.
- [ ] **Phase 6: Interactive Streamlit Web Application** - Interactive UI for single-message spam testing, confidence score, explainability, and comparative metrics dashboard.


## Phase Details

### Phase 1: Environment & Project Foundation

**Goal**: Establish a modular project directory tree, configure dependencies, and create an automated environment health-check utility.
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, ENV-02
**Success Criteria** (what must be TRUE):

  1. Directory structure (`src/`, `data/`, `notebooks/`, `models/`, `app/`) exists and is cleanly organized.
  2. Python dependencies install smoothly via `requirements.txt` without package conflicts.
  3. Environment verification script validates packages and downloads required NLTK resources automatically.

**Plans**: TBD

Plans:

- [x] 01-01-PLAN.md
- [x] 01-01: Project scaffolding, `requirements.txt`, and environment verification script.

### Phase 2: Data Ingestion & NLP Preprocessing Pipeline

**Goal**: Ingest email and SMS datasets, standardize data schema with stratified splits, and build an end-to-end text preprocessing pipeline.
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-02, DATA-03, NLP-01, NLP-02, NLP-03
**Success Criteria** (what must be TRUE):

  1. Data loaders successfully load email and SMS datasets into a normalized pandas DataFrame with standard columns (`text`, `label`).
  2. Dataset split produces stratified train/validation/test partitions preserving spam-to-ham class distribution.
  3. Text cleaner strips HTML, headers, URLs, and punctuation, producing normalized, lemmatized tokens via a scikit-learn compatible transformer.

**Plans**: TBD

Plans:

- [x] 02-01-PLAN.md
- [x] 02-02-PLAN.md

**Wave 1**

- [x] 02-01: Dataset loaders for email and SMS datasets with stratified splitting.

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 02-02: NLP cleaning, tokenization, stopword removal, and lemmatization pipeline transformer.

### Phase 3: Feature Engineering & Feature Union

**Goal**: Implement TF-IDF n-gram vectorization alongside handcrafted spam features (links, caps ratio, trigger keywords) in a unified pipeline.
**Depends on**: Phase 2
**Requirements**: FEAT-01, FEAT-02, FEAT-03
**Success Criteria** (what must be TRUE):

  1. TF-IDF vectorizer extracts unigrams and bigrams with configurable sublinear term frequency and vocabulary caps.
  2. Handcrafted feature extractor accurately calculates link counts, uppercase character ratios, and trigger keyword frequencies.
  3. Unified FeatureUnion/ColumnTransformer merges text and numeric features into a unified matrix ready for model estimators.

**Plans**: TBD

Plans:

- [x] 03-01: TF-IDF n-gram vectorizer configuration and handcrafted feature extractor.
- [x] 03-02: Unified FeatureUnion pipeline integration and transformed matrix caching.

### Phase 4: Model Training Harness & Persistence

**Goal**: Train, tune, and serialize multiple ML classifiers (Multinomial Naive Bayes, Logistic Regression, Linear SVM, and Random Forest / XGBoost) with modular step-by-step training scripts.
**Depends on**: Phase 3
**Requirements**: MODL-01, MODL-02, MODL-03, MODL-04, MODL-05
**Success Criteria** (what must be TRUE):

  1. Multinomial Naive Bayes trains successfully as a fast, calibrated baseline classifier.
  2. Logistic Regression trains with regularized weights and outputs well-calibrated class probabilities.
  3. Linear Support Vector Machine (LinearSVC) trains with high margin separation on high-dimensional text features.
  4. Tree ensemble (Random Forest / XGBoost) trains successfully and captures non-linear feature interactions.
  5. All trained models and feature extractors are serialized cleanly to `models/` with metadata.

**Plans**: TBD

Plans:

- [x] 04-01: Baseline Multinomial Naive Bayes and Logistic Regression training module.
- [x] 04-02: Linear SVM and Random Forest / XGBoost training module and model persistence pipeline.

### Phase 5: Evaluation Suite & Explainability Engine

**Goal**: Benchmark all models across Precision, Recall, F1-Score, and ROC-AUC, generate visualization artifacts, and extract top predictive feature weights for explainability.
**Depends on**: Phase 4
**Requirements**: EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):

  1. Evaluation script produces comprehensive metric comparison table on held-out test data.
  2. Confusion matrix and ROC-AUC curves are computed, rendered, and saved as publication-ready figures.
  3. Explainability engine extracts top spam-indicative and ham-indicative tokens for any trained linear model and specific test email.

**Plans**: TBD

Plans:

- [ ] 05-01: Multi-metric benchmarking harness, confusion matrix, and ROC-AUC plot generator.
- [ ] 05-02: Model explainability and feature contribution extraction module.

### Phase 6: Interactive Streamlit Web Application

**Goal**: Build and launch an intuitive Streamlit web app providing live spam inference, confidence gauges, keyword explainability, and comparative performance dashboard.
**Depends on**: Phase 5
**Requirements**: UI-01, UI-02, UI-03, UI-04
**Success Criteria** (what must be TRUE):

  1. User can paste any email or SMS text into the UI and receive an immediate spam vs. ham verdict.
  2. Interface displays prediction badge, probability score bar, and risk categorization.
  3. UI highlights detected spam trigger words and structural signals from the input text.
  4. Model Comparison dashboard tab presents comparative metric charts and confusion matrix images.

**Plans**: TBD

Plans:

- [ ] 06-01: Streamlit single-message classification interface with confidence gauge and explainability.
- [ ] 06-02: Model comparison tab, visualization embedding, and end-to-end demo polish.

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Environment & Project Foundation | 1/1 | Complete   | 2026-09-06 |
| 2. Data Ingestion & NLP Preprocessing Pipeline | 2/2 | Complete   | 2026-09-06 |
| 3. Feature Engineering & Feature Union | 2/2 | Complete   | 2026-09-06 |
| 4. Model Training Harness & Persistence | 2/2 | Complete   | 2026-09-06 |
| 5. Evaluation Suite & Explainability Engine | 0/2 | Not started | - |
| 6. Interactive Streamlit Web Application | 0/2 | Not started | - |
