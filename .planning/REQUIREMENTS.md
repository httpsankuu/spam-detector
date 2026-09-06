# Requirements: AI-Based Spam Email Detection using Machine Learning

**Defined:** 2026-09-06
**Core Value:** Accurate, well-evaluated, and explainable spam email classification with an intuitive interactive demo and clear step-by-step model comparisons.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Environment & Project Setup

- [x] **ENV-01**: Clean modular project structure (`src/`, `data/`, `notebooks/`, `models/`, `app/`) with `requirements.txt`
- [x] **ENV-02**: Dependency verification and NLTK resource downloader utility

### Dataset Ingestion & Preparation

- [x] **DATA-01**: Email dataset loader (supporting Enron / SpamAssassin / Kaggle spam corpora)
- [x] **DATA-02**: Standardized data schema (`text`, `label` where spam=1, ham=0) with stratified train/test split
- [x] **DATA-03**: SMS Spam Collection (UCI) dataset ingestion module

### Text Preprocessing & NLP Pipeline

- [x] **NLP-01**: Raw text cleaning (HTML tag removal, email headers/urls/punctuation handling, lowercasing)
- [x] **NLP-02**: Tokenization, stopword removal, and lemmatization (using NLTK/WordNetLemmatizer)
- [x] **NLP-03**: Reusable scikit-learn pipeline transformer for seamless training and inference

### Feature Engineering

- [x] **FEAT-01**: Word n-gram TF-IDF vectorization (unigrams and bigrams with configurable vocab limit)
- [x] **FEAT-02**: Handcrafted feature extractors (link/URL count, uppercase character ratio, spam trigger keyword frequency)
- [x] **FEAT-03**: Unified feature union / column transformer combining TF-IDF and dense handcrafted features

### Model Training & Persistence

- [x] **MODL-01**: Multinomial Naive Bayes classifier trained and tuned as baseline
- [x] **MODL-02**: Logistic Regression classifier trained with regularized loss
- [x] **MODL-03**: Linear Support Vector Machine (LinearSVC) classifier trained
- [x] **MODL-04**: Ensemble tree-based classifier (Random Forest / XGBoost) trained
- [x] **MODL-05**: Model serialization & persistence mechanism saving trained pipelines to disk

### Evaluation & Explainability

- [ ] **EVAL-01**: Multi-metric evaluation reporting Precision, Recall, F1-score, and Accuracy on held-out test split
- [ ] **EVAL-02**: Confusion matrices and ROC-AUC curve visualization generation
- [ ] **EVAL-03**: Top spam/ham feature contribution explainability module

### Interactive Streamlit Web Demo

- [ ] **UI-01**: Interactive single-message text tester for user-submitted emails/SMS
- [ ] **UI-02**: Real-time prediction display with spam/ham label, confidence score, and risk indicator
- [ ] **UI-03**: Highlighted explainability breakdown displaying detected spam keywords and feature metrics
- [ ] **UI-04**: Model benchmark comparison tab displaying comparative tables, confusion matrices, and ROC curves

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

- **EXT-01**: Live mailbox integration via IMAP to scan user inboxes
- **EXT-02**: Fine-tuned Transformer/BERT-based classifier comparison
- **EXT-03**: Automated daily retrain pipeline with drift monitoring

## Out of Scope

| Feature | Reason |
|---------|--------|
| Deep learning / heavy LLM fine-tuning | Classical ML and XGBoost are faster to train, lighter to deploy, and more transparent for college viva |
| Real-time email server daemon (SMTP/IMAP hook) | Project goal is ML modeling, analysis, and web demo |
| Enterprise authentication and multi-user database | Single-session local demo is sufficient for college presentation |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 1 | Complete |
| ENV-02 | Phase 1 | Complete |
| DATA-01 | Phase 2 | Complete |
| DATA-02 | Phase 2 | Complete |
| DATA-03 | Phase 2 | Complete |
| NLP-01 | Phase 2 | Complete |
| NLP-02 | Phase 2 | Complete |
| NLP-03 | Phase 2 | Complete |
| FEAT-01 | Phase 3 | Complete |
| FEAT-02 | Phase 3 | Complete |
| FEAT-03 | Phase 3 | Complete |
| MODL-01 | Phase 4 | Complete |
| MODL-02 | Phase 4 | Complete |
| MODL-03 | Phase 4 | Complete |
| MODL-04 | Phase 4 | Complete |
| MODL-05 | Phase 4 | Complete |
| EVAL-01 | Phase 5 | Pending |
| EVAL-02 | Phase 5 | Pending |
| EVAL-03 | Phase 5 | Pending |
| UI-01 | Phase 6 | Pending |
| UI-02 | Phase 6 | Pending |
| UI-03 | Phase 6 | Pending |
| UI-04 | Phase 6 | Pending |

**Coverage:**

- v1 requirements: 23 total
- Mapped to phases: 23
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-06*
*Last updated: 2026-09-06 after initial definition*
