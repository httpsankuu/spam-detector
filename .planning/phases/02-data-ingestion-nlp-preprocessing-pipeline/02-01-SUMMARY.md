---
phase: 02-data-ingestion-nlp-preprocessing-pipeline
plan: 01
subsystem: data-ingestion
tags:
  - dataset-loader
  - email-ingestion
  - sms-ingestion
  - stratified-split
requires:
  - phase: 01-environment-project-foundation
provides:
  - standardized-email-loader
  - standardized-sms-loader
  - stratified-data-splitter
affects:
  - 02-02-PLAN
  - phase-03-feature-engineering
tech-stack:
  added:
    - pandas
    - scikit-learn.model_selection.train_test_split
  patterns:
    - standardized-dataset-schema
    - stratified-proportional-splitting
key-files:
  created:
    - src/data/email_loader.py
    - src/data/sms_loader.py
    - src/data/split_data.py
key-decisions:
  - "Standardized on uniform schema: ['text', 'label'] where 1=spam, 0=ham across both email and SMS"
  - "Embedded realistic starter corpora for email and SMS to ensure immediate out-of-the-box offline testing"
  - "Stratified partitioning with fixed random seed (42) to preserve class balance across train/val/test splits"
requirements-completed:
  - DATA-01
  - DATA-02
  - DATA-03
coverage:
  - deliverable: "Email dataset loader & schema normalization"
    verification:
      kind: "command"
      ref: "python -m src.data.email_loader --sample"
      status: "pass"
    human_judgment: false
  - deliverable: "SMS dataset loader with UCI support"
    verification:
      kind: "command"
      ref: "python -m src.data.sms_loader --sample"
      status: "pass"
    human_judgment: false
  - deliverable: "Stratified dataset partitioning utility"
    verification:
      kind: "command"
      ref: "python -m src.data.split_data --input data/raw/sample_emails.csv"
      status: "pass"
    human_judgment: false
duration: "4 min"
completed: "2026-09-06"
---

# Phase 2 Plan 01: Dataset Ingestion & Partitioning Summary

Implemented robust dataset ingestion, schema normalization, and stratified partitioning modules for both email and SMS spam corpora.

## Accomplishments

1. **Email Dataset Loader (`DATA-01`, `DATA-02`)**:
   - Built `EmailDataLoader` in `src/data/email_loader.py` supporting arbitrary CSV files, raw directory ingestion (Enron/SpamAssassin layout), and an embedded sample corpus of 30 realistic spam/ham emails.
   - Enforced standardized schema `['text', 'label']` with integer labels (1=spam, 0=ham), null cleaning, and deduplication.

2. **SMS Dataset Loader (`DATA-03`)**:
   - Built `SMSDataLoader` in `src/data/sms_loader.py` supporting UCI tab-separated format, automated mirror archive retrieval, and starter SMS collections.
   - Normalized outputs into identical standard schema.

3. **Stratified Train/Val/Test Splitter (`DATA-02`)**:
   - Implemented `split_dataset` and `save_splits` in `src/data/split_data.py` using `sklearn.model_selection.train_test_split` with `stratify=y`.
   - Verified row preservation and consistent class distribution across `train.csv`, `val.csv`, and `test.csv`.

## Verification Results

- `python -m src.data.email_loader --sample`: PASSED (30 sample emails generated)
- `python -m src.data.sms_loader --sample`: PASSED (16 sample SMS generated)
- `python -m src.data.split_data --input data/raw/sample_emails.csv`: PASSED (Train: 21 rows, Val: 3 rows, Test: 6 rows)

## Deviations from Plan

None - plan executed exactly as written.

## Next Steps

Wave 1 complete. Proceed to Wave 2: **Plan 02-02 (NLP cleaning, tokenization, stopword removal, and lemmatization pipeline transformer)**.
