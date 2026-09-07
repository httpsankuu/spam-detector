# Requirements — Milestone v2.0: Hardened and Production-Ready

**Milestone:** v2.0  
**Status:** Active  
**Date:** 2026-09-07  

---

## Reproducibility & Data (REPRO)

- [ ] **REPRO-01**: Developer can run a single `python scripts/bootstrap.py` command on a fresh clone and have all data downloaded, splits generated, and all 5 models trained and serialized with no manual steps.
- [ ] **REPRO-02**: The bootstrap script integrates the real UCI SMS Spam Collection dataset (~5.5k messages) as the primary training corpus, replacing the 30-row synthetic dataset.
- [ ] **REPRO-03**: After bootstrap, `pytest` passes all tests without requiring any additional manual setup.

## Data Loading (DATA)

- [ ] **DATA-01**: `SMSDataLoader.download_uci_dataset()` falls back to the actual bundled sample SMS CSV file (not a non-existent path) when the network download fails, so callers never receive a `FileNotFoundError`.

## Configuration (CFG)

- [ ] **CFG-01**: All model hyperparameters (`LR_C`, `NB_ALPHA`, `SVM_C`, `RF_N_ESTIMATORS`, `XGB_*`, `LR_MAX_ITER`) defined in `config/config.py` are consumed by their respective model factory functions.
- [ ] **CFG-02**: All NLP preprocessing flags (`MIN_WORD_LENGTH`, `REPLACE_URLS`, `REPLACE_EMAILS`, `REPLACE_CURRENCY`, `REPLACE_NUMBERS`) defined in `config/config.py` are consumed by `TextPreprocessor`.
- [ ] **CFG-03**: All split ratios and path constants (`TRAIN_RATIO`, `TRAIN_DATA_PATH`, etc.) defined in `config/config.py` are used by `split_data` instead of hardcoded strings.
- [ ] **CFG-04**: All `try/except ImportError` fallback blocks in `trainer.py`, `metrics.py`, `plots.py`, `explainability.py`, and `split_data.py` are replaced with direct `from config.config import ...` statements.

## Feature Engineering (FEAT)

- [ ] **FEAT-01**: `TFIDFExtractor.fit_transform()` applies the same small-dataset `min_df`/`max_df` guard logic as the separate `fit()` + `transform()` path so both routes produce identical vocabularies.

## Streamlit Application (UI)

- [ ] **UI-01**: The "Clear Text" button correctly resets the text area by storing text in `st.session_state` and using a `key=` parameter on `st.text_area`, so pressing the button visibly clears the input on rerun.
- [ ] **UI-02**: When the `models/` directory is absent or empty, the app displays a friendly informational message ("Run `python scripts/bootstrap.py` first") instead of raising an unhandled exception that crashes the app.
- [ ] **UI-03**: The Feature Attribution expander routes explainability to whichever linear model the user has selected (LR, SVM, or NB), not always Logistic Regression. For non-linear models (RF, XGBoost), a clear note explains that word-level attribution is only available for linear models.
- [ ] **UI-04**: The sidebar icon is served from a local emoji or inline SVG rather than a remote URL (`img.icons8.com`) so the app works fully offline.

## Code Quality & Portability (QUAL)

- [ ] **QUAL-01**: `requirements.txt` pins all package versions (e.g. `scikit-learn==1.9.0`) matching the current working install, with no bare ranges.
- [ ] **QUAL-02**: The SSL bypass in `setup_env.py` is scoped to the NLTK download call only (using a context manager or a local monkey-patch), not applied globally for the entire process lifetime.
- [ ] **QUAL-03**: `test_benchmark.json` stores only summary metrics per model (Accuracy, Precision, Recall, F1, ROC-AUC) — full `y_pred`/`y_prob` arrays are excluded to keep the file small on real datasets.
- [ ] **QUAL-04**: Model metadata `models_saved` paths use forward slashes (via `Path.as_posix()`) so the JSON is portable across Windows and Linux.
- [ ] **QUAL-05**: Unused typing imports removed from all model modules; `digit_ratio` computation documented; `EMAIL_HEADER_RE` regex uses `re.MULTILINE` so `^` matches mid-string line starts; `os.makedirs` in evaluator guards against empty `dirname`.

## Future Requirements (Deferred)

- CI/CD pipeline (GitHub Actions) — run bootstrap + pytest on push — deferred to v3.0
- EDA notebooks in `notebooks/` — deferred to v3.0
- Email dataset support (Enron/SpamAssassin) — deferred to v3.0

## Out of Scope

- Heavy deep learning or LLM fine-tuning — classical ML and gradient boosting only.
- Real-time mailbox daemon / IMAP integration.
- Enterprise multi-user authentication.

---

## Traceability

| REQ-ID | Phase | Plan |
|--------|-------|------|
| REPRO-01, REPRO-02, REPRO-03, DATA-01 | Phase 7 | TBD |
| CFG-01, CFG-02, CFG-03, CFG-04, FEAT-01 | Phase 8 | TBD |
| UI-01, UI-02, UI-03, UI-04 | Phase 9 | TBD |
| QUAL-01, QUAL-02, QUAL-03, QUAL-04, QUAL-05 | Phase 10 | TBD |
