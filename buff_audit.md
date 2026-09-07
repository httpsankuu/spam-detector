# Audit Report — spam-detector

**Date:** September 6, 2026
**Scope:** Full codebase review — `src/`, `app/`, `tests/`, `config/`, artifacts (`models/`, `reports/`, `data/`), and git state.
**Verification:** All 16 tests pass locally (`pytest -v` → 16 passed on Python 3.13, sklearn 1.9.0, xgboost 3.4.1).

---

## ✅ What's solid

- **Clean, modular architecture** matching the documented layout — data → preprocessing → features → models → evaluation → app, all sklearn-compatible (`BaseEstimator`/`TransformerMixin`) so everything bundles into single `.joblib` pipelines.
- **Consistent schema normalization** (`['text','label']`, 1=spam) across Email/SMS loaders; semantic tokens (`httpaddr`, `emailaddr`, `dollar`, `number`) are well done.
- **Good ML hygiene**: stratified splits with leakage guards, negation-preserving stopword removal, `class_weight="balanced"`, fixed seeds, headless matplotlib, lazy NLTK init with fallbacks, edge-case tests.
- **Documentation quality** is high — docstrings, self-check `main()` demos per module, thorough README.

---

## 🔴 Critical

### 1. Fresh-clone reproducibility is broken (models + data are gitignored, tests/app depend on them)

`models/*.joblib`, `models/*.json`, and all CSVs are in `.gitignore` (confirmed via `git ls-files`). Yet:

- `tests/test_pipeline.py` asserts `models/{name}_pipeline.joblib` exists (5 tests) and `SpamExplainer()` raises `FileNotFoundError` on missing model (2 tests) → **`pytest` fails on a fresh clone**.
- `streamlit run` crashes at startup without the model artifacts.
- No CI (no `.github/`), and the README quickstart never documents the chain to regenerate artifacts (`--sample` loaders → `split_data` → `trainer`). A fresh clone has: no data, no models, no tests that pass, and no script that fixes it.

### 2. All models are trained and "benchmarked" on a toy dataset of 30 synthetic emails (21 train / 3 val / 6 test)

`model_metadata.json` confirms `train_samples: 21`, and `test_benchmark.json` reports ~1.0 accuracy/F1 on **6 test rows**. The tracked `reports/test_benchmark.json` + figures present these as meaningful results — they're not. The 16 SMS messages in `data/raw/sample_sms.csv` are **not in the training splits at all**, despite the README claiming email *and* SMS classification. This is the biggest credibility gap for a capstone project.

### 3. `SMSDataLoader.download_uci_dataset()` has a fake fallback

On download failure it logs "Falling back to built-in sample SMS corpus" but returns a path to a **non-existent file** (`data/raw/SMSSpamCollection`), so any caller gets `FileNotFoundError` — there is no actual fallback.

### 4. The "centralized configuration" convention is dead code

~30 constants in `config/config.py` are **never consumed**: all hyperparameters (`LR_C`, `NB_ALPHA`, `SVM_C`, `RF_N_ESTIMATORS`, `XGB_*`, `LR_MAX_ITER`…), all NLP flags (`MIN_WORD_LENGTH`, `REPLACE_URLS`…), `TRAIN_RATIO`, and `MODEL_METADATA_PATH`. Model factories and `TextPreprocessor` hardcode their own defaults, so tuning requires editing source, not config.

---

## 🟠 High / Medium

### 5. Config duplication via `try/except ImportError` fallback blocks

In 5 modules (`trainer`, `metrics`, `plots`, `explainability`, `split_data`) — each with duplicated string defaults, partially wired. Fix by importing `config.config` directly.

### 6. `TFIDFExtractor.fit_transform` is inconsistent with `fit` + `transform`

It skips the tiny-dataset `min_df`/`max_df` guards that `fit` applies, so the two paths can produce different vocabularies on small inputs.

### 7. "Clear Text" button in the Streamlit app does nothing

It assigns a local variable then calls `st.rerun()`, which doesn't reset widget state — the text area re-renders with the old value. Needs a `key=` on `st.text_area` + `st.session_state` manipulation.

### 8. App hard-crashes without local artifacts

`load_all_models()` does `os.listdir(models_dir)` (raises if dir missing) and `get_explainer()` raises — the Benchmarks/Architecture tabs die too, with no graceful "run the trainer first" message.

### 9. Explainability is hardwired to Logistic Regression

Even when the user selects XGBoost/RF in the UI (only disclosed in a small caption).

### 10. `requirements.txt` is unpinned

Bare ranges, `numpy<2.0.0` cap → non-reproducible installs; no lockfile.

### 11. `setup_env.py` disables SSL certificate verification globally

`ssl._create_default_https_context = _create_unverified_context` — a known NLTK-on-Windows workaround, but a MITM smell; worth scoping.

### 12. `test_benchmark.json` embeds full `y_pred`/`y_prob` arrays

Will bloat enormously on a real dataset (fine at 6 rows).

### 13. Duplication across model modules

Identical `evaluate()` and demo `main()` copy-pasted in 4 files; `trainer.py` re-implements metric computation instead of reusing `ModelEvaluator`.

### 14. Model metadata is Windows-specific

`models_saved` paths contain `models\\...` backslashes, breaking portability.

---

## 🟡 Low

- Unused typing imports in every model module; several dead config constants; `TRAIN_RATIO` unused.
- `digit_ratio` uses an undocumented `number_tokens * 3` surrogate weight.
- Sidebar fetches `img.icons8.com` remotely — broken offline.
- `EMAIL_HEADER_RE` is `^`-anchored, so headers preceded by whitespace aren't stripped.
- `os.makedirs(os.path.dirname(output_json))` fails if the path has no directory component (edge case).
- `notebooks/` is empty; the roadmap advertises EDA notebooks that don't exist.

---

## Top recommendations (in order of impact)

1. **Add a bootstrap path for fresh clones** — either track the artifacts (bad practice) or add a `make bootstrap`/`setup_data.py` that generates samples → splits → trains all 5 models, and make the README quickstart run it. Add a CI workflow that does this and runs `pytest`.
2. **Retrain on a real dataset** (the UCI SMS collection, ~5.5k messages) once the downloader is fixed, and regenerate the benchmark/figures so the tracked numbers are honest.
3. **Wire `config/config.py` through the model/preprocessing factories** so the convention is real.
4. **Fix the SMS downloader fallback, the Clear Text button, and app startup error handling** (the quick wins).