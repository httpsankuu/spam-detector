# Phase 9: App Robustness & UX Fixes - Research Notes

## Goal & Objectives
This phase tackles key UX improvements and robustness fixes for the final application state.
Requirement IDs to address: `UI-01`, `UI-02`, `UI-03`, `UI-04`, `QUAL-01` (also referred to as `FIX-07` through `FIX-10`).

## Implementation Details for PLAN

### 1. Fix "Clear Text" Button (UI-01 / FIX-07)
**File to modify:** `app/streamlit_app.py`
- **Issue:** The current Streamlit app attempts to clear the text area by just assigning a local variable `user_input = ""` and calling `st.rerun()`. This doesn't work because `st.text_area` retains its internal widget state unless explicitly tied to `st.session_state`.
- **Solution:** 
  1. Initialize `st.session_state.user_input` in `main()` with the default sample text.
  2. Bind the `st.text_area` to `st.session_state` by providing `key="user_input"`.
  3. When the "Clear Text" button is clicked, explicitly clear it via `st.session_state.user_input = ""` and `st.rerun()`.
  4. Make sure to gracefully handle when the user switches preset dropdowns (update the `session_state` with the new preset value).

### 2. Graceful Startup Error Handling (UI-02 / FIX-08)
**File to modify:** `app/streamlit_app.py`
- **Issue:** `load_all_models()` assumes the `models/` directory exists. On a fresh clone without running `bootstrap.py`, `os.listdir("models")` raises a `FileNotFoundError`, crashing the app.
- **Solution:**
  1. Inside `load_all_models()`, check if `models_dir` exists. If not, return an empty dictionary.
  2. In `main()`, check if `models` is empty. If it is, render a friendly warning: `st.info("Run \`python scripts/bootstrap.py\` first to train models and generate data.")` and call `st.stop()`.

### 3. Route Explainability to User-Selected Model (UI-03 / FIX-09)
**Files to modify:** `app/streamlit_app.py`, `src/evaluation/explainability.py`
- **Issue:** The feature attribution (word-level explainability) is hardcoded to always load `logistic_regression_pipeline.joblib`. Furthermore, `SpamExplainer` crashes if it attempts to extract `coef_` from a non-linear model.
- **Solution:**
  1. Update `SpamExplainer.__init__` in `src/evaluation/explainability.py` to extract coefficients dynamically based on the linear model type:
     - **Logistic Regression**: `classifier.coef_.flatten()`
     - **Linear SVM (CalibratedClassifierCV)**: Average the `estimator.coef_.flatten()` across all `classifier.calibrated_classifiers_`.
     - **Naive Bayes**: Calculate the log probability difference: `(classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]).flatten()`.
     - For non-linear models, set an `is_linear = False` flag and return empty coefficients.
  2. Update `SpamExplainer.explain_text()` to return the `is_linear` flag in its dictionary payload.
  3. In `app/streamlit_app.py`, change `get_explainer()` to accept `model_path` as a parameter and initialize `SpamExplainer` with the currently selected model.
  4. In the Streamlit UI, if `is_linear` is `False`, display a `st.info` block explaining that word-level attribution is only available for linear models (and hide the attribution columns).

### 4. Make App Work Fully Offline (UI-04 / FIX-10)
**File to modify:** `app/streamlit_app.py`
- **Issue:** The sidebar shield icon is requested from a remote URL (`https://img.icons8.com/color/96/000000/shield.png`), meaning the app fails to load the image if run offline.
- **Solution:** Replace `st.image(url)` with a local HTML emoji using `st.markdown("<h1 style='text-align: center;'>🛡️</h1>", unsafe_allow_html=True)` or a similar offline solution.

### 5. Pin Package Versions (QUAL-01)
**File to modify:** `requirements.txt`
- **Issue:** `requirements.txt` currently uses `>` and `>=` operators, which violates the requirement for strictly pinned dependencies.
- **Solution:** Run `pip freeze` in the current environment to capture the exact versions of the packages (e.g., `scikit-learn==1.3.2`) and update `requirements.txt` accordingly.
