# Plan 04-02 Summary: Linear SVM, Ensembles, Multi-Model Training Harness, and Persistence

## Delivered Artifacts
- `src/models/svm_classifier.py`:
  - `SVMClassifier` wrapping `LinearSVC` with probability calibration using `CalibratedClassifierCV`.
  - `build_svm_pipeline()` constructing composite feature + SVM pipeline.
  - Verification: 100% Accuracy and F1 on validation split; live test outputted 80.66% spam probability on urgent phishing sample.
- `src/models/tree_classifier.py`:
  - `EnsembleClassifier` wrapping `RandomForestClassifier` and `xgboost.XGBClassifier` with `get_feature_importances()`.
  - `build_ensemble_pipeline()` for non-linear tree pipelines.
  - Verification: both Random Forest and XGBoost trained and extracted top features (e.g. `hc_word_count`, `httpaddr`, `hc_exclamation_count`).
- `src/models/trainer.py`:
  - `train_and_persist_all()` harness training and validating all 5 models:
    1. `naive_bayes`
    2. `logistic_regression`
    3. `linear_svm`
    4. `random_forest`
    5. `xgboost`
  - Persisted all pipelines as `.joblib` files to `models/`.
  - Saved leaderboard benchmark and training metadata to `models/model_metadata.json`.
- `src/models/__init__.py`: Clean unified exports for all models and training functions.

## Verification
- Validated all 5 models trained on `data/processed/train.csv` and evaluated against `data/processed/val.csv`.
- Confirmed files exist in `models/`:
  - `linear_svm_pipeline.joblib`
  - `logistic_regression_pipeline.joblib`
  - `naive_bayes_pipeline.joblib`
  - `random_forest_pipeline.joblib`
  - `xgboost_pipeline.joblib`
  - `model_metadata.json`
- Verified live inference directly from serialized joblib artifact (`Pred: [1]`, spam probability `0.9127`).
