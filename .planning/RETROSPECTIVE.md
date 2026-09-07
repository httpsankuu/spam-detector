# Project Retrospective

## Milestone: v2.0 — Hardened and Production-Ready

**Shipped:** 2026-09-07
**Phases:** 3 | **Plans:** 6

### What Was Built
- Bootstrap script for fully autonomous end-to-end data fetching and model training
- Unified configuration wiring to replace hardcoded thresholds and fallback blocks
- App robustness improvements (offline emoji, UI clear button session state, graceful startup errors)
- Pinned `requirements.txt` environment file

### What Worked
- Decomposing the configuration constants wiring into multiple smaller plans (08-01, 08-02, 08-03) made the refactoring very clean and testable.
- The `pytest` integration test suite served as an excellent regression catch-all when replacing the config variables across the entire app.

### What Was Inefficient
- `ROADMAP.md` checkboxes fell out of sync during Phase 8 because execution proceeded manually without updating the central roadmap table until milestone closure.

### Patterns Established
- UI logic should always rely on `st.session_state` rather than local loop variables.
- Models should handle internal model metadata via `Path.as_posix()` for cross-platform robustness.
- All hyperparameters MUST be defined in `config/config.py`.

### Key Lessons
- Explicitly routing explainability models by checking for `coef_` vs `feature_log_prob_` prevents opaque crashes in `SpamExplainer`.
- Real-world production data (UCI SMS) requires much stricter boundary checks in `fit_transform` than synthetic datasets.

---

## Cross-Milestone Trends

| Milestone | Total Phases | Completion Time | Key Theme |
|-----------|--------------|-----------------|-----------|
| v1.0 MVP  | 6            | 1 Day           | Core functionality and baseline model metrics |
| v2.0 Prod | 3            | 1 Day           | Robustness, configuration wiring, UX edge cases |
