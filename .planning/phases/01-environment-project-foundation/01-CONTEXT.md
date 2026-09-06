# Phase 1: Environment & Project Foundation - Context

**Gathered:** 2026-09-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish the project directory structure (`src/`, `data/`, `models/`, `app/`, `notebooks/`), configure dependencies via `requirements.txt` for standard `venv` + `pip`, and create an automated environment verification script that checks library installation and downloads required NLTK resources (`punkt`, `stopwords`, `wordnet`) automatically on first run.
</domain>

<decisions>
## Implementation Decisions

### Environment & Package Tooling
- **D-01:** Use standard Python `venv` and `pip` with pinned `requirements.txt` rather than Conda or uv. Provides universal portability on Windows without requiring separate package managers. — **Reversibility:** reversible

### NLP Library Primary Choice
- **D-02:** Use NLTK exclusively (`WordNetLemmatizer`, `stopwords`, `punkt`) for text preprocessing. Skip spaCy to eliminate heavy C-extension/wheel compilation issues on Windows and keep the footprint lightweight. — **Reversibility:** costly — Preprocessing modules and feature extractors will bind to NLTK tokenizers and lemmatizers.

### Corpus Bootstrapping & Setup
- **D-03:** Provide an automated environment health and setup script (`src/utils/setup_env.py` / verification CLI) that verifies Python version (3.10+), checks key imports (pandas, numpy, scikit-learn, xgboost, nltk, streamlit, matplotlib, seaborn), and automatically downloads required NLTK corpora (`punkt`, `stopwords`, `wordnet`, `omw-1.4`) if missing. — **Reversibility:** reversible

### Directory Layout & Packaging
- **D-04:** Adopt a structured modular package layout:
  - `src/`: core logic split into modules:
    - `src/data/` (raw dataset downloaders, loaders, schema validation)
    - `src/preprocessing/` (text cleaner, tokenizer, lemmatizer transformers)
    - `src/features/` (TF-IDF vectorizer, handcrafted feature extractors)
    - `src/models/` (training routines, persistence, predictors)
    - `src/utils/` (environment check, logger, helpers)
  - `app/`: Streamlit web demo UI and visualization dashboards
  - `data/`: `data/raw/` (external corpora) and `data/processed/` (cleaned/split datasets)
  - `models/`: saved model pipelines (`.joblib` files and metadata)
  - `notebooks/`: Jupyter notebooks for exploratory data analysis and demonstrations
  - Root: `requirements.txt`, `.gitignore`, `README.md`, `AGENTS.md`
  — **Reversibility:** costly — Module paths are imported across the entire application.

### Antigravity Discretion
- Specific pinning ranges for requirements.txt (e.g., scikit-learn >= 1.3, xgboost >= 2.0, streamlit >= 1.30) to guarantee compatibility on Python 3.10+.
- Exact CLI flags or function entrypoint for `python -m src.utils.setup_env`.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Specifications
- `.planning/PROJECT.md` — Core value, constraints, and architecture guidelines
- `.planning/REQUIREMENTS.md` — Requirements ENV-01 and ENV-02 definitions
- `.planning/ROADMAP.md` — Phase 1 scope and deliverables

No external specs — requirements fully captured in decisions above.
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Greenfield project. No legacy components exist.

### Established Patterns
- Clean modular Python package pattern with `__init__.py` markers in subdirectories under `src/`.
- Script runnable as modules (`python -m src.utils.setup_env`).

### Integration Points
- Subsequent Phase 2 (Data Ingestion & NLP) will directly build on `src/data/` and `src/preprocessing/` directories created in Phase 1.
- Phase 6 (Streamlit UI) will live in `app/`.
</code_context>

<specifics>
## Specific Ideas
- User specifically requested: "standard venv + pip for environment/package management, NLTK only (WordNetLemmatizer) for NLP preprocessing, an automated setup script that downloads required NLTK resources (punkt, stopwords, wordnet) on first run, and a modular package layout (src/data, src/models, src/preprocessing, app/) rather than a flat structure."
</specifics>

<deferred>
## Deferred Ideas
- None — discussion stayed within Phase 1 scope.
</deferred>

---

*Phase: 1-Environment & Project Foundation*
*Context gathered: 2026-09-06*
