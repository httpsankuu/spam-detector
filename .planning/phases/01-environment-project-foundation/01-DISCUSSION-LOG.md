# Phase 1: Environment & Project Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-06
**Phase:** 01-Environment & Project Foundation
**Areas discussed:** Environment & Package Tooling, NLP Library Choice, Corpus Bootstrapping, Directory Layout

---

## Environment & Package Tooling

| Option | Description | Selected |
|--------|-------------|----------|
| Standard venv + pip | Lightweight, portable across Windows without third-party CLI requirements | ✓ |
| Conda / Mamba | Heavyweight virtual environment manager | |
| uv | Modern Rust-based Python package manager | |

**User's choice:** Standard venv + pip with requirements.txt
**Notes:** User specifically requested standard venv + pip for simplicity and compatibility.

---

## NLP Library Choice

| Option | Description | Selected |
|--------|-------------|----------|
| NLTK (WordNetLemmatizer) | Classical NLP toolkit, lightweight, avoids C-build issues on Windows | ✓ |
| spaCy | Industrial NLP library with pre-trained pipelines | |
| Both | Mixed dependency | |

**User's choice:** NLTK only (WordNetLemmatizer)
**Notes:** Excludes spaCy to avoid heavy downloads and Windows compilation quirks.

---

## Corpus Bootstrapping & Setup

| Option | Description | Selected |
|--------|-------------|----------|
| Automated setup script | Script that verifies environment and auto-downloads punkt, stopwords, wordnet | ✓ |
| On-demand lazy download | Downloads corpora when preprocessing functions are invoked | |

**User's choice:** Automated setup script on first run
**Notes:** Verifies environment and ensures missing NLTK resources are pre-fetched.

---

## Directory Layout

| Option | Description | Selected |
|--------|-------------|----------|
| Modular package layout | Clean separation (src/data, src/preprocessing, src/models, app/) | ✓ |
| Flat structure | Single directory with top-level scripts | |

**User's choice:** Modular package layout
**Notes:** Keeps codebase organized and scalable for step-by-step model walkthroughs.

---

## Antigravity Discretion

- Package version specifiers in `requirements.txt` for scikit-learn, xgboost, nltk, streamlit, and visualizers.
- Exact CLI options for the verification script.

## Deferred Ideas

- None — discussion stayed within Phase 1 scope.
