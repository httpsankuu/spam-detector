# Phase 9: App Robustness & UX Fixes - Implementation Patterns

This document outlines the files to be modified in this phase, their roles, data flows, and the concrete existing code excerpts that will be targeted for updates.

## 1. `app/streamlit_app.py`

**Role:** The main interactive Streamlit web application. It handles the UI components, user input, model inference, and rendering of explainability data and evaluation plots.
**Data Flow:** 
- Reads serialized models (`*.joblib`) from the `models/` directory during startup.
- Manages user input via Streamlit widgets (text area, buttons, select boxes).
- Passes the input text to the loaded models and the `SpamExplainer` class.
- Renders the results (predictions, probabilities, metrics, and diagrams) back to the user.
**Closest Existing Analog:** Existing implementation (Self)

**Concrete Code Excerpts to Target:**

*a) Clear Text Button (Currently does not clear properly):*
```python
        default_text = SAMPLE_PRESETS.get(selected_preset, "")
        user_input = st.text_area(
            "Paste your email / message text below:",
            value=default_text,
            height=160,
            placeholder="e.g. Subject: You won $10,000 cash! Claim your prize now...",
        )

        col_btn1, col_btn2 = st.columns([1, 5])
        with col_btn1:
            analyze_clicked = st.button("🚀 Analyze Message", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("Clear Text", use_container_width=False):
                user_input = ""
                st.rerun()
```

*b) Model Loading (Currently crashes if `models/` is missing):*
```python
@st.cache_resource
def load_all_models() -> Dict[str, Any]:
    """Cache loaded scikit-learn models from disk."""
    models: Dict[str, Any] = {}
    models_dir = "models"
    for filename in os.listdir(models_dir):
        if filename.endswith("_pipeline.joblib"):
```

*c) Explainer Initialization (Currently hardcoded to Logistic Regression):*
```python
@st.cache_resource
def get_explainer() -> SpamExplainer:
    """Initialize cached explainability engine."""
    return SpamExplainer(model_path="models/logistic_regression_pipeline.joblib")
```

*d) Offline Support (Currently using a remote URL for the shield icon):*
```python
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/shield.png", width=70)
        st.markdown("### Model Configuration")
```

---

## 2. `src/evaluation/explainability.py`

**Role:** The explainability engine for the spam detection models, extracting global feature importance and generating local explanations (word-level feature attribution) for specific text inputs.
**Data Flow:** Takes a `model_path` (pointing to a serialized pipeline), extracts the feature vocabulary and classifier coefficients, and processes raw text strings through the pipeline to determine the specific contributions (`x_i * w_i`) to the final prediction.
**Closest Existing Analog:** Existing implementation (Self)

**Concrete Code Excerpts to Target:**

*a) Initialization & Coefficient Extraction (Currently assumes a Logistic Regression pipeline):*
```python
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model pipeline not found at {self.model_path}")

        self.pipeline = joblib.load(self.model_path)
        self.feature_pipeline = self.pipeline.named_steps["features"]
        self.classifier = self.pipeline.named_steps["lr"]

        # Cache feature names and coefficients
        self.feature_names = self.feature_pipeline.get_feature_names_out()
        self.coefficients = self.classifier.coef_.flatten()
        self.intercept = float(self.classifier.intercept_[0])
```

---

## 3. `requirements.txt`

**Role:** Project dependency list.
**Data Flow:** Consumed by Python package managers (like `pip`) to install the exact required environment.
**Closest Existing Analog:** Existing implementation (Self)

**Concrete Code Excerpts to Target:**

*Current Version Ranges (Needs exact pinning `==`):*
```text
pandas>=2.0.0
numpy>=1.24.0,<2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
nltk>=3.8.1
streamlit>=1.30.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```
