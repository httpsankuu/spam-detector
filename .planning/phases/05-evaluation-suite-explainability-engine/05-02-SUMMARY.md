# Plan 05-02 Summary: Local & Global Explainability Engine

## Delivered Artifacts
- `src/evaluation/explainability.py`:
  - `SpamExplainer` extracting:
    - Global feature importance: top positive (spam) weights (`hc_url_count`, `hc_exclamation_count`, `hc_trigger_words_count`, `httpaddr`, `hc_dollar_count`) and top negative (ham) weights (`number number`, `hc_question_count`, `bill`, `note`).
    - Local email decomposition ($x_i \cdot w_i$): computes per-token and per-metric contributions for any specific text.
    - Direct keyword regex detection matching curated triggers in raw text.
    - Structural handcrafted metrics (capitalization ratio, exclamation marks, URL count, digit ratio, word/character lengths).
    - Risk classification: HIGH RISK (>=0.80), MODERATE RISK (>=0.50), LOW RISK (>=0.20), SAFE (<0.20).
- `src/evaluation/__init__.py`: Exported `SpamExplainer`.

## Verification
- Executed on a sample phishing text:
  - Probability: 0.9909 (HIGH RISK).
  - Detected 8 distinct triggers (`cash`, `claim`, `congratulations`, `expires`, `prize`, `urgent`, `winner`, `won`).
  - Correctly attributed largest positive spam contributions to `hc_trigger_words_count` (+2.11) and `hc_exclamation_count` (+1.47).
