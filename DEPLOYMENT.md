# Deployment Guide — Streamlit Cloud

## Prerequisites
- GitHub repository pushed (with `models/`, `data/processed/`, `reports/` included)
- `requirements.txt` present at repository root

## Steps
1. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub.
2. Click **New app**.
3. Select repository, branch `main`, and file path `app/streamlit_app.py`.
4. Click **Deploy**.
5. The app will be live at `https://<your-app>.streamlit.app`.

## Notes
- `models/` contain ~2.4MB of `.joblib` files; Streamlit handles this fine.
- `.streamlit/config.toml` controls theming.
- If models fail to load, verify `models/` is not excluded by `.gitignore`.
