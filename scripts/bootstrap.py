import argparse
import logging
import sys
import urllib.request
import ssl
from pathlib import Path

from config.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TEST_BENCHMARK_JSON,
    RANDOM_SEED,
    TEST_RATIO,
    VAL_RATIO,
    TFIDF_MAX_FEATURES,
)
from src.data.sms_loader import SMSDataLoader
from src.data.split_data import split_dataset, save_splits
from src.models.trainer import train_and_persist_all
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.plots import generate_all_plots
from src.utils.setup_env import (
    check_python_version,
    check_dependencies,
    download_nltk_resources,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bootstrap")

def verify_environment():
    if not check_python_version():
        raise RuntimeError("Python version check failed.")
    deps_ok, _ = check_dependencies()
    if not deps_ok:
        raise RuntimeError("Dependency check failed. Run pip install -r requirements.txt")
    if not download_nltk_resources():
        raise RuntimeError("Failed to download NLTK resources.")

def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end project bootstrap orchestrator.")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download if already exists")
    args = parser.parse_args()

    try:
        # Step 1
        verify_environment()
        print("[1/6] Environment verified")

        # Step 2
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        loader = SMSDataLoader()
        
        sms_path = RAW_DATA_DIR / "SMSSpamCollection"
        if args.skip_download and sms_path.exists():
            print(f"[2/6] Skipping download, using existing file: {sms_path}")
        else:
            sms_path = loader.download_uci_dataset(str(RAW_DATA_DIR))
            print(f"[2/6] SMS dataset located at: {sms_path} ({sms_path.stat().st_size} bytes)")

        # Step 3
        df = loader.load_any_sms_file(sms_path)
        assert list(df.columns) == ['text', 'label'], f"Columns mismatch: {df.columns}"
        assert len(df) >= 16, f"Not enough rows: {len(df)}"
        print(f"[3/6] Loaded {len(df)} messages ({(df['label']==1).sum()} spam, {(df['label']==0).sum()} ham)")

        # Step 4
        train_df, val_df, test_df = split_dataset(df, test_size=TEST_RATIO, val_size=VAL_RATIO, random_state=RANDOM_SEED)
        save_splits(train_df, val_df, test_df, output_dir=str(PROCESSED_DATA_DIR))
        print(f"[4/6] Splits saved to {PROCESSED_DATA_DIR}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        # Step 5
        metadata = train_and_persist_all(
            train_path=str(PROCESSED_DATA_DIR / 'train.csv'),
            val_path=str(PROCESSED_DATA_DIR / 'val.csv'),
            output_dir=str(MODELS_DIR),
            max_features=TFIDF_MAX_FEATURES
        )
        print(f"[5/6] Trained and saved {len(metadata['models_saved'])} model pipelines to {MODELS_DIR}")

        # Step 6
        evaluator = ModelEvaluator(models_dir=str(MODELS_DIR))
        evaluator.evaluate_all(
            test_path=str(PROCESSED_DATA_DIR / 'test.csv'),
            output_json=str(TEST_BENCHMARK_JSON)
        )
        generate_all_plots()
        print(f"[6/6] Benchmark JSON and figures saved to {REPORTS_DIR}")

        print("\n[SUCCESS] Bootstrap complete! Run: streamlit run app/streamlit_app.py")
        return 0

    except Exception as e:
        logger.exception("Bootstrap failed")
        print(f"Bootstrap failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
