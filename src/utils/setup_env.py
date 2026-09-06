"""
setup_env.py - Automated Environment Verification and NLTK Corpus Setup.

This utility verifies Python runtime requirements, inspects critical dependencies,
and automatically downloads required NLTK resources (punkt, stopwords, wordnet)
to guarantee reproducible execution across environments.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import ssl
import sys
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("setup_env")

# Minimum required Python version
MIN_PYTHON_VERSION: Tuple[int, int] = (3, 10)

# Core required dependencies for the project
REQUIRED_PACKAGES: Dict[str, str] = {
    "pandas": "Data manipulation and DataFrame handling",
    "numpy": "Numerical computing and array operations",
    "sklearn": "Machine learning estimators, metrics, and pipelines (scikit-learn)",
    "xgboost": "Gradient boosting decision tree ensemble",
    "nltk": "Natural language toolkit for tokenization and lemmatization",
    "streamlit": "Interactive web demo application framework",
    "matplotlib": "Plotting and visualization of metrics and curves",
    "seaborn": "Statistical graphics and confusion matrix heatmaps",
    "joblib": "Model serialization and persistence",
}

# Required NLTK resources
NLTK_REQUIREMENTS: List[str] = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
]


def check_python_version() -> bool:
    """Verify that current Python version meets the minimum version requirement."""
    current = sys.version_info[:2]
    version_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    if current < MIN_PYTHON_VERSION:
        logger.error(
            f"Python version check FAILED: running {version_str}. "
            f"Requires Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+."
        )
        return False

    logger.info(f"Python version check PASSED: {version_str}")
    return True


def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check availability of required third-party packages.
    
    Returns:
        Tuple of (all_passed: bool, missing_packages: List[str])
    """
    missing: List[str] = []
    logger.info("Verifying installed Python packages...")

    for pkg_name, description in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(pkg_name)
            version = getattr(mod, "__version__", "unknown")
            logger.info(f"  [OK] {pkg_name} (v{version}) - {description}")
        except ImportError:
            logger.warning(f"  [MISSING] {pkg_name} - {description}")
            missing.append(pkg_name)

    if missing:
        logger.error(
            f"Missing {len(missing)} package(s): {', '.join(missing)}.\n"
            f"Run: pip install -r requirements.txt"
        )
        return False, missing

    logger.info("All required packages are successfully installed.")
    return True, []


def configure_ssl_for_nltk() -> None:
    """Handle Windows SSL certificate chain fallbacks for NLTK downloader."""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context


def download_nltk_resources() -> bool:
    """
    Download and verify required NLTK corpora and tokenizers.
    
    Returns:
        True if all downloads succeed, False otherwise.
    """
    try:
        import nltk
    except ImportError:
        logger.error("Cannot download NLTK resources: NLTK is not installed.")
        return False

    configure_ssl_for_nltk()
    logger.info("Checking and downloading required NLTK resources...")
    all_success = True

    for resource in NLTK_REQUIREMENTS:
        try:
            logger.info(f"  Downloading/Verifying NLTK resource: '{resource}'...")
            downloaded = nltk.download(resource, quiet=True)
            if downloaded:
                logger.info(f"  [OK] Resource '{resource}' is ready.")
            else:
                # nltk.download returns True on new download, but may return False/None if already up to date
                logger.info(f"  [OK] Resource '{resource}' verified.")
        except Exception as exc:
            logger.error(f"  [FAIL] Error downloading '{resource}': {exc}")
            all_success = False

    return all_success


def main() -> int:
    """CLI entrypoint for environment verification."""
    parser = argparse.ArgumentParser(
        description="Verify development environment and download required NLTK corpora."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify Python version and package imports, skip NLTK downloads.",
    )
    parser.add_argument(
        "--download-nltk",
        action="store_true",
        help="Force download of NLTK resources without stopping if packages are missing.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" AI-Based Spam Email Detection - Environment Setup")
    print("=" * 60)

    py_ok = check_python_version()
    if not py_ok:
        return 1

    deps_ok, missing = check_dependencies()

    if args.check_only:
        return 0 if deps_ok else 1

    if not deps_ok and not args.download_nltk:
        logger.warning(
            "Skipping NLTK downloads due to missing packages. "
            "Install requirements first or use --download-nltk."
        )
        return 1

    nltk_ok = download_nltk_resources()

    print("=" * 60)
    if deps_ok and nltk_ok:
        logger.info("Environment is fully initialized and ready!")
        return 0
    else:
        logger.error("Environment setup encountered errors.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
