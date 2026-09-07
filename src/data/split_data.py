"""
split_data.py - Stratified Dataset Partitioning Utility.

Partitions standardized spam/ham datasets into train, validation, and test subsets
while strictly preserving class stratification across all splits.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import RANDOM_SEED, TEST_RATIO, VAL_RATIO, PROCESSED_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("split_data")


def split_dataset(
    df: pd.DataFrame,
    test_size: float = TEST_RATIO,
    val_size: float = VAL_RATIO,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into stratified train, validation, and test sets.

    Args:
        df: Standardized DataFrame containing 'text' and 'label' columns
        test_size: Fraction of the dataset allocated to the test set (e.g., 0.2)
        val_size: Fraction of the dataset allocated to the validation set (e.g., 0.1)
        random_state: Seed for reproducible pseudorandom splits

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("DataFrame must contain both 'text' and 'label' columns.")

    if len(df["label"].unique()) < 2:
        raise ValueError("Cannot perform stratified split: DataFrame must contain at least 2 distinct classes.")

    total_eval = test_size + val_size
    if total_eval >= 1.0 or test_size <= 0:
        raise ValueError(f"Invalid split proportions: test={test_size}, val={val_size}.")

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    # Second split: separate validation from train
    if val_size > 0:
        val_relative_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative_size,
            random_state=random_state,
            stratify=train_val_df["label"],
        )
    else:
        train_df = train_val_df
        val_df = pd.DataFrame(columns=df.columns)

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path = PROCESSED_DATA_DIR,
) -> Dict[str, Path]:
    """
    Save train, val, and test splits into CSV files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_file = out_path / "train.csv"
    val_file = out_path / "val.csv"
    test_file = out_path / "test.csv"

    train_df.to_csv(train_file, index=False, encoding="utf-8")
    val_df.to_csv(val_file, index=False, encoding="utf-8")
    test_df.to_csv(test_file, index=False, encoding="utf-8")

    def _dist(subset: pd.DataFrame, name: str) -> str:
        if len(subset) == 0:
            return f"{name}: 0 rows"
        s = int((subset["label"] == 1).sum())
        h = int((subset["label"] == 0).sum())
        ratio = (s / len(subset)) * 100
        return f"{name}: {len(subset)} rows (Ham: {h}, Spam: {s}, {ratio:.1f}% spam)"

    logger.info("Saved stratified data splits:")
    logger.info(f"  {_dist(train_df, 'Train')}")
    logger.info(f"  {_dist(val_df, 'Validation')}")
    logger.info(f"  {_dist(test_df, 'Test')}")

    return {"train": train_file, "val": val_file, "test": test_file}


def main() -> int:
    """CLI execution for dataset splitting."""
    parser = argparse.ArgumentParser(description="Stratified dataset splitter utility")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/sample_emails.csv",
        help="Path to input standardized CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Destination directory for splits",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    in_file = Path(args.input)
    if not in_file.exists():
        logger.error(f"Input file not found: {in_file}. Run email_loader or sms_loader first.")
        return 1

    df = pd.read_csv(in_file)
    logger.info(f"Loaded {len(df)} rows from {in_file}.")

    train_df, val_df, test_df = split_dataset(
        df, test_size=args.test_size, val_size=args.val_size, random_state=args.seed
    )

    save_splits(train_df, val_df, test_df, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
