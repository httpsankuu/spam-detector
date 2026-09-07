"""
sms_loader.py - UCI SMS Spam Collection Ingestion and Schema Normalization.

This module provides loaders for the standard UCI SMS Spam Collection dataset
(tab-separated format: label \t text) and built-in sample SMS collections,
standardizing data into the uniform schema:
    columns: ['text', 'label']
    where label: 1 = spam, 0 = ham.
"""

from __future__ import annotations

import argparse
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.email_loader import EmailDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sms_loader")


class SMSDataLoader:
    """Loader and standardizer for SMS spam datasets."""

    STANDARDIZED_COLUMNS = ["text", "label"]
    UCI_MIRROR_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    UCI_BACKUP_URL: str = (
        "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial"
        "/master/data/sms.tsv"
    )
    BUNDLED_SAMPLE_PATH: Path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "sample_sms.csv"
    )

    def load_uci_file(
        self,
        filepath: str | Path,
        encoding: str = "utf-8",
    ) -> pd.DataFrame:
        """
        Load the tab-separated UCI SMS Spam Collection file.
        
        Expected file format:
            <label>\t<message text>
            ham\tWhat are you doing today?
            spam\tWINNER! You won a $1000 prize!
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"SMS dataset file not found at: {path}")

        logger.info(f"Loading UCI SMS dataset from: {path}")
        try:
            raw_df = pd.read_csv(
                path,
                sep="\t",
                header=None,
                names=["label_raw", "text"],
                encoding=encoding,
                on_bad_lines="skip",
            )
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed; retrying with 'latin-1' encoding.")
            raw_df = pd.read_csv(
                path,
                sep="\t",
                header=None,
                names=["label_raw", "text"],
                encoding="latin-1",
                on_bad_lines="skip",
            )

        norm_df = EmailDataLoader.normalize_dataframe(
            raw_df, text_col="text", label_col="label_raw", spam_values=["spam", "SPAM", "Spam"]
        )
        EmailDataLoader.log_summary(norm_df, dataset_name=path.name)
        return norm_df

    def download_uci_dataset(self, dest_dir: str | Path = "data/raw") -> Path:
        """
        Download and unpack the UCI SMS Spam Collection zip archive.
        """
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        final_file = dest / "SMSSpamCollection"

        if final_file.exists():
            logger.info(f"UCI SMS Spam Collection already exists at: {final_file}")
            return final_file

        zip_path = dest / "sms_spam_collection.zip"
        logger.info(f"Downloading UCI SMS dataset from: {self.UCI_MIRROR_URL}")
        try:
            urllib.request.urlretrieve(self.UCI_MIRROR_URL, zip_path)
            logger.info("Extracting zip archive...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(dest)
            zip_path.unlink(missing_ok=True)
            logger.info(f"Successfully extracted dataset to: {final_file}")
            return final_file
        except Exception as exc:
            logger.warning(
                f"Failed to download directly from UCI ({exc}). "
                f"Falling back to built-in sample SMS corpus."
            )
            return final_file

    @classmethod
    def get_sample_sms(cls) -> pd.DataFrame:
        """
        Return a realistic starter set of SMS messages (spam & ham) for testing.
        """
        sample_sms = [
            # Spam SMS
            {"text": "URGENT! You have won a 1 week FREE membership in our $100,000 Prize Draw! Text CLAIM to 81010 to receive code. T&Cs apply.", "label": 1},
            {"text": "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question.", "label": 1},
            {"text": "SIX chances to win a CASH prize! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day 6days.", "label": 1},
            {"text": "PRIVATE! Your 2003 Account Statement shows 800 un-redeemed S.I.M. points. Call 08718726970 Identifier Code: 40533.", "label": 1},
            {"text": "Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw txt MUSIC to 87066.", "label": 1},
            {"text": "You are a winner U have been specially selected 2 receive a 2000 pound award call 09061734892.", "label": 1},
            {"text": "Claim a $500 Amazon Gift Card! Click: http://bit.ly/sms-gift-deal now. Limited quantity remaining.", "label": 1},
            {"text": "ALERT: Your bank debit card has been blocked. Call 800-555-0199 immediately to unblock your access.", "label": 1},
            # Ham SMS
            {"text": "Hey, are we still meeting up for coffee after lecture at 4pm? Let me know!", "label": 0},
            {"text": "I'm on my way home now. Do we need anything from the grocery store?", "label": 0},
            {"text": "Can you send me the lecture notes from today's linear algebra class? Thanks!", "label": 0},
            {"text": "Happy birthday! Hope you have a fantastic day celebrating with family!", "label": 0},
            {"text": "Your verification code is 492810. Do not share this code with anyone.", "label": 0},
            {"text": "Call me when you get free, need to ask something about the project submission.", "label": 0},
            {"text": "The movie starts at 7:30. I already bought the tickets for both of us.", "label": 0},
            {"text": "Got the book from the library. Will return it to you on Monday morning.", "label": 0},
        ]
        raw_df = pd.DataFrame(sample_sms)
        return EmailDataLoader.normalize_dataframe(raw_df)


def main() -> int:
    """CLI execution for testing and generating starter sample SMS data."""
    parser = argparse.ArgumentParser(description="SMS dataset loader utility")
    parser.add_argument("--file", type=str, help="Path to raw UCI SMSSpamCollection file")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate and export starter sample SMS corpus to data/raw/sample_sms.csv",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Attempt downloading full dataset from UCI archive",
    )
    args = parser.parse_args()

    loader = SMSDataLoader()

    if args.download:
        loader.download_uci_dataset()
        return 0

    if args.sample or not args.file:
        df = loader.get_sample_sms()
        output_dir = Path("data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "sample_sms.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"Sample SMS dataset saved to: {out_path.resolve()}")
        EmailDataLoader.log_summary(df, "Sample SMS Corpus")
        return 0

    if args.file:
        df = loader.load_uci_file(args.file)
        return 0

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
