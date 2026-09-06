"""
email_loader.py - Email Dataset Ingestion and Schema Normalization.

This module provides loaders for email spam datasets (CSV, raw directories, or
built-in starter corpora), standardizing data into a uniform DataFrame schema:
    columns: ['text', 'label']
    where label: 1 = spam, 0 = ham (legitimate).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("email_loader")


class EmailDataLoader:
    """Loader and standardizer for email spam and ham datasets."""

    STANDARDIZED_COLUMNS = ["text", "label"]

    @classmethod
    def normalize_dataframe(
        cls,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
        spam_values: Optional[List[Any]] = None,
    ) -> pd.DataFrame:
        """
        Normalize an arbitrary DataFrame into standard ['text', 'label'] format.

        Args:
            df: Raw input DataFrame
            text_col: Name of column containing email text
            label_col: Name of column containing target label
            spam_values: Values in label_col that represent spam (defaults to [1, '1', 'spam', 'SPAM', 'True', True])

        Returns:
            Standardized DataFrame with ['text', 'label'] (1=spam, 0=ham)
        """
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(
                f"Missing required columns '{text_col}' or '{label_col}'. "
                f"Available columns: {list(df.columns)}"
            )

        if spam_values is None:
            spam_values = [1, "1", "spam", "SPAM", "Spam", True, "True", "true"]

        norm_df = pd.DataFrame()
        norm_df["text"] = df[text_col].astype(str).str.strip()
        norm_df["label"] = df[label_col].apply(
            lambda x: 1 if x in spam_values or str(x).strip().lower() in ["1", "spam", "true"] else 0
        ).astype(int)

        # Drop empty texts or whitespace-only records
        norm_df = norm_df[norm_df["text"].str.len() > 0].copy()
        norm_df.drop_duplicates(subset=["text"], inplace=True)
        norm_df.reset_index(drop=True, inplace=True)

        return norm_df[cls.STANDARDIZED_COLUMNS]

    def load_from_csv(
        self,
        filepath: str | Path,
        text_col: str = "text",
        label_col: str = "label",
        spam_values: Optional[List[Any]] = None,
        encoding: str = "utf-8",
    ) -> pd.DataFrame:
        """
        Load email dataset from a CSV file.

        Args:
            filepath: Path to the CSV file
            text_col: Name of the text column in the CSV
            label_col: Name of the label column in the CSV
            spam_values: Values indicating spam
            encoding: Text encoding (defaults to utf-8)
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Email dataset file not found at: {path}")

        logger.info(f"Loading email dataset from: {path}")
        try:
            raw_df = pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decode failed for {path}. Retrying with 'latin-1' encoding.")
            raw_df = pd.read_csv(path, encoding="latin-1")

        norm_df = self.normalize_dataframe(
            raw_df, text_col=text_col, label_col=label_col, spam_values=spam_values
        )
        self.log_summary(norm_df, dataset_name=path.name)
        return norm_df

    def load_raw_directory(
        self,
        spam_dir: str | Path,
        ham_dir: str | Path,
        encoding: str = "latin-1",
    ) -> pd.DataFrame:
        """
        Load emails from directories containing individual text files (e.g. Enron or SpamAssassin).
        """
        spam_path = Path(spam_dir)
        ham_path = Path(ham_dir)

        if not spam_path.exists() or not ham_path.exists():
            raise FileNotFoundError(f"Spam dir '{spam_path}' or Ham dir '{ham_path}' does not exist.")

        records: List[Dict[str, Any]] = []

        # Load spam
        for file in spam_path.glob("*"):
            if file.is_file():
                try:
                    content = file.read_text(encoding=encoding, errors="ignore")
                    records.append({"text": content, "label": 1})
                except Exception as exc:
                    logger.debug(f"Skipping unreadable spam file {file}: {exc}")

        # Load ham
        for file in ham_path.glob("*"):
            if file.is_file():
                try:
                    content = file.read_text(encoding=encoding, errors="ignore")
                    records.append({"text": content, "label": 0})
                except Exception as exc:
                    logger.debug(f"Skipping unreadable ham file {file}: {exc}")

        raw_df = pd.DataFrame(records)
        norm_df = self.normalize_dataframe(raw_df)
        self.log_summary(norm_df, dataset_name="Raw Directory Ingestion")
        return norm_df

    @classmethod
    def get_sample_corpus(cls) -> pd.DataFrame:
        """
        Return a realistic starter corpus of spam and ham emails for testing and offline execution.
        """
        sample_data = [
            # --- SPAM SAMPLES ---
            {
                "text": "Subject: URGENT: You have won $1,000,000 in the International Lottery! Reply with your account details immediately to claim your cash prize now!",
                "label": 1,
            },
            {
                "text": "Subject: Hot Stock Alert! Guaranteed 500% returns on penny stocks this week. Buy now before the massive explosion!",
                "label": 1,
            },
            {
                "text": "Subject: Security Alert: Your PayPal account has been suspended due to suspicious activity. Click http://secure-verify-paypal.fake-link.com to restore access immediately.",
                "label": 1,
            },
            {
                "text": "Subject: Cheap Meds Online! Viagra, Cialis, Xanax without prescription. 80% discount and discreet overnight delivery. Order today: http://cheap-meds-direct.com",
                "label": 1,
            },
            {
                "text": "Subject: Exclusive invitation to join Crypto Millionaire Bot! Earn $3,500 daily on complete autopilot. Free registration ending tonight.",
                "label": 1,
            },
            {
                "text": "Subject: CONGRATULATIONS! You have been selected for an Apple iPhone 15 Pro giveaway. Confirm your address here to receive tracking number.",
                "label": 1,
            },
            {
                "text": "Subject: Final Notice: IRS Tax Refund of $2,450 is pending. Submit your SSN and bank details at http://irs-refund-portal.net to claim.",
                "label": 1,
            },
            {
                "text": "Subject: Increase your website traffic by 10x in 24 hours! Guaranteed top Google rank with automated backlink blast. Limited discount offer!",
                "label": 1,
            },
            {
                "text": "Subject: Dear beneficiary, I am Barrister Williams representing the estate of late Mr. Smith. A fund of $8.5M is ready for transfer to your account.",
                "label": 1,
            },
            {
                "text": "Subject: Refinance your mortgage at 1.99% fixed APR! Pre-approved regardless of credit score. Apply now at http://fast-loan-approval.com",
                "label": 1,
            },
            {
                "text": "Subject: Immediate action required: Netflix subscription expired. Update payment method at http://netflix-billing-update.info to avoid cancellation.",
                "label": 1,
            },
            {
                "text": "Subject: Work from home and earn $500-$1000 per day stuffing envelopes or reviewing products online! No experience needed.",
                "label": 1,
            },
            {
                "text": "Subject: Weight loss miracle pill: Lose 20 lbs in 14 days without dieting or exercise! 100% natural formula doctor approved.",
                "label": 1,
            },
            {
                "text": "Subject: You have an unpaid parking citation in New York City. Pay penalty fee within 48 hours or face warrant: http://nyc-citation-pay.biz",
                "label": 1,
            },
            {
                "text": "Subject: Limited Time Offer: 90% OFF luxury Rolex, Gucci, and designer replica watches. Shop our private warehouse sale.",
                "label": 1,
            },
            # --- HAM SAMPLES ---
            {
                "text": "Subject: Team Sync Agenda - Thursday 10:00 AM. Hi team, please find attached the slide deck and notes for our sprint retrospective tomorrow morning.",
                "label": 0,
            },
            {
                "text": "Subject: Project Milestone Review. Hi Ankit, I reviewed the model evaluation report you shared. The precision metrics look solid, let us discuss the confusion matrix on Friday.",
                "label": 0,
            },
            {
                "text": "Subject: Your Amazon.com order #112-9843721 has shipped. Estimated delivery is Friday by 8:00 PM. You can track your package in your account.",
                "label": 0,
            },
            {
                "text": "Subject: College Computer Science Dept: Machine Learning Project Submission. Reminder that final project code repositories and presentation slides are due next Monday.",
                "label": 0,
            },
            {
                "text": "Subject: GitHub: [spam-detector] Pull request #4 merged into main by httpsankuu. All continuous integration tests passed successfully.",
                "label": 0,
            },
            {
                "text": "Subject: Lunch tomorrow? Hey, are you free around 1:00 PM to grab lunch near campus? Let me know if that time works for you.",
                "label": 0,
            },
            {
                "text": "Subject: Receipt for your recent Uber ride. Total: $14.25 charged to Visa ending in 4128. Thank you for riding with Uber.",
                "label": 0,
            },
            {
                "text": "Subject: Meeting minutes: Database migration plan. We agreed to execute the PostgreSQL upgrade during the scheduled maintenance window this weekend.",
                "label": 0,
            },
            {
                "text": "Subject: Python 3.12 release notes and deprecation notices. Check out the latest performance improvements in the GIL and typing modules.",
                "label": 0,
            },
            {
                "text": "Subject: Dr. Sharma's Office: Appointment confirmation for Wednesday at 3:30 PM. Please arrive 10 minutes early to complete check-in paperwork.",
                "label": 0,
            },
            {
                "text": "Subject: Monthly electricity utility bill statement. Your statement for September is now available online. Total amount due: $68.40 on October 5th.",
                "label": 0,
            },
            {
                "text": "Subject: Feedback on Machine Learning Lab 4. Your implementation of the TF-IDF vectorizer and Naive Bayes classifier met all rubric requirements.",
                "label": 0,
            },
            {
                "text": "Subject: Flight itinerary confirmation: Delta Air Lines flight DL452 departing JFK at 8:15 AM. Check-in opens 24 hours prior to departure.",
                "label": 0,
            },
            {
                "text": "Subject: Family dinner on Sunday. Mom wanted to know if you can make it to dinner around 7:00 PM this Sunday? Let us know!",
                "label": 0,
            },
            {
                "text": "Subject: Your Spotify Premium subscription receipt. $10.99 has been billed to your payment method for the billing cycle starting today.",
                "label": 0,
            },
        ]

        raw_df = pd.DataFrame(sample_data)
        return cls.normalize_dataframe(raw_df)

    @staticmethod
    def log_summary(df: pd.DataFrame, dataset_name: str = "Dataset") -> None:
        """Print summary statistics for a standardized DataFrame."""
        total = len(df)
        spam_count = int((df["label"] == 1).sum())
        ham_count = int((df["label"] == 0).sum())
        spam_ratio = (spam_count / total * 100) if total > 0 else 0.0

        logger.info(
            f"--- {dataset_name} Summary ---\n"
            f"  Total records: {total}\n"
            f"  Ham (0): {ham_count} ({100 - spam_ratio:.1f}%)\n"
            f"  Spam (1): {spam_count} ({spam_ratio:.1f}%)\n"
            f"  Columns: {list(df.columns)}"
        )


def main() -> int:
    """CLI execution for testing and generating starter sample data."""
    parser = argparse.ArgumentParser(description="Email dataset loader utility")
    parser.add_argument("--csv", type=str, help="Path to custom email CSV file")
    parser.add_argument("--text-col", type=str, default="text", help="Text column name")
    parser.add_argument("--label-col", type=str, default="label", help="Label column name")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate and export starter sample email corpus to data/raw/sample_emails.csv",
    )
    args = parser.parse_args()

    loader = EmailDataLoader()

    if args.sample or not args.csv:
        df = loader.get_sample_corpus()
        output_dir = Path("data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "sample_emails.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"Sample email dataset saved to: {out_path.resolve()}")
        loader.log_summary(df, "Sample Email Corpus")
        return 0

    if args.csv:
        df = loader.load_from_csv(args.csv, text_col=args.text_col, label_col=args.label_col)
        return 0

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
