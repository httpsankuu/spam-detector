"""
cleaner.py - Raw Text Sanitization and Cleaning Pipeline for Emails and SMS.

Removes HTML tags, email headers, URLs, phone numbers, currency symbols, and
non-alphabetic characters while normalizing whitespace and case.
"""

from __future__ import annotations

import re
from typing import Optional


# Precompiled regular expressions for high-throughput regex cleaning
HTML_TAG_RE = re.compile(r"<[^>]+>")
EMAIL_HEADER_RE = re.compile(r"^(subject|from|to|cc|date|reply-to):\s*", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_ADDR_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
CURRENCY_RE = re.compile(r"[$£€¥₹]")
NUMBER_RE = re.compile(r"\b\d+\b")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
EXTRA_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(
    text: Optional[str],
    remove_html: bool = True,
    remove_headers: bool = True,
    replace_urls: bool = True,
    replace_emails: bool = True,
    replace_currency: bool = True,
    replace_numbers: bool = True,
    lowercase: bool = True,
) -> str:
    """
    Clean and sanitize raw email or SMS text.

    Args:
        text: Raw text string (handles None gracefully)
        remove_html: If True, strip HTML tags (e.g. <b>, <a href=...>)
        remove_headers: If True, strip leading header keywords (e.g. Subject:)
        replace_urls: If True, replace URLs with token 'httpaddr'
        replace_emails: If True, replace email addresses with 'emailaddr'
        replace_currency: If True, replace currency signs ($£€) with 'dollar'
        replace_numbers: If True, replace standalone digits with 'number'
        lowercase: If True, convert text to lowercase

    Returns:
        Sanitized text string
    """
    if not isinstance(text, str):
        return ""

    cleaned = text

    # Strip HTML tags
    if remove_html:
        cleaned = HTML_TAG_RE.sub(" ", cleaned)

    # Strip standard email header prefixes
    if remove_headers:
        cleaned = EMAIL_HEADER_RE.sub("", cleaned)

    # Replace URLs
    if replace_urls:
        cleaned = URL_RE.sub(" httpaddr ", cleaned)

    # Replace email addresses
    if replace_emails:
        cleaned = EMAIL_ADDR_RE.sub(" emailaddr ", cleaned)

    # Replace currency symbols (often strong spam triggers)
    if replace_currency:
        cleaned = CURRENCY_RE.sub(" dollar ", cleaned)

    # Replace standalone numbers
    if replace_numbers:
        cleaned = NUMBER_RE.sub(" number ", cleaned)

    # Convert to lowercase
    if lowercase:
        cleaned = cleaned.lower()

    # Remove non-alphabetical punctuation (retaining spaces and letters)
    cleaned = NON_ALPHA_RE.sub(" ", cleaned)

    # Collapse repeated whitespace
    cleaned = EXTRA_WHITESPACE_RE.sub(" ", cleaned).strip()

    return cleaned


def main() -> None:
    """Self-check demo for text cleaner."""
    sample_raw = (
        "Subject: URGENT! You won $1,000,000 in the Lottery! "
        "Visit <b>http://spam-winner.org/claim?id=4920</b> or email claim@lottery.org "
        "to get your cash now!!!"
    )
    print("--- RAW TEXT ---")
    print(sample_raw)
    print("\n--- CLEANED TEXT ---")
    cleaned = clean_text(sample_raw)
    print(cleaned)
    assert "httpaddr" in cleaned, "URL placeholder missing"
    assert "dollar" in cleaned, "Currency placeholder missing"
    assert "emailaddr" in cleaned, "Email placeholder missing"
    assert "<b" not in cleaned, "HTML tag was not removed"
    print("\nClean_text verification passed!")


if __name__ == "__main__":
    main()
