# core/pdf_extractor.py
"""
PDF text extraction and policy field parsing.
"""

import re
from pathlib import Path
from typing import Dict, Optional

import pdfplumber
from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    text_parts = []

    # Primary: pdfplumber
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                # extract tables as CSV-like strings (optional)
                tables = page.extract_tables()
                for table in tables:
                    table_lines = []
                    for row in table:
                        row_text = " | ".join(str(c) for c in row if c and str(c).strip())
                        if row_text:
                            table_lines.append(row_text)
                    if table_lines:
                        text_parts.append("\n".join(table_lines))
    except Exception as e:
        print(f"[WARN] pdfplumber failed: {e}")

    # Fallback: PyPDF
    if not "".join(text_parts).strip():
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        except Exception as e:
            print(f"[WARN] PyPDF failed: {e}")

    return "\n".join(text_parts).strip()


def parse_policy_data(text: str) -> Dict[str, Optional[str]]:
    """Extract common ACORD policy fields from text using regex."""
    normalized = re.sub(r'\s+', ' ', text)
    data = {
        "policy_number": None,
        "insured_name": None,
        "effective_date": None,
        "expiration_date": None,
        "additional_insured": None,
        "policy_type": None,
    }

    # Policy number
    for pat in [
        r'Policy\s*(?:No\.?|Number)[:.\s]+([A-Z0-9\-]+)',
        r'Policy\s*ID[:.\s]+([A-Z0-9\-]+)',
        r'Policy\s*#[:.\s]+([A-Z0-9\-]+)',
    ]:
        m = re.search(pat, normalized, re.I)
        if m:
            data["policy_number"] = m.group(1).strip()
            break

    # Insured name
    for pat in [
        r'Named\s+Insured[:.\s]+([A-Z][A-Za-z0-9\s,\.&]+?)(?=\s+(?:Address|Policy|Effective|For|$))',
        r'Insured[:.\s]+([A-Z][A-Za-z0-9\s,\.&]+?)(?=\s+(?:Address|Policy|$))',
        r'Name\s+of\s+Insured[:.\s]+([A-Z][A-Za-z0-9\s,\.&]+)',
    ]:
        m = re.search(pat, normalized, re.I)
        if m:
            data["insured_name"] = m.group(1).strip()
            break

    # Date patterns: mm/dd/yyyy, mm-dd-yyyy, yyyy-mm-dd
    date_pat = r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{4}|\d{4}[/\-]\d{1,2}[/\-]\d{1,2})'

    # Effective date
    for pat in [
        r'Effective\s+Date[:.\s]+' + date_pat,
        r'Policy\s+Period\s+From[:.\s]+' + date_pat,
    ]:
        m = re.search(pat, normalized, re.I)
        if m:
            data["effective_date"] = m.group(1)
            break

    # Expiration date
    for pat in [
        r'Expiration\s+Date[:.\s]+' + date_pat,
        r'Policy\s+Period\s+To[:.\s]+' + date_pat,
    ]:
        m = re.search(pat, normalized, re.I)
        if m:
            data["expiration_date"] = m.group(1)
            break

    # Additional insured
    if re.search(r'Additional\s+Insured', normalized, re.I):
        data["additional_insured"] = "Yes"

    # Policy type
    for pat in [
        r'Policy\s+Type[:.\s]+([A-Za-z\s]+)',
        r'Form\s+Name[:.\s]+([A-Za-z\s]+)',
    ]:
        m = re.search(pat, normalized, re.I)
        if m:
            data["policy_type"] = m.group(1).strip()
            break

    return data