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
    """Extract common ACORD policy fields from text using line-by-line parsing."""
    data = {
        "policy_number": None,
        "insured_name": None,
        "effective_date": None,
        "expiration_date": None,
        "additional_insured": None,
        "policy_type": None,
    }

    # Process line by line for exact field matches
    lines = text.split('\n')
    
    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        
        # Policy number
        if not data["policy_number"]:
            for prefix in ["policy number:", "policy no:", "policy #:", "policy id:"]:
                if line_lower.startswith(prefix):
                    data["policy_number"] = line_stripped.split(":", 1)[1].strip()
                    break
            if not data["policy_number"]:
                m = re.search(r'Policy\s*(?:No\.?|Number|ID|#)[:.\s]+([A-Z0-9\-]+)', line_stripped, re.I)
                if m:
                    data["policy_number"] = m.group(1).strip()
        
        # Named Insured
        if not data["insured_name"]:
            for prefix in ["named insured:", "insured:", "name of insured:"]:
                if line_lower.startswith(prefix):
                    value = line_stripped.split(":", 1)[1].strip()
                    # Remove trailing artifacts from adjacent lines
                    value = re.sub(r'\s+(Address|Policy|Effective|Mailing).*$', '', value, flags=re.I)
                    if value:
                        data["insured_name"] = value
                    break
        
        # Effective Date
        if not data["effective_date"]:
            for prefix in ["effective date:", "policy period from:"]:
                if line_lower.startswith(prefix):
                    data["effective_date"] = line_stripped.split(":", 1)[1].strip()
                    break
        
        # Expiration Date
        if not data["expiration_date"]:
            for prefix in ["expiration date:", "policy period to:"]:
                if line_lower.startswith(prefix):
                    data["expiration_date"] = line_stripped.split(":", 1)[1].strip()
                    break
        
        # Additional Insured
        if "additional insured" in line_lower:
            if ":" in line_stripped:
                after_colon = line_stripped.split(":", 1)[1].strip().lower()
                if after_colon in ["yes", "y", "true", "x"]:
                    data["additional_insured"] = "Yes"
            elif any(word in line_lower for word in ["yes", "true"]):
                data["additional_insured"] = "Yes"
        
        # Policy Type
        if not data["policy_type"]:
            for prefix in ["policy type:", "form name:"]:
                if line_lower.startswith(prefix):
                    value = line_stripped.split(":", 1)[1].strip()
                    # Stop at next section
                    value = re.sub(r'\s+(POLICY|TERMS|CONDITIONS|SECTION).*$', '', value, flags=re.I)
                    data["policy_type"] = value.strip()
                    break

    # Fallback for insured_name
    if not data["insured_name"] or len(data["insured_name"]) > 100:
        m = re.search(
            r'Named\s+Insured[:.\s]+([A-Z][A-Za-z0-9\s,\.&]+?)(?=\s+(?:Address|Mailing|Policy|Effective|$))', 
            text, re.I
        )
        if m:
            data["insured_name"] = m.group(1).strip()

    # Final cleanup
    for key in data:
        if data[key] and isinstance(data[key], str):
            data[key] = data[key].strip()
            if len(data[key]) > 200:
                data[key] = data[key][:200] + "..."

    return data