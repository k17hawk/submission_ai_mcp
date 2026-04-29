# tools/parsing.py
"""
MCP tools for parsing ACORD submission PDFs and extracting policy data.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from core.pdf_extractor import extract_text_from_pdf, parse_policy_data


async def parse_acord_submission(pdf_path: str) -> Dict[str, Any]:
    """
    MCP Tool: Extract full text and structured policy data from an ACORD PDF.

    Args:
        pdf_path: Path to the ACORD form PDF file.

    Returns:
        Dictionary with keys:
            - text: full extracted text
            - policy_data: parsed policy fields
            - error: optional error message
    """
    try:
        path = Path(pdf_path)
        if not path.exists():
            return {"error": f"File not found: {pdf_path}"}

        text = extract_text_from_pdf(pdf_path)
        policy_data = parse_policy_data(text)

        return {
            "text": text,
            "policy_data": policy_data,
            "error": None
        }
    except Exception as e:
        return {"error": str(e), "text": None, "policy_data": None}


async def extract_policy_data(pdf_path: str) -> Dict[str, Optional[str]]:
    """
    MCP Tool: Extract only structured policy fields (no full text).

    Args:
        pdf_path: Path to the ACORD form PDF file.

    Returns:
        Dictionary of policy fields: policy_number, insured_name, etc.
    """
    try:
        text = extract_text_from_pdf(pdf_path)
        return parse_policy_data(text)
    except Exception as e:
        return {"error": str(e)}