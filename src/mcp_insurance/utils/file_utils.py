"""File and path helpers."""

import os

def ensure_absolute_path(path: str) -> str:
    """Convert relative path to absolute using current working directory."""
    return os.path.abspath(path)

def check_file_exists(path: str) -> bool:
    """Return True if file exists and is a regular file."""
    return os.path.isfile(path)
