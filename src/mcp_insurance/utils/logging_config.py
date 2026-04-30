"""Structured logging configuration (e.g., JSON logs)."""

import logging

def setup_logging(level=logging.INFO):
    """Configure logging with structured output."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # TODO: add JSON formatter if needed
