"""Centralized logging configuration for the submission parsing module"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "mcp_submission_parsing",
    level: int = logging.DEBUG,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Set up and return a logger with console handler.
    
    Args:
        name: Logger name (typically __name__ of the module)
        level: Logging level (default: DEBUG)
        log_file: Optional file path to log to file as well
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

default_logger = setup_logger("mcp_submission_parsing")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Optional logger name (if None, returns default logger)
    
    Returns:
        Logger instance
    """
    if name is None:
        return default_logger
    return setup_logger(name)