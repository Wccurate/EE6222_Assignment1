"""Simple logging setup."""

from __future__ import annotations

import logging
from pathlib import Path


def build_logger(log_file: Path) -> logging.Logger:
    """Create logger writing to both stdout and a file."""
    logger = logging.getLogger("ee6222_dr")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers in repeated CLI/test calls.
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
