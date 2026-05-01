"""Simple logger setup, uses rich for pretty output when available."""
from __future__ import annotations

import logging
import sys


def setup_logger(name: str = "raregraph", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    try:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
    except ImportError:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
