"""
Server-side logging configuration for Mini SOC.
Provides structured, consistent logging across all server modules.
"""
from __future__ import annotations

import logging
import os
import sys


def setup_logging(level: str | None = None) -> logging.Logger:
    """
    Configure and return the root logger for the Mini SOC server.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR). Defaults to
               the LOG_LEVEL env var, or INFO if not set.

    Returns:
        Configured logger instance.
    """
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger("mini_soc")
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.handlers.clear()
    logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logger


# Module-level logger — import this in other server modules
logger = setup_logging()
