# src/utils/logger.py
# ============================================================
# Centralised logging for AIML_Exoplanet pipeline
# Usage:
#   from src.utils.logger import get_logger
#   log = get_logger(__name__)
#   log.info("Processing TIC %s", tic_id)
# ============================================================

import logging
import os
import sys
from datetime import datetime


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """
    Return a named logger that writes to both console and (optionally) a file.

    Parameters
    ----------
    name      : Module name — pass __name__ from calling module.
    log_file  : Absolute path to log file. If None, uses config.LOG_FILE.
                Pass False to disable file logging.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Don't add handlers twice if already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # ── File handler ─────────────────────────────────────────
    if log_file is not False:
        if log_file is None:
            try:
                from config import LOG_FILE
                log_file = LOG_FILE
            except ImportError:
                log_file = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "outputs", "pipeline.log"
                )

        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_tic_logger(tic_id: int | str, module_name: str) -> logging.Logger:
    """
    Convenience wrapper that prefixes every log record with the TIC ID.

    Usage:
        log = get_tic_logger(420814525, __name__)
        log.info("Downloading sector 1")
        # → 2026-10-10 11:00:00 | INFO     | src.ingestion.fetcher | [TIC 420814525] Downloading sector 1
    """
    logger = get_logger(module_name)

    class TICFilter(logging.Filter):
        def filter(self, record):
            record.msg = f"[TIC {tic_id}] {record.msg}"
            return True

    # Apply filter only if not already set for this TIC
    existing_filters = [f for f in logger.filters if isinstance(f, TICFilter)]
    if not existing_filters:
        logger.addFilter(TICFilter())

    return logger
