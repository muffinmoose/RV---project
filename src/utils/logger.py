# utils/logger.py
# Simple logger for 9HPT analysis
# Logs to both terminal and a log file simultaneously

import logging
import sys
from pathlib import Path
from datetime import datetime

from config import RESULTS_DIR


def setup_logger(patient_id: str = "", video_name: str = "") -> logging.Logger:
    """
    Set up and return the project logger.
    Creates a log file under data/results/<patient_id>/

    Usage:
        from utils.logger import setup_logger
        log = setup_logger(patient_id="patient_001", video_name="patient_001camP_1_...")
        log.info("Starting analysis")
        log.warning("Checkerboard not found")
        log.error("Video could not be opened")
    """
    logger = logging.getLogger("9HPT")
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Terminal handler ───────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # ── File handler ──────────────────────────────────────────────────────────
    log_dir = Path(RESULTS_DIR) / (patient_id or "general")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem      = Path(video_name).stem if video_name else "session"
    log_path  = log_dir / f"{stem}_{timestamp}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)   # log everything to file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger started — {log_path}")
    return logger


def get_logger() -> logging.Logger:
    """
    Get the existing logger anywhere in the codebase without passing it around.

    Usage:
        from utils.logger import get_logger
        log = get_logger()
        log.info("something happened")
    """
    return logging.getLogger("9HPT")