# utils/logger.py
# Simple logger for 9HPT analysis
# Logs to both terminal and a log file simultaneously
#
# Usage:
#   log = setup_logger(patient_id="patient_001", video_name="...", log_dir="results/patient_001/video_stem")
#   log.info("Starting analysis")
#   log.warning("Something off")
#   log.error("Critical failure")
#
# From anywhere else in codebase:
#   from utils.logger import get_logger
#   log = get_logger()

import logging
import sys
from pathlib import Path
from datetime import datetime

from config import RESULTS_DIR


def setup_logger(patient_id: str = "", video_name: str = "",
                 log_dir: str = "") -> logging.Logger:
    """
    Set up and return the project logger.
    Logs to terminal (INFO+) and to a .log file (DEBUG+).

    Args:
        patient_id:  used for default log folder if log_dir not provided
        video_name:  used for log filename
        log_dir:     if provided, log file goes here (video-specific subfolder)
    """
    logger = logging.getLogger("9HPT")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if called multiple times
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
    # Use provided log_dir (video subfolder) or fallback to patient folder
    if log_dir:
        out_dir = Path(log_dir)
    else:
        out_dir = Path(RESULTS_DIR) / (patient_id or "general")
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem      = Path(video_name).stem if video_name else "session"
    log_path  = out_dir / f"{stem}_{timestamp}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger started — {log_path}")
    return logger


def get_logger() -> logging.Logger:
    """
    Get existing logger from anywhere in codebase without passing it around.
    Must call setup_logger() first.
    """
    return logging.getLogger("9HPT")