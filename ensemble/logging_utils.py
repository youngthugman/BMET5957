"""Concise console and detailed timestamped file logging."""
import logging
from datetime import datetime
from pathlib import Path


def configure_logging(log_dir: Path, verbose=False):
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"ensemble_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger("ensemble")
    logger.handlers.clear(); logger.setLevel(logging.DEBUG); logger.propagate = False
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    console = logging.StreamHandler(); console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler); logger.addHandler(console)
    return logger, path
