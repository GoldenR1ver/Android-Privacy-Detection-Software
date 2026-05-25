"""Logging helpers for batch and pipeline runs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _FlushStreamHandler(logging.StreamHandler):
    """Stream handler that flushes after each record (visible progress in CLI)."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def configure_cli_logging(
    *,
    verbose: bool = False,
    log_file: Path | None = None,
    logger_name: str = "privacy_audit",
) -> logging.Logger:
    """Configure root privacy_audit logging to stdout (+ optional file). Idempotent."""
    global _CONFIGURED
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger(logger_name)
    root.setLevel(level)
    root.propagate = False

    if _CONFIGURED and not log_file:
        root.setLevel(level)
        return root

    if not _CONFIGURED:
        root.handlers.clear()
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
        stream_handler = _FlushStreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        root.addHandler(stream_handler)
        _CONFIGURED = True

    if log_file is not None:
        log_file = log_file.resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        existing_files = {
            getattr(h, "baseFilename", "")
            for h in root.handlers
            if isinstance(h, logging.FileHandler)
        }
        if str(log_file) not in existing_files:
            formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            root.addHandler(file_handler)

    root.setLevel(level)
    return root


def setup_file_logger(
    name: str,
    log_file: Path,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a logger that writes to *log_file* and stdout."""
    configure_cli_logging(verbose=level <= logging.DEBUG, log_file=log_file, logger_name="privacy_audit")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the configured privacy_audit tree."""
    configure_cli_logging()
    return logging.getLogger(name)
