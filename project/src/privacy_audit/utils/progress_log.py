"""Timed step logging for long-running pipelines."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def log_step(logger: logging.Logger, message: str) -> Iterator[None]:
    """Log step start/end with elapsed seconds."""
    logger.info(">>> %s", message)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception("<<< FAILED %s (%.1fs)", message, elapsed)
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info("<<< done %s (%.1fs)", message, elapsed)


def log_countdown(logger: logging.Logger, label: str, total_sec: int, interval_sec: int = 5) -> None:
    """Log periodic progress during a blocking sleep."""
    if total_sec <= 0:
        return
    interval = max(1, interval_sec if interval_sec > 0 else min(5, total_sec))
    elapsed = 0
    while elapsed < total_sec:
        chunk = min(interval, total_sec - elapsed)
        time.sleep(chunk)
        elapsed += chunk
        logger.info("%s: %d/%ds", label, elapsed, total_sec)
