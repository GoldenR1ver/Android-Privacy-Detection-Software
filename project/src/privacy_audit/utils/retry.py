"""Retry utilities with optional backoff."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def run_with_retry(
    func: Callable[[], T],
    retries: int = 2,
    delay_sec: float = 1.0,
    backoff: float = 2.0,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Execute *func* up to *retries + 1* times with exponential backoff."""
    last_error: Exception | None = None
    wait = max(0.0, delay_sec)
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - runtime dependent
            last_error = exc
            if attempt >= retries:
                break
            if on_retry is not None:
                on_retry(attempt + 1, exc)
            if wait > 0:
                time.sleep(wait)
            wait *= max(1.0, backoff)
    assert last_error is not None
    raise last_error
