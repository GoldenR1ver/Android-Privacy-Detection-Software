"""Thread-safe reactor for Frida callbacks (ported from frida-tools' pattern).

Frida delivers ``script.on("message")`` callbacks on its own thread. Mutating
shared state directly from there is race-prone. The :class:`Reactor` marshals
every callback onto a single worker thread so handlers run serially, mirroring
``frida_tools.reactor.Reactor`` used by ``frida-trace``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional


class Reactor:
    """Single-threaded work queue drained by a background worker.

    Usage::

        reactor = Reactor()
        reactor.start()
        ...
        reactor.schedule(lambda: do_work())
        ...
        reactor.stop()  # drains remaining work then joins
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: "deque[tuple[Callable[[], None], float]]" = deque()
        self._running = False
        self._worker: Optional[threading.Thread] = None

    def schedule(self, fn: Callable[[], None], delay: Optional[float] = None) -> None:
        """Queue ``fn`` to run on the reactor thread (optionally after ``delay`` s)."""
        when = time.time() + (delay or 0.0)
        with self._lock:
            self._pending.append((fn, when))
            self._cond.notify()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(
            target=self._run, name="frida-reactor", daemon=True
        )
        self._worker.start()

    def _next_due(self) -> Optional[float]:
        if not self._pending:
            return None
        return min(when for _, when in self._pending)

    def _run(self) -> None:
        while True:
            ready: list[Callable[[], None]] = []
            with self._lock:
                while self._running and not self._has_due():
                    self._cond.wait(timeout=0.5)
                if not self._running and not self._pending:
                    return
                now = time.time()
                remaining: "deque[tuple[Callable[[], None], float]]" = deque()
                while self._pending:
                    fn, when = self._pending.popleft()
                    if when <= now:
                        ready.append(fn)
                    else:
                        remaining.append((fn, when))
                self._pending = remaining
            for fn in ready:
                try:
                    fn()
                except Exception:  # pragma: no cover - handler must not kill loop
                    pass

    def _has_due(self) -> bool:
        now = time.time()
        return any(when <= now for _, when in self._pending)

    def stop(self, drain_timeout: float = 3.0) -> None:
        """Stop the worker after draining pending work (best-effort)."""
        deadline = time.time() + drain_timeout
        # Give already-queued callbacks a chance to flush.
        while time.time() < deadline:
            with self._lock:
                if not self._pending:
                    break
            time.sleep(0.05)
        with self._lock:
            self._running = False
            self._cond.notify_all()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=2.0)
        self._worker = None
