"""Synchronized Frida capture alongside Appium UI automation."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from privacy_audit.analyzers.dynamic.frida_runner import (
    _attach_session,
    build_hook_registry,
)
from privacy_audit.analyzers.dynamic.hook_loader import wrap_hook_script
from privacy_audit.utils.progress_log import log_countdown

logger = logging.getLogger("privacy_audit.frida")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FridaCaptureSession:
    """
    Hold an attached Frida session while Appium navigates.

    Timeline (real run):
      1. start_capture() — attach + load hooks
      2. wait_manual_grace(sec) — user may open screens manually
      3. (Appium runs externally)
      4. wait_tail(sec) — optional extra capture after UI stops
      5. stop_capture() — unload hooks and return events
    """

    def __init__(
        self,
        app_package: str,
        candidate_pii: list[str],
        taxonomy: dict[str, dict[str, list[str]]],
        hooks_dir: Path,
        launch_mode: str = "attach",
        app_name: str = "",
    ) -> None:
        self.app_package = app_package
        self.app_name = app_name
        self.launch_mode = launch_mode if launch_mode in {"attach", "spawn"} else "attach"
        self.registry = build_hook_registry(candidate_pii, taxonomy, hooks_dir)
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._scripts: list[Any] = []
        self._session: Any = None
        self._device: Any = None
        self._spawn_pid: int | None = None
        self._started = False
        self._error = ""
        self._started_at_utc = ""

    def _append_event(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(
                {
                    "timestamp_utc": _utc_now_iso(),
                    "pii_type": payload.get("pii_type", "unknown"),
                    "hook_file": payload.get("hook_file", ""),
                    "api_name": payload.get("api_name", ""),
                    "args_summary": payload.get("args_summary", ""),
                    "stack_summary": payload.get("stack_summary", ""),
                }
            )

    def _on_message(self, message: dict, _data: bytes | None) -> None:
        payload = message.get("payload", {})
        if isinstance(payload, dict):
            self._append_event(payload)
            return
        if message.get("type") == "error":
            self._append_event(
                {
                    "pii_type": "frida",
                    "hook_file": "",
                    "api_name": "hook_error",
                    "args_summary": str(message.get("description", message)),
                    "stack_summary": str(message.get("stack", ""))[:500],
                }
            )

    def start_capture(self) -> bool:
        """Attach and load hooks. Returns False on failure (error stored)."""
        if self._started:
            return not self._error
        self._started_at_utc = _utc_now_iso()
        try:
            import frida  # type: ignore
        except Exception as exc:
            self._error = f"Frida unavailable: {exc}"
            logger.error(self._error)
            return False

        try:
            logger.info(
                "Frida sync: connecting device, mode=%s package=%s hooks=%d",
                self.launch_mode,
                self.app_package,
                sum(len(v) for v in self.registry.values()),
            )
            self._device = frida.get_usb_device(timeout=8)
            if self.launch_mode == "spawn":
                self._spawn_pid = self._device.spawn([self.app_package])
                self._session = self._device.attach(self._spawn_pid)
            else:
                self._session = _attach_session(self._device, self.app_package, self.app_name)

            loaded_paths: set[str] = set()
            for pii_type, hook_paths in self.registry.items():
                for hook_path in hook_paths:
                    hook_key = str(hook_path.resolve())
                    if hook_key in loaded_paths:
                        continue
                    loaded_paths.add(hook_key)
                    logger.info("Frida sync: load hook %s (%s)", hook_path.name, pii_type)
                    script = self._session.create_script(
                        wrap_hook_script(hook_path.read_text(encoding="utf-8"))
                    )
                    script.on("message", self._on_message)
                    script.load()
                    self._scripts.append(script)
                    self._append_event(
                        {
                            "pii_type": pii_type,
                            "hook_file": str(hook_path),
                            "api_name": "hook_loaded",
                            "args_summary": "",
                            "stack_summary": "",
                        }
                    )

            if self.launch_mode == "spawn" and self._spawn_pid is not None:
                self._device.resume(self._spawn_pid)

            self._started = True
            logger.info("Frida sync: capture started at %s", self._started_at_utc)
            return True
        except Exception as exc:
            if self.launch_mode == "attach":
                label = self.app_name or self.app_package
                self._error = (
                    f"Frida attach failed: {exc}. "
                    f"Open {label} on the emulator and keep it in the foreground, then retry."
                )
            else:
                self._error = f"Frida spawn/attach failed: {exc}"
            logger.error(self._error)
            self._cleanup_scripts()
            return False

    def wait_manual_grace(self, grace_sec: int) -> None:
        """Initial window for manual navigation before Appium automation."""
        if grace_sec <= 0 or self._error:
            return
        logger.info(
            "Frida sync: manual grace %ds — operate the app now; Appium will start after",
            grace_sec,
        )
        log_countdown(logger, "Frida manual grace (before Appium)", grace_sec, interval_sec=1)

    def wait_tail(self, tail_sec: int) -> None:
        """Extra capture after Appium finishes (async API calls may lag)."""
        if tail_sec <= 0 or self._error or not self._started:
            return
        logger.info("Frida sync: post-Appium tail capture %ds", tail_sec)
        log_countdown(logger, "Frida tail (after Appium)", tail_sec, interval_sec=2)

    def stop_capture(self) -> dict[str, Any]:
        """Unload hooks and build frida_result dict."""
        if self._started:
            logger.info("Frida sync: stopping capture events=%d", len(self._events))
        self._cleanup_scripts()
        runtime_events = [
            e
            for e in self._events
            if str(e.get("api_name", "")) not in {"", "hook_loaded", "hook_error"}
        ]
        logger.info(
            "Frida sync: finished runtime_events=%d total_events=%d error=%s",
            len(runtime_events),
            len(self._events),
            self._error or "(none)",
        )
        return {
            "runner": "frida",
            "mode": "real",
            "launch_mode": self.launch_mode,
            "package": self.app_package,
            "registered_hooks": {k: [str(p) for p in v] for k, v in self.registry.items()},
            "events": list(self._events),
            "error": self._error,
            "sync_mode": True,
            "capture_started_at_utc": self._started_at_utc,
            "runtime_event_count": len(runtime_events),
        }

    def _cleanup_scripts(self) -> None:
        for script in self._scripts:
            try:
                script.unload()
            except Exception:
                pass
        self._scripts.clear()
        if self._session is not None:
            try:
                self._session.detach()
            except Exception:
                pass
            self._session = None
