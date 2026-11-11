"""Synchronized Frida capture alongside Appium UI automation.

Rebuilt around the frida-tools host pattern:

* **Reactor** — every ``script.on("message")`` callback is marshalled onto a
  single worker thread (see :mod:`frida_reactor`), so event handling is serial
  and race-free, exactly like ``frida-trace``.
* **Spawn-gated resume** — in ``spawn`` mode the app is spawned *paused*, hook
  scripts are loaded, then the process is *resumed* and we wait for
  ``java_active`` / ``papi_ready`` before the manual-grace window. Cold-start
  identifier reads are caught once ART is live.
* **Data-driven hooking** — ``hook_papi.js`` receives an ``api_targets`` spec
  via RPC and hooks every overload (PAPIMonitor engine) plus wildcard patterns
  (``Java.enumerateMethods``), all emitting the project's unified event schema.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from privacy_audit.analyzers.dynamic.frida_reactor import Reactor
from privacy_audit.analyzers.dynamic.frida_runner import (
    PRE_ATTACH_WARMUP_SEC,
    _attach_session,
    build_hook_registry,
    build_papi_spec,
    resolve_code_prefixes,
    spawn_package_cold,
)
from privacy_audit.analyzers.dynamic.hook_loader import (
    is_frida_runtime_api,
    papi_agent_path,
    resolve_global_hooks,
    wrap_hook_script,
)
from privacy_audit.utils.progress_log import log_countdown

logger = logging.getLogger("privacy_audit.frida")

# After spawn resume, ART must start before Java.perform / papi hooks activate.
SPAWN_POST_RESUME_READY_SEC = 30.0
PAPI_READY_TIMEOUT_ATTACH_SEC = 18.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _install_papi_hooks(script: Any, spec: list[dict[str, str]]) -> int:
    """Call the agent's installhooks RPC export (frida 16+ uses exports_sync)."""
    exports = getattr(script, "exports_sync", None) or getattr(script, "exports", None)
    if exports is None:
        raise RuntimeError("Frida script exposes no RPC exports")
    return int(exports.installhooks(spec))


class FridaCaptureSession:
    """Hold a Frida session open while Appium navigates.

    Timeline (real run):
      1. start_capture()      — attach/spawn(paused) + load all hooks + install papi
                                (spawn) resume only after hooks are ready
      2. wait_manual_grace()  — user may trigger screens manually
      3. (Appium runs externally)
      4. wait_tail()          — extra capture for async API/network calls
      5. stop_capture()       — unload hooks, stop reactor, return events
    """

    def __init__(
        self,
        app_package: str,
        candidate_pii: list[str],
        taxonomy: dict[str, Any],
        hooks_dir: Path,
        launch_mode: str = "attach",
        app_name: str = "",
        code_prefixes: list[str] | None = None,
        pre_attach_warm_sec: float = PRE_ATTACH_WARMUP_SEC,
        spawn_fallback_attach: bool = True,
        frida_options: dict[str, Any] | None = None,
    ) -> None:
        self.app_package = app_package
        self.app_name = app_name
        self.code_prefixes = code_prefixes or resolve_code_prefixes(app_package)
        self.pre_attach_warm_sec = pre_attach_warm_sec
        self.requested_launch_mode = launch_mode if launch_mode in {"attach", "spawn"} else "attach"
        self.launch_mode = self.requested_launch_mode
        self.spawn_fallback_attach = spawn_fallback_attach
        self.frida_options = frida_options or {}
        self.hooks_dir = hooks_dir
        self.registry = build_hook_registry(candidate_pii, taxonomy, hooks_dir)
        self.papi_spec = build_papi_spec(candidate_pii, taxonomy)
        self.global_hooks = resolve_global_hooks(hooks_dir)
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._reactor = Reactor()
        self._scripts: list[Any] = []
        self._session: Any = None
        self._device: Any = None
        self._spawn_pid: int | None = None
        self._started = False
        self._error = ""
        self._started_at_utc = ""
        self._papi_installed = 0
        self._papi_ready = False
        self._java_active = False

    # ---- event handling (runs on reactor thread) -------------------------

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

    def _handle_message(self, message: dict, _data: bytes | None) -> None:
        payload = message.get("payload", {})
        if isinstance(payload, dict):
            api_name = str(payload.get("api_name", ""))
            if api_name == "papi_ready":
                self._papi_ready = True
            elif api_name == "java_active":
                self._java_active = True
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

    def _on_message(self, message: dict, data: bytes | None) -> None:
        # Marshal onto the reactor thread (frida-tools pattern).
        self._reactor.schedule(lambda: self._handle_message(message, data))

    # ---- script loading --------------------------------------------------

    def _load_wrapped_hook(self, hook_path: Path, pii_type: str) -> None:
        logger.info("Frida sync: load hook %s (%s)", hook_path.name, pii_type)
        script = self._session.create_script(
            wrap_hook_script(hook_path.read_text(encoding="utf-8"), hook_file=str(hook_path))
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

    def _load_papi_agent(self) -> None:
        agent = papi_agent_path(self.hooks_dir)
        if agent is None or not self.papi_spec:
            return
        logger.info(
            "Frida sync: load data-driven agent %s (%d targets)",
            agent.name,
            len(self.papi_spec),
        )
        # Loaded RAW (not wrapped): the agent manages its own Java-ready wait.
        script = self._session.create_script(agent.read_text(encoding="utf-8"))
        script.on("message", self._on_message)
        script.load()
        self._scripts.append(script)
        try:
            self._papi_installed = _install_papi_hooks(script, self.papi_spec)
            logger.info("Frida sync: papi installhooks accepted=%d", self._papi_installed)
        except Exception as exc:
            logger.warning("Frida sync: papi installhooks failed: %s", exc)

    def _hook_runtime_is_ready(self) -> bool:
        """True once ART is up and data-driven hooks have finished installing."""
        if self._papi_ready:
            return True
        if not self.papi_spec and self._java_active:
            return True
        return False

    def _wait_hook_runtime_ready(self, timeout_sec: float, *, phase: str) -> bool:
        deadline = time.monotonic() + max(0.5, timeout_sec)
        while time.monotonic() < deadline:
            if self._hook_runtime_is_ready():
                logger.info(
                    "Frida sync: hook runtime ready (%s) java_active=%s papi_ready=%s",
                    phase,
                    self._java_active,
                    self._papi_ready,
                )
                return True
            time.sleep(0.15)
        logger.warning(
            "Frida sync: hook runtime not ready within %.1fs after %s "
            "(java_active=%s papi_ready=%s; proceeding anyway)",
            timeout_sec,
            phase,
            self._java_active,
            self._papi_ready,
        )
        return False

    def _begin_spawn_session(self) -> None:
        assert self._device is not None
        self._spawn_pid = spawn_package_cold(self._device, self.app_package)
        self._session = self._device.attach(self._spawn_pid)

    # ---- lifecycle -------------------------------------------------------

    def start_capture(self) -> bool:
        """Attach/spawn, install all hooks, then (spawn) resume. False on failure."""
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
                "Frida sync: connecting device, mode=%s package=%s pii_hooks=%d "
                "global_hooks=%d papi_targets=%d",
                self.launch_mode,
                self.app_package,
                sum(len(v) for v in self.registry.values()),
                len(self.global_hooks),
                len(self.papi_spec),
            )
            self._reactor.start()
            self._device = frida.get_usb_device(timeout=8)

            if self.launch_mode == "spawn":
                logger.info(
                    "Frida sync: cold spawn %s (force-stop + paused hook install)",
                    self.app_package,
                )
                try:
                    self._begin_spawn_session()
                except Exception as spawn_exc:
                    if self.spawn_fallback_attach:
                        logger.warning(
                            "Frida sync: spawn failed (%s); falling back to attach",
                            spawn_exc,
                        )
                        self.launch_mode = "attach"
                        self._spawn_pid = None
                        self._session = _attach_session(
                            self._device,
                            self.app_package,
                            self.app_name,
                            self.code_prefixes,
                            pre_warm_sec=self.pre_attach_warm_sec,
                            accept_java_timeout_on_display=bool(
                                self.frida_options.get(
                                    "frida_accept_java_timeout_on_display", True
                                )
                            ),
                        )
                    else:
                        raise
            else:
                self._session = _attach_session(
                    self._device,
                    self.app_package,
                    self.app_name,
                    self.code_prefixes,
                    pre_warm_sec=self.pre_attach_warm_sec,
                    accept_java_timeout_on_display=bool(
                        self.frida_options.get(
                            "frida_accept_java_timeout_on_display", True
                        )
                    ),
                )

            loaded_paths: set[str] = set()
            for pii_type, hook_paths in self.registry.items():
                for hook_path in hook_paths:
                    hook_key = str(hook_path.resolve())
                    if hook_key in loaded_paths:
                        continue
                    loaded_paths.add(hook_key)
                    self._load_wrapped_hook(hook_path, pii_type)

            for hook_path in self.global_hooks:
                hook_key = str(hook_path.resolve())
                if hook_key in loaded_paths:
                    continue
                loaded_paths.add(hook_key)
                self._load_wrapped_hook(hook_path, "network_transmission")

            self._load_papi_agent()

            # Spawn: scripts load while paused, but Java/papi only run after resume.
            if self.launch_mode == "spawn" and self._spawn_pid is not None:
                logger.info(
                    "Frida sync: resuming spawned pid=%s (then wait for Java/papi)",
                    self._spawn_pid,
                )
                self._device.resume(self._spawn_pid)
                self._wait_hook_runtime_ready(
                    SPAWN_POST_RESUME_READY_SEC,
                    phase="spawn_post_resume",
                )
            else:
                self._wait_hook_runtime_ready(
                    PAPI_READY_TIMEOUT_ATTACH_SEC,
                    phase="attach",
                )

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
            self._reactor.stop()
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
        """Extra capture after Appium finishes (async API/network calls may lag)."""
        if tail_sec <= 0 or self._error or not self._started:
            return
        logger.info("Frida sync: post-Appium tail capture %ds", tail_sec)
        log_countdown(logger, "Frida tail (after Appium)", tail_sec, interval_sec=2)

    def stop_capture(self) -> dict[str, Any]:
        """Unload hooks, stop the reactor, and build the frida_result dict."""
        if self._started:
            logger.info("Frida sync: stopping capture events=%d", len(self._events))
        self._cleanup_scripts()
        # Drain any in-flight events scheduled on the reactor before snapshotting.
        self._reactor.stop()
        with self._lock:
            events_snapshot = list(self._events)
        runtime_events = [
            e
            for e in events_snapshot
            if is_frida_runtime_api(str(e.get("api_name", "")))
        ]
        java_active_count = sum(
            1 for e in events_snapshot if str(e.get("api_name", "")) == "java_active"
        )
        logger.info(
            "Frida sync: finished runtime_events=%d java_active=%d total_events=%d "
            "papi_targets=%d error=%s",
            len(runtime_events),
            java_active_count,
            len(events_snapshot),
            self._papi_installed,
            self._error or "(none)",
        )
        return {
            "runner": "frida",
            "mode": "real",
            "launch_mode": self.launch_mode,
            "requested_launch_mode": self.requested_launch_mode,
            "package": self.app_package,
            "registered_hooks": {k: [str(p) for p in v] for k, v in self.registry.items()},
            "global_hooks": [str(p) for p in self.global_hooks],
            "papi_target_count": len(self.papi_spec),
            "papi_installed_accepted": self._papi_installed,
            "events": events_snapshot,
            "error": self._error,
            "sync_mode": True,
            "capture_started_at_utc": self._started_at_utc,
            "runtime_event_count": len(runtime_events),
            "java_active_count": java_active_count,
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
