"""Frida execution layer with hook auto-registration."""

from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from privacy_audit.analyzers.dynamic.hook_loader import JAVA_READY_PROBE, wrap_hook_script
from privacy_audit.utils.progress_log import log_countdown

logger = logging.getLogger("privacy_audit.frida")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_hook_registry(
    candidate_pii: list[str],
    taxonomy: dict[str, dict[str, list[str]]],
    hooks_dir: Path,
) -> dict[str, list[Path]]:
    """Resolve hook files by pii type from taxonomy configuration."""
    registry: dict[str, list[Path]] = {}
    for pii_type in candidate_pii:
        meta = taxonomy.get(pii_type, {})
        hook_names = meta.get("hooks", [])
        paths: list[Path] = []
        for hook_name in hook_names:
            hook_path = hooks_dir / hook_name
            if hook_path.exists():
                paths.append(hook_path)
        if paths:
            registry[pii_type] = paths
    return registry


def _simulate_events(registry: dict[str, list[Path]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for pii_type, hook_paths in registry.items():
        for hook in hook_paths:
            events.append(
                {
                    "timestamp_utc": _utc_now_iso(),
                    "pii_type": pii_type,
                    "hook_file": str(hook),
                    "api_name": hook.stem,
                    "args_summary": "dry_run_args",
                    "stack_summary": "dry_run_stack",
                }
            )
    return events


def _pids_from_adb_pidof(app_package: str) -> list[int]:
    try:
        proc = subprocess.run(
            ["adb", "shell", "pidof", app_package],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        logger.debug("adb pidof failed: %s", exc)
        return []
    return [int(tok) for tok in (proc.stdout or "").split() if tok.isdigit()]


def _pick_jvm_pid(
    pids: list[int],
    device: Any,
    app_package: str,
    app_name: str = "",
) -> int | None:
    """
    MuMu: pidof lists parent first (Frida name often ``Keep``); the child
    ``com.gotokeep.keep`` frequently refuses frida-agent injection.
    """
    if not pids:
        return None
    names: dict[int, str] = {}
    for proc in device.enumerate_processes():
        pid = int(getattr(proc, "pid", 0) or 0)
        if pid in pids:
            names[pid] = (getattr(proc, "name", "") or "").strip()
    display = (app_name or "").strip()
    if display:
        for pid in pids:
            if names.get(pid) == display:
                return pid
    for pid in pids:
        if names.get(pid) == app_package:
            continue
        return pid
    return pids[0]


def _main_pid_from_adb(
    app_package: str,
    device: Any | None = None,
    app_name: str = "",
) -> int | None:
    """Resolve attach PID via adb pidof (+ Frida name disambiguation on MuMu)."""
    pids = _pids_from_adb_pidof(app_package)
    if not pids:
        return None
    if len(pids) == 1:
        return pids[0]
    if device is not None:
        return _pick_jvm_pid(pids, device, app_package, app_name=app_name)
    return pids[0]


def _resolve_attach_candidates(
    device: Any,
    app_package: str,
    app_name: str = "",
) -> list[tuple[int, int, str]]:
    """Return (score, pid, label) sorted best-first; package-named processes only."""
    display = (app_name or "").strip()
    scored: list[tuple[int, int, str]] = []
    for proc in device.enumerate_processes():
        pid = int(getattr(proc, "pid", 0) or 0)
        if not pid:
            continue
        identifier = (getattr(proc, "identifier", "") or "").strip()
        name = (getattr(proc, "name", "") or "").strip()
        label = name or identifier
        score: int | None = None
        if display and name == display and name != app_package:
            score = 0
        elif identifier == app_package and name == identifier:
            score = 0
        elif identifier == app_package and ":" not in name:
            score = 1
        elif name == app_package:
            score = 3
        elif identifier == app_package:
            score = 2
        if score is None:
            continue
        lower = label.lower()
        # Sub-processes are not the ART runtime we need for Java hooks.
        if ":" in name and name != app_package:
            continue
        if ":core" in lower or ":push" in lower or "webview" in lower or "sandbox" in lower:
            score += 5
        scored.append((score, pid, label))
    scored.sort(key=lambda row: (row[0], row[1]))
    return scored


def _process_name_map(device: Any, pids: list[int]) -> dict[int, str]:
    names: dict[int, str] = {}
    for proc in device.enumerate_processes():
        pid = int(getattr(proc, "pid", 0) or 0)
        if pid in pids:
            names[pid] = (getattr(proc, "name", "") or "").strip()
    return names


def _verify_java_session(session: Any, timeout_sec: float = 20.0) -> bool:
    """Return True when ART is up and Java.perform works in the attached process."""
    import threading

    ready = threading.Event()
    ok = False

    def on_message(message: dict, _data: bytes | None) -> None:
        nonlocal ok
        payload = message.get("payload", {})
        if isinstance(payload, dict) and payload.get("api_name") == "java_ready":
            ok = True
            ready.set()

    script = session.create_script(JAVA_READY_PROBE)
    script.on("message", on_message)
    script.load()
    ready.wait(timeout=timeout_sec)
    try:
        script.unload()
    except Exception:
        pass
    return ok


def _ordered_attach_pids(
    device: Any,
    app_package: str,
    app_name: str,
) -> list[tuple[int, str]]:
    """Build attach order: pidof list first, then scored Frida candidates."""
    ordered: list[tuple[int, str]] = []
    seen: set[int] = set()
    names = _process_name_map(device, _pids_from_adb_pidof(app_package))
    preferred = _main_pid_from_adb(app_package, device=device, app_name=app_name)
    pidof_list = _pids_from_adb_pidof(app_package)
    if preferred and preferred in pidof_list:
        pidof_list = [preferred] + [p for p in pidof_list if p != preferred]
    for pid in pidof_list:
        if pid not in seen:
            ordered.append((pid, names.get(pid, app_package)))
            seen.add(pid)
    for score, pid, label in _resolve_attach_candidates(device, app_package, app_name):
        if pid not in seen:
            ordered.append((pid, label))
            seen.add(pid)
    return ordered


def _attach_session(device: Any, app_package: str, app_name: str = "") -> Any:
    """Attach to a PID that accepts frida-agent and exposes Java (MuMu-safe)."""
    errors: list[str] = []
    targets = _ordered_attach_pids(device, app_package, app_name)
    if not targets:
        raise RuntimeError(
            f"no process found for package={app_package!r} app_name={app_name!r}; "
            f"open the app and keep it in foreground before stage3"
        )
    logger.info("Frida attach order: %s", targets[:8])
    display = (app_name or "").strip()
    for pid, label in targets:
        for attempt in range(1, 4):
            session = None
            try:
                session = device.attach(pid, persist_timeout=30)
                # MuMu: parent process is often labeled "Keep"; ART warms up after attach.
                if display and label == display:
                    logger.info(
                        "Frida attached pid=%s label=%s attempt=%d "
                        "(MuMu parent; hooks will wait for Java)",
                        pid,
                        label,
                        attempt,
                    )
                    return session
                java_timeout = 25.0 if label == app_package else 20.0
                if _verify_java_session(session, timeout_sec=java_timeout):
                    logger.info(
                        "Frida attached pid=%s label=%s attempt=%d (java_ready)",
                        pid,
                        label,
                        attempt,
                    )
                    return session
                errors.append(f"pid({pid},{label})#{attempt}: java_not_ready")
                logger.warning(
                    "Frida attached pid=%s (%s) but Java not ready; trying next",
                    pid,
                    label,
                )
                session.detach()
                continue
            except Exception as exc:
                err = f"pid({pid},{label})#{attempt}: {exc}"
                errors.append(err)
                logger.warning("Frida attach failed %s", err)
                if session is not None:
                    try:
                        session.detach()
                    except Exception:
                        pass
                if attempt < 3:
                    time.sleep(2)
    detail = "; ".join(errors)
    raise RuntimeError(
        f"unable to attach with Java to {app_package!r} (app_name={app_name!r}): {detail}. "
        f"On MuMu, keep {app_name or app_package} in foreground; avoid attaching only "
        f"the child process named {app_package!r} if injection is refused."
    )


def run_frida_hooks(
    app_package: str,
    candidate_pii: list[str],
    taxonomy: dict[str, dict[str, list[str]]],
    hooks_dir: Path,
    duration_sec: int = 8,
    dry_run: bool = True,
    launch_mode: str = "attach",
    app_name: str = "",
) -> dict[str, Any]:
    """Attach Frida hooks and collect runtime events."""
    registry = build_hook_registry(candidate_pii, taxonomy, hooks_dir)
    logger.info(
        "Frida hooks start package=%s mode=%s dry_run=%s duration_sec=%d hook_types=%d",
        app_package,
        launch_mode,
        dry_run,
        duration_sec,
        len(registry),
    )
    if dry_run:
        return {
            "runner": "frida",
            "mode": "dry_run",
            "launch_mode": launch_mode,
            "package": app_package,
            "registered_hooks": {k: [str(p) for p in v] for k, v in registry.items()},
            "events": _simulate_events(registry),
            "error": "",
        }

    try:
        import frida  # type: ignore
    except Exception as exc:
        logger.error("Frida unavailable: %s", exc)
        return {
            "runner": "frida",
            "mode": "real",
            "launch_mode": launch_mode,
            "package": app_package,
            "registered_hooks": {k: [str(p) for p in v] for k, v in registry.items()},
            "events": [],
            "error": f"Frida unavailable: {exc}",
        }

    if launch_mode not in {"attach", "spawn"}:
        launch_mode = "attach"

    events: list[dict[str, Any]] = []
    scripts: list[Any] = []
    error = ""
    session = None
    pid: int | None = None
    device: Any = None

    def on_message(message: dict, _data: bytes | None) -> None:
        payload = message.get("payload", {})
        if isinstance(payload, dict):
            events.append(
                {
                    "timestamp_utc": _utc_now_iso(),
                    "pii_type": payload.get("pii_type", "unknown"),
                    "hook_file": payload.get("hook_file", ""),
                    "api_name": payload.get("api_name", ""),
                    "args_summary": payload.get("args_summary", ""),
                    "stack_summary": payload.get("stack_summary", ""),
                }
            )

    try:
        logger.info("Connecting Frida USB device (timeout=8s)...")
        device = frida.get_usb_device(timeout=8)
        logger.info("Frida device connected: %s", getattr(device, "id", device))
        if launch_mode == "spawn":
            logger.info("Spawning package=%s", app_package)
            pid = device.spawn([app_package])
            session = device.attach(pid)
            should_resume = True
        else:
            logger.info(
                "Attaching package=%s app_name=%s (keep app in foreground)",
                app_package,
                app_name or "(none)",
            )
            session = _attach_session(device, app_package, app_name=app_name)
            should_resume = False
            logger.info("Frida attach OK")
        for pii_type, hook_paths in registry.items():
            for hook_path in hook_paths:
                logger.info("Loading hook pii_type=%s file=%s", pii_type, hook_path.name)
                script_text = wrap_hook_script(hook_path.read_text(encoding="utf-8"))
                script = session.create_script(script_text)
                script.on("message", on_message)
                script.load()
                scripts.append(script)
                events.append(
                    {
                        "timestamp_utc": _utc_now_iso(),
                        "pii_type": pii_type,
                        "hook_file": str(hook_path),
                        "api_name": "hook_loaded",
                        "args_summary": "",
                        "stack_summary": "",
                    }
                )
        if should_resume and pid is not None:
            logger.info("Resuming spawned pid=%s", pid)
            device.resume(pid)
        capture_sec = max(1, int(duration_sec))
        logger.info("Collecting Frida events for %ds (operate the app to trigger APIs)...", capture_sec)
        log_countdown(logger, "Frida capture", capture_sec, interval_sec=5)
    except Exception as exc:  # pragma: no cover - runtime dependent
        if launch_mode == "attach":
            error = (
                f"Frida attach failed: {exc}. "
                f"Open and log into {app_package} first, keep it in foreground, then retry."
            )
        else:
            error = f"Frida runtime failed: {exc}"
        logger.error("%s", error)
    finally:
        logger.info("Detaching Frida session scripts=%d events=%d", len(scripts), len(events))
        for script in scripts:
            try:
                script.unload()
            except Exception:
                pass
        if session is not None:
            try:
                session.detach()
            except Exception:
                pass

    runtime_events = [
        e for e in events if str(e.get("api_name", "")) not in {"", "hook_loaded", "hook_error"}
    ]
    logger.info(
        "Frida finished package=%s hook_loaded=%d runtime_events=%d error=%s",
        app_package,
        len(events),
        len(runtime_events),
        error or "(none)",
    )
    return {
        "runner": "frida",
        "mode": "real",
        "launch_mode": launch_mode,
        "package": app_package,
        "registered_hooks": {k: [str(p) for p in v] for k, v in registry.items()},
        "events": events,
        "error": error,
        "sync_mode": False,
        "runtime_event_count": len(runtime_events),
    }
