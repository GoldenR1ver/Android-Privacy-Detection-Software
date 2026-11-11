"""Frida execution layer with hook auto-registration."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Extra ART class-prefix hints when Android package != applicationId / obfuscated brand.
_PACKAGE_CODE_PREFIX_HINTS: dict[str, list[str]] = {
    "com.lvsongguo": ["com.xiaohe", "com.bytedance.health"],
    "com.gotokeep.keep": ["com.gotokeep"],
}

JAVA_PROBE_ATTEMPTS = 56  # 56 * 250ms ~= 14s waiting for Java.available
JAVA_PROBE_INTERVAL_MS = 250
JAVA_CLASS_SCAN_BUDGET_MS = 4000
JAVA_VERIFY_TIMEOUT_SEC = 18.0
PRE_ATTACH_WARMUP_SEC = 5.0
ATTACH_ROUNDS = 2  # re-resolve PIDs + relaunch after process exit during probe

from privacy_audit.analyzers.dynamic.app_launch import adb_launch_package
from privacy_audit.analyzers.dynamic.frida_reactor import Reactor
from privacy_audit.analyzers.dynamic.hook_loader import (
    is_frida_runtime_api,
    papi_agent_path,
    resolve_global_hooks,
    wrap_hook_script,
)
from privacy_audit.utils.progress_log import log_countdown

logger = logging.getLogger("privacy_audit.frida")


def _install_papi_hooks(script: Any, spec: list[dict[str, str]]) -> int:
    """Call the agent's installhooks RPC export (frida 16+ uses exports_sync)."""
    exports = getattr(script, "exports_sync", None) or getattr(script, "exports", None)
    if exports is None:
        raise RuntimeError("Frida script exposes no RPC exports")
    return int(exports.installhooks(spec))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_hook_registry(
    candidate_pii: list[str],
    taxonomy: dict[str, Any],
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


def build_papi_spec(
    candidate_pii: list[str],
    taxonomy: dict[str, Any],
) -> list[dict[str, str]]:
    """Build the data-driven hook_papi.js spec from taxonomy ``api_targets``.

    Each spec entry is one of::

        {"pii_type": ..., "category": ..., "clazz": ..., "method": ...}
        {"pii_type": ..., "category": ..., "enumerate": "pkg.Class!method*"}
    """
    spec: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pii_type in candidate_pii:
        meta = taxonomy.get(pii_type, {})
        for target in meta.get("api_targets", []) or []:
            if not isinstance(target, dict):
                continue
            if target.get("enumerate"):
                key = ("enum", str(target["enumerate"]))
                if key in seen:
                    continue
                seen.add(key)
                spec.append(
                    {
                        "pii_type": pii_type,
                        "category": pii_type,
                        "enumerate": str(target["enumerate"]),
                    }
                )
            elif target.get("clazz") and target.get("method"):
                key = (str(target["clazz"]), str(target["method"]))
                if key in seen:
                    continue
                seen.add(key)
                spec.append(
                    {
                        "pii_type": pii_type,
                        "category": pii_type,
                        "clazz": str(target["clazz"]),
                        "method": str(target["method"]),
                    }
                )
    return spec


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


def resolve_code_prefixes(app_package: str, extra: list[str] | None = None) -> list[str]:
    """Build ordered unique class-prefix list for app_ready probing."""
    ordered: list[str] = []
    seen: set[str] = set()
    for prefix in [app_package, *(extra or []), *_PACKAGE_CODE_PREFIX_HINTS.get(app_package, [])]:
        token = (prefix or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def parse_app_package_entry(value: Any) -> tuple[str, list[str], dict[str, Any]]:
    """Parse one app_packages.json entry -> (package, code_prefixes, frida_options)."""
    if isinstance(value, str):
        package = value.strip()
        return package, resolve_code_prefixes(package), {}
    if isinstance(value, dict):
        package = str(value.get("package", "")).strip()
        if not package:
            raise ValueError("app_packages entry dict missing non-empty 'package'")
        extras = [str(p).strip() for p in (value.get("code_prefixes") or []) if str(p).strip()]
        options = {
            k: v
            for k, v in value.items()
            if k not in {"package", "code_prefixes"}
        }
        return package, resolve_code_prefixes(package, extras), options
    raise ValueError(f"unsupported app_packages entry type: {type(value).__name__}")


def load_app_frida_config(app_name: str, packages_file: Path) -> tuple[str, list[str], dict[str, Any]]:
    """Load Android package, code_prefixes, and Frida options for app_ready probing."""
    if not packages_file.exists():
        raise FileNotFoundError(f"App package map not found: {packages_file}")
    data = json.loads(packages_file.read_text(encoding="utf-8"))
    if app_name not in data:
        known = ", ".join(sorted(data.keys())[:12])
        raise KeyError(f"Unknown app {app_name!r} in {packages_file}. Known: {known}...")
    return parse_app_package_entry(data[app_name])


def load_frida_tier_file(tier_file: Path) -> dict[str, Any]:
    """Load ``input/frida_tier.json`` (returns empty dict if missing)."""
    if not tier_file.exists():
        return {}
    data = json.loads(tier_file.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def get_app_frida_tier_entry(app_name: str, tier_file: Path) -> dict[str, Any]:
    """Merge per-app tier entry with file-level defaults."""
    data = load_frida_tier_file(tier_file)
    defaults = dict(data.get("defaults") or {})
    apps = data.get("apps") or {}
    entry = dict(defaults)
    if isinstance(apps.get(app_name), dict):
        entry.update(apps[app_name])
    return entry


def resolve_frida_launch_mode(
    app_name: str,
    tier_file: Path | None,
    cli_mode: str = "auto",
) -> str:
    """Resolve attach vs spawn from CLI override or frida_tier.json."""
    mode = (cli_mode or "auto").strip().lower()
    if mode in {"attach", "spawn"}:
        return mode
    if tier_file is not None and tier_file.exists():
        preferred = str(
            get_app_frida_tier_entry(app_name, tier_file).get("preferred_launch_mode", "attach")
        ).strip().lower()
        if preferred in {"attach", "spawn"}:
            return preferred
    return "attach"


def should_enable_frida_for_consent(
    app_name: str,
    tier_file: Path | None,
    *,
    cli_enable: bool | None = None,
    cli_disable: bool = False,
) -> bool:
    """Whether consent batch should pass --enable-frida (supplementary to mitm-primary)."""
    if cli_disable:
        return False
    if cli_enable is True:
        return True
    # Consent audit defaults to mitm-primary; Frida only when explicitly requested.
    return False


def resolve_frida_capture_seconds(
    app_name: str,
    tier_file: Path | None,
    *,
    grace_default: int,
    tail_default: int,
) -> tuple[int, int]:
    """Per-app grace/tail from tier file with CLI defaults as fallback."""
    grace, tail = grace_default, tail_default
    if tier_file is not None and tier_file.exists():
        entry = get_app_frida_tier_entry(app_name, tier_file)
        if entry.get("frida_grace_sec") is not None:
            grace = int(entry["frida_grace_sec"])
        if entry.get("frida_tail_sec") is not None:
            tail = int(entry["frida_tail_sec"])
    return max(0, grace), max(0, tail)


def spawn_package_cold(device: Any, app_package: str, *, grant_permissions: bool = True) -> int:
    """Force-stop, optionally grant runtime perms, then spawn paused."""
    from privacy_audit.analyzers.dynamic.app_launch import (
        adb_force_stop_package,
        adb_grant_common_permissions,
    )

    adb_force_stop_package(app_package)
    time.sleep(0.4)
    if grant_permissions:
        adb_grant_common_permissions(app_package)
    pid = int(device.spawn([app_package]))
    logger.info("Frida cold spawn package=%s pid=%s (paused)", app_package, pid)
    return pid


def _pids_from_adb_pidof(app_package: str) -> list[int]:
    from privacy_audit.utils.adb_device import adb_run

    try:
        proc = adb_run(["shell", "pidof", app_package], timeout=8)
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


def _app_ready_probe(code_prefixes: list[str]) -> str:
    """Lightweight JS probe: wait for Java, then async class-prefix scan (capped).

    Avoids ``enumerateLoadedClassesSync()`` over the full classpath — that can
    take minutes and triggers anti-frida kills on apps like 小荷健康.
    """
    prefixes_json = json.dumps(code_prefixes or [])
    return (
        r"""
(function () {
  var attempts = 0;
  var prefixes = %s;
  var maxAttempts = %d;
  var intervalMs = %d;
  var classScanBudgetMs = %d;
  var sent = false;
  function sendOnce(apiName) {
    if (sent) { return; }
    sent = true;
    send({ api_name: apiName, pii_type: "frida", hook_file: "probe" });
  }
  function classMatchesPrefix(className) {
    for (var j = 0; j < prefixes.length; j++) {
      if (className.indexOf(prefixes[j]) === 0) { return true; }
    }
    return false;
  }
  function scanPrefixes() {
    if (!prefixes.length) {
      sendOnce("app_ready");
      return;
    }
    var found = false;
    Java.perform(function () {
      try {
        Java.enumerateLoadedClasses({
          onMatch: function (name) {
            if (found) { return; }
            if (classMatchesPrefix(name)) {
              found = true;
              sendOnce("app_ready");
            }
          },
          onComplete: function () {
            if (!found) { sendOnce("java_only"); }
          }
        });
      } catch (e) {
        sendOnce("java_only");
      }
    });
    setTimeout(function () {
      if (!sent) { sendOnce("java_only"); }
    }, classScanBudgetMs);
  }
  function waitForJava() {
    attempts += 1;
    if (typeof Java === "undefined" || !Java.available) {
      if (attempts < maxAttempts) { setTimeout(waitForJava, intervalMs); }
      else { sendOnce("java_timeout"); }
      return;
    }
    scanPrefixes();
  }
  waitForJava();
})();
"""
        % (
            prefixes_json,
            JAVA_PROBE_ATTEMPTS,
            JAVA_PROBE_INTERVAL_MS,
            JAVA_CLASS_SCAN_BUDGET_MS,
        )
    )


_PROBE_TERMINAL_STATES = frozenset({"app_ready", "java_only", "java_timeout"})


def _is_display_name_label(label: str, app_name: str, app_package: str) -> bool:
    """MuMu often exposes the launcher label (e.g. 小荷健康) instead of the package."""
    display = (app_name or "").strip()
    return bool(display) and label == display and label != app_package


def _is_session_gone_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "session is gone" in text or "unable to find process" in text


def _verify_java_session(
    session: Any,
    code_prefixes: list[str] | None = None,
    timeout_sec: float = JAVA_VERIFY_TIMEOUT_SEC,
) -> str:
    """Probe the attached process.

    Returns ``app_ready`` | ``java_only`` | ``java_timeout`` | ``not_ready``.
    """
    import threading

    prefixes = code_prefixes or []
    ready = threading.Event()
    state = "not_ready"

    def on_message(message: dict, _data: bytes | None) -> None:
        nonlocal state
        payload = message.get("payload", {})
        if isinstance(payload, dict):
            api = payload.get("api_name")
            if api in _PROBE_TERMINAL_STATES:
                state = str(api)
                ready.set()

    script = session.create_script(_app_ready_probe(prefixes))
    script.on("message", on_message)
    script.load()
    ready.wait(timeout=timeout_sec)
    try:
        script.unload()
    except Exception:
        pass
    return state


def _accept_java_probe_state(
    state: str,
    label: str,
    app_name: str,
    app_package: str,
    *,
    accept_java_timeout_on_display: bool = True,
) -> bool:
    """Whether to proceed with hook install for this attach candidate."""
    if state == "app_ready":
        return True
    if state == "java_only" and _is_display_name_label(label, app_name, app_package):
        return True
    if (
        state == "java_timeout"
        and accept_java_timeout_on_display
        and _is_display_name_label(label, app_name, app_package)
    ):
        return True
    return False


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


def _attach_session(
    device: Any,
    app_package: str,
    app_name: str = "",
    code_prefixes: list[str] | None = None,
    *,
    pre_warm_sec: float = PRE_ATTACH_WARMUP_SEC,
    accept_java_timeout_on_display: bool = True,
) -> Any:
    """Attach to a PID that accepts frida-agent and exposes Java (MuMu-safe)."""
    prefixes = code_prefixes or resolve_code_prefixes(app_package)
    if pre_warm_sec > 0:
        logger.info(
            "Frida pre-attach warm-up %.0fs (prefixes=%s)",
            pre_warm_sec,
            prefixes[:5],
        )
        time.sleep(pre_warm_sec)
    errors: list[str] = []
    targets = _ordered_attach_pids(device, app_package, app_name)
    if not targets:
        raise RuntimeError(
            f"no process found for package={app_package!r} app_name={app_name!r}; "
            f"open the app and keep it in foreground before dynamic capture"
        )
    logger.info("Frida attach order: %s", targets[:8])
    fallback_session: Any = None  # Java present but app classes unconfirmed
    fallback_label = ""

    def _drop(sess: Any) -> None:
        if sess is not None:
            try:
                sess.detach()
            except Exception:
                pass

    for round_idx in range(ATTACH_ROUNDS):
        if round_idx > 0:
            logger.info(
                "Frida attach round %d/%d: relaunch %s and refresh PIDs",
                round_idx + 1,
                ATTACH_ROUNDS,
                app_package,
            )
            adb_launch_package(app_package)
            time.sleep(3)
            targets = _ordered_attach_pids(device, app_package, app_name)
            if not targets:
                break
            logger.info("Frida attach order (refreshed): %s", targets[:8])
            errors.clear()
            fallback_session = None
            fallback_label = ""

        for pid, label in targets:
            for attempt in range(1, 3):
                session = None
                try:
                    session = device.attach(pid, persist_timeout=30)
                    state = _verify_java_session(session, prefixes)
                    if _accept_java_probe_state(
                        state,
                        label,
                        app_name,
                        app_package,
                        accept_java_timeout_on_display=accept_java_timeout_on_display,
                    ):
                        if fallback_session is not session:
                            _drop(fallback_session)
                        logger.info(
                            "Frida attached pid=%s label=%s round=%d attempt=%d "
                            "probe=%s prefixes=%s",
                            pid,
                            label,
                            round_idx + 1,
                            attempt,
                            state,
                            prefixes[:4],
                        )
                        return session
                    if state == "java_only":
                        if fallback_session is None:
                            fallback_session = session
                            fallback_label = label
                            logger.info(
                                "Frida pid=%s label=%s probe=java_only (prefixes=%s); "
                                "keeping as fallback",
                                pid,
                                label,
                                prefixes[:4],
                            )
                        else:
                            _drop(session)
                        break
                    if state == "java_timeout":
                        main_pid = _main_pid_from_adb(
                            app_package, device=device, app_name=app_name
                        )
                        if fallback_session is None and main_pid is not None and pid == main_pid:
                            fallback_session = session
                            fallback_label = label
                            logger.info(
                                "Frida pid=%s label=%s probe=java_timeout on main PID; "
                                "keeping session for hook install (attach may still work "
                                "after scripts load)",
                                pid,
                                label,
                            )
                        else:
                            _drop(session)
                        break
                    errors.append(f"pid({pid},{label})#r{round_idx + 1}a{attempt}: {state}")
                    logger.warning(
                        "Frida attached pid=%s (%s) probe=%s; trying next",
                        pid,
                        label,
                        state,
                    )
                    _drop(session)
                    break
                except Exception as exc:
                    err = f"pid({pid},{label})#r{round_idx + 1}a{attempt}: {exc}"
                    errors.append(err)
                    logger.warning("Frida attach failed %s", err)
                    if session is not None:
                        _drop(session)
                    if _is_session_gone_error(exc):
                        break
                    if attempt < 2:
                        time.sleep(2)
    if fallback_session is not None:
        logger.warning(
            "Java probe inconclusive for %s; using attach fallback (label=%s). "
            "Proceeding to hook install — prefer spawn mode if hooks stay silent.",
            app_package,
            fallback_label,
        )
        return fallback_session
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
    code_prefixes: list[str] | None = None,
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
    papi_spec = build_papi_spec(candidate_pii, taxonomy)
    global_hooks = resolve_global_hooks(hooks_dir)
    if dry_run:
        sim_events = _simulate_events(registry)
        for hook in global_hooks:
            sim_events.append(
                {
                    "timestamp_utc": _utc_now_iso(),
                    "pii_type": "network_transmission",
                    "hook_file": str(hook),
                    "api_name": "OkHttp.execute",
                    "args_summary": "dry_run_network",
                    "stack_summary": "",
                }
            )
        for entry in papi_spec[:8]:
            sim_events.append(
                {
                    "timestamp_utc": _utc_now_iso(),
                    "pii_type": entry.get("pii_type", "unknown"),
                    "hook_file": "hook_papi.js",
                    "api_name": entry.get("clazz", entry.get("enumerate", "papi")) + "."
                    + entry.get("method", "*"),
                    "args_summary": "dry_run_papi",
                    "stack_summary": "",
                }
            )
        return {
            "runner": "frida",
            "mode": "dry_run",
            "launch_mode": launch_mode,
            "package": app_package,
            "registered_hooks": {k: [str(p) for p in v] for k, v in registry.items()},
            "global_hooks": [str(p) for p in global_hooks],
            "papi_target_count": len(papi_spec),
            "events": sim_events,
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
    events_lock = threading.Lock()
    scripts: list[Any] = []
    error = ""
    session = None
    pid: int | None = None
    device: Any = None
    papi_installed = 0
    reactor = Reactor()

    def append_event(payload: dict[str, Any]) -> None:
        with events_lock:
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

    def on_message(message: dict, _data: bytes | None) -> None:
        payload = message.get("payload", {})
        if isinstance(payload, dict):
            reactor.schedule(lambda: append_event(payload))

    def load_wrapped(hook_path: Path, pii_type: str) -> None:
        logger.info("Loading hook pii_type=%s file=%s", pii_type, hook_path.name)
        script = session.create_script(
            wrap_hook_script(hook_path.read_text(encoding="utf-8"), hook_file=str(hook_path))
        )
        script.on("message", on_message)
        script.load()
        scripts.append(script)
        append_event(
            {
                "pii_type": pii_type,
                "hook_file": str(hook_path),
                "api_name": "hook_loaded",
                "args_summary": "",
                "stack_summary": "",
            }
        )

    try:
        reactor.start()
        logger.info("Connecting Frida USB device (timeout=8s)...")
        device = frida.get_usb_device(timeout=8)
        logger.info("Frida device connected: %s", getattr(device, "id", device))
        if launch_mode == "spawn":
            logger.info("Cold spawning package=%s (paused for gated hook install)", app_package)
            pid = spawn_package_cold(device, app_package)
            session = device.attach(pid)
            should_resume = True
        else:
            logger.info(
                "Attaching package=%s app_name=%s (keep app in foreground)",
                app_package,
                app_name or "(none)",
            )
            session = _attach_session(
                device,
                app_package,
                app_name=app_name,
                code_prefixes=code_prefixes,
            )
            should_resume = False
            logger.info("Frida attach OK")
        for pii_type, hook_paths in registry.items():
            for hook_path in hook_paths:
                load_wrapped(hook_path, pii_type)
        for hook_path in global_hooks:
            load_wrapped(hook_path, "network_transmission")
        agent = papi_agent_path(hooks_dir)
        if agent is not None and papi_spec:
            logger.info("Loading data-driven agent %s (%d targets)", agent.name, len(papi_spec))
            papi_script = session.create_script(agent.read_text(encoding="utf-8"))
            papi_script.on("message", on_message)
            papi_script.load()
            scripts.append(papi_script)
            try:
                papi_installed = _install_papi_hooks(papi_script, papi_spec)
            except Exception as exc:
                logger.warning("papi installhooks failed: %s", exc)
        # Spawn: resume first, then wait for Java/papi (paused process has no ART yet).
        if should_resume and pid is not None:
            logger.info("Resuming spawned pid=%s (then wait for Java/papi)", pid)
            device.resume(pid)
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                with events_lock:
                    api_names = {str(e.get("api_name", "")) for e in events}
                if "papi_ready" in api_names or (
                    "java_active" in api_names and not papi_spec
                ):
                    break
                time.sleep(0.15)
            else:
                logger.warning(
                    "Spawn post-resume: java_active/papi_ready not seen within 30s",
                )
        elif papi_spec or registry:
            deadline = time.monotonic() + 18.0
            while time.monotonic() < deadline:
                with events_lock:
                    api_names = {str(e.get("api_name", "")) for e in events}
                if "papi_ready" in api_names or (
                    "java_active" in api_names and not papi_spec
                ):
                    break
                time.sleep(0.15)
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
        reactor.stop()

    with events_lock:
        events_snapshot = list(events)
    runtime_events = [
        e for e in events_snapshot if is_frida_runtime_api(str(e.get("api_name", "")))
    ]
    logger.info(
        "Frida finished package=%s total_events=%d runtime_events=%d papi_targets=%d error=%s",
        app_package,
        len(events_snapshot),
        len(runtime_events),
        len(papi_spec),
        error or "(none)",
    )
    return {
        "runner": "frida",
        "mode": "real",
        "launch_mode": launch_mode,
        "package": app_package,
        "registered_hooks": {k: [str(p) for p in v] for k, v in registry.items()},
        "global_hooks": [str(p) for p in global_hooks],
        "papi_target_count": len(papi_spec),
        "papi_installed_accepted": papi_installed,
        "events": events_snapshot,
        "error": error,
        "sync_mode": False,
        "runtime_event_count": len(runtime_events),
    }

