"""ADB helpers to launch an app and verify it is in the foreground."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from privacy_audit.utils.adb_device import adb_run

logger = logging.getLogger("privacy_audit.appium")

# Packages where Appium must not auto-resolve launcher activity at session start.
PACKAGES_SKIP_APPIUM_AUTO_LAUNCH: frozenset[str] = frozenset(
    {
        "com.lvsongguo",  # 小荷健康
    }
)

_FOCUS_PKG_RE = re.compile(r"mCurrentFocus=Window\{[^}]+\s+u\d+\s+([^\s/]+)/")
_RESUMED_PKG_RE = re.compile(r"mResumedActivity:.*?([a-zA-Z0-9_.]+)/")

# Runtime permissions to pre-grant so feature use actually reads the PII (and
# Frida hooks fire). Granting avoids the app short-circuiting on a denied check.
FUNCTION_TRIGGER_PERMISSIONS: tuple[str, ...] = (
    "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.ACCESS_COARSE_LOCATION",
    "android.permission.READ_PHONE_STATE",
    "android.permission.READ_CONTACTS",
)


def _adb_shell(
    args: list[str],
    *,
    timeout: float = 12.0,
    adb_serial: str | None = None,
) -> str:
    try:
        proc = adb_run(["shell", *args], timeout=timeout, serial=adb_serial)
    except Exception as exc:
        logger.debug("adb shell %s failed: %s", args[:3], exc)
        return ""
    if proc.returncode != 0:
        logger.debug("adb shell rc=%s stderr=%s", proc.returncode, (proc.stderr or "")[:200])
    return (proc.stdout or "") + (proc.stderr or "")


def adb_foreground_package() -> str:
    """Best-effort current foreground package from dumpsys."""
    for args in (
        ["dumpsys", "window", "displays"],
        ["dumpsys", "activity", "activities"],
    ):
        text = _adb_shell(args)
        if not text:
            continue
        for line in text.splitlines():
            if "mCurrentFocus" in line:
                match = _FOCUS_PKG_RE.search(line)
                if match:
                    return match.group(1).strip()
            if "mResumedActivity" in line:
                match = _RESUMED_PKG_RE.search(line)
                if match:
                    return match.group(1).strip()
    return ""


def adb_force_stop_package(
    app_package: str,
    *,
    adb_serial: str | None = None,
) -> bool:
    """Force-stop app so spawn mode gets a clean cold start."""
    if not app_package:
        return False
    out = _adb_shell(["am", "force-stop", app_package], adb_serial=adb_serial)
    ok = "Error" not in out and "Exception" not in out
    if ok:
        logger.info("adb force-stop ok package=%s", app_package)
    else:
        logger.debug("adb force-stop package=%s out=%s", app_package, out[:120])
    return ok


def adb_close_after_dynamic_verification(
    app_package: str,
    *,
    adb_serial: str | None = None,
) -> bool:
    """Force-stop the target app after Stage3 so batch runs do not leak foreground state."""
    if not app_package:
        return False
    ok = adb_force_stop_package(app_package, adb_serial=adb_serial)
    if ok:
        logger.info("post_capture close ok package=%s", app_package)
    else:
        logger.warning("post_capture close failed package=%s", app_package)
    return ok


def adb_launch_package(app_package: str, *, wait_sec: float = 1.8) -> bool:
    """Bring app to foreground via launcher intent (does not require Appium session)."""
    if not app_package:
        return False
    commands = [
        [
            "monkey",
            "-p",
            app_package,
            "-c",
            "android.intent.category.LAUNCHER",
            "1",
        ],
        [
            "am",
            "start",
            "-a",
            "android.intent.action.MAIN",
            "-c",
            "android.intent.category.LAUNCHER",
            "-p",
            app_package,
        ],
    ]
    for cmd in commands:
        out = _adb_shell(cmd)
        if "Error" in out and "does not exist" in out:
            continue
        time.sleep(wait_sec)
        fg = adb_foreground_package()
        if fg == app_package:
            logger.info("adb_launch ok package=%s (foreground confirmed)", app_package)
            return True
        logger.info(
            "adb_launch tried %s; foreground=%s (expected %s)",
            " ".join(cmd[:4]),
            fg or "(unknown)",
            app_package,
        )
    fg = adb_foreground_package()
    if fg == app_package:
        return True
    logger.warning(
        "adb_launch could not confirm foreground for %s (saw %s)",
        app_package,
        fg or "(unknown)",
    )
    return False


def adb_grant_common_permissions(
    app_package: str,
    *,
    permissions: tuple[str, ...] = FUNCTION_TRIGGER_PERMISSIONS,
    steps: list[str] | None = None,
) -> dict[str, bool]:
    """Best-effort ``pm grant`` of runtime permissions before function triggers.

    Many apps gate a real PII read behind a granted permission; without the
    grant the feature path returns early and the Frida hook never fires. Grant
    failures are non-fatal (permission may not be declared / already fixed).
    """
    results: dict[str, bool] = {}
    if not app_package:
        return results
    for perm in permissions:
        out = _adb_shell(["pm", "grant", app_package, perm])
        ok = "Error" not in out and "Exception" not in out and "not a changeable" not in out
        results[perm] = ok
        if steps is not None:
            steps.append(f"grant[{perm.split('.')[-1]}]={'ok' if ok else 'skip'}")
    logger.info(
        "adb grant permissions package=%s granted=%d/%d",
        app_package,
        sum(1 for v in results.values() if v),
        len(results),
    )
    return results


def ensure_target_app_foreground(
    app_package: str,
    *,
    steps: list[str] | None = None,
    driver: Any | None = None,
    max_attempts: int = 3,
) -> bool:
    """
    Ensure the target package is foreground before / during Appium automation.

    Uses adb launch first, then optional Appium activate_app.
    """
    if not app_package:
        return False
    log_steps = steps if steps is not None else []
    for attempt in range(1, max_attempts + 1):
        fg = adb_foreground_package()
        if fg == app_package:
            log_steps.append(f"foreground_ok[{app_package}]=attempt{attempt}")
            return True
        log_steps.append(
            f"foreground_launch_attempt{attempt}:was={fg or 'unknown'}:target={app_package}"
        )
        adb_launch_package(app_package)
        if driver is not None:
            try:
                driver.activate_app(app_package)
                time.sleep(1.0)
            except Exception as exc:
                log_steps.append(f"activate_app_attempt{attempt}=fail({exc})")
        fg = adb_foreground_package()
        current = ""
        if driver is not None:
            try:
                current = str(getattr(driver, "current_package", "") or "")
            except Exception:
                current = ""
        if fg == app_package or current == app_package:
            log_steps.append(f"foreground_ok[{app_package}]=post_activate{attempt}")
            return True
        time.sleep(0.8)
    log_steps.append(f"foreground_fail[{app_package}]")
    return False
